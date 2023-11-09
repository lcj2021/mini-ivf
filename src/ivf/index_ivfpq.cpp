#include <index_ivfpq.hpp>
#include <omp.h>
#include <cassert>
#include <iostream>
#include <timer.hpp>
#include <distance.hpp>


extern "C" {
    void omp_init_lock(omp_lock_t *);
    void omp_set_lock(omp_lock_t *);
    void omp_unset_lock(omp_lock_t *);
    void omp_destroy_lock(omp_lock_t *);
};


namespace index {

namespace ivf {


DistanceTable::DistanceTable(size_t M, size_t Ks): kp_(Ks), data_(M * Ks) {}


inline void DistanceTable::SetValue(size_t m, size_t ks, float val)
{
    data_[m * kp_ + ks] = val;
}


inline distance_t DistanceTable::GetValue(size_t m, size_t ks) const
{
    return data_[m * kp_ + ks];
}


template <typename vector_dimension_t> 
IndexIVFPQ<vector_dimension_t>::IndexIVFPQ(
    size_t N, size_t D, size_t L, size_t kc, size_t kp, size_t mc, size_t mp, size_t dc, size_t dp, // index config
    const std::string & index_path, 
    const std::string & db_path,
    const std::string & name = "",
    IndexStatus status = IndexStatus::LOCAL
): N_(N), D_(D), L_(L), kc_(kc), kp_(kp), mc_(mc), mp_(mp), dc_(dc), dp_(dp),
index_path_(index_path), db_path_(db_path), name_(name), status_(status), is_trained_(false), nsamples_(0), seed_(-1)
{
    assert( dc_ == D_ && mc_ == 1 ); // CQ is for all vector

    cq_ = nullptr;
    pq_ = nullptr;

    std::cout << "SIMD support: " << g_simd_architecture << std::endl;

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::Populate(const std::vector<vector_dimension_t> & raw_data)
{
    postint_lists_.clear();
    postint_lists_.resize(kc_);
    segments_.clear();
    segments_.resize(kc_);

    const size_t N = raw_data.size() / D_;

    for (auto & postint_list: postint_lists_)
    {
        postint_list.reserve(N / kc_); // Roughly malloc
    }
    for (auto & segment: segments_)
    {
        segment.reserve(N / kc_); // Roughly malloc
    }

    InsertIvf(raw_data);
}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::Train(const std::vector<vector_dimension_t> & raw_data)
{
    assert( nsamples_ && seed_ != -1 && "nsample != 0 and seed != -1" );
    const Nt = raw_data.size() / D_;
    const size_t min_nsamples = 100000;
    const size_t cq_train_times = 12;
    const size_t pq_train_times = 6;
    
    if (nsamples_ < min_nsamples) {
        nsamples_ = min_nsamples;
        std::cout << "nsamples_ < min_nsamples, set nsamples_ to " << nsamples_ << std::endl;
    }
    if (nsamples_ > Nt) {
        nsamples_ = Nt;
        std::cout << "nsamples_ > Nt, set nsamples_ to " << nsamples_ << std::endl;
    }
    if (nsamples_ > N_) {
        nsamples_ = N_;
        std::cout << "nsamples_ > N_, set nsamples_ to " << nsamples_ << std::endl;
    }

    std::vector<vector_id_t> ids(Nt)
    std::vector<vector_id_t> ids(N_);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 default_random_engine(seed_);
    std::shuffle(ids.begin(), ids.end(), default_random_engine);

    std::unique_ptr<std::vector<vector_dimension_t>> traindata = std::make_unique<std::vector<vector_dimension_t>>();
    traindata->reserve(nsamples_ * D_);

    for (size_t k = 0; k < nsamples_; k++)
    {
        const size_t id = ids[k];
        traindata->insert(traindata->end(), raw_data.begin() + id * D_, raw_data.begin() + (id + 1) * D_);
    }

    cq_ = std::make_unique<::Quantizer<vector_dimension_t>>(D_, nsamples_, mc_, kc_);
    cq_->Fit(*traindata, cq_train_times, seed_);
    centers_cq_ = cq_->GetCentroids()[0];
    labels_cq_ = cq_->GetAssignments()[0];

    pq_ = std::make_unique<::Quantizer<vector_dimension_t>>(D_, nsamples_, mp_, kp_);
    pq_->Fit(*traindata, pq_train_times, seed_);
    centers_pq_ = pq_->GetCentroids();
    labels_pq_ = pq_->GetAssignments();

    is_trained_ = true;
}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::LoadIndex()
{

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::WriteIndex()
{

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::LoadSegments()
{

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::LoadSegments(const std::vector<cluster_id_t> & book)
{

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::WriteSegments()
{

}


template <typename vector_dimension_t> void 
IndexIVFPQ<vector_dimension_t>::InsertIvf(const std::vector<vector_dimension_t> & raw_data)
{
    const auto & pqcodes = pq_->Encode(raw_data);

    std::vector<omp_lock_t> locks(kc_);

    for (size_t i = 0; i < kc_; i++)
    {
        omp_init_lock(&locks[i]);
    }

    std::cout << "Start to insert pqcodes to IVFPQ index" << std::endl;
    utils::Timer timer;
    timer.Start();

    #pragma omp parallel for
    for (size_t n = 0; n < N_; n++)
    {
        std::vector<vector_dimension_t> nth_raw_data(raw_data.begin() + n * D_, raw_data.begin() + (n + 1) * D_);
        cluster_id_t id = cq_->Predict(nth_raw_data, 0); // CQ's subspace is the vector.
        omp_set_lock(&locks[id]);
        postint_lists_[id].emplace_back(n);
        omp_unset_lock(&locks[id]);
    }

    for (size_t i = 0; i < kc_; i++)
    {
        omp_destroy_lock(&locks[i]);
    }

    #pragma omp parallel for
    for (size_t no = 0; no < kc_; no++)
    {
        for (const auto & id: postint_lists_[no])
        {
            const auto & nth_code = pqcodes[id];
            segments_[no].insert(segments_[no].end(), nth_code.begin(), nth_code.end());
        }
    }

    timer.Stop();
    std::cout << "Finish inserting pqcodes to IVFPQ index, time cost: " << timer.GetTime() << " s" << std::endl;

}


template <typename vector_dimension_t> void 
IndexIVFPQ<vector_dimension_t>::SetTrainingConfig(size_t nsamples, int seed)
{
    nsamples_ = nsamples;
    seed_ = seed;
}



// template <typename vector_dimension_t> const std::vector<vector_dimension_t> 
// IndexIVFPQ<vector_dimension_t>::GetSingleCode(size_t list_no, size_t offset) const
// {
    
// }

};

};