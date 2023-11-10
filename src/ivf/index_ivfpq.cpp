#include <index_ivfpq.hpp>
#include <omp.h>
#include <cassert>
#include <iostream>
#include <timer.hpp>
#include <distance.hpp>
#include <unordered_set>
#include <vector_io.hpp>


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


inline float DistanceTable::GetValue(size_t m, size_t ks) const
{
    return data_[m * kp_ + ks];
}


template <typename vector_dimension_t> 
IndexIVFPQ<vector_dimension_t>::IndexIVFPQ(
    size_t N, size_t D, size_t L,    // ivfpq-info
    size_t kc, size_t mc, size_t dc, // level1-index-info
    size_t kp, size_t mp, size_t dp, // level2-index-info
    const std::string & index_path, 
    const std::string & db_path,
    const std::string & name = "",
    IndexStatus status = IndexStatus::LOCAL
): 
N_(N), D_(D), L_(L), 
kc_(kc), mc_(mc), dc_(dc),
kp_(kp), mp_(mp), dp_(dp),
index_path_(index_path), db_path_(db_path), 
name_(name), status_(status),
nsamples_(0), seed_(-1),
cq_(dc_, mc_, kc_), pq_(dp_, mp_, kp_)
{
    assert( dc_ == D_ && mc_ == 1 ); 
    assert( mp_ * dp_ == D_ );

    if (index_path_.back() != '/') index_path_ += "/";
    if (db_path_.back() != '/') db_path_ += "/";

    std::cout << "SIMD support: " << g_simd_architecture << std::endl;

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::Populate(const std::vector<vector_dimension_t> & raw_data)
{
    assert( status_ == IndexStatus::LOCAL && "Index must be local." );

    /// @brief adjust postinglist and segments 
    {
        postint_lists_.clear();
        postint_lists_.resize(kc_);
        segments_.clear();
        segments_.resize(kc_);
    }

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
    assert( status_ == IndexStatus::LOCAL && "Index must be local." );
    assert( nsamples_ && seed_ != -1 && "nsample != 0 and seed != -1" );

    const size_t Nt = raw_data.size() / D_;
    const size_t min_nsamples = 100'000;
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

    cq_.Fit(*traindata, cq_train_times, seed_);
    centers_cq_ = cq_.GetCentroids()[0];
    // labels_cq_ = cq_.GetAssignments()[0];

    pq_.Fit(*traindata, pq_train_times, seed_);
    centers_pq_ = pq_.GetCentroids();
    // labels_pq_ = pq_.GetAssignments();

}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::LoadIndex()
{
    /// @brief Load CQ codebook.
    {
        cq_.Load(index_path_ + cq_prefix);
        centers_cq_ = cq_.GetCentroids()[0];
        std::cout << "CQ codebook loaded." << std::endl;
    }

    /// @brief Load PQ codebook.
    {
        pq_.Load(index_path_ + pq_prefix);
        centers_pq_ = pq_.GetCentroids();
        std::cout << "PQ codebook loaded." << std::endl;
    }
}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::WriteIndex()
{
    assert( status_ == IndexStatus::LOCAL && "Index must be local." );

    cq_.Write(index_path_ + cq_prefix);
    pq_.Write(index_path_ + pq_prefix);
}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::LoadSegments()
{
    std::cout << "Loading all segments..." << std::endl;

    /// @brief adjust postinglist and segments 
    if (postint_lists_.empty())
    {
        postint_lists_.resize(kc_);
        segments_.resize(kc_);
    }

    for (cluster_id_t id = 0; id < kc_; id++)
    {
        if (postint_lists_[id].empty())
        {
            utils::VectorIO<vector_id_t>::LoadFromFile(postint_lists_[id], db_path_ + id_prefix + std::to_string(id));
            utils::VectorIO<uint8_t>::LoadFromFile(segments_[id], db_path_ + vector_prefix + std::to_string(id));
        }
    }
}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::LoadSegments(const std::vector<cluster_id_t> & book)
{
    std::cout << "Loading " << book.size << " segments..." << std::endl;

    /// @brief adjust postinglist and segments 
    if (postint_lists_.empty())
    {
        postint_lists_.resize(kc_);
        segments_.resize(kc_);
    }

    std::unordered_set<cluster_id_t> new_book_set(book.begin(), book.end());
    for (cluster_id_t id = 0; id < kc_; id++)
    {
        if (postint_lists_[id].size()) { 
            if (new_book_set.count(id)) 
            { 
                new_book_set.erase(id); 
            }
            else 
            {
                std::vector<vector_id_t>().swap(postint_lists_[id]);
                std::vector<uint8_t>().swap(segments_[id]);
            }
        }
    }
    
    for (const auto & id: new_book_set)
    {
        utils::VectorIO<vector_id_t>::LoadFromFile(postint_lists_[id], db_path_ + id_prefix + std::to_string(id) + suffix_id);
        utils::VectorIO<uint8_t>::LoadFromFile(segments_[id], db_path_ + vector_prefix + std::to_string(id) + suffix_vector);
    }
}


template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::WriteSegments()
{
    assert( status_ == IndexStatus::LOCAL && "Index must be local." );

    std::vector<size_t> postint_lists_size(kc_);

    for (cluster_id_t id = 0; id < kc_; id++)
    {
        postint_lists_size[id] = postint_lists_[id].size();
        utils::VectorIO<uint8_t>::WriteToFile( segments_[id], { postint_lists_size[id], mp_ }, db_path_ + vector_prefix + std::to_string(id) );
    }

    utils::VectorIO<size_t>::WriteToFile(postint_lists_size, {1, kc_}, dp_path_ + "posting_lists_size");
}


template <typename vector_dimension_t> void 
IndexIVFPQ<vector_dimension_t>::InsertIvf(const std::vector<vector_dimension_t> & raw_data)
{
    const auto & pqcodes = pq_.Encode(raw_data);

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
        cluster_id_t id = cq_.PredictOne(nth_raw_data, 0); // CQ's subspace is the vector.
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



template <typename vector_dimension_t> inline float
IndexIVFPQ<vector_dimension_t>::ADist(const DistanceTable & dtable, const uint8_t * code) const{
    
    float dist = 0.0;

    for (size_t i = 0; i < mp_; i++)
    {
        dist += dtable.GetValue(i, code[i]);
    }

    return dist;

}



template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::WritePostingLists() const
{
    for (cluster_id_t id = 0; id < kc_; id++)
    {
        utils::VectorIO<vector_id_t>::WriteToFile(
            postint_lists_[id], { 1, postint_lists_[id].size() }, 
            db_path_ + id_prefix + std::to_string(id)
        );
    }
}



template <typename vector_dimension_t> void
IndexIVFPQ<vector_dimension_t>::TopKID (
    size_t k, 
    const std::vector<vector_dimension_t> & query, 
    const std::vector<cluster_id_t> & book,
    std::vector<vector_id_t> & vid,
    std::vector<float> & dist
)
{
    assert( cq_.Ready() && pq_.Ready() && "CQ and PQ must be initialized." );

    vid.clear(); dist.clear();

    // #pragma omp parallel for reduction(+:num_searched_segments, num_searched_vectors) num_threads(num_threads_)
    DistanceTable dtable = DTable(query);
    std::vector<std::pair<vector_id_t, float>> score;
    score.reserve(L_);

    size_t num_searched_segments = book.size();
    size_t num_searched_vectors = 0;

    for (size_t i = 0; i < book.size() && num_searched_vectors < L_; i++) {
        cluster_id_t cid = book[i];
        const auto & posting_list = postint_lists_[cid];
        num_searched_vectors += posting_list.size();
        for (size_t j = 0; j < posting_list.size() && num_searched_vectors < L_; j++)
        {
            vector_id_t n = posting_list[j];
            score.emplace_back(n, ADist(dtable, (segments_[cid].data() + j * mp_))); /* the j-th vector of segment[cid] */
        }
    }

}



template <typename vector_dimension_t> inline DistanceTable
IndexIVFPQ<vector_dimension_t>::DTable(const std::vector<vector_dimension_t> & vec) const
{
    DistanceTable dtable(mp_, kp_);

    for (size_t m = 0; m < mp_; m++)
    {
        for (size_t ks = 0; ks < kp_; ks++)
        {
            dtable.SetValue(m, ks, vec_L2sqr(&(vec[m*dp_]), centers_pq_[m][ks].data(), dp_));
        }
    }

    return dtable;
}



};

};