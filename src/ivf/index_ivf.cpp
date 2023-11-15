#include <ivf/index_ivf.hpp>
#include <utils/vector_io.hpp>
#include <utils/stimer.hpp>
#include <distance.hpp>

#include <omp.h>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <numeric>
#include <random>
#include <algorithm>
#include <memory>

extern "C" {
    void omp_init_lock(omp_lock_t *);
    void omp_set_lock(omp_lock_t *);
    void omp_unset_lock(omp_lock_t *);
    void omp_destroy_lock(omp_lock_t *);
};

namespace index {

namespace ivf {


template <typename vector_dimension_t> 
IndexIVF<vector_dimension_t>::IndexIVF(
    size_t N, size_t D, size_t L,
    size_t kc, // level-1-config
    const std::string & index_path,
    const std::string & db_path,
    const std::string & name,
    IndexStatus status
): N_(N), D_(D), L_(L),
   kc_(kc), mc_(1), dc_(D),
   IvfBase <std::vector<vector_dimension_t>, vector_dimension_t> (index_path, db_path, name, status),
   nsamples_(0), seed_(-1),
   cq_(D, mc_, kc_)
{
    if (this->index_path_.back() != '/') this->index_path_ += "/";
    if (this->db_path_.back() != '/') this->db_path_ += "/";

    std::cout << "SIMD support: " << g_simd_architecture << std::endl;
}


template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::Populate(const std::vector<vector_dimension_t> & raw_data)
{
    assert( this->status_ == IndexStatus::LOCAL && "Index must be local." );
    assert( cq_.Ready() && "CQ must be ready.");

    /// @brief adjust postinglist and segments 
    {
        posting_lists_.clear();
        posting_lists_.resize(kc_);
        this->segments_.clear();
        this->segments_.resize(kc_);
    }

    const size_t N = raw_data.size() / D_;
    assert( N == N_ && "Number of data points must match." );

    for (auto & posting_list : posting_lists_)
    {
        posting_list.reserve(N / kc_);              // Roughly malloc
    }

    for (auto & segment : this->segments_)
    {
        segment.reserve(N / kc_);                   // Roughly malloc
    }

    InsertIvf(raw_data);
}


template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::Train(const std::vector<vector_dimension_t> & raw_data)
{
    assert( this->status_ == IndexStatus::LOCAL && "Index must be local." );
    assert( nsamples_ && seed_ != -1 && "Please set nsamples and seed." );

    const size_t Nt = raw_data.size() / D_;
    const size_t cq_train_times = 12;

    if (nsamples_ > Nt) {
        nsamples_ = Nt;
        std::cout << "nsamples_ > Nt, set nsamples_ to " << nsamples_ << std::endl;
    }

    std::vector<vector_id_t> ids(Nt);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 default_random_engine(seed_);
    std::shuffle(ids.begin(), ids.end(), default_random_engine);

    std::unique_ptr<std::vector<vector_dimension_t>> traindata = std::make_unique<std::vector<vector_dimension_t>>();
    traindata->reserve(nsamples_ * D_);

    for (size_t k = 0; k < nsamples_; k++)
    {
        const vector_id_t id = ids[k];
        traindata->insert(traindata->end(), raw_data.begin() + (id * D_), raw_data.begin() + (id + 1) * D_);
    }

    cq_.Fit(*traindata, cq_train_times, seed_);
    centers_cq_ = cq_.GetCentroids()[0];
}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::LoadIndex()
{
    std::cout << "Loading index..." << std::endl;
    /// @brief Load CQ codebook.
    {
        cq_.LoadCenters(this->index_path_ + cq_centers_);
        centers_cq_ = cq_.GetCentroids()[0];
        std::cout << "CQ codebook loaded." << std::endl;
    }
}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::WriteIndex()
{
    assert( this->status_ == IndexStatus::LOCAL && "Index must be local." );

    cq_.WriteCenters(this->index_path_ + cq_centers_);
}


template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::LoadSegments()
{
    std::cout << "Loading all segments..." << std::endl;

    /// @brief adjust postinglist and segments 
    if (posting_lists_.empty())
    {
        posting_lists_.resize(kc_);
        this->segments_.resize(kc_);
    }

    for (cluster_id_t id = 0; id < kc_; id++)
    {
        if (posting_lists_[id].empty())
        {
            utils::VectorIO<vector_id_t>::LoadFromFile(posting_lists_[id], this->db_path_ + id_prefix_ + std::to_string(id));
            utils::VectorIO<vector_dimension_t>::LoadFromFile(this->segments_[id], this->db_path_ + vector_prefix_ + std::to_string(id));
        }
    }
}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::LoadSegments(const std::vector<cluster_id_t> & book)
{
    std::cout << "Loading " << book.size() << " segments..." << std::endl;

    /// @brief adjust postinglist and segments 
    if (posting_lists_.empty())
    {
        posting_lists_.resize(kc_);
        this->segments_.resize(kc_);
    }

    std::unordered_set<cluster_id_t> new_book_set(book.begin(), book.end());
    for (cluster_id_t id = 0; id < kc_; id++)
    {
        if (posting_lists_[id].size()) { 
            if (new_book_set.count(id)) 
            { 
                new_book_set.erase(id); 
            }
            else 
            {
                std::vector<vector_id_t>().swap(posting_lists_[id]);
                std::vector<vector_dimension_t>().swap(this->segments_[id]);
            }
        }
    }
    
    for (const auto & id: new_book_set)
    {
        utils::VectorIO<vector_id_t>::LoadFromFile(posting_lists_[id], this->db_path_ + id_prefix_ + std::to_string(id));
        utils::VectorIO<vector_dimension_t>::LoadFromFile(this->segments_[id], this->db_path_ + vector_prefix_ + std::to_string(id));
    }
}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::WriteSegments()
{
    assert( this->status_ == IndexStatus::LOCAL && "Index must be local." );

    std::vector<size_t> posting_lists_size(kc_);

    for (cluster_id_t id = 0; id < kc_; id++)
    {
        posting_lists_size[id] = posting_lists_[id].size();
        utils::VectorIO<vector_dimension_t>::WriteToFile( this->segments_[id], { posting_lists_size[id], D_ }, this->db_path_ + vector_prefix_ + std::to_string(id) );
    }

    utils::VectorIO<size_t>::WriteToFile(posting_lists_size, {1, kc_}, this->db_path_ + "posting_lists_size");
}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::InsertIvf(const std::vector<vector_dimension_t> & raw_data)
{
    assert( N_ == raw_data.size() / D_ );
    std::vector<omp_lock_t> locks(kc_);

    for (size_t i = 0; i < kc_; i++)
    {
        omp_init_lock(&locks[i]);
    }

    std::cout << "Start to insert pqcodes to IVF index" << std::endl;
    utils::STimer timer;
    timer.Start();

    #pragma omp parallel for schedule(dynamic, 1)
    for (vector_id_t n = 0; n < N_; n++)
    {
        std::vector<vector_dimension_t> nth_raw_data(raw_data.begin() + n * D_, raw_data.begin() + (n + 1) * D_);
        cluster_id_t id = cq_.PredictOne(nth_raw_data, 0); // CQ's subspace is the vector.
        omp_set_lock(&locks[id]);
        posting_lists_[id].emplace_back(n);
        omp_unset_lock(&locks[id]);
    }

    for (size_t i = 0; i < kc_; i++)
    {
        omp_destroy_lock(&locks[i]);
    }

    #pragma omp parallel for
    for (size_t no = 0; no < kc_; no++)
    {
        for (const auto & id: posting_lists_[no])
        {
            this->segments_[no].insert( this->segments_[no].end(), 
                raw_data.begin() + id * D_,
                raw_data.begin() + (id + 1) * D_
            );
        }
    }

    timer.Stop();
    std::cout << "Finish inserting pqcodes to IVF index, time cost: " << timer.GetTime() << " s" << std::endl;
}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::SetTrainingConfig(size_t nsamples, int seed)
{
    nsamples_ = nsamples;
    seed_ = seed;
}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::WritePostingLists() const
{
    for (cluster_id_t id = 0; id < kc_; id++)
    {
        utils::VectorIO<vector_id_t>::WriteToFile(
            posting_lists_[id], { 1, posting_lists_[id].size() }, 
            this->db_path_ + id_prefix_ + std::to_string(id)
        );
    }
}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::TopKID (
    size_t k, 
    const std::vector<vector_dimension_t> & query, 
    const std::vector<cluster_id_t> & book,
    std::vector<vector_id_t> & vid,
    std::vector<float> & dist
)
{
    assert( this->Ready() && "Index must be initialized before querying." );

    vid.clear(); dist.clear();

    std::vector<std::pair<vector_id_t, float>> score;
    score.reserve(L_);

    size_t num_searched_segments = 0;
    size_t num_searched_vectors = 0;

    for (size_t i = 0; i < book.size() && num_searched_vectors < L_; i++) {
        cluster_id_t cid = book[i];
        const auto & posting_list = posting_lists_[cid];
        for (size_t j = 0; j < posting_list.size() && num_searched_vectors < L_; j++)
        {
            score.emplace_back (
                posting_list[j],
                vec_L2sqr(query.data(), this->segments_[cid].data() + j * D_, D_)
            );
            num_searched_vectors ++;
        }
        num_searched_segments ++;
    }

    size_t actual_k = std::min(score.size(), k);
    std::partial_sort ( score.begin(), score.begin() + actual_k, score.end(), 
        [](const std::pair<vector_id_t, float> & a, const std::pair<vector_id_t, float> & b) {
            return a.second < b.second;
        }
    );

    vid.reserve(actual_k);
    dist.reserve(actual_k);
    for ( const auto & [id, d]: score ) {
        vid.emplace_back(id);
        dist.emplace_back(d);
    }

    std::cout << "num_searched_segments: " << num_searched_segments << std::endl;
    std::cout << "num_searched_vectors: " << num_searched_vectors << std::endl;
}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::TopKID (
    size_t k,
    const std::vector<std::vector<vector_dimension_t>> & queries,
    const std::vector<std::vector<cluster_id_t>> & books,
    std::vector<std::vector<vector_id_t>> & vids,
    std::vector<std::vector<float>> & dists
)
{
    // assert( this->Ready() && "Index must be initialized before querying." );

    vids.clear(); dists.clear();
    vids.resize(queries.size());
    dists.resize(queries.size());

    size_t num_searched_segments = 0;
    size_t num_searched_vectors = 0;

    #pragma omp parallel for reduction(+: num_searched_segments, num_searched_vectors) num_threads(this->num_threads_) schedule(dynamic, 1)
    for (size_t i = 0; i < queries.size(); i++)
    {
        const auto & query = queries[i];
        const auto & book = books[i];
        auto & vid = vids[i];
        auto & dist = dists[i];

        size_t local_num_searched_segments = 0;
        size_t local_num_searched_vectors = 0;

        std::vector<std::pair<vector_id_t, float>> score;
        score.reserve(L_);

        for (size_t j = 0; j < book.size() && local_num_searched_vectors < L_; j++)
        {
            cluster_id_t cid = book[j];
            const auto & posting_list = posting_lists_[cid];
            for (size_t v = 0; v < posting_list.size() && local_num_searched_vectors < L_; v++)
            {
                score.emplace_back (
                    posting_list[v],
                    vec_L2sqr(query.data(), this->segments_[cid].data() + v * D_, D_)
                );
                local_num_searched_vectors ++;                
            }
            local_num_searched_segments ++;
        }
        num_searched_segments += local_num_searched_segments;
        num_searched_vectors += local_num_searched_vectors;

        size_t actual_k = std::min(score.size(), k);
        std::partial_sort( score.begin(), score.begin() + actual_k, score.end(),
            [](const std::pair<vector_id_t, float> & a, const std::pair<vector_id_t, float> & b) {
                return a.second < b.second;
            }
        );

        vid.resize(actual_k);
        dist.resize(actual_k);
        for ( size_t j = 0; j < actual_k; j++ ) {
            const auto & [id, d] = score[j];
            vid[j] = id;
            dist[j] = d;
        }
    }

    std::cout << "num_searched_segments: " << num_searched_segments << std::endl;
    std::cout << "num_searched_vectors: " << num_searched_vectors << std::endl;

}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::TopWID(
    size_t w, 
    const std::vector<vector_dimension_t> & query,
    std::vector<cluster_id_t> & book
)
{
    // assert( cq_.Ready() && "CQ must be initialized." );

    std::vector<std::pair<cluster_id_t, float>> score;
    score.resize(kc_);
    for (cluster_id_t cid = 0; cid < kc_; cid++)
    {
        score[cid].first = cid;
        score[cid].second = vec_L2sqr(query.data(), centers_cq_[cid].data(), D_);
    }

    size_t actual_w = std::min(w, kc_);
    std::partial_sort(score.begin(), score.begin() + actual_w, score.end(),
        [](const std::pair<cluster_id_t, float> & a, const std::pair<cluster_id_t, float> & b) {
            return a.second < b.second;
        }
    );

    book.clear();
    book.resize(actual_w);
    for (size_t i = 0; i < actual_w; i++)
    {
        book[i] = (score[i].first);
    }

}



template <typename vector_dimension_t> void
IndexIVF<vector_dimension_t>::TopWID(
    size_t w, 
    const std::vector<std::vector<vector_dimension_t>> & queries,
    std::vector<std::vector<cluster_id_t>> & books
)
{
    // assert( cq_.Ready() && "CQ must be initialized." );

    size_t actual_w = std::min(w, kc_);
    books.clear();
    books.resize(queries.size(), std::vector<cluster_id_t>(actual_w));

    #pragma omp parallel for num_threads(this->num_threads_) schedule(dynamic, 1)
    for (size_t i = 0; i < queries.size(); i++)
    {
        const auto & query = queries[i];
        auto & book = books[i];

        std::vector<std::pair<cluster_id_t, float>> score;
        score.resize(kc_);
        for (cluster_id_t cid = 0; cid < kc_; cid++)
        {
            score[cid].first = cid;
            score[cid].second = vec_L2sqr(query.data(), centers_cq_[cid].data(), D_);
        }

        std::partial_sort(score.begin(), score.begin() + actual_w, score.end(),
            [](const std::pair<cluster_id_t, float> & a, const std::pair<cluster_id_t, float> & b) {
                return a.second < b.second;
            }
        );

        for (size_t j = 0; j < actual_w; j++)
        {
            book[j] = (score[j].first);
        }
    }
}



template <typename vector_dimension_t> bool IndexIVF<vector_dimension_t>::Ready()
{
    return cq_.Ready() && posting_lists_.size() == kc_ && this->segments_.size() == kc_;
}


};

};