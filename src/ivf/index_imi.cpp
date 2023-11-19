#include <ivf/index_imi.hpp>
#include <utils/vector_io.hpp>
#include <utils/stimer.hpp>
#include <utils/resize.hpp>
#include <vector_ops.hpp>

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
IndexIMI<vector_dimension_t>::IndexIMI(
    size_t N, size_t D, size_t L,    // ivfpq-info
    size_t kc,                       // level1-index-info
    size_t kp, size_t mp,            // level2-index-info
    const std::string & index_path, 
    const std::string & db_path,
    const std::string & name,
    IndexStatus status
):
N_(N), D_(D), L_(L),
kc_(kc), mc_(1), dc_(D),
kp_(kp), mp_(mp), dp_(D / mp),
IvfBase <std::vector<vector_dimension_t>, vector_dimension_t> (index_path, db_path, name, status),
nsamples_(0), seed_(-1), num_buckets_(1),
cq_(D, mc_, kc_), pq_(D, mp_, kp_)
{
    assert( mp_ * dp_ == D_ );

    assert( mp_ <= (sizeof(size_t) / sizeof(uint8_t)) && "We only support mp <= 8" );

    if (this->index_path_.back() != '/') this->index_path_ += "/";
    if (this->db_path_.back() != '/') this->db_path_ += "/";

    for (size_t i = 0; i < mp_; i++)
    {
        num_buckets_ *= kp_;
    }

    std::cout << "SIMD support: " << g_simd_architecture << std::endl;
}

/// @brief Get bucket ID from pqcode, 
///        bucket id = code[mp-1] * kp^(mp-1) + code[mp-2] * kp^{mp-2} + ... + code[0] * kp^0
/// @tparam vector_dimension_t 
/// @param code 
/// @return  
template <typename vector_dimension_t> inline size_t
IndexIMI<vector_dimension_t>::PQCode2BucketID(const uint8_t * code)
{
    size_t bucket_id = 0;
    for (size_t i = mp_-1; i >= 0; i--)
    {
        bucket_id *= kp_;
        bucket_id += code[i];
    }
}

/// bucket id = code[mp-1] * kp^(mp-1) + code[mp-2] * kp^{mp-2} + ... + code[0] * kp^0
template <typename vector_dimension_t> inline void 
IndexIMI<vector_dimension_t>::BucketID2PQCode(size_t bid, std::vector<uint8_t> & code)
{
    assert( code.size() == mp_ && "code size should be equal to mp_" );

    for (size_t i = 0; i < mp_; i++)
    {
        code[i] = bid % kp_;
        bid /= kp_;
    }
}


template <typename vector_dimension_t> bool IndexIMI<vector_dimension_t>::Ready() 
{  
    return cq_.Ready() && pq_.Ready() && posting_lists_.size() == kc_ && this->segments_.size() == kc_;
}


template <typename vector_dimension_t> void
IndexIMI<vector_dimension_t>::Populate(const std::vector<vector_dimension_t> & raw_data)
{
    assert( this->status_ == IndexStatus::LOCAL && "Index must be local." );
    assert( cq_.Ready() && pq_.Ready() && "CQ && PQ must be ready.");

    /// @brief adjust posting-list and segments
    {
        this->segments_.clear();
        this->segments_.resize(kc_);
        posting_lists_.clear();
        posting_lists_.resize(kc_);

        bucket_begin_.clear();
        bucket_begin_.resize(kc_, std::vector<size_t>(num_buckets_ + 1, 0));
    }

    const size_t N = raw_data.size() / D_;
    assert( N_ == N && "Index size mismatch." );

    for (auto & posting_list: posting_lists_)
    {
        posting_list.reserve(N / kc_);          // Roughly malloc
    }
    for (auto & segment: this->segments_)
    {
        segment.reserve(N / kc_);               // Roughly malloc
    }

    /// @brief Insert DB data into index.
    InsertIvf(raw_data);

}



template <typename vector_dimension_t> void
IndexIMI<vector_dimension_t>::Train(const std::vector<vector_dimension_t> & raw_data)
{
    assert( this->status_ == IndexStatus::LOCAL && "Index must be local." );
    assert( nsamples_ && seed_ != -1 && "Please set nsamples and seed." );

    const size_t Nt = raw_data.size() / D_;
    const size_t cq_train_times = 12;
    const size_t pq_train_times = 6;

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

    pq_.Fit(*traindata, pq_train_times, seed_);
    centers_pq_ = pq_.GetCentroids();

}


template <typename vector_dimension_t> void
IndexIMI<vector_dimension_t>::LoadIndex()
{
    std::cout << "Loading index..." << std::endl;
    /// @brief Load CQ codebook.
    {
        cq_.LoadCenters(this->index_path_ + cq_centers_);
        centers_cq_ = cq_.GetCentroids()[0];
        std::cout << "CQ codebook loaded." << std::endl;
    }

    /// @brief Load PQ codebook.
    {
        pq_.LoadCenters(this->index_path_ + pq_centers_);
        centers_pq_ = pq_.GetCentroids();
        std::cout << "PQ codebook loaded." << std::endl;
    }

    /// @brief Load Bucket Begin File
    {
        std::vector<size_t> bbf;
        auto [kc, sz] = utils::VectorIO<size_t>::LoadFromFile(bbf, this->index_path_ + bucket_begin_file_);
        assert( kc == kc_ && sz == num_buckets_ + 1 && "Invalid bucket begin file.");
        bucket_begin_ = utils::Resize<size_t>::Nest(bbf, kc, sz);
    }
}


template <typename vector_dimension_t> void
IndexIMI<vector_dimension_t>::WriteIndex()
{
    assert( this->status_ == IndexStatus::LOCAL && "Index must be local." );

    cq_.WriteCenters(this->index_path_ + cq_centers_);
    pq_.WriteCenters(this->index_path_ + pq_centers_);
    
    /// @brief Write bucket begin matrix
    {
        auto bucket_begin_flatten = utils::Resize<size_t>::Flatten(bucket_begin_);
        utils::VectorIO<size_t>::WriteToFile(bucket_begin_flatten, {kc_, num_buckets_ + 1}, this->index_path_ + bucket_begin_file_);
    }
}


template <typename vector_dimension_t> void
IndexIMI<vector_dimension_t>::LoadSegments()
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
IndexIMI<vector_dimension_t>::LoadSegments(const std::vector<cluster_id_t> & book)
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
IndexIMI<vector_dimension_t>::WriteSegments()
{
    assert( this->status_ == IndexStatus::LOCAL && "Index must be local." );

    std::vector<size_t> posting_lists_size(kc_);

    for (cluster_id_t id = 0; id < kc_; id++)
    {
        posting_lists_size[id] = posting_lists_[id].size();
        utils::VectorIO<vector_dimension_t>::WriteToFile( this->segments_[id], { posting_lists_size[id], mp_ }, this->db_path_ + vector_prefix_ + std::to_string(id) );
        utils::VectorIO<vector_id_t>::WriteToFile(posting_lists_[id], { 1, posting_lists_size[id] }, this->db_path_ + id_prefix_ + std::to_string(id));
    }

    utils::VectorIO<size_t>::WriteToFile(posting_lists_size, {1, kc_}, this->db_path_ + posting_lists_size_file_);
}



template <typename vector_dimension_t> void 
IndexIMI<vector_dimension_t>::InsertIvf(const std::vector<vector_dimension_t> & raw_data)
{
    assert( N_ == raw_data.size() / D_ );
    const auto & pqcodes = pq_.Encode(raw_data);

    std::vector<omp_lock_t> locks(kc_);

    for (size_t i = 0; i < kc_; i++)
    {
        omp_init_lock(&locks[i]);
    }

    std::cout << "Start to insert data to IMI index" << std::endl;
    utils::STimer timer;
    timer.Start();

    #pragma omp parallel for
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

    #pragma omp parallel for schedule(dynamic, 1)
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
    std::cout << "Finish inserting data to IMI index, time cost: " << timer.GetTime() << " s" << std::endl;

    /// @brief Reorder the data
    std::cout << "Start to reorder segments." << std::endl;
    timer.Start();

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < kc_; i++)
    {
        auto & segment          = this->segments_[i];
        auto & posting_list     = posting_lists_[i];
        auto & bucket_begin     = bucket_begin_[i];

        const auto & codes      = pqcodes[i];
        const size_t psize      = posting_list.size();

        std::vector<size_t> bucket_size(num_buckets_, 0);
        std::vector<size_t> bucket_ids(psize);
        for (size_t j = 0; j < psize; j++)
        {
            const uint8_t * pqcode = codes.data() + j * mp_;
            bucket_ids[j] = PQCode2BucketID(pqcode);
            bucket_size[bucket_ids[j]] ++;
        }

        for (size_t j = 1; j <= num_buckets_; j++)
        {
            bucket_begin[j] = bucket_begin[j-1] + bucket_size[j-1];
        }

        /// Temporary buffer.
        auto bucket_begin_copy_ptr  = std::make_unique<std::vector<size_t>> (bucket_begin.begin(), bucket_begin.end());
        auto segment_temp_ptr       = std::make_unique<std::vector<vector_dimension_t>>(segment.begin(), segment.end());
        auto posting_list_temp_ptr  = std::make_unique<std::vector<vector_id_t>>(posting_list.begin(), posting_list.end());
        auto bucket_begin_copy      = *bucket_begin_copy_ptr;
        auto segment_temp           = *segment_temp_ptr;
        auto posting_list_temp      = *posting_list_temp_ptr;

        std::cout << "Cp Finished" << std::endl;

        for (size_t j = 0; j < psize; j++)
        {
            const auto & bid = bucket_ids[j];
            auto & beg = bucket_begin_copy[bid];
            posting_list_temp[beg] = posting_list[j];
            std::copy_n(segment.begin() + j * D_, D_, segment_temp.begin() + beg * D_);
            beg ++;
        }

        /// Swap Reorder Buffer
        posting_list.swap(posting_list_temp);
        segment.swap(segment_temp);

        std::cout << "Cluster " << i << " finished." << std::endl;
    }

    timer.Stop();
    std::cout << "Finish reorder processing, time cost: " << timer.GetTime() << " s" << std::endl;

}



template <typename vector_dimension_t> void 
IndexIMI<vector_dimension_t>::SetTrainingConfig(size_t nsamples, int seed)
{
    nsamples_ = nsamples;
    seed_ = seed;
}



template <typename vector_dimension_t> inline float
IndexIMI<vector_dimension_t>::ADist(const DistanceTable & dtable, const uint8_t * code) const{
    
    float dist = 0.0;

    for (size_t i = 0; i < mp_; i++)
    {
        dist += dtable.GetValue(i, code[i]);
    }

    return dist;
}




template <typename vector_dimension_t> inline DistanceTable
IndexIMI<vector_dimension_t>::DTable(const std::vector<vector_dimension_t> & vec) const
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



template <typename vector_dimension_t> void 
IndexIMI<vector_dimension_t>::TopKID (
    size_t k, 
    const std::vector<vector_dimension_t> & query, 
    const std::vector<cluster_id_t> & book,
    std::vector<vector_id_t> & vid,
    std::vector<float> & dist
)
{
    assert( this->Ready() && "Index must be initialized before top-k query" );

    vid.clear(); dist.clear();

    DistanceTable dtable = DTable(query);
    std::vector<std::pair<size_t, float>> bucket_score(num_buckets_);
    /// @brief get top-nbs buckets
    {
        std::vector<size_t> code(mp_, 0);
        float dist = 0;
        for (size_t i = 0; i < mp_; i++)
        {
            dist += dtable.GetValue(i, 0);
        }
        /// Get Bucket Score
        for (size_t bid = 0; bid < num_buckets_; bid ++)
        {
            // store 
            bucket_score[bid] = { bid, dist };
            // update dist
            dist = dist - dtable.GetValue(0, code[0]) + dtable.GetValue(0, (++code[0]) % kp_);
            for (size_t i = 1; i < mp_; i++)
            {
                if (code[i - 1] / kp_) {
                    code[i - 1] = 0;
                    dist = dist - dtable.GetValue(i, code[i]) + dtable.GetValue(i, (++code[i]) % kp_);
                }
            }
        }
        /// Sort
        std::sort(bucket_score.begin(), bucket_score.end(),
            [](const std::pair<size_t, float> & a, const std::pair<size_t, float> & b) {
                return a.second < b.second;
            }
        );
    }

    /// Traverse buckets
    std::vector<std::pair<vector_id_t, float>> score;
    score.reserve(L_);

    size_t num_searched_segments = 0;
    size_t num_searched_vectors = 0;

    for (size_t i = 0; i < book.size() && num_searched_vectors < L_; i++)
    {
        const auto & cid = book[i];
        const auto & posting_list = posting_lists_[cid];
        const auto & segment = this->segments_[cid];
        const auto & bbeg = bucket_begin_[cid];
        for (size_t j = 0; j < num_buckets_ && num_searched_vectors < L_; j++)
        {
            const auto & bid = bucket_score[j].first;
            for (size_t k = bbeg[bid], end = bbeg[bid + 1]; k < end && num_searched_vectors < L_; k++)
            {
                score.emplace_back (
                    posting_list[k],
                    vec_L2sqr(query.data(), segment.data() + k * D_, D_)
                );
                num_searched_vectors ++;
            }
            num_searched_segments ++;
        }
    }

    size_t ak = std::min(score.size(), k);
    std::partial_sort( score.begin(), score.begin() + ak, score.end(),
        [](const std::pair<vector_id_t, float> & a, const std::pair<vector_id_t, float> & b) {
            return a.second < b.second;
        }
    );
    vid.resize(ak); dist.resize(ak);
    
    for (size_t i = 0; i < ak; i++)
    {
        vid[i] = score[i].first;
        dist[i] = score[i].second;
    }

    // std::cout << "num_searched_segments: " << num_searched_segments << std::endl;
    // std::cout << "num_searched_vectors: " << num_searched_vectors << std::endl;
}




template <typename vector_dimension_t> void 
IndexIMI<vector_dimension_t>::TopKID(
    size_t k,
    const std::vector<std::vector<vector_dimension_t>> & queries,
    const std::vector<std::vector<cluster_id_t>> & books,
    std::vector<std::vector<vector_id_t>> & vids,
    std::vector<std::vector<float>> & dists
)
{
    assert( this->Ready() && "Index not ready");

    vids.clear(); dists.clear();
    vids.resize(queries.size()); dists.resize(queries.size());

    size_t num_searched_segments = 0;
    size_t num_searched_vectors = 0;

    #pragma omp parallel for reduction(+: num_searched_segments, num_searched_vectors) num_threads(this->num_threads_)
    for (size_t q = 0; q < queries.size(); q++)
    {
        const auto & query = queries[q];
        const auto & book = books[q];
        auto & vid = vids[q];
        auto & dist = dists[q];

        size_t local_num_searched_segments = 0;
        size_t local_num_searched_vectors = 0;

        DistanceTable dtable = DTable(query);
        std::vector<std::pair<size_t, float>> bucket_score(num_buckets_);
        /// @brief get top-nbs buckets
        {
            std::vector<size_t> code(mp_, 0);
            float dist = 0;
            for (size_t i = 0; i < mp_; i++)
            {
                dist += dtable.GetValue(i, 0);
            }
            /// Get Bucket Score
            for (size_t bid = 0; bid < num_buckets_; bid ++)
            {
                // store 
                bucket_score[bid] = { bid, dist };
                // update dist
                dist = dist - dtable.GetValue(0, code[0]) + dtable.GetValue(0, (++code[0]) % kp_);
                for (size_t i = 1; i < mp_; i++)
                {
                    if (code[i - 1] / kp_) {
                        code[i - 1] = 0;
                        dist = dist - dtable.GetValue(i, code[i]) + dtable.GetValue(i, (++code[i]) % kp_);
                    }
                }
            }
            /// Sort
            std::sort(bucket_score.begin(), bucket_score.end(),
                [](const std::pair<size_t, float> & a, const std::pair<size_t, float> & b) {
                    return a.second < b.second;
                }
            );
        }

        /// Traverse buckets
        std::vector<std::pair<vector_id_t, float>> score;
        score.reserve(L_);

        for (size_t i = 0; i < book.size() && local_num_searched_vectors < L_; i++)
        {
            const auto & cid = book[i];
            const auto & posting_list = posting_lists_[cid];
            const auto & segment = this->segments_[cid];
            const auto & bbeg = bucket_begin_[cid];
            for (size_t j = 0; j < num_buckets_ && local_num_searched_vectors < L_; j++)
            {
                const auto & bid = bucket_score[j].first;
                for (size_t k = bbeg[bid], end = bbeg[bid + 1]; k < end && local_num_searched_vectors < L_; k++)
                {
                    score.emplace_back (
                        posting_list[k],
                        vec_L2sqr(query.data(), segment.data() + k * D_, D_)
                    );
                    local_num_searched_vectors ++;
                }
                local_num_searched_segments ++;
            }
        }

        num_searched_segments += local_num_searched_segments;
        num_searched_vectors += local_num_searched_vectors;

        size_t ak = std::min(score.size(), k);
        std::partial_sort( score.begin(), score.begin() + ak, score.end(),
            [](const std::pair<vector_id_t, float> & a, const std::pair<vector_id_t, float> & b) {
                return a.second < b.second;
            }
        );
        vid.resize(ak); dist.resize(ak);
        
        for (size_t i = 0; i < ak; i++)
        {
            vid[i] = score[i].first;
            dist[i] = score[i].second;
        }
    }

    // std::cout << "num_searched_segments: " << num_searched_segments << std::endl;
    // std::cout << "num_searched_vectors: " << num_searched_vectors << std::endl;
}



template <typename vector_dimension_t> void
IndexIMI<vector_dimension_t>::TopWID(
    size_t w, 
    const std::vector<vector_dimension_t> & query,
    std::vector<cluster_id_t> & book
)
{
    assert( cq_.Ready() && "CQ must be initialized." );

    std::vector<std::pair<cluster_id_t, float>> score;
    score.resize(kc_);
    for (cluster_id_t cid = 0; cid < kc_; cid++)
    {
        score[cid] = { cid, vec_L2sqr(query.data(), centers_cq_[cid].data(), D_) };
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




/// @brief Get TopK vector ID (KNN) with one single thread.
/// @tparam vector_dimension_t 
/// @param w 
/// @param queries 
/// @param books 
template <typename vector_dimension_t> void
IndexIMI<vector_dimension_t>::TopWID(
    size_t w, 
    const std::vector<std::vector<vector_dimension_t>> & queries,
    std::vector<std::vector<cluster_id_t>> & books
)
{
    assert( cq_.Ready() && "CQ must be initialized." );

    books.clear();
    books.resize(queries.size());
    size_t actual_w = std::min(w, kc_);

    #pragma omp parallel for num_threads(this->num_threads_)
    for (size_t i = 0; i < queries.size(); i++)
    {
        const auto & query = queries[i];
        auto & book = books[i];

        std::vector<std::pair<cluster_id_t, float>> score;
        score.resize(kc_);
        for (cluster_id_t cid = 0; cid < kc_; cid++)
        {
            score[cid] = { cid, vec_L2sqr(query.data(), centers_cq_[cid].data(), D_) };
        }

        std::partial_sort(score.begin(), score.begin() + actual_w, score.end(),
            [](const std::pair<cluster_id_t, float> & a, const std::pair<cluster_id_t, float> & b) {
                return a.second < b.second;
            }
        );

        book.resize(actual_w);
        for (size_t j = 0; j < actual_w; j++)
        {
            book[i] = (score[j].first);
        }

    }
}



template <typename vector_dimension_t> void 
IndexIMI<vector_dimension_t>::Search (
    size_t k, size_t w,
    const std::vector<vector_dimension_t> & query,
    std::vector<vector_id_t> & vid,
    std::vector<float> & dist
)
{
    std::vector<cluster_id_t> book;
    TopWID(w, query, book);
    TopKID(k, query, book, vid, dist);
}


template <typename vector_dimension_t> void
IndexIMI<vector_dimension_t>::Search(
    size_t k, size_t w,
    const std::vector<std::vector<vector_dimension_t>> & queries,
    std::vector<std::vector<vector_id_t>> & vids,
    std::vector<std::vector<float>> & dists
)
{
    #pragma omp parallel for num_threads(this->num_threads_) schedule(dynamic, 1)
    for (size_t i = 0; i < queries.size(); i++)
    {
        std::vector<cluster_id_t> book;
        TopWID(w, queries[i], book);
        TopKID(k, queries[i], book, vids[i], dists[i]);
    }
}



}; // namespace ivf

}; // namespace index