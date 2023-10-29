#include "index_ivfpq.hpp"

#include <unordered_set>

using namespace Toy;

extern "C" {
    void omp_init_lock(omp_lock_t *);
    void omp_set_lock(omp_lock_t *);
    void omp_unset_lock(omp_lock_t *);
    void omp_destroy_lock(omp_lock_t *);
}

IVFPQConfig::IVFPQConfig(
    size_t N, size_t D, 
    size_t L, 
    size_t kc, size_t kp, 
    size_t mc, size_t mp, 
    size_t dc, size_t dp, 
    std::string index_path, std::string db_path
) : N_(N), D_(D), L_(L), 
    kc(kc), kp(kp), 
    mc(mc), mp(mp), 
    dc(dc), dp(dp), 
    index_path(index_path), db_path(db_path)
{}

IndexIVFPQ::IndexIVFPQ(const IVFPQConfig& cfg, size_t nq, bool verbose)
    : N_(cfg.N_), D_(cfg.D_), L_(cfg.L_), nq(nq), 
    kc(cfg.kc), kp(cfg.kp), mc(cfg.mc), mp(cfg.mp), dc(cfg.dc), dp(cfg.dp)
{
    verbose_ = verbose;
    assert(dc == D_ && mc == 1);

    cq_ = nullptr;
    pq_ = nullptr;

    if (verbose_) {
        // Check which SIMD functions are used. See distance.h for this global variable.
        std::cout << "SIMD support: " << g_simd_architecture << std::endl;
    }
}

void 
IndexIVFPQ::train(const std::vector<float>& rawdata, int seed, size_t nsamples)
{
    size_t Nt_ = rawdata.size() / D_;
    if (nsamples < 100'000) nsamples = 100'000;
    if (nsamples > Nt_) nsamples = Nt_;
    if (nsamples > N_) nsamples = N_;

    if (verbose_) std::cout << "Training index with " << nsamples << " samples" << std::endl;
    std::unique_ptr<std::vector<float>> traindata;

    std::vector<size_t> ids(N_);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 default_random_engine(seed);
    std::shuffle(ids.begin(), ids.end(), default_random_engine);

    traindata = std::make_unique<std::vector<float>>();
    traindata->reserve(nsamples * D_);

    for (size_t k = 0; k < nsamples; ++k) {
        size_t id = ids[k];
        traindata->insert(traindata->end(), 
                rawdata.begin() + id * D_, rawdata.begin() + (id + 1) * D_);
    }

    cq_ = std::make_unique<Quantizer::Quantizer>(D_, nsamples, mc, kc, true);
    cq_->fit(*traindata, 12, seed);
    centers_cq_ = cq_->get_centroids()[0];      // Because mc == 1
    labels_cq_ = cq_->get_assignments()[0];

    pq_ = std::make_unique<Quantizer::Quantizer>(D_, nsamples, mp, kp, true);
    pq_->fit(*traindata, 6, seed);
    centers_pq_ = pq_->get_centroids();
    labels_pq_ = pq_->get_assignments();

    is_trained_ = true;
}

void 
IndexIVFPQ::insert_ivf(const std::vector<float>& rawdata)
{
    const auto& pqcodes = pq_->encode(rawdata);

    std::vector<omp_lock_t> locks(kc);

    for (int i = 0; i < kc; ++i) {
        omp_init_lock(&locks[i]);
    }

    #pragma omp parallel for
    for (size_t n = 0; n < N_; ++n) {
        const auto& vec = nth_raw_vector(rawdata, n);
        int id = cq_->predict_one(vec, 0);
        omp_set_lock(&locks[id]);
        posting_lists_[id].emplace_back(n);
        omp_unset_lock(&locks[id]);
    }

    for (int i = 0; i < kc; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    #pragma omp parallel for
    for (size_t no = 0; no < kc; ++no) {
        for (const auto& id : posting_lists_[no]) {
            const auto& nth_code = pqcodes[id];
            db_codes_[no].insert(db_codes_[no].end(), nth_code.begin(), nth_code.end());
        }
    }
}

void
IndexIVFPQ::load_from_book(const std::vector<uint32_t>& book, std::string cluster_path)
{
    if (verbose_) { std::cout << "Start to load from book" << std::endl; }

    if (cluster_path.back() != '/') {
        cluster_path += "/";
    }

    size_t ncentroid = book.size();
    posting_lists_.clear();
    posting_lists_.resize(kc);
    db_codes_.clear();
    db_codes_.resize(kc);

    std::string prefix_vector = "pqcode_", prefix_id = "id_";
    std::string suffix_vector = ".ui8vecs", suffix_id = ".uivecs";

    std::unordered_set<uint32_t> new_book_set(book.begin(), book.end());
    for (size_t id = 0; id < kc; ++id) {
        if (!posting_lists_[id].size()) continue;
        if (new_book_set.count(id)) {
            new_book_set.erase(id);
        } else {
            std::vector<uint32_t>().swap(posting_lists_[id]);
            std::vector<uint8_t>().swap(db_codes_[id]);
        }
    }

    for (const auto& id : new_book_set) {
        load_from_file_binary<uint32_t>(posting_lists_[id], cluster_path + prefix_id + std::to_string(id) + suffix_id);
        load_from_file_binary<uint8_t>(db_codes_[id], cluster_path + prefix_vector + std::to_string(id) + suffix_vector);
    }

    if (verbose_) {
        std::cout << N_ << " new vectors are added." << std::endl;
    }
}

void 
IndexIVFPQ::load_cq_codebook(std::string cq_codebook_path)
{
    if (verbose_) { std::cout << "Start to load cq codebook" << std::endl; }

    if (cq_codebook_path.back() != '/') {
        cq_codebook_path += "/";
    }

    std::string cq_suffix = "cq_";

    cq_ = std::make_unique<Quantizer::Quantizer>(D_, 200'000, mc, kc, true);
    cq_->load(cq_codebook_path + cq_suffix);

    centers_cq_ = cq_->get_centroids()[0];      // Because mc == 1
    std::cerr << "CQ codebook loaded.\n";
}

void 
IndexIVFPQ::load_pq_codebook(std::string pq_codebook_path)
{
    if (verbose_) { std::cout << "Start to load pq codebook" << std::endl; }

    if (pq_codebook_path.back() != '/') {
        pq_codebook_path += "/";
    }

    std::string pq_suffix = "pq_";
    /**
     * @todo load should change the mp, kp from file. 
    */
    pq_ = std::make_unique<Quantizer::Quantizer>(D_, 200'000, mp, kp, true);
    pq_->load(pq_codebook_path + pq_suffix);

    centers_pq_ = pq_->get_centroids();
    std::cerr << "PQ codebook loaded.\n";
}

void
IndexIVFPQ::top_w_id(
    int w, 
    const std::vector<std::vector<float>>& queries,
    std::vector<std::vector<uint32_t>>& topw
)
{
    if (cq_ == nullptr) {
        std::cerr << "Coarse quantizer not initialized yet!" << std::endl;
        throw;
    }
    
    topw.resize(queries.size(), std::vector<uint32_t>(w));

    for (size_t n = 0; n < queries.size(); ++n) {
        const auto& query = queries[n];
        assert(query.size() == D_);

        std::vector<std::pair<size_t, float>> scores_coarse(centers_cq_.size());
        for (size_t no = 0; no < kc; ++no) {
            scores_coarse[no] = {no, fvec_L2sqr(query.data(), centers_cq_[no].data(), D_)};
        }

        std::partial_sort(scores_coarse.begin(), scores_coarse.begin() + w, scores_coarse.end(),
            [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                return a.second < b.second;
            }
        );

        scores_coarse.resize(w);
        scores_coarse.shrink_to_fit();
        for (size_t i = 0; i < w; ++i) {
            size_t no = scores_coarse[i].first;
            topw[n][i] = no;
        }
    }
}

void
IndexIVFPQ::top_k_id(
    int k, 
    const std::vector<std::vector<float>>& queries, 
    const std::vector<std::vector<uint32_t>>& topw, 
    std::vector<std::vector<uint32_t>>& topk_id,
    std::vector<std::vector<float>>& topk_dist,
    int num_threads
)
{
    if (pq_ == nullptr || cq_ == nullptr) {
        std::cerr << "Product quantizer not initialized yet!" << std::endl;
        throw;
    }

    topk_id.resize(queries.size());
    topk_dist.resize(queries.size());
    
    size_t num_searched_cluster = 0;
    size_t num_searched_vector = 0;

    #pragma omp parallel for\
    reduction(+:num_searched_cluster, num_searched_vector) num_threads(num_threads)
    for (size_t n = 0; n < queries.size(); ++n) {
        const auto& query = queries[n];
        // assert(query.size() == D_);
        DistanceTable dtable = DTable(query);

        std::vector<std::pair<uint32_t, float>> scores;
        scores.reserve(L_);
        for (const auto& no : topw[n]) {
            // assert(no < 1000);
            size_t posting_lists_len = posting_lists_[no].size();
            num_searched_cluster++;

            for (size_t idx = 0; idx < posting_lists_len; ++idx) {
                const auto& n = posting_lists_[no][idx];
                scores.emplace_back(n, ADist(dtable, no, idx));
                num_searched_vector++;
            }
        }
        size_t searched_cnt = std::min(scores.size(), (size_t)k);
        std::partial_sort(scores.begin(), scores.begin() + searched_cnt, scores.end(),
            [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                return a.second < b.second;
            }
        );
        scores.resize(searched_cnt);
        scores.shrink_to_fit();
        for (const auto& [id, d] : scores) {
            topk_id[n].emplace_back(id);
            topk_dist[n].emplace_back(d);
        }
    }
    std::cerr << "num_searched_cluster: " << num_searched_cluster << '\n';
    std::cerr << "num_searched_vector: " << num_searched_vector << '\n';
}

void 
IndexIVFPQ::populate(const std::vector<float>& rawdata)
{
    assert(rawdata.size() / D_ == N_);
    if (!is_trained_ || centers_cq_.empty()) {
        std::cerr << "Error. train() must be called before running populate(vecs=X).\n";
        throw;
    }

    if (verbose_) { std::cout << "Start to update posting lists" << std::endl; }

    posting_lists_.clear();
    posting_lists_.resize(kc);
    db_codes_.clear();
    db_codes_.resize(kc);

    for (auto& posting_list : posting_lists_) {
        posting_list.reserve(N_ / kc);  // Roughly malloc
    }
    for (auto& code : db_codes_) {
        code.reserve(N_ / kc);  // Roughly malloc
    }
    insert_ivf(rawdata);

    if (verbose_) {
        std::cout << N_ << " new vectors are added." << std::endl;
    }
}

void 
IndexIVFPQ::load_index(std::string index_path)
{
    if (index_path.back() != '/') {
        index_path += "/";
    }

    load_cq_codebook(index_path);
    load_pq_codebook(index_path);

    is_trained_ = true;
}

void
IndexIVFPQ::write_index(std::string index_path)
{
    if (index_path.back() != '/') {
        index_path += "/";
    }
    std::string cq_suffix = "cq_", pq_suffix = "pq_";
    cq_->write(index_path + cq_suffix);
    pq_->write(index_path + pq_suffix);
}


void
IndexIVFPQ::query_baseline(
    const std::vector<float>& query,
    std::vector<size_t>& nnid,
    std::vector<float>& dist,
    size_t& searched_cnt,
    int topk,
    int L,
    int id, 
    int W
)
{
    DistanceTable dtable = DTable(query);

    std::vector<std::pair<size_t, float>> scores_coarse(centers_cq_.size());
    for (size_t no = 0; no < kc; ++no) {
        scores_coarse[no] = {no, fvec_L2sqr(query.data(), centers_cq_[no].data(), D_)};
    }

    std::partial_sort(scores_coarse.begin(), scores_coarse.begin() + W, scores_coarse.end(),
        [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
            return a.second < b.second;
        }
    );

    assert(query.size() == D_);

    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(L);
    size_t coarse_cnt = 0;
    for (const auto& score_coarse : scores_coarse) {
        size_t no = score_coarse.first;
        size_t posting_lists_len = posting_lists_[no].size();

        for (size_t idx = 0; idx < posting_lists_len; ++idx) {
            const auto& n = posting_lists_[no][idx];
            scores.emplace_back(n, ADist(dtable, no, idx));
        }

        coarse_cnt++;
        if (coarse_cnt == W) {
            searched_cnt = scores.size();
            topk = std::min(topk, (int)searched_cnt);
            std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                    return a.second < b.second;
                }
            );
            scores.resize(topk);
            scores.shrink_to_fit();
            for (size_t i = 0; i < scores.size(); ++i) {
                const auto& [id, d] = scores[i];
                nnid[i] = id;
                dist[i] = d;
            }
            return;
        }
    }
}

void
IndexIVFPQ::query_obs(
    const std::vector<float>& query,
    const std::vector<int>& gt,
    std::vector<size_t>& nnid,
    std::vector<float>& dist,
    size_t& searched_cnt,
    int topk,
    int L,
    int id
)
{
    std::vector<std::pair<size_t, float>> scores_coarse(centers_cq_.size());
    DistanceTable dtable = DTable(query);

    for (size_t no = 0; no < kc; ++no) {
        scores_coarse[no] = {no, fvec_L2sqr(query.data(), centers_cq_[no].data(), D_)};
    }

    std::unordered_set<int> gt_set;
    gt_set = std::unordered_set<int>(gt.begin(), gt.end());

    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(L);
    int coarse_cnt = 0;
    printf("===== Query %d =====\n", id);
    for (const auto& score_coarse : scores_coarse) {
        size_t no = score_coarse.first;
        size_t hit_count = 0, posting_lists_len = posting_lists_[no].size();

        for (size_t idx = 0; idx < posting_lists_len; ++idx) {
            const auto& n = posting_lists_[no][idx];
            if (gt_set.count(n)) {
                hit_count ++;
            }
            scores.emplace_back(n, ADist(dtable, no, idx));
        }

        std::cerr << std::fixed << std::setprecision(2) 
                << ' ' << hit_count << std::endl;

        coarse_cnt++;
    }
    searched_cnt = scores.size();
    topk = std::min(topk, (int)searched_cnt);
    std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
        [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
            return a.second < b.second;
        }
    );
    scores.resize(topk);
    scores.shrink_to_fit();
    for (size_t i = 0; i < scores.size(); ++i) {
        const auto& [id, d] = scores[i];
        nnid[i] = id;
        dist[i] = d;
    }
    return;
}

void
IndexIVFPQ::query_exhausted(
    const std::vector<float>& query,
    const std::vector<int>& gt,
    std::vector<size_t>& nnid,
    std::vector<float>& dist,
    size_t& searched_cnt,
    int topk,
    int L,
    int id
)
{
    if (write_trainset_path_ == "") {
        std::cerr << "write_trainset_ not set!" << std::endl;
        throw;
    }

    DistanceTable dtable = DTable(query);

    std::vector<std::pair<size_t, float>> scores_coarse(centers_cq_.size());
    for (size_t no = 0; no < kc; ++no) {
        scores_coarse[no] = {no, fvec_L2sqr(query.data(), centers_cq_[no].data(), D_)};
    }

    std::unordered_set<int> gt_set;
    gt_set = std::unordered_set<int>(gt.begin(), gt.end());
    assert(query.size() == D_);
    // for (size_t d = 0; d < D_; ++d) {
    //     queryraw_[id * D_ + d] = query[d];
    // }

    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(L);
    int coarse_cnt = 0;
    for (const auto& score_coarse : scores_coarse) {
        size_t no = score_coarse.first;
        size_t hit_count = 0, posting_lists_len = posting_lists_[no].size();

        for (size_t idx = 0; idx < posting_lists_len; ++idx) {
            const auto& n = posting_lists_[no][idx];
            if (gt_set.count(n)) {
                hit_count ++;
            }
            scores.emplace_back(n, ADist(dtable, no, idx));
        }

        coarse_cnt++;
    }
    searched_cnt = scores.size();
    topk = std::min(topk, (int)searched_cnt);
    std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
        [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
            return a.second < b.second;
        }
    );
    scores.resize(topk);
    scores.shrink_to_fit();
    for (size_t i = 0; i < scores.size(); ++i) {
        const auto& [id, d] = scores[i];
        nnid[i] = id;
        dist[i] = d;
    }
    return;
}


void 
IndexIVFPQ::write_trainset()
{
    auto dataset_name = write_trainset_path_;
    if (dataset_name.back() != '/') {
        dataset_name += "/";
    }
    std::string prefix;
    if (write_trainset_type_ == 0) {
        prefix = "train_";
    } else if (write_trainset_type_ == 1) {
        prefix = "tuning_";
    } else {
        throw;
    }
    std::string f_suffix = ".fvecs", i_suffix = ".ivecs";
    // write_to_file_binary(L, {nq, kc}, dataset_name + prefix + "l" + f_suffix);
    // write_to_file_binary(R, {nq, kc}, dataset_name + prefix + "r" + f_suffix);
    // write_to_file_binary(Q, {nq, kc}, dataset_name + prefix + "q" + f_suffix);
    // write_to_file_binary(queryraw_, {nq, D_}, dataset_name + prefix + "query" + f_suffix);

    // write_to_file_binary(radius_, {nq, kc}, dataset_name + prefix + "radius" + i_suffix);
}

void 
IndexIVFPQ::write_cluster_vector()
{
    auto dataset_name = write_cluster_vector_path_;
    if (dataset_name.back() != '/') {
        dataset_name += "/";
    }
    std::string prefix = "pqcode_";
    std::string f_suffix = ".fvecs", ui8_suffix = ".ui8vecs", ul_suffix = ".ulvecs";
    std::vector<size_t> posting_lists_lens(kc);
    for (size_t no = 0; no < kc; ++no) {
        size_t posting_lists_len = posting_lists_[no].size();
        auto cluster_vector_name = dataset_name + prefix + std::to_string(no) + ui8_suffix;
        write_to_file_binary(db_codes_[no], {posting_lists_len, mp}, cluster_vector_name);
        posting_lists_lens[no] = posting_lists_len;
    }

    write_to_file_binary(posting_lists_lens, {1, kc}, dataset_name + "posting_lists_lens" + ul_suffix);
}

void 
IndexIVFPQ::write_cluster_id()
{
    auto dataset_name = write_cluster_id_path_;
    if (dataset_name.back() != '/') {
        dataset_name += "/";
    }
    std::string prefix = "id_";
    std::string f_suffix = ".fvecs", ui_suffix = ".uivecs";
    for (size_t no = 0; no < kc; ++no) {
        size_t posting_lists_len = posting_lists_[no].size();
        auto cluster_id_name = dataset_name + prefix + std::to_string(no) + ui_suffix;
        write_to_file_binary(posting_lists_[no], {1, posting_lists_len}, cluster_id_name);
    }
}

void
IndexIVFPQ::finalize()
{
    if (write_trainset_path_ != "") {
        write_trainset();
    }

    if (write_cluster_vector_path_ != "") {
        write_cluster_vector();
    }

    if (write_cluster_id_path_ != "") {
        write_cluster_id();
    }
}

void 
IndexIVFPQ::show_statistics()
{

}


DistanceTable 
IndexIVFPQ::DTable(const std::vector<float>& vec) const
{
    const auto& v = vec;
    // Ds: Dimension of each sub-space
    size_t Ds = centers_pq_[0][0].size();
    assert((size_t) v.size() == mp * Ds);
    DistanceTable dtable(mp, kp);
    for (size_t m = 0; m < mp; ++m) {
        for (size_t ks = 0; ks < kp; ++ks) {
            dtable.set_value(m, ks, fvec_L2sqr(&(v[m * Ds]), centers_pq_[m][ks].data(), Ds));
        }
    }
    return dtable;
}

float 
IndexIVFPQ::ADist(const DistanceTable& dtable, const std::vector<uint8_t>& code) const
{
    assert(code.size() == mp);
    float dist = 0;
    for (size_t m = 0; m < mp; ++m) {
        uint8_t ks = code[m];
        dist += dtable.get_value(m, ks);
    }
    return dist;
}

float 
IndexIVFPQ::ADist(const DistanceTable& dtable, size_t list_no, size_t offset) const
{
    float dist = 0;
    for (size_t m = 0; m < mp; ++m) {
        uint8_t ks = nth_vector_mth_element(list_no, offset, m);
        dist += dtable.get_value(m, ks);
    }
    return dist;
}

std::vector<uint8_t> 
IndexIVFPQ::get_single_code(size_t list_no, size_t offset) const
{
    return std::vector<uint8_t>(db_codes_[list_no].begin() + offset * mp, 
                            db_codes_[list_no].begin() + (offset + 1) * mp);
}

template<typename T>
const std::vector<T> 
IndexIVFPQ::nth_raw_vector(const std::vector<T>& long_code, size_t n) const
{
    return std::vector<T>(long_code.begin() + n * D_, long_code.begin() + (n + 1) * D_);
}

uint8_t 
IndexIVFPQ::nth_vector_mth_element(const std::vector<uint8_t>& long_code, size_t n, size_t m) const
{
    return long_code[ n * mp + m];
}

uint8_t 
IndexIVFPQ::nth_vector_mth_element(size_t list_no, size_t offset, size_t m) const
{
    return db_codes_[list_no][ offset * mp + m];
}

void IndexIVFPQ::set_cluster_vector_path(std::string cluster_vector_path) 
{
    write_cluster_vector_path_ = cluster_vector_path;
}

void IndexIVFPQ::set_cluster_id_path(std::string cluster_id_path) 
{
    write_cluster_id_path_ = cluster_id_path;
}

void IndexIVFPQ::set_trainset_path(std::string trainset_path, int trainset_type)
{
    write_trainset_path_ = trainset_path;
    write_trainset_type_ = trainset_type;
}