#include <index_ivf.hpp>

using namespace toy;

extern "C" {
    void omp_init_lock(omp_lock_t *);
    void omp_set_lock(omp_lock_t *);
    void omp_unset_lock(omp_lock_t *);
    void omp_destroy_lock(omp_lock_t *);
}

IVFConfig::IVFConfig(
    size_t N, size_t D, size_t L, 
    size_t kc, 
    size_t mc, 
    size_t dc, 
    std::string index_path, std::string db_path
) : N_(N), D_(D), L_(L), 
    kc(kc), mc(mc), dc(dc),
    index_path(index_path), db_path(db_path)
{}

template <typename T>
IndexIVF<T>::IndexIVF(const IVFConfig& cfg, size_t nq, bool verbose)
    : N_(cfg.N_), D_(cfg.D_), L_(cfg.L_), nq(nq), 
    kc(cfg.kc), mc(cfg.mc), dc(cfg.dc)
{
    verbose_ = verbose;
    assert(dc == D_ && mc == 1);

    cq_ = nullptr;

    if (verbose_) {
        // Check which SIMD functions are used. See distance.hpp for this global variable.
        std::cout << "SIMD support: " << g_simd_architecture << std::endl;
    }
}

template <typename T>
void IndexIVF<T>::Train(const std::vector<T>& rawdata, int seed, size_t nsamples)
{
    size_t Nt_ = rawdata.size() / D_;
    if (nsamples < 100'000) nsamples = 100'000;
    if (nsamples > Nt_) nsamples = Nt_;
    if (nsamples > N_) nsamples = N_;

    if (verbose_) std::cout << "Training index with " << nsamples << " samples" << std::endl;
    std::unique_ptr<std::vector<T>> traindata;

    std::vector<size_t> ids(N_);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 default_random_engine(seed);
    std::shuffle(ids.begin(), ids.end(), default_random_engine);

    traindata = std::make_unique<std::vector<T>>();
    traindata->reserve(nsamples * D_);

    for (size_t k = 0; k < nsamples; ++k) {
        size_t id = ids[k];
        traindata->insert(traindata->end(), 
                rawdata.begin() + id * D_, rawdata.begin() + (id + 1) * D_);
    }

    cq_ = std::make_unique<Quantizer::Quantizer<T>>(D_, nsamples, mc, kc, true);
    cq_->fit(*traindata, 12, seed);
    centers_cq_ = cq_->get_centroids()[0];      // Because mc == 1
    labels_cq_ = cq_->GetAssignments()[0];

    is_trained_ = true;
}

template <typename T> 
void IndexIVF<T>::InsertIvf(const std::vector<T>& rawdata)
{
    std::vector<omp_lock_t> locks(kc);

    for (int i = 0; i < kc; ++i) {
        omp_init_lock(&locks[i]);
    }

    std::cerr << "Start to insert rawdata to IVF index" << std::endl;
    Timer timer_insert_ivf;
    timer_insert_ivf.Start();

    #pragma omp parallel for
    for (size_t n = 0; n < N_; ++n) {
        // const auto& vec = NthRawVector(rawdata, n);
        // int id = cq_->predict_one(vec.data(), 0);
        int id = cq_->predict_one(rawdata.data() + n * D_, 0);
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
            // const auto& nth_code = NthRawVector(rawdata, id);
            // db_codes_[no].insert(db_codes_[no].end(), nth_code.begin(), nth_code.end());
            db_codes_[no].insert(db_codes_[no].end(), rawdata.data() + id * D_, rawdata.data() + (id + 1) * D_);
        }
    }
    timer_insert_ivf.Stop();
    std::cerr << "Time of inserting rawdata to IVF index: " << timer_insert_ivf.GetTime() << " s" << std::endl;
}

template <typename T> 
void IndexIVF<T>::LoadCqCodebook(std::string cq_codebook_path)
{
    if (verbose_) { std::cout << "Start to load cq codebook" << std::endl; }

    if (cq_codebook_path.back() != '/') {
        cq_codebook_path += "/";
    }

    std::string cq_suffix = "cq_";

    cq_ = std::make_unique<Quantizer::Quantizer<T>>(D_, 200'000, mc, kc, true);
    cq_->Load(cq_codebook_path + cq_suffix);

    centers_cq_ = cq_->get_centroids()[0];      // Because mc == 1
    std::cerr << "CQ codebook loaded.\n";
}

template <typename T> 
void IndexIVF<T>::Populate(const std::vector<T>& rawdata)
{
    assert(rawdata.size() / D_ == N_);
    if (!is_trained_ || centers_cq_.empty()) {
        std::cerr << "Error. Train() must be called before running Populate(vecs=X).\n";
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
    InsertIvf(rawdata);

    if (verbose_) {
        std::cout << N_ << " new vectors are added." << std::endl;
    }
}

template <typename T>
void IndexIVF<T>::LoadIndex(std::string index_path)
{
    if (index_path.back() != '/') {
        index_path += "/";
    }

    LoadCqCodebook(index_path);

    is_trained_ = true;
}

template <typename T> 
void IndexIVF<T>::WriteIndex(std::string index_path)
{
    if (index_path.back() != '/') {
        index_path += "/";
    }
    std::string cq_suffix = "cq_";
    cq_->Write(index_path + cq_suffix);
}

template <typename T>
void IndexIVF<T>::QueryBaseline(
    const std::vector<T>& query,
    std::vector<size_t>& nnid,
    std::vector<float>& dist,
    size_t& searched_cnt,
    int topk,
    int L,
    int id, 
    int W
) 
{
    std::vector<std::pair<size_t, float>> scores_coarse(centers_cq_.size());
    for (size_t no = 0; no < kc; ++no) {
        scores_coarse[no] = {no, fvec_L2sqr(query.data(), centers_cq_[no].data(), D_)};
    }

    W = std::min(W, (int)kc);
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
            scores.emplace_back(n, fvec_L2sqr(query.data(), GetSingleCode(no, idx), D_));
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

template <typename T>
const T*
IndexIVF<T>::GetSingleCode(size_t list_no, size_t offset) const
{
    return db_codes_[list_no].data() + offset * D_;
}

template<typename T>
const std::vector<T> 
IndexIVF<T>::NthRawVector(const std::vector<T>& long_code, size_t n) const
{
    return std::vector<T>(long_code.begin() + n * D_, long_code.begin() + (n + 1) * D_);
}


template class IndexIVF<float>;
template class IndexIVF<uint8_t>;