#ifndef INDEX_IVFPQ_H
#define INDEX_IVFPQ_H

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <fstream>
#include "util.h"
#include "quantizer.h"
#include "distance.h"
#include <omp.h>

namespace Toy {

/**
 * Configuration structure
 * @param N_ the number of data
 * @param D_ the number of dimensions
 * @param W_ the number of bucket involed when searching is performed
 * @param L_ the expected number of candidates involed when searching is performed
 * @param kc the number of coarse quantizer's (nlist) centers. Default: 100
 * @param mc the number of subspace for coarse quantizer. mc must be 1
 * @param dc the dimensions of subspace for coarse quantize and product quantizer. dc must be D_.  dp = D_ / mp
 * @param db_path path to the DB files
 * @param db_prefix the prefix of DB files
 */
struct IVFConfig {
	size_t N_, D_, W_, L_, kc, mc, dc, segs;
	// std::string db_prefix;
	// std::string db_path;
    IVFConfig(size_t N, size_t D, size_t W, size_t L, 
                size_t kc, 
                size_t mc, 
                size_t dc, size_t segs) : N_(N), D_(D), W_(W), L_(L), 
                kc(kc), mc(mc), dc(dc), segs(segs) {}
};

class IndexIVF {
public:
    IndexIVF(const IVFConfig& cfg, size_t nq, bool verbose, bool write_trainset);

    void populate(const std::vector<float>& rawdata);
    void train(const std::vector<float>& traindata, int seed, bool need_split);

    // IVF baseline
    void
    query_baseline(const std::vector<float>& query,
                    std::vector<size_t>& nnid,
                    std::vector<float>& dist,
                    size_t& searched_cnt,
                    int topk,
                    int L,
                    int id);

    // IVF RF
    void
    query_pred(const std::vector<float>& query,
                const std::vector<int>& pred_radius,
                std::vector<size_t>& nnid,
                std::vector<float>& dist,
                size_t& searched_cnt,
                int cut,
                int topk,
                int L,
                int id);

    void insert_ivf(const std::vector<float>& rawdata);

    const std::vector<float> get_single_code(size_t list_no, size_t offset) const;
    // Given a long (N * M) codes, pick up n-th code
    template<typename T>
    const std::vector<T> nth_raw_vector(const std::vector<T>& long_code, size_t n) const;
    // Member variables
    size_t N_, D_, W_, L_, nq, kc, mc, dc, segs;
    bool verbose_, write_trainset_, is_trained_;

    std::unique_ptr<Quantizer::Quantizer> cq_;

    std::vector<std::vector<float>> centers_cq_;
    std::vector<int> labels_cq_;


    std::vector<std::vector<float>> db_codes_; // binary codes, size nlist
    std::vector<std::vector<int>> posting_lists_;  // (NumList, any)
    std::vector<std::vector<float>> posting_dist_lists_;  // (NumList, any)

    std::vector<float> L, R, Q, distance_, queryraw_;
    std::vector<int> radius_;

    std::vector<std::vector<std::pair<int, int>>> direction_cnt;
};


IndexIVF::IndexIVF(const IVFConfig& cfg, size_t nq, bool verbose, bool write_trainset)
    : N_(cfg.N_), D_(cfg.D_), W_(cfg.W_), L_(cfg.L_), nq(nq), 
    kc(cfg.kc), mc(cfg.mc), dc(cfg.dc), segs(cfg.segs)
{
    verbose_ = verbose;
    write_trainset_ = write_trainset;
    assert(dc == D_ && mc == 1);

    cq_ = nullptr;

    if (write_trainset) {
        L.resize(nq * kc);
        R.resize(nq * kc);
        Q.resize(nq * kc);
        distance_.resize(nq * kc);
        radius_.resize(nq * kc);

        queryraw_.resize(nq * D_);
    }

    if (verbose_) {
        // Check which SIMD functions are used. See distance.h for this global variable.
        std::cout << "SIMD support: " << g_simd_architecture << std::endl;
    }
}

void 
IndexIVF::train(const std::vector<float>& rawdata, int seed, bool need_split)
{
    std::unique_ptr<std::vector<float>> traindata;
    size_t Nt_ = rawdata.size() / D_;
    if (need_split) {
        Nt_ = std::min(N_, (size_t)200'000);
        // Nt_ = std::min(N_, (size_t)10'000);      // for test only
        std::vector<size_t> ids(N_);
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937 default_random_engine(seed);
        std::shuffle(ids.begin(), ids.end(), default_random_engine);

        traindata = std::make_unique<std::vector<float>>();
        traindata->reserve(Nt_ * D_);

        for (size_t k = 0; k < Nt_; ++k) {
            size_t id = ids[k];
            traindata->insert(traindata->end(), 
                    rawdata.begin() + id * D_, rawdata.begin() + (id + 1) * D_);
        }
    } else {
        traindata = std::make_unique<std::vector<float>>(rawdata.data(), rawdata.data() + rawdata.size());
    }

    cq_ = std::make_unique<Quantizer::Quantizer>(D_, Nt_, mc, kc, true);
    cq_->fit(*traindata, 12, seed);
    centers_cq_ = cq_->get_centroids()[0];      // Because mc == 1
    labels_cq_ = cq_->get_assignments()[0];

    // std::vector<size_t> labels_cnt(kc);
    // for (auto label : labels_cq_) {
    //     labels_cnt[label] ++;
    // }
    // std::sort(labels_cnt.begin(), labels_cnt.end());
    // for (auto cnt : labels_cnt) {
    //     std::cout << cnt << '\n';
    // }

    is_trained_ = true;
}

void 
IndexIVF::insert_ivf(const std::vector<float>& rawdata)
{
#pragma omp parallel for
    for (size_t n = 0; n < N_; ++n) {
        int id = cq_->predict_one(nth_raw_vector(rawdata, n), 0);
        #pragma omp critical
        {
            posting_lists_[id].emplace_back(n);
            posting_dist_lists_[id].emplace_back(0);
        }
    }

#pragma omp parallel for
    for (size_t no = 0; no < kc; ++no) {
        auto& plist = posting_lists_[no];
        const auto& center = centers_cq_[no];
        auto& pdlist = posting_dist_lists_[no];

        /**
         * Which distance should be used? Real distance or PQ distance? 
         * @todo Finish Quantizer.asynm_dist()
         * @todo Finish Quantizer.synm_dist()
         * */ 
        std::vector<size_t> indices(plist.size());
        std::iota(indices.begin(), indices.end(), 0);
        for (size_t i = 0; i < pdlist.size(); ++i) {
            pdlist[i] = fvec_L2sqr(center.data(), nth_raw_vector(rawdata, plist[i]).data(), D_);
        }

        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return pdlist[a] < pdlist[b];
        });
        auto sorted_pdlist = pdlist;
        auto sorted_plist = plist;
        for (size_t i = 0; i < plist.size(); ++i) {
            sorted_pdlist[i] = pdlist[indices[i]];
            sorted_plist[i] = plist[indices[i]];
        }
        std::swap(pdlist, sorted_pdlist);
        std::swap(plist, sorted_plist);
        std::vector<float>().swap(sorted_pdlist);
        std::vector<int>().swap(sorted_plist);

        for (const auto& id : plist) {
            const auto& nth_code = nth_raw_vector(rawdata, id);
            db_codes_[no].insert(db_codes_[no].end(), nth_code.begin(), nth_code.end());
        }
    }
}


void 
IndexIVF::populate(const std::vector<float>& rawdata)
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
    posting_dist_lists_.clear();
    posting_dist_lists_.resize(kc);

    for (auto& posting_list : posting_lists_) {
        posting_list.reserve(N_ / kc);  // Roughly malloc
    }
    for (auto& posting_dist_list : posting_dist_lists_) {
        posting_dist_list.reserve(N_ / kc);  // Roughly malloc
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
IndexIVF::query_baseline(const std::vector<float>& query,
                                            std::vector<size_t>& nnid,
                                            std::vector<float>& dist,
                                            size_t& searched_cnt,
                                            int topk,
                                            int L,
                                            int id)
{
    std::vector<std::pair<size_t, float>> scores_coarse(centers_cq_.size());
    for (size_t no = 0; no < kc; ++no) {
        scores_coarse[no] = {no, fvec_L2sqr(query.data(), centers_cq_[no].data(), D_)};
    }

    std::partial_sort(scores_coarse.begin(), scores_coarse.begin() + W_, scores_coarse.end(),
                      [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b){return a.second < b.second;});

    assert(query.size() == D_);

    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(L);
    size_t coarse_cnt = 0;
    for (const auto& score_coarse : scores_coarse) {
        size_t no = score_coarse.first;
        size_t posting_lists_len = posting_lists_[no].size();

        for (size_t idx = 0; idx < posting_lists_len; ++idx) {
            const auto& n = posting_lists_[no][idx];
            scores.emplace_back(n, fvec_L2sqr(query.data(), get_single_code(no, idx).data(), D_));
        }

        coarse_cnt++;
        if (coarse_cnt == W_) {
            searched_cnt = scores.size();
            topk = std::min(topk, (int)scores.size());
            std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                                [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b){return a.second < b.second;});
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
IndexIVF::query_pred(const std::vector<float>& query,
                                            const std::vector<int>& pred_radius,
                                            std::vector<size_t>& nnid,
                                            std::vector<float>& dist,
                                            size_t& searched_cnt,
                                            int cut,
                                            int topk,
                                            int L,
                                            int id)
{
    std::vector<std::pair<size_t, float>> scores_coarse(centers_cq_.size());
    for (size_t no = 0; no < kc; ++no) {
        scores_coarse[no] = {no, fvec_L2sqr(query.data(), centers_cq_[no].data(), D_)};
    }

    std::partial_sort(scores_coarse.begin(), scores_coarse.begin() + W_, scores_coarse.end(),
                    [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b){return a.second < b.second;});

    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(L);
    int coarse_cnt = 0;
    for (const auto& score_coarse : scores_coarse) {
        size_t no = score_coarse.first;
        size_t posting_lists_len = posting_lists_[no].size();

        int radius = pred_radius[no];
        if (1 <= radius && radius < cut) {
            radius = cut;
        }
        auto cq = (float)score_coarse.second / posting_dist_lists_[no].back();
        float radius_f = (float)radius / segs;
        float bl_f = cq - radius_f;
        float br_f = cq + radius_f;
        bl_f -= (float)3.5 / segs;
        br_f -= (float)3.5 / segs;
        bl_f = std::min(1.0f, bl_f);
        bl_f = std::max(0.0f, bl_f);
        br_f = std::min(1.0f, br_f);
        br_f = std::max(0.0f, br_f);
        size_t bl = bl_f * posting_lists_len;
        size_t br = br_f * posting_lists_len;
        if (radius == 0) {
            goto end_of_current_cluster;
        }
        for (size_t idx = bl; idx < br; ++idx) {
            const auto& n = posting_lists_[no][idx];
            scores.emplace_back(n, fvec_L2sqr(query.data(), get_single_code(no, idx).data(), D_));
        }

end_of_current_cluster: 
        coarse_cnt++;
        if (coarse_cnt == W_) {
            searched_cnt = scores.size();
            topk = std::min(topk, (int)searched_cnt);
            std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) {
                    return a.second < b.second;
            });
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

const std::vector<float> 
IndexIVF::get_single_code(size_t list_no, size_t offset) const
{
    return std::vector<float>(db_codes_[list_no].begin() + offset * D_, 
                            db_codes_[list_no].begin() + (offset + 1) * D_);
}

template<typename T>
const std::vector<T> 
IndexIVF::nth_raw_vector(const std::vector<T>& long_code, size_t n) const
{
    return std::vector<T>(long_code.begin() + n * D_, long_code.begin() + (n + 1) * D_);
}

} // namespace Toy


#endif