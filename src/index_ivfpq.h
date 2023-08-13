#ifndef INDEX_IVFPQ_H
#define INDEX_IVFPQ_H

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <fstream>
#include <mutex>
#include "pqkmeans.h"
#include "util.h"
#include "quantizer.h"
#include <omp.h>

namespace Toy {

/**
 * Configuration structure
 * @param N_ the number of data
 * @param D_ the number of dimensions
 * @param W_ the number of bucket involed when searching is performed
 * @param L_ the expected number of candidates involed when searching is performed
 * @param kc, kp the number of coarse quantizer (nlist) and product quantizer's centers (1 << nbits). Default: 100, 256
 * @param mc, mp the number of subspace for coarse quantize and product quantizer. mc must be 1
 * @param dc, dp the dimensions of subspace for coarse quantize and product quantizer. dc must be D_.  dp = D_ / mp
 * @param db_path path to the DB files
 * @param db_prefix the prefix of DB files
 */
struct IVFPQConfig {
	size_t N_, D_, W_, L_, kc, kp, mc, mp, dc, dp;
	// std::string db_prefix;
	// std::string db_path;
    IVFPQConfig(size_t N, size_t D, size_t W, size_t L, 
                size_t kc, size_t kp, 
                size_t mc, size_t mp, 
                size_t dc, size_t dp) : N_(N), D_(D), W_(W), L_(L), 
                kc(kc), kp(kp), mc(mc), mp(mp), dc(dc), dp(dp) {}
};

struct DistanceTable{
    // Helper structure. This is identical to vec<vec<float>> dt(M, vec<float>(Ks))
    DistanceTable() {}
    DistanceTable(size_t M, size_t Ks) : kp(Ks), data_(M * Ks) {}
    void SetVal(size_t m, size_t ks, float val) {
        data_[m * kp + ks] = val;
    }
    float GetVal(size_t m, size_t ks) const {
        return data_[m * kp + ks];
    }
    size_t kp;
    std::vector<float> data_;
};

class IndexIVFPQ {
public:
    IndexIVFPQ(const IVFPQConfig& cfg, bool verbose, bool write_trainset);

    void Reconfigure(int nlist, int iter);
    void populate(const std::vector<float> &rawdata);
    void train(const std::vector<float>& traindata, int seed, bool need_split);

    std::pair<std::vector<size_t>, std::vector<float>> query(const std::vector<float>& query,
                                                             const std::vector<int>& gt,
                                                             int topk,
                                                             int L);
    void Clear();

    void UpdatePostingLists(size_t num);
    void insert_ivf(const std::vector<float>& rawdata);
    DistanceTable DTable(const std::vector<float>& vec) const;
    float ADist(const DistanceTable& dtable, const std::vector<uint8_t> &code) const;
    float ADist(const DistanceTable& dtable, size_t list_no, size_t offset) const;
    float ADist(const DistanceTable& dtable, const std::vector<uint8_t> &flattened_codes, size_t n) const;
    std::pair<std::vector<size_t>, std::vector<float>> PairVectorToVectorPair(const std::vector<std::pair<size_t, float>> &pair_vec) const;

    // Property getter
    size_t GetN() const {return flattened_codes_.size() / mp;}
    size_t GetNumList() const {return centers_cq_.size();}

    std::vector<uint8_t> get_single_code(size_t list_no, size_t offset) const;
    // Given a long (N * M) codes, pick up n-th code
    template<typename T>
    std::vector<T> nth_raw_vector(const std::vector<T> &long_code, size_t n) const;
    // Given a long (N * M) codes, pick up m-th element from n-th code
    uint8_t nth_vector_mth_element(const std::vector<uint8_t> &long_code, size_t n, size_t m) const;
    uint8_t nth_vector_mth_element(size_t list_no, size_t offset, size_t m) const;
    std::vector<uint8_t>  encode(const std::vector<float> &vec) const;
    // Member variables
    size_t N_, D_, W_, L_, kc, kp, mc, mp, dc, dp;
    bool verbose_, write_trainset_, is_trained_;

    std::unique_ptr<Quantizer::Quantizer> cq_, pq_;

    std::vector<std::vector<float>> centers_cq_;
    std::vector<int> labels_cq_;

    std::vector<std::vector<std::vector<float>>> centers_pq_;
    std::vector<std::vector<int>> labels_pq_;


    std::vector<std::vector<std::vector<float>>> codewords_;  // (M, Ks, Ds)
    std::vector<std::vector<uint8_t>> coarse_centers_;  // (kc, mc) => (kc, D_)
    std::vector<uint8_t> flattened_codes_;  // (N, M) PQ codes are flattened to N * M long array
    std::vector<std::vector<uint8_t>> codes_; // binary codes, size nlist
    std::vector<std::vector<int>> posting_lists_;  // (NumList, any)
    std::vector<std::vector<float>> posting_dist_lists_;  // (NumList, any)

    std::ofstream write_centroid_word;
    std::ofstream write_centroid_distribution;
    std::ofstream write_l, write_r;
    std::ofstream write_queryword;
};


IndexIVFPQ::IndexIVFPQ(const IVFPQConfig& cfg, bool verbose, bool write_trainset)
    : N_(cfg.N_), D_(cfg.D_), W_(cfg.W_), L_(cfg.L_), 
    kc(cfg.kc), kp(cfg.kp), mc(cfg.mc), mp(cfg.mp), dc(cfg.dc), dp(cfg.dp)
{
    verbose_ = verbose;
    write_trainset_ = write_trainset;
    assert(dc == D_ && mc == 1);

    cq_ = nullptr;
    pq_ = nullptr;

    if (write_trainset) {
        std::string dataset_name = "sift1m_";
        write_centroid_word = std::ofstream(dataset_name + "centroid_word.csv");
        write_centroid_distribution = std::ofstream(dataset_name + "centroid_distribution.csv");
        write_l = std::ofstream(dataset_name + "l.csv");
        write_r = std::ofstream(dataset_name + "r.csv");
        write_queryword = std::ofstream(dataset_name + "queryword.csv");
    }

    if (verbose_) {
        // Check which SIMD functions are used. See distance.h for this global variable.
        std::cout << "SIMD support: " << g_simd_architecture << std::endl;
    }
}

void 
IndexIVFPQ::train(const std::vector<float>& rawdata, int seed, bool need_split)
{
    std::vector<float>* traindata = nullptr;
    size_t Nt_ = rawdata.size() / D_;
    if (need_split) {
        Nt_ = std::min(N_, (size_t)200'000);
        // Nt_ = std::min(N_, (size_t)10'000);      // for test only
        std::vector<size_t> ids(N_);
        std::iota(ids.begin(), ids.end(), 0);
        std::mt19937 default_random_engine(seed);
        std::shuffle(ids.begin(), ids.end(), default_random_engine);
        traindata = new std::vector<float>();
        traindata->reserve(Nt_ * D_);
        for (size_t k = 0; k < Nt_; ++k) {
            size_t id = ids[k];
            traindata->insert(traindata->end(), rawdata.begin() + id * D_, rawdata.begin() + (id + 1) * D_);
        }
    } else {
        traindata = const_cast<std::vector<float>*>(&rawdata);
    }

    cq_ = std::make_unique<Quantizer::Quantizer>(D_, Nt_, mc, kc, 10, true);
    cq_->fit(*traindata, 10, seed);
    centers_cq_ = cq_->get_centroids()[0];      // Because mc == 1
    labels_cq_ = cq_->get_assignments()[0];

    std::vector<size_t> labels_cnt(kc);
    for (auto label : labels_cq_) {
        labels_cnt[label] ++;
    }
    std::sort(labels_cnt.begin(), labels_cnt.end());
    for (auto cnt : labels_cnt) {
        std::cout << cnt << '\n';
    }

    pq_ = std::make_unique<Quantizer::Quantizer>(D_, Nt_, mp, kp, 10, true);
    pq_->fit(*traindata, 5, seed);
    centers_pq_ = pq_->get_centroids();
    labels_pq_ = pq_->get_assignments();

    if (traindata != &rawdata) delete traindata;
    is_trained_ = true;
}

void 
IndexIVFPQ::Reconfigure(int nlist, int iter)
{
    assert(0 < nlist);
    assert((size_t) nlist <= GetN());

    // ===== (1) Sampling vectors for pqk-means =====
    // Since clustering takes time, we use a subset of all codes for clustering.
    size_t len_for_clustering = std::min(GetN(), (size_t) nlist * 5'000);
    if (verbose_) {
        std::cout << "The number of vectors used for training of coarse centers: " << len_for_clustering << std::endl;
    }
    std::vector<size_t> ids_for_clustering(GetN());  // This can be large and might be the bottle neck of memory consumption
    std::iota(ids_for_clustering.begin(), ids_for_clustering.end(), 0);
    std::shuffle(ids_for_clustering.begin(), ids_for_clustering.end(), std::default_random_engine(123));
    ids_for_clustering.resize(len_for_clustering);
    ids_for_clustering.shrink_to_fit();  // For efficient memory usage

    std::vector<uint8_t> flattened_codes_randomly_picked;  // size=len_for_clustering
    flattened_codes_randomly_picked.reserve(len_for_clustering * mp);
    for (const auto &id : ids_for_clustering) {  // Pick up vectors to construct a training set
        std::vector<uint8_t> code = nth_raw_vector(flattened_codes_, id);
        flattened_codes_randomly_picked.insert(flattened_codes_randomly_picked.end(),
                                               code.begin(), code.end());
    }
    assert(flattened_codes_randomly_picked.size() == len_for_clustering * mp);

    Timer timer1, timer2;

    // ===== (2) Run pqk-means =====
    timer1.start();
    if (verbose_) {std::cout << "Start to run PQk-means" << std::endl;}
    pqkmeans::PQKMeans clustering_instance(codewords_, nlist, iter, verbose_);
    clustering_instance.fit(flattened_codes_randomly_picked);


    // ===== (3) Update coarse centers =====
    coarse_centers_ = clustering_instance.get_centroids();
    assert(coarse_centers_.size() == (size_t) nlist);
    assert(coarse_centers_[0].size() == mp);
    timer1.stop();


    // ===== (4) Update posting lists =====
    timer2.start();
    if (verbose_) {std::cout << "Start to update posting lists" << std::endl;}

    UpdatePostingLists(GetN());
    timer2.stop();

    std::cout << timer1.get_time() << ' ' << timer2.get_time() << " | " 
        << timer1.get_time() / (timer1.get_time() + timer2.get_time()) << '\n';
}

void 
IndexIVFPQ::insert_ivf(const std::vector<float>& rawdata)
{
    // // ===== (1) Construct a dummy pqkmeans class for computing Symmetric Distance =====
    // pqkmeans::PQKMeans clustering_instance(codewords_, (int)GetNumList(), 0, true);
    // clustering_instance.set_centroids(coarse_centers_);

    // ===== (2) Update posting lists =====
    // std::vector<size_t> assign(num);
    // puts("########################");
    // std::cout << num << ' ' << *std::max_element(labels_cq_.begin(), labels_cq_.end()) << '\n';
    // std::cout << labels_cq_.size() << '\n';
    // std::cout << posting_lists_.size() << '\n';
    // std::cout << posting_dist_lists_.size() << '\n';
    std::cout << nth_raw_vector(rawdata, 0).size() << '\n';
    assert(nth_raw_vector(rawdata, 0).size() == D_);
#pragma omp parallel for
    for (size_t n = 0; n < N_; ++n) {
        int id = cq_->predict_one(nth_raw_vector(rawdata, n), 0);
        #pragma omp critical
        {
            posting_lists_[id].emplace_back(n);
            posting_dist_lists_[id].emplace_back(0);
        }
    }

// #pragma omp parallel for
//     for (size_t no = 0; no < kc; ++no) {
//         auto &plist = posting_lists_[no];
//         const auto &center = coarse_centers_[no];
//         std::sort(plist.begin(), plist.end(), [&](const auto &a, const auto& b) {
//             return clustering_instance.SymmetricDistance(center, nth_raw_vector(flattened_codes_, a))
//                 < clustering_instance.SymmetricDistance(center, nth_raw_vector(flattened_codes_, b));
//         });
//         auto &pdlist = posting_dist_lists_[no];
//         // Which distance should be used? Real distance or PQ distance? 
//         for (size_t i = 0; i < pdlist.size(); ++i) {
//             pdlist[i] = clustering_instance.SymmetricDistance(center, nth_raw_vector(flattened_codes_, plist[i]));
//         }
//         for (const auto& id : plist) {
//             const auto& nth_code = nth_raw_vector(flattened_codes_, id);
//             codes_[no].insert(codes_[no].end(), nth_code.begin(), nth_code.end());
//         }
//     }

    // if (write_trainset_) {
    //     auto &write_d = write_centroid_distribution;
    //     auto &write_w = write_centroid_word;
    //     for (int no = 0; no < kc; ++no) {
    //         const auto& centroidword = coarse_centers_[no];
    //         for (const auto& word : centroidword) write_w << (int)word << ",";
    //         const auto& pdlist = posting_dist_lists_[no];
    //         size_t step = pdlist.size() / 20;
    //         for (size_t i = 0; i < 20; ++i) {
    //             size_t idx = i * step;
    //             write_d << pdlist[idx] << ",";
    //         }
    //         write_d << std::endl;
    //         write_w << std::endl;
    //     }
    // }
}


void 
IndexIVFPQ::populate(const std::vector<float> &rawdata)
{
    assert(rawdata.size() / D_ == N_);
    if (!is_trained_ || centers_cq_.empty()) {
        std::cerr << "Error. train() must be called before running populate(vecs=X).\n";
        throw;
    }

    // ===== (1) Encode rawdata to PQcodes =====
//     std::vector<std::vector<uint8_t>> pqcodes(N_, std::vector<uint8_t>(mp));
// #pragma omp parallel for
//     for (size_t n = 0; n < N_; ++n) {
//         pqcodes[n] = encode(nth_raw_vector(rawdata, n));
//     }
    const auto& pqcodes = pq_->encode(rawdata);

    if (verbose_) {
        std::cout << N_ << " new vectors are added." << std::endl;
    }

    // ===== (2) Update posting lists =====
    if (verbose_) { std::cout << "Start to update posting lists" << std::endl; }

    posting_lists_.clear();
    posting_lists_.resize(kc);
    codes_.clear();
    codes_.resize(kc);
    posting_dist_lists_.clear();
    posting_dist_lists_.resize(kc);

    for (auto &posting_list : posting_lists_) {
        posting_list.reserve(N_ / kc);  // Roughly malloc
    }
    for (auto &posting_dist_list : posting_dist_lists_) {
        posting_dist_list.reserve(N_ / kc);  // Roughly malloc
    }
    for (auto &code : codes_) {
        code.reserve(N_ / kc);  // Roughly malloc
    }
    insert_ivf(rawdata);
}

std::pair<std::vector<size_t>, std::vector<float> > 
IndexIVFPQ::query(const std::vector<float> &query,
                                            const std::vector<int> &gt,
                                            int topk,
                                            int L)
{
    assert((size_t) topk <= GetN());
    assert(topk <= L && (size_t) L <= GetN());

    // ===== (1) Create dtable =====
    DistanceTable dtable = DTable(query);

    // ===== (2) Compare to coarse centers and sort the results =====
    std::vector<std::pair<size_t, float>> scores_coarse(coarse_centers_.size());
    size_t nlist = GetNumList();
//#pragma omp parallel for
    for (size_t no = 0; no < nlist; ++no) {
        scores_coarse[no] = {no, ADist(dtable, coarse_centers_[no])};
    }

    // Timer timer1, timer2;
    // ===== (3) Partial sort the coarse results. =====
    // timer1.start();
    std::partial_sort(scores_coarse.begin(), scores_coarse.begin() + W_, scores_coarse.end(),
                      [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b){return a.second < b.second;});
    // timer1.stop();

    std::unordered_set<int> gt_set;
    if (write_trainset_) {
        const auto &gt_v = gt;
        for (auto i = 0; i < gt_v.size(); ++i) {
            gt_set.insert(gt_v[i]);
        }
        const auto& queryword = pq_->encode(query);
        for (const auto& word : query) {
            write_queryword << (int)word << ",";
        }
        write_queryword << std::endl;
    }

    // ===== (4) Traverse posting list =====
    // timer2.start();
    std::vector<std::pair<size_t, float>> scores;
    scores.reserve(L);
    int coarse_cnt = 0;
    // std::mutex mutex;
    for (const auto &score_coarse : scores_coarse) {
        size_t no = score_coarse.first;
        coarse_cnt++;
        size_t bl = posting_lists_[no].size(), br = 0;
        size_t hit_count = 0, posting_lists_len = posting_lists_[no].size();

        // [TODO] parallelized
        // Doesn't benefit 
// #pragma omp parallel for if(posting_lists_len > 10000)
        for (size_t idx = 0; idx < posting_lists_len; ++idx) {
            // std::lock_guard<std::mutex> lock(mutex);
            const auto &n = posting_lists_[no][idx];
            if (write_trainset_ && gt_set.count(n)) {
                bl = std::min(bl, idx);
                br = std::max(br, idx);
                hit_count ++;
            }
            // ===== (6) Evaluate n =====
            /**
             * scores.emplace_back(n, ADist(dtable, flattened_codes_, n));
             * 3x Faster than ADist(dtable, flattened_codes_, n)
             * Memory locality!
            */
            scores.emplace_back(n, ADist(dtable, no, idx));
        }

        // std::cout << std::fixed << std::setprecision(2)
        //             << (double)bl / posting_lists_[no].size() << ' ' 
        //             << (double)br / posting_lists_[no].size() << "\t"
        //             << hit_count << ' ' << score_coarse.second << '\n';
        if (write_trainset_) {
            write_l << std::fixed << std::setprecision(4) << (double)bl / posting_lists_[no].size() << ",";
            write_r << std::fixed << std::setprecision(4) << (double)br / posting_lists_[no].size() << ",";
        }

        if ( (size_t) coarse_cnt == W_ && scores.size() >= (unsigned long) topk) {
            // ===== (8) Sort them =====
            std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                                [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b){return a.second < b.second;});
            scores.resize(topk);
            scores.shrink_to_fit();
            // for (auto [id, dist] : scores) {
            //     std::cout << id << ' ' << dist << '\n';
            // }

            // std::cout << '\n';
            if (write_trainset_) {
                write_l << std::endl;
                write_r << std::endl;
            }
            // timer2.stop();
            // std::cout << timer1.get_time() << ' ' << timer2.get_time() << " | " 
            //             << timer2.get_time() / (timer1.get_time() + timer2.get_time()) << '\n';
            return PairVectorToVectorPair(scores);  // return pair<vec, vec>
        }
    }
    return std::pair<std::vector<size_t>, std::vector<float>>({}, {});
}

void 
IndexIVFPQ::Clear()
{
    coarse_centers_.clear();
    flattened_codes_.clear();
    posting_lists_.clear();
}

void 
IndexIVFPQ::UpdatePostingLists(size_t num)
{
    // // ===== (1) Construct a dummy pqkmeans class for computing Symmetric Distance =====
    // pqkmeans::PQKMeans clustering_instance(codewords_, (int)GetNumList(), 0, true);
    // clustering_instance.set_centroids(coarse_centers_);

    // ===== (2) Update posting lists =====
    std::vector<size_t> assign(num);
    // puts("########################");
    // std::cout << num << ' ' << *std::max_element(labels_cq_.begin(), labels_cq_.end()) << '\n';
    // std::cout << labels_cq_.size() << '\n';
    // std::cout << posting_lists_.size() << '\n';
    // std::cout << posting_dist_lists_.size() << '\n';
#pragma omp parallel for
    for (size_t n = 0; n < num; ++n) {
        // assign[n] = cq_->predict_one();
        #pragma omp critical
        {
            posting_lists_[assign[n]].emplace_back(n);
            posting_dist_lists_[assign[n]].emplace_back(0);
        }
    }


//     size_t nlist = GetNumList();
// #pragma omp parallel for
//     for (size_t no = 0; no < nlist; ++no) {
//         auto &plist = posting_lists_[no];
//         const auto &center = coarse_centers_[no];
//         std::sort(plist.begin(), plist.end(), [&](const auto &a, const auto& b) {
//             return clustering_instance.SymmetricDistance(center, nth_raw_vector(flattened_codes_, a))
//                 < clustering_instance.SymmetricDistance(center, nth_raw_vector(flattened_codes_, b));
//         });
//         auto &pdlist = posting_dist_lists_[no];
//         for (size_t i = 0; i < pdlist.size(); ++i) {
//             pdlist[i] = clustering_instance.SymmetricDistance(center, nth_raw_vector(flattened_codes_, plist[i]));
//         }
//         for (const auto& id : plist) {
//             const auto& nth_code = nth_raw_vector(flattened_codes_, id);
//             codes_[no].insert(codes_[no].end(), nth_code.begin(), nth_code.end());
//         }
//     }

    // if (write_trainset_) {
    //     auto &write_d = write_centroid_distribution;
    //     auto &write_w = write_centroid_word;
    //     for (int no = 0; no < kc; ++no) {
    //         const auto& centroidword = coarse_centers_[no];
    //         for (const auto& word : centroidword) write_w << (int)word << ",";
    //         const auto& pdlist = posting_dist_lists_[no];
    //         size_t step = pdlist.size() / 20;
    //         for (size_t i = 0; i < 20; ++i) {
    //             size_t idx = i * step;
    //             write_d << pdlist[idx] << ",";
    //         }
    //         write_d << std::endl;
    //         write_w << std::endl;
    //     }
    // }
}

DistanceTable 
IndexIVFPQ::DTable(const std::vector<float> &vec) const
{
    const auto &v = vec;
    size_t Ds = codewords_[0][0].size();
    assert((size_t) v.size() == mp * Ds);
    DistanceTable dtable(mp, kp);
    for (size_t m = 0; m < mp; ++m) {
        for (size_t ks = 0; ks < kp; ++ks) {
            dtable.SetVal(m, ks, fvec_L2sqr(&(v[m * Ds]), codewords_[m][ks].data(), Ds));
        }
    }
    return dtable;
}

float 
IndexIVFPQ::ADist(const DistanceTable &dtable, const std::vector<uint8_t> &code) const
{
    assert(code.size() == mp);
    float dist = 0;
    for (size_t m = 0; m < mp; ++m) {
        uint8_t ks = code[m];
        dist += dtable.GetVal(m, ks);
    }
    return dist;
}

float 
IndexIVFPQ::ADist(const DistanceTable &dtable, const std::vector<uint8_t> &flattened_codes, size_t n) const
{
    float dist = 0;
    for (size_t m = 0; m < mp; ++m) {
        uint8_t ks = nth_vector_mth_element(flattened_codes, n, m);
        dist += dtable.GetVal(m, ks);
    }
    return dist;
}

float 
IndexIVFPQ::ADist(const DistanceTable &dtable, size_t list_no, size_t offset) const
{
    float dist = 0;
    for (size_t m = 0; m < mp; ++m) {
        uint8_t ks = nth_vector_mth_element(list_no, offset, m);
        dist += dtable.GetVal(m, ks);
    }
    return dist;
}

std::pair<std::vector<size_t>, std::vector<float> > 
IndexIVFPQ::PairVectorToVectorPair(const std::vector<std::pair<size_t, float> > &pair_vec) const
{
    std::pair<std::vector<size_t>, std::vector<float>> vec_pair(std::vector<size_t>(pair_vec.size()), std::vector<float>(pair_vec.size()));
    for(size_t n = 0, N = pair_vec.size(); n < N; ++n) {
        vec_pair.first[n] = pair_vec[n].first;
        vec_pair.second[n] = pair_vec[n].second;
    }
    return vec_pair;
}

std::vector<uint8_t> 
IndexIVFPQ::get_single_code(size_t list_no, size_t offset) const
{
    return std::vector<uint8_t>(codes_[list_no].begin() + offset * mp, 
                            codes_[list_no].begin() + (offset + 1) * mp);
}

template<typename T>
std::vector<T> 
IndexIVFPQ::nth_raw_vector(const std::vector<T> &long_code, size_t n) const
{
    return std::vector<T>(long_code.begin() + n * D_, long_code.begin() + (n + 1) * D_);
}

uint8_t 
IndexIVFPQ::nth_vector_mth_element(const std::vector<uint8_t> &long_code, size_t n, size_t m) const
{
    return long_code[ n * mp + m];
}

uint8_t 
IndexIVFPQ::nth_vector_mth_element(size_t list_no, size_t offset, size_t m) const
{
    return codes_[list_no][ offset * mp + m];
}

} // namespace Toy


#endif