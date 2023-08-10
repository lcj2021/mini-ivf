#pragma once

#include <iostream>
#include <iomanip>
#include <cassert>
#include <unordered_set>
#include <mutex>
#include "./pqkmeans.h"
#include "./distance.h"
// #include "./inverted_lists.h"
#include "./util.h"
#include "product_quantizer.h"

namespace Toy {


struct DistanceTable{
    // Helper structure. This is identical to vec<vec<float>> dt(M, vec<float>(Ks))
    DistanceTable() {}
    DistanceTable(size_t M, size_t Ks) : Ks_(Ks), data_(M * Ks) {}
    void SetVal(size_t m, size_t ks, float val) {
        data_[m * Ks_ + ks] = val;
    }
    float GetVal(size_t m, size_t ks) const {
        return data_[m * Ks_ + ks];
    }
    size_t Ks_;
    std::vector<float> data_;
};



class IndexIVFPQ {
public:
    IndexIVFPQ(const std::vector<std::vector<std::vector<float>>> &codewords, 
                        size_t D, size_t nlist, size_t M, size_t nbits, 
                        bool verbose);

    //void SetCodewords(const py::array_t<float> &codewords);  // This should be called first
    void Reconfigure(int nlist, int iter);
    void AddCodes(const std::vector<std::vector<uint8_t>> &codes, bool update_flag);
    void train();

    std::pair<std::vector<size_t>, std::vector<float>> query(const std::vector<float> &query,
                                                             const std::vector<int> &gt,
                                                             int topk,
                                                             int L,
                                                             int nprobe);
    void Clear();

    void UpdatePostingLists(size_t num);
    DistanceTable DTable(const std::vector<float> &vec) const;
    float ADist(const DistanceTable &dtable, const std::vector<uint8_t> &code) const;
    float ADist(const DistanceTable &dtable, size_t list_no, size_t offset) const;
    float ADist(const DistanceTable &dtable, const std::vector<uint8_t> &flattened_codes, size_t n) const;
    std::pair<std::vector<size_t>, std::vector<float>> PairVectorToVectorPair(const std::vector<std::pair<size_t, float>> &pair_vec) const;

    // Property getter
    size_t GetN() const {return flattened_codes_.size() / M_;}
    size_t GetNumList() const {return coarse_centers_.size();}

    std::vector<uint8_t> get_single_code(size_t list_no, size_t offset) const;
    // Given a long (N * M) codes, pick up n-th code
    std::vector<uint8_t> NthCode(const std::vector<uint8_t> &long_code, size_t n) const;
    // Given a long (N * M) codes, pick up m-th element from n-th code
    uint8_t NthCodeMthElement(const std::vector<uint8_t> &long_code, std::size_t n, size_t m) const;
    uint8_t NthCodeMthElement(std::size_t list_no, size_t offset, size_t m) const;

    // Member variables
    size_t M_, Ks_;
    bool verbose_;
    ProductQuantizer pq_;
    std::vector<std::vector<std::vector<float>>> codewords_;  // (M, Ks, Ds)
    std::vector<std::vector<uint8_t>> coarse_centers_;  // (NumList, M)
    std::vector<uint8_t> flattened_codes_;  // (N, M) PQ codes are flattened to N * M long array
    // ArrayInvertedLists ivl;
    std::vector<std::vector<uint8_t>> codes_; // binary codes, size nlist
    std::vector<std::vector<int>> posting_lists_;  // (NumList, any)
    std::vector<std::vector<float>> posting_dist_lists_;  // (NumList, any)
    std::vector<std::vector<std::array<int, 2>>> posting_lr_lists_;
    // std::vector<std::vector<std::vector<float>>> distance_matrices_among_codewords_;
};


IndexIVFPQ::IndexIVFPQ(const std::vector<std::vector<std::vector<float>>> &codewords, 
                        size_t D, size_t nlist, size_t M, size_t nbits, 
                        bool verbose) : pq_(D, M, nbits)
{
    verbose_ = verbose;
    const auto &r = codewords;  // codewords must have ndim=3, with non-writable
    M_ = (size_t) r.size();
    Ks_ = (size_t) r[0].size();
    size_t Ds = (size_t) r[0][0].size();
    codewords_.resize(M_, 
        std::vector<std::vector<float>>(Ks_, 
        std::vector<float>(Ds)));

    for (ssize_t m = 0; m < M_; ++m) {
        for (ssize_t ks = 0; ks < Ks_; ++ks) {
            for (ssize_t ds = 0; ds < Ds; ++ds) {
                codewords_[m][ks][ds] = r[m][ks][ds];
            }
        }
    }

    if (verbose_) {
        // Check which SIMD functions are used. See distance.h for this global variable.
        std::cout << "SIMD support: " << g_simd_architecture << std::endl;
    }
}

void 
IndexIVFPQ::Reconfigure(int nlist, int iter)
{
    assert(0 < nlist);
    assert((size_t) nlist <= GetN());

    // ===== (1) Sampling vectors for pqk-means =====
    // Since clustering takes time, we use a subset of all codes for clustering.
    size_t len_for_clustering = std::min(GetN(), (size_t) nlist * 2000);
    // size_t len_for_clustering = std::min(GetN(), (size_t)(GetN() * 0.1));
    if (verbose_) {
        std::cout << "The number of vectors used for training of coarse centers: " << len_for_clustering << std::endl;
    }
    // Prepare a random set of integers, drawn from [0, ..., N-1], where the cardinality of the set is len_for_clustering
    std::vector<size_t> ids_for_clustering(GetN());  // This can be large and might be the bottle neck of memory consumption
    std::iota(ids_for_clustering.begin(), ids_for_clustering.end(), 0);
    std::shuffle(ids_for_clustering.begin(), ids_for_clustering.end(), std::default_random_engine(123));
    ids_for_clustering.resize(len_for_clustering);
    ids_for_clustering.shrink_to_fit();  // For efficient memory usage

    std::vector<uint8_t> flattened_codes_randomly_picked;  // size=len_for_clustering
    flattened_codes_randomly_picked.reserve(len_for_clustering * M_);
    for (const auto &id : ids_for_clustering) {  // Pick up vectors to construct a training set
        std::vector<uint8_t> code = NthCode(flattened_codes_, id);
        flattened_codes_randomly_picked.insert(flattened_codes_randomly_picked.end(),
                                               code.begin(), code.end());
    }
    assert(flattened_codes_randomly_picked.size() == len_for_clustering * M_);

    Timer timer1, timer2;

    // ===== (2) Run pqk-means =====
    timer1.start();
    if (verbose_) {std::cout << "Start to run PQk-means" << std::endl;}
    pqkmeans::PQKMeans clustering_instance(codewords_, nlist, iter, verbose_);
    clustering_instance.fit(flattened_codes_randomly_picked);


    // ===== (3) Update coarse centers =====
    coarse_centers_ = clustering_instance.GetClusterCenters();
    assert(coarse_centers_.size() == (size_t) nlist);
    assert(coarse_centers_[0].size() == M_);
    timer1.stop();


    // ===== (4) Update posting lists =====
    timer2.start();
    if (verbose_) {std::cout << "Start to update posting lists" << std::endl;}
    posting_lists_.clear();
    posting_lists_.resize(nlist);
    codes_.clear();
    codes_.resize(nlist);
    posting_dist_lists_.clear();
    posting_dist_lists_.resize(nlist);
    posting_lr_lists_.clear();
    posting_lr_lists_.resize(nlist);

    for (auto &posting_list : posting_lists_) {
        posting_list.reserve(GetN() / nlist);  // Roughly malloc
    }
    for (auto &posting_dist_list : posting_dist_lists_) {
        posting_dist_list.reserve(GetN() / nlist);  // Roughly malloc
    }
    for (auto &code : codes_) {
        code.reserve(GetN() / nlist);  // Roughly malloc
    }
    UpdatePostingLists(GetN());
    timer2.stop();

    std::cout << timer1.get_time() << ' ' << timer2.get_time() << " | " 
        << timer1.get_time() / (timer1.get_time() + timer2.get_time()) << '\n';
}

void 
IndexIVFPQ::AddCodes(const std::vector<std::vector<uint8_t>> &codes, bool update_flag)
{
    // (1) Add new input codes to flatted_codes. This imply pushes back the elements.
    // After that, if update_flg=true, (2) update posting lists for the input codes.
    // Note that update_flag should be true in usual cases. It should be false
    // if (1) this is the first call of AddCodes (i.e., calling in add_configure()),
    // of (2) you've decided to call reconfigure() manually after add()

    if (update_flag && coarse_centers_.empty()) {
        std::cerr << "Error. reconfigure() must be called before running add(vecs=X, update_posting_lists=True)."
                  << "If this is the first addition, please call add_configure(vecs=X)" << std::endl;
        throw;
    }

    // ===== (1) Add codes to flattened_codes =====
    const auto &r = codes; // codes must have ndim=2; with non-writeable
    size_t N = (size_t) r.size();
    std::cout << (size_t) r[0].size() << '\n';
    assert(M_ == (size_t) r[0].size());
    size_t N0 = GetN();
    flattened_codes_.resize( (N0 + N) * M_);
    for (size_t n = 0; n < N; ++n) {
        for (size_t m = 0; m < M_; ++m) {
            flattened_codes_[ (N0 + n) * M_ + m] = r[n][m];
        }
    }
    if (verbose_) {
        std::cout << N << " new vectors are added." << std::endl;
        std::cout << "Total number of codes is " << GetN() << std::endl;
    }

    // ===== (2) Update posting lists =====
    if (update_flag) {
        if (verbose_) { std::cout << "Start to update posting lists" << std::endl; }
        UpdatePostingLists(N);
    }
}

std::pair<std::vector<size_t>, std::vector<float> > 
IndexIVFPQ::query(const std::vector<float> &query,
                                            const std::vector<int> &gt,
                                            int topk,
                                            int L,
                                            int nprobe)
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

    // const auto &gt_v = gt;
    // std::unordered_set<int> gt_set;
    // for (auto i = 0; i < gt_v.size(); ++i) {
    //     gt_set.insert(gt_v[i]);
    // }

    // Timer timer1, timer2;
    // ===== (3) Partial sort the coarse results. =====
    // timer1.start();
    size_t w = (size_t) std::round((double) L * GetNumList() / GetN()) + 3;  // Top w posting lists to be considered. +3 just in case. 
    if (nlist < w) {  // If w is bigger than the original nlist, let's set back nlist
        w = nlist;
    }

    w = nprobe;
    std::partial_sort(scores_coarse.begin(), scores_coarse.begin() + w, scores_coarse.end(),
                      [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b){return a.second < b.second;});
    // timer1.stop();

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
            // if (gt_set.count(n)) {
            //     bl = std::min(bl, idx);
            //     br = std::max(br, idx);
            //     hit_count ++;
            // }
            // ===== (6) Evaluate n =====
            /**
             * scores.emplace_back(n, ADist(dtable, flattened_codes_, n));
             * 3x Faster than ADist(dtable, flattened_codes_, n)
             * Memory locality!
            */
            scores.emplace_back(n, ADist(dtable, no, idx));
        }
        // std::cout << "# After " << coarse_cnt << " coarses search\n";

        // posting_lr_lists_[no].emplace_back(std::array<int, 2>{(int)bl, (int)br});
        // std::cout << std::fixed << std::setprecision(2)
        //             << (double)bl / posting_lists_[no].size() << ' ' 
        //             << (double)br / posting_lists_[no].size() << "\t"
        //             << hit_count << ' ' << score_coarse.second << '\n';
        if ( (size_t) coarse_cnt == w && scores.size() >= (unsigned long) topk) {
            // ===== (8) Sort them =====
            std::partial_sort(scores.begin(), scores.begin() + topk, scores.end(),
                                [](const std::pair<size_t, float> &a, const std::pair<size_t, float> &b){return a.second < b.second;});
            scores.resize(topk);
            scores.shrink_to_fit();
            // for (auto [id, dist] : scores) {
            //     std::cout << id << ' ' << dist << '\n';
            // }

            // std::cout << '\n';
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
    // Update (add) identifiers to posting lists, from codes[start] to codes[start + num -1]
    // This just add IDs, so be careful to call this (e.g., the same IDs will be added if you call
    // this funcs twice at the same time, that would be not expected behavior)
    assert(num <= GetN());

    // ===== (1) Construct a dummy pqkmeans class for computing Symmetric Distance =====
    pqkmeans::PQKMeans clustering_instance(codewords_, (int)GetNumList(), 0, true);
    clustering_instance.SetClusterCenters(coarse_centers_);
    // distance_matrices_among_codewords_ = clustering_instance.distance_matrices_among_codewords_;

    // ===== (2) Update posting lists =====
    std::vector<size_t> assign(num);

    std::mutex mutex;
#pragma omp parallel for
    for (size_t n = 0; n < num; ++n) {
        const auto &nth_code = NthCode(flattened_codes_, n);
        assign[n] = clustering_instance.predict_one(nth_code);
        {
            std::lock_guard<std::mutex> lock(mutex);
            posting_lists_[assign[n]].emplace_back(n);
            posting_dist_lists_[assign[n]].emplace_back(0);
        }
    }


    size_t nlist = GetNumList();
#pragma omp parallel for
    for (size_t no = 0; no < nlist; ++no) {
        auto &plist = posting_lists_[no];
        const auto &center = coarse_centers_[no];
        std::sort(plist.begin(), plist.end(), [&](const auto &a, const auto& b) {
            return clustering_instance.SymmetricDistance(center, NthCode(flattened_codes_, a))
                < clustering_instance.SymmetricDistance(center, NthCode(flattened_codes_, b));
        });
        auto &pdlist = posting_dist_lists_[no];
        for (size_t i = 0; i < pdlist.size(); ++i) {
            pdlist[i] = clustering_instance.SymmetricDistance(center, NthCode(flattened_codes_, plist[i]));
        }
        for (const auto& id : plist) {
            const auto& nth_code = NthCode(flattened_codes_, id);
            codes_[no].insert(codes_[no].end(), nth_code.begin(), nth_code.end());
        }
    }
}

DistanceTable 
IndexIVFPQ::DTable(const std::vector<float> &vec) const
{
    const auto &v = vec;
    size_t Ds = codewords_[0][0].size();
    assert((size_t) v.size() == M_ * Ds);
    DistanceTable dtable(M_, Ks_);
    for (size_t m = 0; m < M_; ++m) {
        for (size_t ks = 0; ks < Ks_; ++ks) {
            dtable.SetVal(m, ks, fvec_L2sqr(&(v[m * Ds]), codewords_[m][ks].data(), Ds));
        }
    }
    return dtable;
}

float 
IndexIVFPQ::ADist(const DistanceTable &dtable, const std::vector<uint8_t> &code) const
{
    assert(code.size() == M_);
    float dist = 0;
    for (size_t m = 0; m < M_; ++m) {
        uint8_t ks = code[m];
        dist += dtable.GetVal(m, ks);
    }
    return dist;
}

float 
IndexIVFPQ::ADist(const DistanceTable &dtable, const std::vector<uint8_t> &flattened_codes, size_t n) const
{
    float dist = 0;
    for (size_t m = 0; m < M_; ++m) {
        uint8_t ks = NthCodeMthElement(flattened_codes, n, m);
        dist += dtable.GetVal(m, ks);
    }
    return dist;
}

float 
IndexIVFPQ::ADist(const DistanceTable &dtable, size_t list_no, size_t offset) const
{
    float dist = 0;
    for (size_t m = 0; m < M_; ++m) {
        uint8_t ks = NthCodeMthElement(list_no, offset, m);
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
    return std::vector<uint8_t>(codes_[list_no].begin() + offset * M_, 
                            codes_[list_no].begin() + (offset + 1) * M_);
}

std::vector<uint8_t> 
IndexIVFPQ::NthCode(const std::vector<uint8_t> &long_code, size_t n) const
{
    return std::vector<uint8_t>(long_code.begin() + n * M_, long_code.begin() + (n + 1) * M_);
}

uint8_t 
IndexIVFPQ::NthCodeMthElement(const std::vector<uint8_t> &long_code, std::size_t n, size_t m) const
{
    return long_code[ n * M_ + m];
}

uint8_t 
IndexIVFPQ::NthCodeMthElement(size_t list_no, size_t offset, size_t m) const
{
    return codes_[list_no][ offset * M_ + m];
}

} // namespace Toy
