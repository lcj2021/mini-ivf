#pragma once
// From Faiss

#include "inverted_lists.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include "assert.h"

InvertedListsIterator::~InvertedListsIterator() {}

/*****************************************
 * InvertedLists implementation
 ******************************************/

InvertedLists::InvertedLists(size_t nlist, size_t code_size)
        : nlist(nlist), code_size(code_size), use_iterator(false) {}

InvertedLists::~InvertedLists() {}

idx_t InvertedLists::get_single_id(size_t list_no, size_t offset) const {
    assert(offset < list_size(list_no));
    const idx_t* ids = get_ids(list_no);
    idx_t id = ids[offset];
    release_ids(list_no, ids);
    return id;
}

void InvertedLists::release_codes(size_t, const uint8_t*) const {}

void InvertedLists::release_ids(size_t, const idx_t*) const {}

void InvertedLists::prefetch_lists(const idx_t*, int) const {}

const uint8_t* InvertedLists::get_single_code(size_t list_no, size_t offset)
        const {
    assert(offset < list_size(list_no));
    return get_codes(list_no) + offset * code_size;
}

size_t InvertedLists::add_entry(
        size_t list_no,
        idx_t theid,
        const uint8_t* code) {
    return add_entries(list_no, 1, &theid, code);
}

void InvertedLists::update_entry(
        size_t list_no,
        size_t offset,
        idx_t id,
        const uint8_t* code) {
    update_entries(list_no, offset, 1, &id, code);
}

void InvertedLists::reset() {
    for (size_t i = 0; i < nlist; i++) {
        resize(i, 0);
    }
}

void InvertedLists::print_stats() const {
    std::vector<int> sizes(40);
    for (size_t i = 0; i < nlist; i++) {
        for (size_t j = 0; j < sizes.size(); j++) {
            if ((list_size(i) >> j) == 0) {
                sizes[j]++;
                break;
            }
        }
    }
    for (size_t i = 0; i < sizes.size(); i++) {
        if (sizes[i]) {
            printf("list size in < %zu: %d instances\n",
                   static_cast<size_t>(1) << i,
                   sizes[i]);
        }
    }
}

size_t InvertedLists::compute_ntotal() const {
    size_t tot = 0;
    for (size_t i = 0; i < nlist; i++) {
        tot += list_size(i);
    }
    return tot;
}

/*****************************************
 * ArrayInvertedLists implementation
 ******************************************/

ArrayInvertedLists::ArrayInvertedLists(size_t nlist, size_t code_size)
        : InvertedLists(nlist, code_size) {
    ids.resize(nlist);
    codes.resize(nlist);
}

size_t ArrayInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* code) {
    if (n_entry == 0)
        return 0;
    assert(list_no < nlist);
    size_t o = ids[list_no].size();
    ids[list_no].resize(o + n_entry);
    memcpy(&ids[list_no][o], ids_in, sizeof(ids_in[0]) * n_entry);
    codes[list_no].resize((o + n_entry) * code_size);
    memcpy(&codes[list_no][o * code_size], code, code_size * n_entry);
    return o;
}

size_t ArrayInvertedLists::list_size(size_t list_no) const {
    assert(list_no < nlist);
    return ids[list_no].size();
}

const uint8_t* ArrayInvertedLists::get_codes(size_t list_no) const {
    assert(list_no < nlist);
    return codes[list_no].data();
}

const idx_t* ArrayInvertedLists::get_ids(size_t list_no) const {
    assert(list_no < nlist);
    return ids[list_no].data();
}

void ArrayInvertedLists::resize(size_t list_no, size_t new_size) {
    ids[list_no].resize(new_size);
    codes[list_no].resize(new_size * code_size);
}

void ArrayInvertedLists::update_entries(
        size_t list_no,
        size_t offset,
        size_t n_entry,
        const idx_t* ids_in,
        const uint8_t* codes_in) {
    assert(list_no < nlist);
    assert(n_entry + offset <= ids[list_no].size());
    memcpy(&ids[list_no][offset], ids_in, sizeof(ids_in[0]) * n_entry);
    memcpy(&codes[list_no][offset * code_size], codes_in, code_size * n_entry);
}

void ArrayInvertedLists::permute_invlists(const idx_t* map) {
    std::vector<std::vector<uint8_t>> new_codes(nlist);
    std::vector<std::vector<idx_t>> new_ids(nlist);

    for (size_t i = 0; i < nlist; i++) {
        size_t o = map[i];
        assert(o < nlist);
        std::swap(new_codes[i], codes[o]);
        std::swap(new_ids[i], ids[o]);
    }
    std::swap(codes, new_codes);
    std::swap(ids, new_ids);
}

ArrayInvertedLists::~ArrayInvertedLists() {}
