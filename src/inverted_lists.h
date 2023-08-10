#pragma once
#include <vector>
#include <cstdint>

using idx_t = size_t;

struct InvertedListsIterator {
    virtual ~InvertedListsIterator();
    virtual bool is_available() const = 0;
    virtual void next() = 0;
    virtual std::pair<idx_t, const uint8_t*> get_id_and_codes() = 0;
};

struct InvertedLists {
    size_t nlist;     ///< number of possible key values
    size_t code_size; ///< code size per vector in bytes. Default by 8
    bool use_iterator;

    InvertedLists(size_t nlist, size_t code_size);

    virtual ~InvertedLists();

    /// used for BlockInvertedLists, where the codes are packed into groups
    /// and the individual code size is meaningless
    static const size_t INVALID_CODE_SIZE = static_cast<size_t>(-1);

    /*************************
     *  Read only functions */

    /// get the size of a list
    virtual size_t list_size(size_t list_no) const = 0;

    /** get the codes for an inverted list
     * must be released by release_codes
     *
     * @return codes    size list_size * code_size
     */
    virtual const uint8_t* get_codes(size_t list_no) const = 0;

    /** get the ids for an inverted list
     * must be released by release_ids
     *
     * @return ids      size list_size
     */
    virtual const idx_t* get_ids(size_t list_no) const = 0;

    /// release codes returned by get_codes (default implementation is nop
    virtual void release_codes(size_t list_no, const uint8_t* codes) const;

    /// release ids returned by get_ids
    virtual void release_ids(size_t list_no, const idx_t* ids) const;

    /// @return a single id in an inverted list
    virtual idx_t get_single_id(size_t list_no, size_t offset) const;

    /// @return a single code in an inverted list
    /// (should be deallocated with release_codes)
    virtual const uint8_t* get_single_code(size_t list_no, size_t offset) const;

    /// prepare the following lists (default does nothing)
    /// a list can be -1 hence the signed long
    virtual void prefetch_lists(const idx_t* list_nos, int nlist) const;

    /*************************
     * writing functions     */

    /// add one entry to an inverted list
    virtual size_t add_entry(size_t list_no, idx_t theid, const uint8_t* code);

    virtual size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) = 0;

    virtual void update_entry(
            size_t list_no,
            size_t offset,
            idx_t id,
            const uint8_t* code);

    virtual void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) = 0;

    virtual void resize(size_t list_no, size_t new_size) = 0;

    virtual void reset();

    /*************************
     * statistics            */

    /// display some stats about the inverted lists
    void print_stats() const;

    /// sum up list sizes
    size_t compute_ntotal() const;
};

/// simple (default) implementation as an array of inverted lists
struct ArrayInvertedLists : InvertedLists {
    std::vector<std::vector<uint8_t>> codes; // binary codes, size nlist
    std::vector<std::vector<idx_t>> ids;     ///< Inverted lists for indexes

    ArrayInvertedLists(size_t nlist, size_t code_size);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const idx_t* get_ids(size_t list_no) const override;

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    void resize(size_t list_no, size_t new_size) override;

    /// permute the inverted lists, map maps new_id to old_id
    void permute_invlists(const idx_t* map);

    ~ArrayInvertedLists() override;
};