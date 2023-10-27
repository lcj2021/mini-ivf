#ifndef INCLUDE_MINIIVFTYPE_HPP
#define INCLUDE_MINIIVFTYPE_HPP

#include <stdint.h>


namespace MiniIVF {

/// @brief Numeric Type
using cluster_id_t      =   uint32_t;
using vector_id_t       =   uint32_t;
using vector_dim_t      =   float;
using querynode_id_t    =   uint8_t;
using asym_dist_t       =   float;
using history_score_t   =   uint64_t;

};

#endif // ! INCLUDE_MINIIVFTYPE_HPP