#ifndef INCLUDE_INDEX_HPP
#define INCLUDE_INDEX_HPP

#include <stdint.h>

namespace index {

using cluster_id_t  = uint32_t;
using vector_id_t = uint32_t;
using querynode_id_t = uint8_t;

enum IndexStatus { LOCAL, DISTRIBUTED };

};

#endif