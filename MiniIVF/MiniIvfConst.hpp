#ifndef INCLUDE_IVFCONST_HPP
#define INCLUDE_IVFCONST_HPP

#include "MiniIvfType.hpp"
#include <numeric>

namespace MiniIVF {

const size_t            MAX_CLUSTER_NUM             = 10000;
const querynode_id_t    MAX_QUERYNODE_NUM           = 100;
const querynode_id_t    NULL_QUERYNODE              = MAX_QUERYNODE_NUM + 1;
const querynode_id_t    GLOBAL_QUERYNODE            = MAX_QUERYNODE_NUM + 2;
const asym_dist_t       MAX_ASYM_DISTANCE           = 1E18;
const unsigned int      MAX_REMOTE_CALL_TIMEOUT_MS  = 1'000'000'000;
const size_t            MAX_MESSAGE_LENGTH          = std::numeric_limits<size_t>::max();

/// @attention sleep instance pragma: ms in windows, sec in linux
#ifdef _WIN32
const unsigned int      OND_SECOND           = 2000;
#elif _WIN64
const unsigned int      OND_SECOND           = 2000;
#else
const unsigned int      OND_SECOND           = 2;
#endif


};

#endif // ! INCLUDE_IVFCONST_HPP