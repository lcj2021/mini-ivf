#ifndef INCLUDE_MINIIVFINTERFACE_HPP
#define INCLUDE_MINIIVFINTERFACE_HPP

#include <RCF/Idl.hpp>
#include <SF/vector.hpp>
#include <SF/utility.hpp>

#include <string>

#include "MiniIvfType.hpp"

#include <index_ivfpq.hpp>

using namespace MiniIVF;
using namespace std;

RCF_BEGIN(I_MiniIvfService, "I_MiniIvfService")

    RCF_METHOD_V1(
        void, 
        loadSegments, 
            vector<cluster_id_t>
    );

    RCF_METHOD_V0(
        void,
        loadCodeBook
    );

    RCF_METHOD_V11(
        void,
        indexInit,
            size_t, size_t, size_t, size_t, size_t, 
            size_t, size_t, size_t, size_t,
            string, string 
    );

    RCF_METHOD_V5(
        void,
        runQueries,
            size_t ,
            vector<vector<vector_dim_t>>,
            vector<vector<cluster_id_t>>,
            vector<vector<vector_id_t>> &,
            vector<vector<asym_dist_t>> &
    );

    RCF_METHOD_V2(
        void,
        addFile,
            string,
            string
    );
    
RCF_END(I_MiniIvfService);

#endif // ! INCLUDE_MINIIVFINTERFACE_HPP