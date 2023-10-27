#include "QueryNode.hpp"
#include "MiniIvfPrint.hpp"

#include "index_ivfpq.hpp"
#include "binary_io.hpp"

#include <string>
#include <stdlib.h>

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        MiniIVF::Printer::err("Please input query node IP, Port and querynode thread number.\n");
        exit(1);
    }

    RCF::RcfInit rcf_init;
    MiniIVF::QueryNode qn( argv[1], { argv[1], atoi(argv[2]) } );
    qn.setNumThreads(atoi(argv[3]));

    while (true) sleep(1000);

    return 0;
}