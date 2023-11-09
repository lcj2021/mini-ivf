#include <vector_io.hpp>
#include <iostream>

using namespace std;

int main()
{
    std::vector<uint32_t> data = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109};
    utils::VectorIO<uint32_t>::WriteToFile(data, {2, 5}, "test.ui32vecs");
    std::vector<uint32_t> data2;
    auto shape = utils::VectorIO<uint32_t>::LoadFromFile(data2, "test.ui32vecs");
    cout << shape.first << " " << shape.second << endl;
    for (const auto & x: data2) cout << x << " "; cout << endl;
    auto shape1 = utils::VectorIO<uint32_t>::LoadFromFile(data2, "test.ui32vecs", 1);
    cout << shape1.first << " " << shape1.second << endl;
    for (const auto & x: data2) cout << x << " "; cout << endl;

    return 0;
}