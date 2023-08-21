#ifndef HDF5_IO_H
#define HDF5_IO_H

#include <iostream>
#include <vector>
#include <assert.h>
#include <H5Cpp.h>

template <typename T>
hid_t get_hdf5_type() {
    static_assert(std::is_void_v<T>, "Unsupported HDF5 data type");
    return 0; // Default value, replace with appropriate action
}

template<>
hid_t get_hdf5_type<char>() { return H5T_NATIVE_CHAR; }
template<>
hid_t get_hdf5_type<unsigned char>() { return H5T_NATIVE_UCHAR; }
template<>
hid_t get_hdf5_type<short int>() { return H5T_NATIVE_SHORT; }
template<>
hid_t get_hdf5_type<unsigned short int>() { return H5T_NATIVE_USHORT; }
template<>
hid_t get_hdf5_type<int>() { return H5T_NATIVE_INT; }
template<>
hid_t get_hdf5_type<unsigned int>() { return H5T_NATIVE_UINT; }
template<>
hid_t get_hdf5_type<long>() { return H5T_NATIVE_LONG; }
template<>
hid_t get_hdf5_type<unsigned long>() { return H5T_NATIVE_ULONG; }
template<>
hid_t get_hdf5_type<float>() { return H5T_NATIVE_FLOAT; }
template<>
hid_t get_hdf5_type<double>() { return H5T_NATIVE_DOUBLE; }
template<>
hid_t get_hdf5_type<long double>() { return H5T_NATIVE_LDOUBLE; }


template <typename T>
H5::DataType get_hdf5_datatype() {
    if (std::is_same<T, int>::value)                return H5::PredType::NATIVE_INT;
    else if (std::is_same<T, unsigned char>::value) return H5::PredType::NATIVE_UCHAR;
    else if (std::is_same<T, float>::value)         return H5::PredType::NATIVE_FLOAT;
    else if (std::is_same<T, double>::value)        return H5::PredType::NATIVE_DOUBLE;
    else {
        throw std::runtime_error("Unsupported type for HDF5");
    }
}

// C++ API style
template<typename T>
void write_to_file_hdf5(const std::vector<T>& data, std::pair<size_t, size_t> dimension, const std::string &filename, const std::string &dataset_name) {
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        hsize_t dims[2] = {dimension.first, dimension.second};
        H5::DataSpace dataspace(2, dims);
        H5::DataSet dataset = file.createDataSet(dataset_name, get_hdf5_datatype<T>(), dataspace);
        dataset.write(&data[0], get_hdf5_datatype<T>());
        printf("%s[%s]: [%llu x %llu] has written!\n", filename.data(), dataset_name.data(), dims[0], dims[1]);
    } catch (const H5::Exception& e) {
        std::cerr << "Exception caught: " << e.getDetailMsg() << std::endl;
        throw;
    }
}

// return the dimention of corresponding dataset
template<typename T>
std::pair<size_t, size_t> load_from_file_hdf5(std::vector<T>& data, const std::string &filename, const std::string &dataset_name) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims_out[2];
        dataspace.getSimpleExtentDims(dims_out, NULL);

        data.resize(dims_out[0] * dims_out[1]);
        printf("%s[%s]: [%llu x %llu] has loaded!\n", filename.data(), dataset_name.data(), dims_out[0], dims_out[1]);

        dataset.read(&data[0], get_hdf5_datatype<T>());

        return {dims_out[0], dims_out[1]};
    } catch (const H5::Exception& e) {
        std::cerr << "Exception caught: " << e.getDetailMsg() << std::endl;
        throw;
    }
}

#endif