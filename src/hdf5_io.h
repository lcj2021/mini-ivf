#pragma once

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


// return the dimention of corresponding dataset
template<typename T>
std::pair<size_t, size_t> load_from_file(std::vector<T>& data, const std::string &filename, const std::string &dataset_name) {
    hid_t file_id = H5Fopen(filename.data(), H5F_ACC_RDWR, H5P_DEFAULT);
    assert(file_id >= 0 && "Error opening hdf5 file.");

    hid_t dataset_id;
#if H5Dopen_vers == 2
    dataset_id = H5Dopen2(file_id, dataset_name.data(), H5P_DEFAULT);
#else
    dataset_id = H5Dopen(file_id, dataset_name.data());
#endif
    assert(dataset_id >= 0 && "Error opening dataset in file.");

    hid_t space_id = H5Dget_space(dataset_id);

    hsize_t dims_out[2];
    H5Sget_simple_extent_dims(space_id, dims_out, NULL);

    data.resize(dims_out[0] * dims_out[1]);
    printf("%s[%s]: [%llu x %llu]\n", filename.data(), dataset_name.data(), dims_out[0], dims_out[1]);

    herr_t status = H5Dread(dataset_id, get_hdf5_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
    assert(status >= 0 && "Error reading data");

    H5Sclose(space_id);
    H5Dclose(dataset_id);
    // H5Fclose(file_id);
    return {dims_out[0], dims_out[1]};
}