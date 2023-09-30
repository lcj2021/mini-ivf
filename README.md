# Mini IVF(PQ) for ANNS

Hello there! This is a mini toy for IVF (inverted file index), along with its PQ (Product Quantizer) version. 

This repo is still under development, but the basic IVF index is built and saved, and the query functionality is already implemented.

Both [hdf5](https://github.com/erikbern/ann-benchmarks/tree/main#data-sets) and [.vec](http://corpus-texmex.irisa.fr/) datasets are supported. 


## Requirement
HDF5 (parallel mode on) is required

Download and configure

```shell
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.N/hdf5-1.N.N/src/<distribution>.tar.gz
gzip -cd <distribution>.tar.gz | tar xvf -

CC=mpicc  CXX=mpicxx \
./configure --prefix=/hdf5/install_path \
--enable-parallel
```

Compile and install

```shell
make -j 32
make install -j 32
```

## Compile

```shell
mkdir build
cd build
cmake ..
make -j4
```

## Usage

You can edit dataset and configuration in tests/test_ivfpq_sift1m_baseline.cpp. 

`configuration` includes kc (the `k` of coarse quantizer)...

`test_ivfpq_sift1m_baseline` require `nprobe` (`nprobe` <= `kc`) from command line

```shell
cd tests
./test_ivfpq_sift1m_baseline 30
```