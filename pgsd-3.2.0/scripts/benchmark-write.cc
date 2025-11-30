#include "pgsd.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <thread>
#include <type_traits>

#ifdef _WIN32
#include <io.h>
#define fsync _commit
#else // linux / mac
#include <unistd.h>
#endif


int main(int argc, char** argv) // NOLINT
    {
    int rank, nprocs;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    long unsigned int _rank = static_cast<std::make_unsigned<decltype(rank)>::type>(rank);
    long unsigned int _nprocs = static_cast<std::make_unsigned<decltype(nprocs)>::type>(nprocs);

    const size_t n_keys = 17;
    const size_t n_frames = 100;
    const size_t key_size = static_cast<const size_t>(1024) * static_cast<const size_t>(1024);
    size_t keys_per_rank = floor( key_size / nprocs );

    if ( _rank < key_size%_nprocs ){
        keys_per_rank = keys_per_rank + 1;
    }

    size_t offset = 0;
    size_t alloffsets[nprocs];
    MPI_Allgather(&keys_per_rank, 1, my_MPI_SIZE_T, alloffsets, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    for( int j = 0; j < rank; j++ ){
        offset += alloffsets[j];
    }

    // std::cout << "Rank: " << rank << " key_size " << key_size << std::endl;
    // std::cout << "Rank: " << rank << " keys_per_rank " << keys_per_rank << std::endl;
    // std::cout << "Rank: " << rank << " offset " << offset << std::endl;
    

    std::vector<double> data(keys_per_rank);

    for (size_t i = 0; i < keys_per_rank; i++)
        {
        data[i] = (double)i;
        }

    std::vector<std::string> names;
    for (size_t i = 0; i < n_keys; i++)
        {
        std::ostringstream s;
        s << "quantity/" << i;
        names.push_back(s.str());
        }


    
    std::cout << "Writing test.gsd with: " << n_keys << " keys, " << n_frames << " frames, "
              << "and " << key_size << " double(s) per key on " << nprocs << " rank(s)" << std::endl;
    std::cout << "Rank: " << rank << " with " << keys_per_rank << " double(s) per key" << std::endl;
    
    pgsd_handle handle;
    
    // concatenate filename
    const std::string str_fn = "test" + std::to_string(nprocs) + ".gsd";
    const char *filename = str_fn.c_str();
    if ( rank == 0 )
        {
        std::cout << "Filename " << filename << std::endl;
        }

    auto starttime = std::chrono::high_resolution_clock::now();

    pgsd_create_and_open(&handle, filename, "app", "schema", 0, PGSD_OPEN_APPEND, 0);
    for (size_t frame = 0; frame < n_frames / 2; frame++)
        {
        MPI_Barrier(MPI_COMM_WORLD);
        for (auto const& name : names)
            {
            pgsd_write_chunk(&handle, 
                             name.c_str(), 
                             PGSD_TYPE_DOUBLE, 
                             keys_per_rank, // N 
                             1, // M 
                             key_size, // N_global
                             1, // M_global
                             offset,
                             key_size*1, // global_size
                             true, // write all 
                             0, 
                             data.data());
            }

        pgsd_end_frame(&handle);
        }

    auto t1 = std::chrono::high_resolution_clock::now();

    for (size_t frame = 0; frame < n_frames / 2; frame++)
        {
        MPI_Barrier(MPI_COMM_WORLD);
        for (auto const& name : names)
            {
            pgsd_write_chunk(&handle, 
                             name.c_str(), 
                             PGSD_TYPE_DOUBLE, 
                             keys_per_rank, // N 
                             1, // M 
                             key_size, // N_global
                             1, // M_global
                             offset,
                             key_size, // global_size
                             true, // write all 
                             0, 
                             data.data());
            }

        pgsd_end_frame(&handle);
        }

    auto t2 = std::chrono::high_resolution_clock::now();
    
    pgsd_close(&handle);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto endtime = std::chrono::high_resolution_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);
    std::chrono::duration<double> time_span
        = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    double time_per_key = time_span.count() / double(n_keys) / double(n_frames / double(2));

    MPI_Allreduce(MPI_IN_PLACE, &time_per_key, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    const double us = 1e-6;
    if ( rank == 0 )
        {            
        std::cout << "Write time: " << time_per_key / us * 1./nprocs << " microseconds/key." << std::endl;
        std::cout << "Write time: " << time_per_key / us * n_keys * 1./nprocs << " microseconds/frame."
                  << std::endl;
        }

    double mb_per_second
        = double(key_size * 8 + static_cast<const size_t>(32) * static_cast<const size_t>(2))
          / 1048576.0 / time_per_key;
    
    std::chrono::duration<double> abs_time_span
        = std::chrono::duration_cast<std::chrono::duration<double>>(endtime - starttime);
    double abs_time = time_span.count();

    const double total_mb
        = double(n_keys * n_frames * key_size * 8 + static_cast<const size_t>(32) * static_cast<const size_t>(2))
          / 1000000;

    MPI_Allreduce(MPI_IN_PLACE, &mb_per_second, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if ( rank == 0 )
        {
        std::cout << "MB/s: " << mb_per_second << " MB/s." << std::endl;
        std::cout << "Total time required: " << abs_time << " seconds for " << total_mb << " MB or " << total_mb/1000 << " GB" << std::endl; 
        std::cout << "!!!!!!!!!!!! Finished Writing to File !!!!!!!!!!!!" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    pgsd_open(&handle, filename, PGSD_OPEN_READONLY);
    struct pgsd_name_buffer file_names = handle.file_names;
    //struct pgsd_name_buffer frame_names = handle.frame_names;


    uint64_t number_of_frames = pgsd_get_nframes(&handle);
    if ( rank == 0 )
        {
        std::cout << "Frames: " << number_of_frames << std::endl;
        std::cout << "Names: " << file_names.n_names << std::endl;
        }
    // std::cout << "Frame names: " << frame_names.n_names << std::endl;
    
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    pgsd_close(&handle);
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // std::cout << "Rank: " << rank << " DONE!" << std::endl;
    MPI_Finalize();
    }
