#include "pgsd.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <thread>

int main(int argc, char** argv) // NOLINT
    {
    int rank, nprocs;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    long unsigned int _rank = static_cast<std::make_unsigned<decltype(rank)>::type>(rank);
    long unsigned int _nprocs = static_cast<std::make_unsigned<decltype(nprocs)>::type>(nprocs);

    std::vector<char> data;

    // concatenate filename
    const std::string str_fn = "test" + std::to_string(nprocs) + ".gsd";
    const char *filename = str_fn.c_str();
    std::cout << "Filename " << filename << std::endl;
    
    auto starttime = std::chrono::high_resolution_clock::now();

    pgsd_handle handle;
    pgsd_open(&handle, filename, PGSD_OPEN_READONLY);
    std::cout << "Rank " << rank << " finished opening file!" << std::endl;
    
    size_t n_frames = pgsd_get_nframes(&handle);
    size_t n_names = pgsd_get_nnames(&handle);
    
    std::vector<std::string> names;
    for (size_t i = 0; i < n_names; i++)
        {
        std::ostringstream s;
        s << "quantity/" << i;
        names.push_back(s.str());
        }

    // Only valid on rank 0, not optimal
    const pgsd_index_entry* entry0 = pgsd_find_chunk(&handle, (size_t)0, names[0].c_str());
    
    // Following would be better solution, but error! TODO
    // const pgsd_index_entry* c_entry = pgsd_find_chunk(&handle, (size_t)0, names[0].c_str());
    // pgsd_index_entry* entry0 = const_cast<pgsd_index_entry*>(c_entry); 
    // pgsd_bcast_index_entry(entry0);
    
    // Get the offsets depending on the number of entries in the 0-th frame of the file
    // This is only to be used in controlled benchmark circumstances, since here we 
    // know the number of keys stays constant
    size_t key_size;
    if (rank == 0){
        key_size = entry0->N;
    }
    MPI_Bcast(&key_size, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);

    size_t keys_per_rank = floor( key_size / nprocs );

    if ( _rank < key_size%_nprocs ){
        keys_per_rank = keys_per_rank + 1;
    }
    size_t offset = 0;
    size_t alloffsets[nprocs];

    MPI_Allgather(&keys_per_rank, 1, my_MPI_SIZE_T, alloffsets, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    for( int j = 0; j < rank; j++ ){
        offset += alloffsets[j];
    }
    std::cout << "Rank: " << rank << " key_size " << key_size << std::endl;
    std::cout << "Rank: " << rank << " keys_per_rank " << keys_per_rank << std::endl;
    std::cout << "Rank: " << rank << " offset " << offset << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Reading test.gsd with: " << n_names << " keys (n_names) and " << n_frames << " frames."
              << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    for (size_t frame = 0; frame < n_frames; frame++)
        {
        for (auto const& name : names)
            {
            const pgsd_index_entry* e = pgsd_find_chunk(&handle, frame, name.c_str()); 
            size_t m_N, m_M;
            uint8_t m_type;
            if ( is_root() )
                {
                m_N = e->N;
                m_M = e->M;
                m_type = e->type; 
                }
            MPI_Bcast(&m_N, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
            MPI_Bcast(&m_M, 1, my_MPI_SIZE_T, 0, MPI_COMM_WORLD);
            MPI_Bcast(&m_type, 1, MPI_UINT8_T, 0, MPI_COMM_WORLD);
            size_t N_per_rank = floor( m_N / nprocs );
    
            if ( _rank < m_N%_nprocs ){
                N_per_rank = N_per_rank + 1;
            }
            size_t m_offset = 0;
            size_t m_alloffsets[nprocs];
            MPI_Allgather(&N_per_rank, 1, my_MPI_SIZE_T, m_alloffsets, 1, my_MPI_SIZE_T, MPI_COMM_WORLD);

            for( int j = 0; j < rank; j++ ){
                m_offset += m_alloffsets[j];
            }

            if (data.empty())
                {
                data.resize(N_per_rank * m_M * pgsd_sizeof_type((pgsd_type)m_type));
                }
            pgsd_read_chunk(&handle, data.data(), e, N_per_rank, m_M, m_offset, true);
            }

        }

    auto t2 = std::chrono::high_resolution_clock::now();
    pgsd_close(&handle);
    auto endtime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_span
        = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    double time_per_key = time_span.count() / double(n_names) / double(n_frames);

    std::chrono::duration<double> abs_time_span
        = std::chrono::duration_cast<std::chrono::duration<double>>(endtime - starttime);
    double abs_time = time_span.count();

    const double total_mb
        = double(n_names * n_frames * key_size * 8 + static_cast<const size_t>(32) * static_cast<const size_t>(2))
          / 1000000;

    const double us = 1e-6;
    if ( rank == 0 )
        {
        std::cout << "Sequential read time: " << time_per_key / us << " microseconds/key." << std::endl;
        std::cout << "Total time required: " << abs_time << " seconds for " << total_mb << " MB or " << total_mb/1000 << " GB" << std::endl; 
        std::cout << "!!!!!!!!!!!! Finished Reading to File !!!!!!!!!!!!" << std::endl;
    }
    MPI_Finalize();
    }
