# Copyright (c) 2016-2026 The Regents of the University of Michigan
# Part of PGSD, released under the BSD 2-Clause License.

"""Write a PGSD test file in parallel. Run under ``mpirun -n <ranks>``.

Usage: python pgsd_mpi_writer.py <scenario> <filename>

Scenarios:
    basic        Uneven particle partition, 4 frames.
    empty_rank   Like basic, but rank 1 owns no particles.
    many_frames  150 frames, forces index growth (pgsd_expand_file_index).
    small_buffer Tiny write buffer, forces repeated pgsd_flush_write_buffer calls.
    append       Write 2 frames, close, reopen with mode 'a' and write the rest.
"""

import os
import sys

import numpy
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgsd_testdata as td  # noqa: E402

import pgsd.fl  # noqa: E402


def write_frame(f, frame, counts, rank):
    """Write one frame: root-only chunks plus distributed particle fields."""
    n_global = int(counts.sum())
    f.write_chunk('configuration/step', numpy.array([td.step(frame)]), write_all=False)
    f.write_chunk('configuration/dimensions', numpy.array([3], dtype=numpy.uint8),
                  write_all=False)
    f.write_chunk('configuration/box', td.box(), write_all=False)
    f.write_chunk('particles/N', numpy.array([n_global], dtype=numpy.uint32),
                  write_all=False)
    f.write_chunk('particles/types', td.types_chunk(), write_all=False)
    for name in td.DISTRIBUTED_FIELDS:
        f.write_chunk(name, td.local_field(name, frame, counts, rank), offset=counts,
                      rank=rank, write_all=True)
    f.write_chunk('log/value', td.log_value(frame), write_all=False)
    f.end_frame()


def main():
    scenario, fname = sys.argv[1], sys.argv[2]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    counts = td.partition(comm.Get_size(), scenario)
    nf = td.n_frames(scenario)

    f = pgsd.fl.open(name=fname, mode='w', application='pgsd-test',
                     schema='hoomd', schema_version=[1, 4])
    if scenario == 'small_buffer':
        f.maximum_write_buffer_size = 64
    first_part = 2 if scenario == 'append' else nf
    for frame in range(first_part):
        write_frame(f, frame, counts, rank)
    f.close()

    if scenario == 'append':
        comm.Barrier()
        f = pgsd.fl.open(name=fname, mode='a')
        for frame in range(first_part, nf):
            write_frame(f, frame, counts, rank)
        f.close()

    comm.Barrier()
    if rank == 0:
        print('PGSD_WRITER_OK', flush=True)


if __name__ == '__main__':
    main()
