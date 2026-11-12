# Copyright (c) 2016-2026 The Regents of the University of Michigan
# Part of PGSD, released under the BSD 2-Clause License.

"""Deterministic test data shared by the MPI writer and the pytest checks.

The global particle data of every frame is a pure function of the frame index and
the global particle count, so the serial reader can reconstruct exactly what the
MPI ranks wrote without any communication.
"""

import numpy

# Particle fields written distributed over all ranks: name -> (dtype, M)
DISTRIBUTED_FIELDS = {
    'particles/position': (numpy.float32, 3),
    'particles/velocity': (numpy.float64, 3),
    'particles/typeid': (numpy.uint32, 1),
    'particles/image': (numpy.int32, 3),
    'particles/density': (numpy.float32, 1),
    'particles/pressure': (numpy.float64, 1),
    'particles/auxiliary1': (numpy.float32, 3),
    'particles/slength': (numpy.float32, 1),
}


def partition(n_ranks, scenario):
    """Number of particles owned by each rank (deliberately uneven)."""
    counts = numpy.array([3 + 2 * i for i in range(n_ranks)], dtype=numpy.uint32)
    if scenario == 'empty_rank' and n_ranks > 1:
        counts[1] = 0
    return counts


def n_frames(scenario):
    """Number of frames written in a scenario."""
    if scenario == 'many_frames':
        # enough index entries to force pgsd_expand_file_index several times
        return 150
    return 4


def global_field(name, frame, n_global):
    """Full (N_global, M) array of a distributed field at a frame."""
    dtype, m = DISTRIBUTED_FIELDS[name]
    seed = sum(ord(c) for c in name)
    base = numpy.arange(n_global * m, dtype=numpy.float64).reshape(n_global, m)
    values = base * 0.5 + frame * 1000.0 + seed
    if numpy.issubdtype(dtype, numpy.integer):
        values = (base + frame * 7 + seed) % 1000
        if dtype == numpy.int32:
            values = values - 500
    arr = numpy.ascontiguousarray(values.astype(dtype))
    return arr if m > 1 else arr.reshape(n_global)


def local_field(name, frame, counts, rank):
    """The slice of a distributed field owned by one rank."""
    n_global = int(counts.sum())
    start = int(counts[:rank].sum())
    stop = start + int(counts[rank])
    return numpy.ascontiguousarray(global_field(name, frame, n_global)[start:stop])


def step(frame):
    """Time step stored at a frame."""
    return numpy.uint64(100 * frame + 5)


def box():
    """Simulation box (root-only chunk)."""
    return numpy.array([10.0, 11.0, 12.0, 0.0, 0.0, 0.0], dtype=numpy.float32)


TYPES = ['F', 'S', 'W']


def types_chunk():
    """particles/types encoded the same way gsd.hoomd and HOOMD-blue encode it."""
    wid = max(len(w) for w in TYPES) + 1
    b = numpy.array(TYPES, dtype=numpy.dtype((bytes, wid)))
    return b.view(dtype=numpy.int8).reshape(len(b), wid)


def log_value(frame):
    """A root-only logged quantity."""
    return numpy.array([frame * 0.25, -frame * 1.5], dtype=numpy.float64)
