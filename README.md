# PGSD — Parallelized General Simulation Data

**PGSD** is an MPI-parallel file I/O library for SPH (Smoothed Particle Hydrodynamics)
simulation output. It is a fork of the
[GSD library](https://github.com/glotzerlab/gsd) from the Glotzer Group, adapted to
eliminate the serial bottleneck of collecting all particle data on rank 0 before writing.

Current version: **3.2.0**

---

## Overview

Standard GSD writes simulation snapshots by gathering all particle data to a single root
rank, writing it serially, then broadcasting. For large particle counts this becomes the
dominant cost. PGSD replaces every POSIX `read`/`write` call with `MPI_File_*` collective
or independent I/O, so all ranks contribute data simultaneously.

Key design points:

| Feature | GSD | PGSD |
|---|---|---|
| File I/O | POSIX (serial) | MPI-IO (parallel) |
| Index / namelist | root only | root only (gathered from all ranks) |
| Particle data collection | gathered to root | each rank writes its own partition |
| Particle ordering | sorted by ID | not sorted (unstructured per timestep) |
| File size | minimal | larger (all fields rewritten every step) |

Because particle data is **not sorted**, every field (including particle ID) is written at
every timestep even if unchanged. This trades file size for parallel write throughput.

---

## File format

PGSD files use the `.gsd` extension and are binary-compatible with the GSD v1/v2 format.
Each file consists of:

- **Header** (256 bytes) — magic number, schema, version, index/namelist offsets
- **Namelist block** — flat array of 64-byte null-terminated chunk names
- **Index block** — array of `pgsd_index_entry` structs (frame, N, location, M, id, type, flags)
- **Data chunks** — raw binary arrays, one per (frame, name) pair

All multi-rank writes use `MPI_File_write_at` with per-rank offsets computed from an
`MPI_Allgather` of each rank's buffer size.

---

## Python API

Three Python modules are provided:

| Module | Description |
|---|---|
| `pgsd.fl` | Low-level Cython wrapper around the C library. Create, read and write files. |
| `pgsd.hoomd` | High-level HOOMD schema interface (`HOOMDTrajectory`, `Frame`, `open`, `read_log`). |
| `pgsd.pypgsd` | Pure-Python read-only reader (no C compilation required). |

### Quick example — reading a trajectory

```python
import pgsd.fl
import pgsd.hoomd

f = pgsd.fl.open(name='simulation.gsd', mode='r',
                 application='pgsd.hoomd', schema='hoomd',
                 schema_version=[1, 0])
t = pgsd.hoomd.HOOMDTrajectory(f)

for snapshot in t:
    pos = snapshot.particles.position   # numpy array, shape (N, 3)
    vel = snapshot.particles.velocity
```

### Quick example — pure Python reader (no C compiler)

```python
import pgsd.pypgsd
import pgsd.hoomd

with pgsd.pypgsd.PGSDFile(open('simulation.gsd', 'rb')) as f:
    t = pgsd.hoomd.HOOMDTrajectory(f)
    pos = t[0].particles.position
```

---

## Prerequisites

| Dependency | Purpose |
|---|---|
| MPI (OpenMPI ≥ 4 or MPICH ≥ 3) | Parallel I/O — required |
| CMake ≥ 3.14 | Build system — required |
| Python ≥ 3.8 | Python bindings — required |
| Cython ≥ 0.29 | Compiles `fl.pyx` — required |
| NumPy ≥ 1.18 | Array interface — required |
| mpi4py ≥ 3.0 | MPI from Python (`pgsd.hoomd`) — required |

For the recommended conda environment see the parent `hoomd-sph3` project.

---

## Build

```bash
cd pgsd-3.2.0
mkdir build && cd build

# Use MPI compiler wrappers explicitly
CC=/usr/bin/mpicc CXX=/usr/bin/mpicxx cmake ..
make -j$(nproc)
```

After building, add the build directory to your Python path:

```bash
export PYTHONPATH=/path/to/pgsd-3.2.0/build:$PYTHONPATH
```

---

## Differences from upstream GSD

1. **MPI-IO** — all POSIX I/O replaced with `MPI_File_*` calls; the file handle is an
   `MPI_File` instead of a POSIX file descriptor.
2. **Distributed writes** — each rank writes its own particle partition directly to the
   correct file offset; no data is ever gathered to root.
3. **Root-managed metadata** — index entries and the namelist are managed exclusively by
   rank 0; sizes are gathered from all ranks via `MPI_Allgather` before writing.
4. **No particle sorting** — particles are written in rank order; fields that did not
   change logically are still rewritten each frame.
5. **Larger files** — because all fields are rewritten unconditionally every frame, output
   files are significantly larger than equivalent GSD files.

---

## Developer

[David Krach](https://www.mib.uni-stuttgart.de/institute/team/Krach/) —
<david.krach@mib.uni-stuttgart.de>
