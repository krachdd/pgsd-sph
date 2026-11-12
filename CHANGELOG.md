# PGSD Changelog

## [Unreleased] — include_hoomd722 (GSD 5.0.1 / HOOMD-blue 7.2.2)

### File layer (`pgsd/pgsd/pgsd.c`, `pgsd/pgsd/pgsd.h`, `pgsd/pgsd/libpgsd.pxd`)

* New files are written with file format version **2.1** (same as GSD 5.x).
* Added `PGSD_TYPE_CHARACTER` (string chunks, file format 2.1).
* Opening a 2.x file with mode `'r+'`/`'a'` silently upgrades it to the latest minor
  version (2.1); read-only opens never modify the file and versions are never downgraded.
* Not ported: the GSD 4.0 buffering refactor (no implicit flush in `end_frame`/`find_chunk`/
  `read_chunk`, merged frame/buffer index, removal of `index_entries_to_buffer`) and the
  `fsync` calls. PGSD keeps its MPI-IO buffering/flush semantics, which HOOMD-blue's
  `GSDDumpWriterMPI`/`GSDReaderMPI` rely on.

### Python (`pgsd/pgsd/fl.pyx`, `pgsd/pgsd/hoomd.py`, `pgsd/pgsd/__init__.py`)

* `fl.PGSDFile.read_chunk` reads `CHARACTER` chunks and returns them as `str`.
* `pgsd.hoomd` accepts hoomd schema versions 1.x and 2.x (files written by gsd 5.x,
  including `precision='double'`).
* Fixed `pgsd.hoomd.open` and `pgsd.hoomd.read_log`: they raised `TypeError` because the
  application string used the `pgsd.version` module instead of `pgsd.version.version`.
* Removed the `SIGTERM` handler (as in GSD 5.0 and HOOMD-blue 6.1).

### Tests (`pgsd/pgsd/test/`)

* New MPI test suite `test_pgsd.py` (with `pgsd_mpi_writer.py`, `pgsd_testdata.py`),
  enabled in CMake. It writes files with `mpirun` on 1, 2 and 4 ranks (uneven partitions,
  an empty rank, 150 frames to grow the index, a tiny write buffer, append mode) and
  checks them serially with `pgsd.fl`, `pgsd.hoomd` and gsd 5.x (`gsd-sph`). It also
  covers version upgrade/rejection, CHARACTER chunks and reading gsd 5.x SPH files in
  single and double precision.
* `test_fl.py`, `test_hoomd.py` and `test_largefile.py` are unmodified GSD 3.2 tests that
  import `gsd`, not `pgsd`; they are no longer copied to the build directory.

## [Release 1.0] — 2026-03-09

### Code improvements (`pgsd/pgsd/pgsd.c`, `pgsd/pgsd/pgsd.h`)

Eight correctness, performance, and portability issues were fixed.

---

#### Fix 1 — Remove redundant `MPI_Barrier` before every `MPI_Bcast` in `bcast_*` helpers

**Files:** `pgsd.c`

`MPI_Bcast` is already a collective operation: it does not return on any rank
until all ranks have called it and the data has been delivered.  The nine
`bcast_*` helper functions (`bcast_file_size`, `bcast_frame_index_size`,
`bcast_index_allocated_entries`, `bcast_index_location`,
`bcast_namelist_allocated_entries`, `bcast_namelist_location`,
`bcast_number_of_names_file`, `bcast_number_of_names_frame`, `bcast_retval`)
each called `MPI_Barrier` immediately before `MPI_Bcast`, adding a redundant
global synchronisation point on every broadcast.  Removed the barrier from
all nine functions.

---

#### Fix 2 — Replace VLA `allbuffers[nprocs]` with `malloc` in `pgsd_flush_write_buffer`

**Files:** `pgsd.c`

```c
// before
size_t allbuffers[nprocs];

// after
size_t* allbuffers = malloc(nprocs * sizeof(size_t));
if (allbuffers == NULL)
    return PGSD_ERROR_MEMORY_ALLOCATION_FAILED;
// ... free(allbuffers) added before every return path
```

Variable-length arrays (VLAs) were made optional in C11 and removed in C23.
Stack-allocating a per-rank buffer is also unsafe for large process counts.
Replaced with heap allocation and added `free(allbuffers)` at all four return
paths inside the function.

---

#### Fix 3 — Remove dead `bytes_written` variable; check `MPI_File_write_at` return in `pgsd_write_chunk`

**Files:** `pgsd.c`

```c
// before — bytes_written was always == size, check was unreachable
size_t bytes_written = size;
MPI_File_write_at(...);          // return value discarded
if (bytes_written == -1 || bytes_written != size)   // always false
    return PGSD_ERROR_IO;
MPI_Allreduce(&bytes_written, ...);

// after — actual error check on the MPI call
int mpi_err = MPI_File_write_at(...);
if (mpi_err != MPI_SUCCESS)
    return PGSD_ERROR_IO;
MPI_Allreduce(&size, ...);       // size used directly
```

The dead error check masked any real I/O failures on the write path.

---

#### Fix 4 — Remove redundant nested `is_root()` checks

**Files:** `pgsd.c`

Three places had a second `if (is_root())` guard inside a block already
gated on `if (is_root())`:

- `pgsd_flush_name_buffer`: two inner guards removed (around `write_at 5`
  and `write_at header`).
- `pgsd_flush`: one inner guard removed (around `write_at 7`).
- `pgsd_flush_name_buffer` (else branch): inner guard around `write_at 6`
  removed.

---

#### Fix 5 — Check `MPI_File_read_at` return in `pgsd_index_buffer_map`; remove redundant `MPI_File_seek`

**Files:** `pgsd.c`

```c
// before — error silently ignored; MPI_File_seek is redundant (read_at uses
//           absolute positioning)
MPI_File_seek(handle->fh, handle->header.index_location, MPI_SEEK_SET);
MPI_File_read_at(handle->fh, handle->header.index_location, ...);

// after
int io_ret = MPI_File_read_at(handle->fh, handle->header.index_location, ...);
if (io_ret != MPI_SUCCESS)
    return PGSD_ERROR_IO;
```

A failed index read would previously go undetected, leading to corrupt data
being used silently.

---

#### Fix 6 — Convert `pgsd_sizeof_type` from if-else chain to `switch`

**Files:** `pgsd.c`

```c
// before — 10-branch if-else chain
if (type == PGSD_TYPE_UINT8) { val = sizeof(uint8_t); }
else if ...

// after — switch with direct returns
switch (type) {
    case PGSD_TYPE_UINT8:  return sizeof(uint8_t);
    ...
    default:               return 0;
}
```

A `switch` over an enum is more idiomatic C and allows the compiler to
generate a jump table.

---

#### Fix 7 — Cache `rank` and `nprocs` in `pgsd_handle`; eliminate repeated `MPI_Comm_rank` calls

**Files:** `pgsd.h`, `pgsd.c`

Added two fields to `pgsd_handle`:

```c
int rank;   ///< MPI rank in MPI_COMM_WORLD
int nprocs; ///< Number of MPI processes in MPI_COMM_WORLD
```

Initialised once at the top of `pgsd_initialize_handle` via
`MPI_Comm_rank` / `MPI_Comm_size`.  Replaced every `is_root()` call
throughout `pgsd.c` with `(rank == 0)` using the cached value from
`handle->rank`, eliminating ~20+ redundant `MPI_Comm_rank` invocations per
write/flush cycle across the following functions:

- `pgsd_flush_write_buffer`
- `pgsd_flush_name_buffer`
- `pgsd_flush`
- `pgsd_write_chunk`
- `pgsd_read_chunk`
- `pgsd_find_chunk`
- `pgsd_close`
- `pgsd_initialize_handle`
- `pgsd_expand_file_index`
- `pgsd_initialize_file`

---

#### Fix 8 — Clean up `pgsd_bcast_index_entry`

**Files:** `pgsd.c`

- Removed two redundant `MPI_Barrier` calls (one before and one after
  `MPI_Bcast`; the collective itself provides the needed synchronisation).
- Removed unused local `rank` variable (queried via `MPI_Comm_rank` but never
  used).
- Fixed typo: `blocklegth` → `blocklength`.

---

### Benchmark results

**Setup:** local machine, 16 cores, single node, NVMe storage.
**Workload:** 17 keys × 100 frames × 1,048,576 doubles/key ≈ 14.26 GB total.
**Binary:** `benchmark-write` / `benchmark-read` in `pgsd/scripts/`.

#### Write benchmark (`benchmark-write`)

Each rank writes its own partition of each key directly to the correct file
offset via `MPI_File_write_at`.  Total data volume is constant across all
rank counts.

| Ranks | Throughput (MB/s) | Total time (s) |
|------:|------------------:|---------------:|
|     1 |             167.0 |           40.7 |
|     2 |             168.9 |           40.3 |
|     4 |             168.9 |           40.3 |
|     8 |             167.1 |           40.7 |

Throughput is flat across ranks: the bottleneck is disk bandwidth, not
coordination overhead.  Scaling to more ranks does not increase total I/O
volume, so no speedup is expected on a single node with a single storage
device.

