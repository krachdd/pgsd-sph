.. Copyright (c) 2016-2023 The Regents of the University of Michigan
.. Part of PGSD, released under the BSD 2-Clause License.

Installation
============

**pgsd** must be compiled from source. It requires an MPI implementation and uses
MPI-IO for all file operations. There are no pre-built binary packages.

Prerequisites
-------------

**Required:**

* **MPI** — OpenMPI ≥ 4.0 or MPICH ≥ 3.4 (must include ``mpicc`` / ``mpicxx`` wrappers)
* **CMake** ≥ 3.14
* **Python** ≥ 3.8
* **Cython** ≥ 0.29
* **NumPy** ≥ 1.18
* **mpi4py** ≥ 3.0

**Optional:**

* **pyevtk** — needed for the ``test_pgsd2vtu.py`` conversion script

Install prerequisites (Ubuntu / Debian)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   $ sudo apt install libopenmpi-dev openmpi-bin cmake python3-dev
   $ pip install cython numpy mpi4py

Install prerequisites via conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The recommended approach is to use the conda environment from the parent
``hoomd-sph3`` project. To create a minimal standalone environment:

.. code-block:: bash

   $ conda create -n pgsd-env python=3.10 cython numpy mpi4py cmake \
         -c conda-forge
   $ conda activate pgsd-env
   $ conda install -c conda-forge openmpi

Compile from source
-------------------

1. Enter the source directory::

      $ cd pgsd-3.2.0

2. Configure with CMake, passing the MPI compiler wrappers explicitly::

      $ mkdir build && cd build
      $ CC=/usr/bin/mpicc CXX=/usr/bin/mpicxx cmake ..

   .. tip::

      If ``mpicc`` / ``mpicxx`` are not in ``/usr/bin``, find them with
      ``which mpicc`` and substitute the correct path.

   .. tip::

      When using a conda environment, activate it first and pass the prefix::

         $ export CMAKE_PREFIX_PATH=$CONDA_PREFIX
         $ CC=$CONDA_PREFIX/bin/mpicc CXX=$CONDA_PREFIX/bin/mpicxx cmake ..

3. Build::

      $ make -j$(nproc)

   The Cython extension ``pgsd/fl.so`` is built in the ``build/`` directory.

4. Add the build directory to your Python path::

      $ export PYTHONPATH=/path/to/pgsd-3.2.0/build:$PYTHONPATH

   Or add this line to your shell profile / conda activation script.

5. Verify the installation::

      $ python3 -c "import pgsd.fl; print('pgsd.fl OK')"
      $ python3 -c "import pgsd.hoomd; print('pgsd.hoomd OK')"

CMake options
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``CMAKE_C_COMPILER``
     - system default
     - Set to MPI C wrapper, e.g. ``mpicc``
   * - ``CMAKE_CXX_COMPILER``
     - system default
     - Set to MPI C++ wrapper, e.g. ``mpicxx``
   * - ``CMAKE_BUILD_TYPE``
     - ``Release``
     - Use ``Debug`` for development builds
   * - ``CMAKE_C_FLAGS``
     - (empty)
     - Pass ``-march=native`` to optimise for your CPU

Incremental builds
------------------

After modifying C or Cython source, re-run only the build step — CMake
reconfigures automatically when needed:

.. code-block:: bash

   $ make -j$(nproc) -C build

Embedding PGSD in your project
--------------------------------

Using the C library
^^^^^^^^^^^^^^^^^^^

**pgsd** is implemented in two files. Copy ``pgsd/pgsd.h`` and ``pgsd/pgsd.c``
into your project and link against your MPI library. No other dependencies are
required for the C layer.

.. code-block:: cmake

   find_package(MPI REQUIRED)
   add_library(pgsd pgsd.c)
   target_link_libraries(pgsd PUBLIC MPI::MPI_C)

Using the pure Python reader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For **read-only** access without compiling C code, copy these three files from
the ``pgsd/`` directory:

* ``pgsd/__init__.py``
* ``pgsd/pypgsd.py``
* ``pgsd/hoomd.py``

These implement a pure Python reader for PGSD/HOOMD files. Use it as follows:

.. code-block:: python

   import pgsd.pypgsd
   import pgsd.hoomd

   with pgsd.pypgsd.PGSDFile(open('file.gsd', 'rb')) as f:
       t = pgsd.hoomd.HOOMDTrajectory(f)
       pos = t[0].particles.position

.. note::

   The pure Python reader is **read-only**. For write access, the compiled
   ``pgsd.fl`` module is required.

Troubleshooting
---------------

``ImportError: No module named 'pgsd.fl'``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Cython extension was not found. Make sure ``PYTHONPATH`` includes the
``build/`` directory that contains the compiled ``pgsd/fl.so``.

``RuntimeError: Not a PGSD file``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file was not written by PGSD, or is corrupted. Check that the magic number
at offset 0 is ``0x65DF65DF65DF65DF``.

MPI errors on open
^^^^^^^^^^^^^^^^^^^

Ensure that the number of MPI ranks used to **read** the file matches the
number of ranks that **wrote** it, or use the pure Python reader for
single-process access.

``UnicodeDecodeError`` when opening a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file was not opened in binary mode. Use ``open('file.gsd', 'rb')``.
