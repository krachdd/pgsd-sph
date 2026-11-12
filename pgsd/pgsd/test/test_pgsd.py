# Copyright (c) 2016-2026 The Regents of the University of Michigan
# Part of PGSD, released under the BSD 2-Clause License.

"""Tests for the PGSD MPI-IO file layer and its GSD 5.x compatibility.

Files are written in parallel by ``pgsd_mpi_writer.py`` launched with ``mpirun`` on
1, 2 and 4 ranks and then checked serially with ``pgsd.fl``, ``pgsd.hoomd`` and, when
available, the upstream-compatible ``gsd`` 5.x package (``gsd-sph``).

Environment variables:
    PGSD_TEST_MPIRUN  mpirun executable to use (default: ``mpirun`` on PATH).
    PGSD_TEST_RANKS   comma separated rank counts (default: ``1,2,4``).
"""

import os
import pathlib
import shutil
import struct
import subprocess
import sys

import numpy
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pgsd_testdata as td  # noqa: E402

import pgsd.fl  # noqa: E402
import pgsd.hoomd  # noqa: E402

WRITER = pathlib.Path(__file__).with_name('pgsd_mpi_writer.py')
RANKS = [int(r) for r in os.environ.get('PGSD_TEST_RANKS', '1,2,4').split(',')]
SCENARIOS = ['basic', 'empty_rank', 'many_frames', 'small_buffer', 'append']

# byte offset of the uint32 file version in the (P)GSD header:
# magic, index_location, index_allocated_entries, namelist_location,
# namelist_allocated_entries (5 x uint64), schema_version (uint32), file version (uint32)
VERSION_OFFSET = 5 * 8 + 4


def _mpirun():
    exe = os.environ.get('PGSD_TEST_MPIRUN') or shutil.which('mpirun')
    if exe is None:
        pytest.skip('mpirun not found')
    cmd = [exe]
    version = subprocess.run([exe, '--version'], capture_output=True, text=True)
    if 'Open MPI' in version.stdout + version.stderr:
        cmd.append('--oversubscribe')
    return cmd


def write_parallel(tmp_path, scenario, n_ranks):
    """Run the MPI writer and return the written file name."""
    fname = tmp_path / f'{scenario}_{n_ranks}.gsd'
    cmd = _mpirun() + ['-n', str(n_ranks), sys.executable, str(WRITER), scenario,
                       str(fname)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert 'PGSD_WRITER_OK' in result.stdout
    return fname, td.partition(n_ranks, scenario), td.n_frames(scenario)


def check_file_layer(f, counts, nf):
    """Check every chunk of every frame through a pgsd.fl / gsd.fl file object."""
    n_global = int(counts.sum())
    assert f.nframes == nf
    assert f.schema == 'hoomd'
    assert f.schema_version == (1, 4)
    for frame in range(nf):
        step = f.read_chunk(frame=frame, name='configuration/step')
        assert step.dtype == numpy.uint64
        numpy.testing.assert_array_equal(step, [td.step(frame)])
        numpy.testing.assert_array_equal(
            f.read_chunk(frame=frame, name='configuration/box'), td.box())
        numpy.testing.assert_array_equal(
            f.read_chunk(frame=frame, name='particles/N'), [n_global])
        numpy.testing.assert_array_equal(
            f.read_chunk(frame=frame, name='particles/types'), td.types_chunk())
        numpy.testing.assert_array_equal(
            f.read_chunk(frame=frame, name='log/value'), td.log_value(frame))
        for name, (dtype, _m) in td.DISTRIBUTED_FIELDS.items():
            data = f.read_chunk(frame=frame, name=name)
            assert data.dtype == dtype, name
            numpy.testing.assert_array_equal(
                data, td.global_field(name, frame, n_global), err_msg=name)


@pytest.mark.parametrize('n_ranks', RANKS)
@pytest.mark.parametrize('scenario', SCENARIOS)
def test_parallel_write_pgsd_read(tmp_path, scenario, n_ranks):
    """Files written on N ranks read back exactly with pgsd.fl."""
    fname, counts, nf = write_parallel(tmp_path, scenario, n_ranks)
    with pgsd.fl.open(name=fname, mode='r') as f:
        assert f.pgsd_version == (2, 1)
        check_file_layer(f, counts, nf)


@pytest.mark.parametrize('n_ranks', RANKS)
@pytest.mark.parametrize('scenario', ['basic', 'empty_rank', 'many_frames', 'append'])
def test_parallel_write_gsd5_read(tmp_path, scenario, n_ranks):
    """Files written by PGSD on N ranks are valid GSD 5.x files."""
    gsd_fl = pytest.importorskip('gsd.fl')
    gsd_version = pytest.importorskip('gsd.version')
    assert tuple(int(v) for v in gsd_version.version.split('.')[:2]) >= (5, 0)
    fname, counts, nf = write_parallel(tmp_path, scenario, n_ranks)
    with gsd_fl.open(name=fname, mode='r') as f:
        assert f.gsd_version == (2, 1)
        check_file_layer(f, counts, nf)


@pytest.mark.parametrize('n_ranks', RANKS)
def test_parallel_write_hoomd_read(tmp_path, n_ranks):
    """pgsd.hoomd reconstructs SPH frames written on N ranks."""
    fname, counts, nf = write_parallel(tmp_path, 'basic', n_ranks)
    n_global = int(counts.sum())
    with pgsd.hoomd.open(name=fname, mode='r') as traj:
        assert len(traj) == nf
        for frame in range(nf):
            snap = traj[frame]
            assert snap.configuration.step == td.step(frame)
            numpy.testing.assert_array_equal(snap.configuration.box, td.box())
            assert snap.particles.N == n_global
            assert snap.particles.types == td.TYPES
            for attr in ['position', 'velocity', 'typeid', 'image', 'density',
                         'pressure', 'auxiliary1', 'slength']:
                numpy.testing.assert_array_equal(
                    getattr(snap.particles, attr),
                    td.global_field('particles/' + attr, frame, n_global),
                    err_msg=attr)
            # fields not written fall back to the schema defaults
            numpy.testing.assert_array_equal(snap.particles.mass,
                                             numpy.ones(n_global, numpy.float32))
            numpy.testing.assert_array_equal(snap.particles.energy,
                                             numpy.zeros(n_global, numpy.float32))
            numpy.testing.assert_array_equal(snap.log['value'], td.log_value(frame))


def _write_serial_file(fname):
    with pgsd.fl.open(name=fname, mode='w', application='pgsd-test', schema='hoomd',
                      schema_version=[1, 4]) as f:
        f.write_chunk('configuration/step', numpy.array([1], dtype=numpy.uint64),
                      write_all=False)
        f.end_frame()


def _set_file_version(fname, major, minor):
    with open(fname, 'r+b') as raw:
        raw.seek(VERSION_OFFSET)
        raw.write(struct.pack('<I', (major << 16) | minor))


def test_new_files_are_version_2_1(tmp_path):
    """New PGSD files use the GSD 5.x file format version 2.1."""
    fname = tmp_path / 'v21.gsd'
    _write_serial_file(fname)
    with pgsd.fl.open(name=fname, mode='r') as f:
        assert f.pgsd_version == (2, 1)


def test_version_upgrade_on_append(tmp_path):
    """Opening a 2.0 file writable upgrades it to 2.1; read-only keeps 2.0."""
    fname = tmp_path / 'v20.gsd'
    _write_serial_file(fname)
    _set_file_version(fname, 2, 0)

    with pgsd.fl.open(name=fname, mode='r') as f:
        assert f.pgsd_version == (2, 0)
    with pgsd.fl.open(name=fname, mode='r') as f:
        assert f.pgsd_version == (2, 0)

    with pgsd.fl.open(name=fname, mode='a') as f:
        f.write_chunk('configuration/step', numpy.array([2], dtype=numpy.uint64),
                      write_all=False)
        f.end_frame()
    with pgsd.fl.open(name=fname, mode='r') as f:
        assert f.pgsd_version == (2, 1)
        assert f.nframes == 2
        numpy.testing.assert_array_equal(
            f.read_chunk(frame=1, name='configuration/step'), [2])


def test_reject_future_major_version(tmp_path):
    """Files with an unknown major version are rejected."""
    fname = tmp_path / 'v30.gsd'
    _write_serial_file(fname)
    _set_file_version(fname, 3, 0)
    with pytest.raises(RuntimeError):
        pgsd.fl.open(name=fname, mode='r')


def test_read_gsd5_character_chunk(tmp_path):
    """PGSD reads string (CHARACTER) chunks written by GSD 5.x."""
    gsd_fl = pytest.importorskip('gsd.fl')
    fname = tmp_path / 'chars.gsd'
    with gsd_fl.open(name=fname, mode='w', application='gsd', schema='hoomd',
                     schema_version=[2, 0]) as f:
        f.write_chunk('log/label', 'SPH run 42')
        f.write_chunk('configuration/step', numpy.array([3], dtype=numpy.uint64))
        f.end_frame()
    with pgsd.fl.open(name=fname, mode='r') as f:
        assert f.read_chunk(frame=0, name='log/label') == 'SPH run 42'
        numpy.testing.assert_array_equal(
            f.read_chunk(frame=0, name='configuration/step'), [3])


@pytest.mark.parametrize('precision', ['single', 'double'])
def test_read_gsd5_hoomd_file(tmp_path, precision):
    """pgsd.hoomd reads schema 2.0 SPH files written by gsd 5.x (gsd-sph)."""
    gsd_hoomd = pytest.importorskip('gsd.hoomd')
    if not hasattr(gsd_hoomd.ParticleData(), 'density'):
        pytest.skip('gsd is not the SPH-extended gsd-sph package')
    dtype = numpy.float32 if precision == 'single' else numpy.float64
    frame = gsd_hoomd.Frame()
    frame.configuration.step = 7
    frame.particles.N = 3
    frame.particles.types = ['F', 'S']
    frame.particles.typeid = [0, 1, 1]
    frame.particles.position = numpy.array([[0.1, 0.2, 0.3], [1, 2, 3], [-1, -2, -3]])
    frame.particles.density = numpy.array([1000.1, 999.9, 1000.0 + 1 / 3])
    frame.particles.auxiliary4 = numpy.array([[1 / 3, 0, 0], [0, 1 / 7, 0], [0, 0, 1]])
    fname = tmp_path / f'gsd5_{precision}.gsd'
    with gsd_hoomd.open(name=fname, mode='w', precision=precision) as traj:
        traj.append(frame)

    with pgsd.hoomd.open(name=fname, mode='r') as traj:
        assert traj.file.schema_version == (2, 0)
        snap = traj[0]
        assert snap.configuration.step == 7
        assert snap.particles.types == ['F', 'S']
        numpy.testing.assert_array_equal(snap.particles.typeid, [0, 1, 1])
        for attr in ['position', 'density', 'auxiliary4']:
            value = getattr(snap.particles, attr)
            assert value.dtype == dtype, attr
            numpy.testing.assert_array_equal(
                value, getattr(frame.particles, attr).astype(dtype), err_msg=attr)
