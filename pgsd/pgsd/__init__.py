# Copyright (c) 2016-2023 The Regents of the University of Michigan
# Part of GSD, released under the BSD 2-Clause License.

"""The PGSD main module.

The main package :py:mod:`pgsd` is the root package. It holds the submodules
:py:mod:`pgsd.fl` and :py:mod:`pgsd.hoomd`, but does not import them by default.
You must explicitly import these modules before use::

    import pgsd.fl

    import pgsd.hoomd
"""

from . import version  # noqa: F401

# The SIGTERM handler was removed, matching GSD 5.0 (glotzerlab/gsd#501) and
# HOOMD-blue 6.1. Applications that need graceful shutdown on SIGTERM must
# install their own handler.
