#!/usr/bin/env python3

"""----------------------------------------------------------
maintainer: dkrach, david.krach@mib.uni-stuttgart.de
-----------------------------------------------------------"""

# --- HEADER ---------------------------------------------------
import sys
sys.path.append("./pgsd-3.2.0/build")
sys.path.append("./pgsd-3.2.0/build/pgsd")
import pgsd.hoomd
import pgsd.fl
import pgsd.pypgsd
import numpy as np
import os
from pyevtk.hl import pointsToVTK as vtk
#--------------------------------------------------------------

# Input GSD file
f = pgsd.fl.PGSDFile(name = sys.argv[1], mode = 'r', application = "HOOMD-SPH", schema = "hoomd", schema_version = [1,0])
# f = gsd.pygsd.GSDFile(open('log.gsd', 'rb'))

# Parse GSD file into a trajectory object
t = pgsd.hoomd.HOOMDTrajectory(f)
# Run loop over all snapshots
count = 0
for snapshot in t:
   count += 1
   print(count)
   pname = sys.argv[1].replace('.gsd','')
