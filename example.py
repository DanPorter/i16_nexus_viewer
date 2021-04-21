"""
i16 nexus viewer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import i16_nexus_viewer as nv

file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\810002.nxs"  # eta scan with pilatus
cv_file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\857991.nxs"  # trajectory scan/ cvscan/ kthZebra
files = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\%d.nxs"  # eta scan with pilatus
datadir0 = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks"
datadir = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus"  # eta scan with pilatus
livedata = r"\\data.diamond.ac.uk\i16\data\2021\cm28156-1\%d.nxs"  # 879419
liveexp = r"\\data.diamond.ac.uk\i16\data\2021\cm28156-1"
#datadir = [r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks", r"\\data.diamond.ac.uk\i16\data\2020\cm26473-1"]
rsmap = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\rsmap_872996_201215_101906.nxs"
rsmap = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\872996-pilatus3_100k-files\rsmap_872996_201215_101906.nxs"

example_files1 = [f for f in os.listdir(r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks") if f.endswith('.nxs')]
example_files2 = [f for f in os.listdir(r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus") if f.endswith('.nxs')]
example_range = [r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\%d.nxs" % fn for fn in range(794932, 794947, 1)]


exp = nv.i16.experiment([datadir0, datadir])
#scan = exp(810002)
scan = exp(877619)  # merlin

print(scan('eta'))
scan.plot('axes', 'nroi[31, 31]')
scan.fit('axes', 'nroi[31, 31]', plot_result=True)


