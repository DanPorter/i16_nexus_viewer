"""
i16 nexus viewer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import i16_nexus_viewer as nv
import i16_nexus_viewer.plotting_matplotlib as pm

file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\810002.nxs"  # eta scan with pilatus
cv_file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\857991.nxs"  # trajectory scan/ cvscan/ kthZebra
files = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\%d.nxs"  # eta scan with pilatus
datadir = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus"  # eta scan with pilatus
livedata = r"\\data.diamond.ac.uk\i16\data\2021\cm28156-1\%d.nxs"  # 879419
liveexp = r"\\data.diamond.ac.uk\i16\data\2021\cm28156-1"
#datadir = [r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks", r"\\data.diamond.ac.uk\i16\data\2020\cm26473-1"]
rsmap = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\rsmap_872996_201215_101906.nxs"
rsmap = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\872996-pilatus3_100k-files\rsmap_872996_201215_101906.nxs"

example_files1 = [f for f in os.listdir(r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks") if f.endswith('.nxs')]
example_files2 = [f for f in os.listdir(r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus") if f.endswith('.nxs')]
example_range = [r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\%d.nxs" % fn for fn in range(794932, 794947, 1)]

# Create object
#nl = nv.NexusLoader(file)
#rm = nv.NexusLoader(rsmap)
#d = nv.Scan(file)

# Create experiment
#exp = nv.i16.experiment(datadir)
#exp.update_mode()
#exp.get_addresses(0)
#print(exp)

#scan = exp.scan(0)

from i16_nexus_viewer.scan_loader import Scan
scan = Scan(file)
print(scan)

print(scan('eta'))
scan.plot('axes', 'nroi[31,31]')


#print(output)

#allscan = exp.scans(exp.allscannumbers(), variables=['Ta', 'chi'])
#print(allscan)

#d = pm.ScanPlotsMatplotlib(file)

#nls = nv.nexus_loader.NexusMultiLoader([nv.NexusLoader(f) for f in example_range])


#ds_str = nl.dataset('entry1/title')
#ds_tim = nl.dataset('entry1/start_time')
#ds_val = nl.dataset('/entry1/instrument/source/energy')
#ds_arr = nl.dataset('entry1/measurement/eta')
#ds_vol = rm.dataset('/processed/reciprocal_space/volume')

