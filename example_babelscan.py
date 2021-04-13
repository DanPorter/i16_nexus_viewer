"""
i16 nexus viewer
"""

import os
import numpy as np
import matplotlib.pyplot as plt


from i16_nexus_viewer.babelscan import file_loader, FolderMonitor

file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\810002.nxs"  # eta scan with pilatus
cv_file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\857991.nxs"  # trajectory scan/ cvscan/ kthZebra
files = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\%d.nxs"  # eta scan with pilatus
datadir = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus"  # eta scan with pilatus
livedata = r"\\data.diamond.ac.uk\i16\data\2021\cm28156-1\%d.nxs"  # 879419
#datadir = [r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks", r"\\data.diamond.ac.uk\i16\data\2020\cm26473-1"]
rsmap = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\rsmap_872996_201215_101906.nxs"
rsmap = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\872996-pilatus3_100k-files\rsmap_872996_201215_101906.nxs"

scan = file_loader(file)
print(scan)

x, y, dy, xlab, ylab = scan.get_plot_data('axes', 'nroi[31,31]', '/Transmission', np.sqrt)

plt.figure()
plt.errorbar(x, y, dy, fmt='-o')
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(scan.title())
plt.show()

exp = FolderMonitor(datadir)
d = exp.scan(0)
print(d)

scan_range = range(794932, 794947, 1)  # datadir, sperp, spara, eta scans
scans = exp.scans(scan_range, ['sperp', 'spara'])
print(scans)

