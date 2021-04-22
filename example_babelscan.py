"""
i16 nexus viewer
"""

import os
import numpy as np
import matplotlib.pyplot as plt


from i16_nexus_viewer.babelscan import file_loader, FolderMonitor

file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\810002.nxs"  # eta scan with pilatus
cv_file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\857991.nxs"  # trajectory scan/ cvscan/ kthZebra
im_file = r'C:\\Users\\dgpor\\OneDrive - Diamond Light Source Ltd\\I16\\Nexus_Format\\example_nexus\\872996.nxs'  # hkl scan with data
dat_file = r'C:\\Users\\dgpor\\OneDrive - Diamond Light Source Ltd\\I16\\Nexus_Format\\example_nexus\\872996.dat'
files = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\%d.nxs"  # eta scan with pilatus
datadir = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus"  # eta scan with pilatus
livedata = r"\\data.diamond.ac.uk\i16\data\2021\cm28156-1\%d.nxs"  # 879419
#datadir = [r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks", r"\\data.diamond.ac.uk\i16\data\2020\cm26473-1"]
rsmap = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\rsmap_872996_201215_101906.nxs"
rsmap = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\872996-pilatus3_100k-files\rsmap_872996_201215_101906.nxs"

scan = file_loader(im_file)
volume = scan.volume()
print('%r, %s' % (scan, scan.find_image()))
print(volume)
print(np.max(volume))
print(volume.peak_search())

scan1 = file_loader(dat_file)
volume1 = scan1.volume()
print('\n%r' % scan1)
print(volume1)
print(np.max(volume1))
print(volume1.peak_search())

scan2 = file_loader(file)
volume2 = scan2.volume()
print('\n%r, %s' % (scan2, scan2.find_image()))
print(volume2)
print(np.max(volume2))
print(volume2.peak_search())

scan3 = file_loader(rsmap)
volume3 = scan3.volume()
print('\n%r, %s' % (scan3, scan3.find_image()))
print(volume3)
print(np.max(volume3))
print(volume3.peak_search())


scan = file_loader(file, debug=['namespace', 'hdf', 'eval'])
scan.add2namespace(['count_time', 'counttime', 'Time', 't'], None, 'count_time')
print(scan)
print('\n\n')
print(scan('count_time'))
print('\n\n')
print(scan('nroi[31,31]'))


exp = FolderMonitor(datadir)
d = exp.scan(0)
print(d)

x, y, dy, xlab, ylab = scan.get_plot_data('axes', 'nroi[31,31]', '/Transmission', np.sqrt)

plt.figure()
plt.errorbar(x, y, dy, fmt='-o')
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(scan.title())
plt.show()

scan_range = range(794932, 794947, 1)  # datadir, sperp, spara, eta scans
scans = exp.scans(scan_range, ['sperp', 'spara'])
print(scans)



