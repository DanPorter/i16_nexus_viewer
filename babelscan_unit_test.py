"""
Unit test for babelscan
"""

import numpy as np
import matplotlib.pyplot as plt


from i16_nexus_viewer import babelscan


print('####################################################')
print('############## babelscan unit tests ################')
print('####################################################')
print('\nbabelscan version: %s (%s)' % (babelscan.__version__, babelscan.__date__))

file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\810002.nxs"  # eta scan with pilatus
cv_file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\857991.nxs"  # trajectory scan/ cvscan/ kthZebra
im_file = r'C:\\Users\\dgpor\\OneDrive - Diamond Light Source Ltd\\I16\\Nexus_Format\\example_nexus\\872996.nxs'  # hkl scan with data
dat_file = r'C:\\Users\\dgpor\\OneDrive - Diamond Light Source Ltd\\I16\\Nexus_Format\\example_nexus\\872996.dat'
datadir = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus"  # eta scan with pilatus
rsmap = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus\872996-pilatus3_100k-files\rsmap_872996_201215_101906.nxs"


print('\n\n############ Missing count_time Tests ##############')
scan = babelscan.file_loader(file, debug='all')
scan.add2namespace(['count_time', 'counttime', 'Time', 't'], None, 'count_time')
print(scan)
print('\n\n')
print(scan('count_time'))
print('\n\n')
print(scan('nroi[31,31]'))


print('\n\n############### FolderMonitor Tests ################')
exp = babelscan.FolderMonitor(datadir)
scan = exp.scan(0)
print(scan)


print('\n\n##################### Plot Tests ###################')
scan = exp.scan(794940)
x, y, dy, xlab, ylab = scan.get_plot_data('axes', 'nroi_peak[31,31]', '/count_time/Transmission', np.sqrt)

plt.figure()
plt.errorbar(x, y, dy, fmt='-o')
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(scan.title())

scan = exp(877619)  # merlin
scan.fit('axes', 'nroi_peak[31, 31]')
scan.plot('axes', ['nroi_peak[31, 31]', 'fit'])
print(scan.string('amplitude'))

scan.plot.plot_image('sum', clim=[0, 100])
plt.show()


print('\n\n################# MultiScan Tests ##################')
scan_range = range(794932, 794947, 1)  # datadir, sperp, spara, eta scans
scans = exp.scans(scan_range, ['sperp', 'spara'])
print(scans)


print('\n\n################### Volume Tests ###################')
scan = babelscan.file_loader(im_file)
volume = scan.volume()
print('%r, %s' % (scan, scan.find_image()))
print(volume)
print(np.max(volume))
print(volume.peak_search())

scan1 = babelscan.file_loader(dat_file)
volume1 = scan1.volume()
print('\n%r' % scan1)
print(volume1)
print(np.max(volume1))
print(volume1.peak_search())

scan2 = babelscan.file_loader(file)
volume2 = scan2.volume()
print('\n%r, %s' % (scan2, scan2.find_image()))
print(volume2)
print(np.max(volume2))
print(volume2.peak_search())

scan3 = babelscan.file_loader(rsmap)
volume3 = scan3.volume()
print('\n%r, %s' % (scan3, scan3.find_image()))
print(volume3)
print(np.max(volume3))
print(volume3.peak_search())