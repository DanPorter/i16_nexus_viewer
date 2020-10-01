"""
i16 nexus viewer
"""

import numpy as np
import matplotlib.pyplot as plt
import i16_nexus_viewer as nv

file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\810002.nxs"  # eta scan with pilatus
cv_file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\857991.nxs"  # trajectory scan/ cvscan/ kthZebra
files = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\%d.nxs"  # eta scan with pilatus
datadir = r"C:\Users\dgpor\OneDrive - Diamond Light Source Ltd\I16\Nexus_Format\example_nexus"  # eta scan with pilatus
#datadir = [r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks", r"\\data.diamond.ac.uk\i16\data\2020\cm26473-1"]

exp = nv.i16.experiment(datadir)
print(exp)

scn = exp.allscannumbers()
print('starting')
scans = exp.loadscans(scn, 'sx,sy')
print('loaded')
print(scans.info())

#exp.printscans(, 'Ta,Tb')

#print(exp.scandata(821760, 'entry1/title'))

#scan = exp.loadscan(810002)
#print(scan)



#print(exp)
#scans = exp.loadscans([794935, 794940])

#scan.plot_image(31)
#plt.show()


