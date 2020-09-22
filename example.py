"""
i16 nexus viewer
"""

import numpy as np
import matplotlib.pyplot as plt
import i16_nexus_viewer as nv

file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\810002.nxs"  # eta scan with pilatus
cv_file = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\857991.nxs"  # trajectory scan/ cvscan/ kthZebra
files = r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks\%d.nxs"  # eta scan with pilatus

exp = nv.i16.experiment(r"C:\Users\dgpor\Dropbox\Python\ExamplePeaks")

scan = exp.loadscan(857991)
print(scan.scan_number)

print(exp)
scans = exp.loadscans([794935, 794940])

scan.plot_image(31)
plt.show()


