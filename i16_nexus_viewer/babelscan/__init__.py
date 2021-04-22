"""
babelscan: a generic class for reading many types of scan data that you don't need to put in your ear.
A generic class for reading data files from scan data

requirements: numpy, h5py, imageio
Optional requirements: matplotlib, lmfit

Usage:
  from babelscan import file_loader
  scan1 = file_loader('some/file.nxs', **options)  # creates Scan class
  scan2 = file_loader('another/file.dat', **options)
  scans = scan1 + scan2  # creates MultiScan class

  # Folder monitor:
  from babelscan import FolderMonitor
  mon = FolderMonitor('/some/folder', **options)
  scan = mon.scan(0)  # creates scan from latest file in folder

Scan class
  Scan class contains an internal namespace where each dataset can contain multiple names. Calling the instantiated
  class searches the namespace (and then the file) for a dataset that matches the name given.
    output = scan('name')
  Certain names are reserved and will automatically search the namespace and file for default values:
    scan('axes')  # returns default scanned dataset (x-axis)
    scan('signal')  # returns default signal dataset (y-axis)
  Operations will be evaluated*:
    scan('name1 + name2')  # returns the result of the operation
  For scans with detector images, regions of interest can be generated:
    scan('nroi[127,127,31,31]')  # creates region of interest on image and returs array of sum of roi at each point

  Functions
    scan.title()  # returns str title of scan
    scan.label()  # returns short scan identifier
    scan.scan_command()  # returns definition of the scan command (if available)
    scan.value('name')  # always returns a single value
    scan.string('name')  # returns 'name = value' string
    scan.array(['name1', 'name2'], array_length)  # returns square array of several datasets
    scan.image(idx)  # if available, returns detector image
    scan.string_format('format {name:fmt}')  # returns string formated from namespace
    scan.get_plot_data(xname, yname)  # return data for plotting with errors and labels
    scan.plot(xname, yname)  # create plot**
    scan.fit(xname, yname)  # fit data to peak***

MultiScan class
  MultiScan class is a holder for multiple scans, allowing operations to be performed on all scans in the class.
    scans = MultiScan([scan1, scan2], **options)
    scans = scan1 + scan2
    scans = babelscan.load_files(['file1', 'file2'], **options)
  Works in the same way as underlying scan class - calling the class will return a list of datasets from the scans.
    [output1, output2] = scans('name')

  Functions
    scans.add_variable('name')  # add default parameter that changes between scans, displayed in print(scans)
    scans.array('name')  # return 2D array of scan data
    scans.griddata('name')  # generate 2D square grid of single values for each scan

FolderMonitor class
  FolderMonitor watches a folder (or several) and allows easy loading of files by scan number:
    fm = FolderMonitor('/some/folder', filename_format='%d.nxs')
    scan = fm.scan(12345) # loads '/some/folder/12345.nxs'
    scan = fm.scan(0)  # loads most recent file
    scans = fm.scans(range(12340, 12345))  # MultiScan of several files

  Functions:
    fm.allscanfiles()  # return list of all scan files in folder(s)
    fm.allscannumbers()  # list of all scan numbers in folder(s)
    fm.updating_scan(12345)  # the resulting Scan class will reload on each operation

*functions using eval only available when the "EVAL_MODE" setting is active.
**functions using plot only available if "MATPLOTLIB_PLOTTING" setting is active and matplotlib installed
***functions using fitting only available if lmfit installed

By Dan Porter, PhD
Diamond
2021

Version 0.3.0
Last updated: 22/04/21

Version History:
13/04/21 0.1.0  Version History started.
16/04/21 0.2.0  Added instrument and other additions to Scan, changed container.py to folder_monitor.py
22/04/21 0.3.0  Changed _get_data search path and added _default_values dict, added volume.py, settings.py
"""


__version__ = "0.3.0"
__date__ = "22/04/2021"


from .__settings__ import EVAL_MODE
from .babelscan import Scan, MultiScan
from .hdf import HdfScan
from .dat import DatScan
from .csv import CsvScan
from .folder_monitor import create_scan, file_loader, hdf_loader, load_files, FolderMonitor
from .instrument import Instrument
