"""
omniscan: generic object for reading scan files
A generic class for reading data files from scan data

Usage:
  from omniscan import file_loader
  scan1 = file_loader('some/file.nxs', **options)  # creates Scan class
  scan2 = file_loader('another/file.dat', **options)
  scans = scan1 + scan2  # creates MultiScan class

  # Folder monitor:
  from omniscan import FolderMonitor
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

MultiScan class
  MultiScan class is a holder for multiple scans, allowing operations to be performed on all scans in the class.
    scans = MultiScan([scan1, scan2], **options)
    scans = scan1 + scan2
    scans = omniscan.load_files(['file1', 'file2'], **options)
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

By Dan Porter, PhD
Diamond
2021

Version 0.1.0
Last updated: 13/04/21

Version History:
13/04/21 0.1.0  Version History started.
"""

from .__settings__ import EVAL_MODE
from .omniscan import Scan, MultiScan
from .hdf import HdfScan
from .dat import DatScan
from .csv import CsvScan
from .container import create_scan, file_loader, load_files, FolderMonitor

print(EVAL_MODE)

__version__ = "0.1.0"
__date__ = "13/04/2021"

