"""
Define Experiment and Beamline objects
"""

import os
import glob
import numpy as np

from .nexus_scan import Scan, MultiScan
from .functions_nexus import scanfile2number, DEFAULTS, DEFAULTS_HELP, defaults_help
from .functions_nexus import load, dataset_addresses


class Beamline:
    """
    Beamline Class
    Contains beamline defaults for file names and data structure
    """
    def __init__(self, name):
        self.beamline_name = name
        self.defaults = DEFAULTS.copy()
        self.defaults_help = DEFAULTS_HELP.copy()

    def experiment(self, data_directory, working_directory='.', title=None):
        """Create an Experiment object using standard options for this beamline"""
        exp = Experiment(data_directory, working_directory, title, self)
        return exp

    def __repr__(self):
        return 'Beamline(%s)' % self.beamline_name

    def __str__(self):
        out = self.__repr__()
        out += defaults_help(self.defaults, self.defaults_help)
        return out

    def setup(self, **kwargs):
        """
        Set beamline configuration for nexus files, analysis and output
        See self.__str__() for options
        :return: None
        """
        if len(kwargs) == 0:
            print('Parameter options:')
            print(self.__str__())
        self.defaults.update(kwargs)


class Experiment:
    """
    Experiment class
    Contains data directories and scan loading functions
    """
    def __init__(self, data_directory, working_directory='.', title=None, instrument=None):
        if instrument is None:
            instrument = Beamline('None')
        self.instrument = instrument
        data_directory = np.asarray(data_directory).reshape(-1)
        self.data_directories = data_directory
        self.working_directory = working_directory

        if title is None:
            title = os.path.basename(data_directory[0])
        self.title = title

        self._defaults = self.instrument.defaults.copy()
        self._defaults_help = self.instrument.defaults_help.copy()

        self.dataset_addresses = None
        self._updatemode = False
        self._auto_normalise = True

    def __repr__(self):
        return 'Experiment(%s)' % self.title

    def __str__(self):
        out = 'Experiment: %s\n' % self.title
        out += 'Data directories:\n  '
        out += '\n  '.join(self.data_directories)
        out += '\nWorking directory:\n  %s\n' % os.path.abspath(self.working_directory)
        scan_numbers = self.allscannumbers()
        out += 'Number of scans: %d\nFirst scan: %d\nLast scan: %d\n' % \
               (len(scan_numbers), scan_numbers[0], scan_numbers[-1])
        return out

    def defaults(self, **kwargs):
        """Set or display the experiment default parameters"""
        if len(kwargs) == 0:
            return defaults_help(self._defaults, self._defaults_help)
        self._defaults.update(kwargs)

    def auto_normalise(self, toggle=None):
        """Toggle automomatic normalisation of parameters"""
        if toggle is None:
            toggle = not self._auto_normalise
        self._auto_normalise = toggle

    def set_title(self, name):
        """Set experiment title"""
        self.title = name

    def add_data_directory(self, data_directory):
        data_directory = np.asarray(data_directory).reshape(-1)
        self.data_directories = np.append(self.data_directories, data_directory)

    def get_addresses(self, scan_number=0, filename=None):
        """Get addresses from scan file"""
        if filename is None:
            filename = self.getfile(scan_number)
        with load(filename) as hdf_group:
            addresses = dataset_addresses(hdf_group)
        self.dataset_addresses = addresses

    def update_mode(self, update=None):
        """
        When update mode is True, files will search their entire structure
        This is useful when working in a live directory or with files from different times when the nexus files
        don't have the same structure.
        :param update: True/False or None to toggle
        """
        if update is None:
            if self._updatemode:
                update = False
            else:
                update = True
        print('Update mode is: %s' % update)
        self._updatemode = update

    def lastscan(self):
        """
        Get the latest scan number from the current experiment directory (self.data_directory[-1])
        Return None if no scans found.
        """
        """
        if not os.path.isdir(self.data_directories[-1]):
            print("I can't find the directory: {}".format(self.data_directories[0]))
            return None

        # Get all data files in folder
        ls = glob.glob('%s/*.nxs' % (self.data_directories[-1]))
        ls = np.sort(ls)

        if len(ls) < 1:
            print("No files in directory: {}".format(self.data_directories[-1]))
            return None

        newest = ls[-1]  # file with highest number
        # newest = max(ls, key=os.path.getctime) # file created last
        num = scanfile2number(newest)
        return num
        """
        return self.allscannumbers()[-1]

    def allscanfiles(self):
        """
        Return list of all scan files in the data directories
        """
        filelist = []
        for directory in self.data_directories:
            filelist += glob.glob('%s/*.nxs' % directory)
        filelist = np.sort(filelist)
        return filelist

    def allscannumbers(self):
        """
        Return a list of all scan numbers in the data directories
        """
        filelist = self.allscanfiles()
        return [scanfile2number(file) for file in filelist if
                os.path.basename(file) == self._defaults['filename_format'] % scanfile2number(file)]

    def getfile(self, scan_number):
        """
        Convert int scan number to file
        :param scan_number: int : scan number, scans < 1 will look for the latest scan
        :return: filename or '' if scan doesn't appear in directory
        """
        scan_number = np.asarray(scan_number, dtype=int).reshape(-1)
        scan_number = scan_number[0]
        if scan_number < 1:
            scan_number = self.lastscan() + scan_number

        for directory in self.data_directories:
            filename = os.path.join(directory, self._defaults['filename_format'] % scan_number)
            if os.path.isfile(filename):
                return filename
        raise Exception('Scan number: %s doesn\'t exist' % scan_number)

    def scan(self, scan_number=0, filename=None):
        """
        Generate Scan object for given scan using either scan number or filename.
        :param scan_number: int
        :param filename: str : scan filename
        :return: Scan object
        """

        if filename is None:
            filename = self.getfile(scan_number)

        if os.path.isfile(filename):
            return Scan(filename, self, updatemode=self._updatemode)
        raise Exception('Scan number: %s doesn\'t exist' % scan_number)
    loadscan = readscan = scan

    def scans(self, scan_numbers, filenames=None, axes='axes', signal='signal', variables=None, shape=None):
        """
        Generate MultiScan object for given range of scans using either scan number or filename.
        :param scan_numbers: list of int scan numbers
        :param filenames: str : list of str scan filenames
        :param axes: default axis of each Loader
        :param signal: default signal of each loader
        :param variables: str or list of str values that change in each file
        :param shape: tuple shape of data, allowing for multiple dimesion arrays (nLoaders, scany, scanx)
        :return: MultiScan object
        """
        if filenames is None:
            scan_numbers = np.asarray(scan_numbers).reshape(-1)
            filenames = [self.getfile(scn) for scn in scan_numbers]
        else:
            filenames = np.asarray(filenames).reshape(-1)
        scans = [Scan(filename, self, updatemode=self._updatemode) for filename in filenames]
        return MultiScan(scans, axes, signal, variables, shape)
    loadscans = readscans = scans

    def scandata(self, scan_numbers, address):
        """
        Fast return of data from scan number(s)
        :param scan_numbers: int or list : scan numbers to get data
        :param address: str : nexus address, name or command
        :return: data
        """
        scan_numbers = np.asarray(scan_numbers).reshape(-1)
        out = []
        for scan in scan_numbers:
            out += [Scan(self.getfile(scan))(address)]
        if len(scan_numbers) == 1:
            return out[0]
        return out

    def printscan(self, scan_number=0, filename=None):
        scan = self.scan(scan_number, filename)
        print(scan)

    def printscans(self, scan_numbers, variables=None):
        """print data for each scan"""
        if variables is None:
            variables = self._defaults['cmd']
        print(self.scans(scan_numbers, variables=variables))

