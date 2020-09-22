"""
Define Experiment and Beamline objects
"""

import os
import glob
import numpy as np
from copy import deepcopy

from .nexus_scan import Scan, MultiScans
from .functions_nexus import scanfile2number


class Beamline:
    """
    Beamline Class
    Contains beamline defaults for file names and data structure
    """
    nexus_string_address = '/entry1'
    nexus_array_address = '/entry1/measurement'
    nexus_value_address = '/entry1/before_scan'
    nexus_filename_format = '%06d.nxs'

    label_format = '%s: %s'

    normalisation_names = ['sum', 'maxval', 'roi2_sum', 'roi1_sum']
    normalisation_command = '%s/Transmission/count_time'
    error_names = [('sum', 'roi2_sum', 'roi1_sum')]
    error_functions = ['np.sqrt(%s+1)']
    duplicate_namespace_names = {
        'en': 'energy',
        'temp': 'Ta',
    }

    def __init__(self, name):
        self.beamline_name = name

    def experiment(self, data_directory, working_directory='.', title=None):
        """Create an Experiment object using standard options for this beamline"""
        exp = Experiment(data_directory, working_directory, title, self)
        exp.setup(
            self.nexus_string_address,
            self.nexus_array_address,
            self.nexus_value_address,
            self.nexus_filename_format,
            self.label_format,
            self.normalisation_names,
            self.normalisation_command,
            self.error_names,
            self.error_functions,
            self.duplicate_namespace_names
        )
        return exp

    def __repr__(self):
        return 'Beamline(%s)' % self.beamline_name

    def setup(self,
              string_address='/entry1',
              array_address='/entry1/measurement',
              value_address='/entry1/before_scan',
              filename_format='%06d.nxs',
              label_format='%s: %s',
              normalisation_names=None,
              normalisation_command='%s/Transmission/count_time',
              error_names=None,
              error_functions=None,
              duplicate_namespace_names=None,
              ):
        """
        Set beamline configuration for nexus files, analysis and output
        :param string_address: str or list of str : hdf5 address(es) in nexus files for strings
        :param array_address: str or list of str : hdf5 address(es) in nexus files for arrays
        :param value_address:  str or list of str : hdf5 address(es) in nexus files for values
        :param filename_format: str : filename format, must accept single int '%d'
        :param label_format: str : output label format, must accept 2*str, e.g. '%s : %s'
        :param normalisation_names: list : list of scanable names to automatically normalise
        :param normalisation_command: str : command format, must accept single str, e.g. '10*%s'
        :param error_names: list of lists of str : each element is a list of names that use error functions below
        :param error_functions: list of str : each element is a command format, must accept single str, e.g. 'sqrt(%s)'
        :param duplicate_namespace_names: dict : dataset names to dubplicate with different names {old_name: new_name}
        :return: None
        """
        if duplicate_namespace_names is None:
            duplicate_namespace_names = {'en': 'energy', 'temp': 'Ta'}
        if error_functions is None:
            error_functions = ['np.sqrt(%s+1)']
        if error_names is None:
            error_names = [('sum', 'roi2_sum', 'roi1_sum')]
        if normalisation_names is None:
            normalisation_names = ['sum', 'maxval', 'roi2_sum', 'roi1_sum']
        self.nexus_value_address = deepcopy(value_address)
        self.nexus_array_address = deepcopy(array_address)
        self.nexus_string_address = deepcopy(string_address)
        self.nexus_filename_format = deepcopy(filename_format)
        self.label_format = deepcopy(label_format)
        self.normalisation_names = deepcopy(normalisation_names)
        self.normalisation_command = deepcopy(normalisation_command)
        self.error_names = deepcopy(error_names)
        self.error_functions = deepcopy(error_functions)
        self.duplicate_namespace_names = deepcopy(duplicate_namespace_names)


class Experiment(Beamline):
    """
    Experiment class
    Contains data directories and scan loading functions
    """
    def __init__(self, data_directory, working_directory='.', title=None, beamline=None):
        if beamline is None:
            beamline = Beamline('None')
        self.beamline = beamline
        super().__init__(self.beamline.beamline_name)
        data_directory = np.asarray(data_directory).reshape(-1)
        self.data_directories = data_directory
        self.working_directory = working_directory

        if title is None:
            title = os.path.basename(data_directory[0])
        self.title = title

    def __repr__(self):
        return 'Experiment(%s)' % self.title

    def __str__(self):
        out = 'Experiment: %s\n' % self.title
        out += 'Data directories:\n  '
        out += '\n  '.join(self.data_directories)
        out += '\nWorking directory:\n  %s\n' % self.working_directory
        scan_numbers = self.allscannumbers()
        out += 'Number of scans: %d\nFirst scan: %d\nLast scan: %d\n' % \
               (len(scan_numbers), scan_numbers[0], scan_numbers[-1])
        return out

    def set_title(self, name):
        """Set experiment title"""
        self.title = name

    def add_data_directory(self, data_directory):
        data_directory = np.asarray(data_directory).reshape(-1)
        self.data_directories = np.append(self.data_directories, data_directory)

    def lastscan(self):
        """
        Get the latest scan number from the current experiment directory (self.data_directory[-1])
        Return None if no scans found.
        """

        if os.path.isdir(self.data_directories[-1]) == False:
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
        return [scanfile2number(file) for file in filelist]

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

        filename = ''
        for directory in self.data_directories:
            filename = os.path.join(directory, self.nexus_filename_format % scan_number)
            if os.path.isfile(filename): break
        return filename

    def loadscan(self, scan_number=0, filename=None):
        """
        Generate Scan object for given scan using either scan number or filename.
        :param scan_number: int
        :param filename: str : scan filename
        :return:
        """
        if filename is None:
            filename = self.getfile(scan_number)
        if os.path.isfile(filename):
            return ExperimentScan(filename, self)
        else:
            print('Scan does not exist: %s' % filename)
            return None

    def printscan(self, scan_number=0, filename=None):
        scan = self.loadscan(scan_number, filename)
        print(scan)

    def loadscans(self, scan_numbers):
        """
        Return multi-scan object
        """
        scanlist = [self.loadscan(scan) for scan in scan_numbers]
        return MultiScans([scan for scan in scanlist if scan])


class ExperimentScan(Scan):
    """
    Version of Scan class called from Experiment
    """
    def __init__(self, filename, experiment):
        self.experiment = experiment
        super().__init__(filename, value_address=self.experiment.nexus_value_address,
                         array_address=self.experiment.nexus_array_address,
                         string_address=self.experiment.nexus_string_address)

        self.filename_format = self.experiment.nexus_filename_format
        self.label_format = self.experiment.label_format
        self.normalisation_names = self.experiment.normalisation_names
        self.normalisation_command = self.experiment.normalisation_command
        self.error_names = self.experiment.error_names
        self.error_functions = self.experiment.error_functions
        self.duplicate_namespace_names = self.experiment.duplicate_namespace_names
