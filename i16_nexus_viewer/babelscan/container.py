"""
Folder Monitor
"""

import os
import glob
import numpy as np

from . import functions as fn
from .babelscan import Scan, MultiScan
from .hdf import HdfScan
from .dat import DatScan
from .csv import CsvScan


def create_scan(data, headers, alternate_names=None, **kwargs):
    """
    Create data holder instance
    :param data: list of data
    :param headers: list of headers for data
    :param alternate_names: dict of aternate names to headers (or None)
    :param kwargs: other options
    :return: Scan
    """
    name2data = {n: d for n,d in zip(headers, data)}
    return Scan(name2data, alternate_names, **kwargs)


def file_loader(filename, **kwargs):
    """
    Load any file type as Scan class
    :param filename: .dat, .csv, .hdf
    :param kwargs: options
    :return: Scan
    """
    name, ext = os.path.splitext(filename)
    if ext.lower() in ['.dat']:
        return DatScan(filename, **kwargs)
    elif ext.lower() in ['.csv']:
        return CsvScan(filename, **kwargs)
    else:
        return HdfScan(filename, **kwargs)


def load_files(filenames, variables=None, **kwargs):
    """
    Load multiple files as MultiScan class
    :param filenames: list of filenames
    :param variables: str or list of str names that vary in each scan
    :param kwargs: options
    :return: MultiScan
    """
    holders = [file_loader(file, **kwargs) for file in filenames]
    return MultiScan(holders, variables)


class FolderMonitor:
    """
    Monitors a folder or several folders for files following a particular format
    """
    def __init__(self, data_directory, working_directory='.', **kwargs):
        data_directory = np.asarray(data_directory).reshape(-1)
        self._data_directories = data_directory
        self._working_directory = working_directory

        self._options = kwargs
        if 'title' in kwargs:
            self.title = self._options.pop('title')
        else:
            self.title = os.path.basename(data_directory[0])

        self._options['FolderTitle'] = self.title
        if 'title_command' not in kwargs:
            self._options['title_command'] = '{FolderTitle} #{scan_number}'

        if 'filename_format' in kwargs:
            self._filename_format = kwargs['filename_format']
        else:
            self._filename_format = '%06d.nxs'
        self._updatemode = False

    def __repr__(self):
        return 'FolderMonitor(%s)' % self.title

    def __str__(self):
        out = 'Folder Monitor: %s\n' % self.title
        out += 'Data directories:\n  '
        out += '\n  '.join(self._data_directories)
        out += '\nWorking directory:\n  %s\n' % os.path.abspath(self._working_directory)
        scanfiles = self.allscanfiles()
        out += 'Number of files: %d\nFirst file: %s\nLast file: %s\n' % \
               (len(scanfiles), scanfiles[0], scanfiles[-1])
        return out

    def set_title(self, name):
        """Set experiment title"""
        self.title = name
        self._options['FolderTitle'] = self.title

    def options(self, **kwargs):
        """Set or display options"""
        if len(kwargs) == 0:
            # return options
            out = 'Options:\n'
            for key, item in self._options.items():
                out += '%20s : %s\n' % (key, item)
            return out
        self._options.update(kwargs)

    def set_format(self, filename_format='%06d.nxs'):
        """Set the file format to monitor"""
        self._filename_format = filename_format

    def add_data_directory(self, data_directory):
        data_directory = np.asarray(data_directory, dtype=str).reshape(-1)
        self._data_directories = np.append(self._data_directories, data_directory)

    def set_working_directory(self, working_directory):
        """Set the directory to save output too"""
        self._working_directory = working_directory

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

    def latest_scan_number(self):
        """
        Get the latest scan number from the current experiment directory (self.data_directory[-1])
        Return None if no scans found.
        """
        return self.allscannumbers()[-1]
    latest = latest_scan_number

    def allscanfiles(self):
        """
        Return list of all scan files in the data directories
        """
        spec, ext = os.path.splitext(self._filename_format)
        filelist = []
        for directory in self._data_directories:
            filelist += glob.glob('%s/*%s' % (directory, ext))
        filelist = np.sort(filelist)
        return filelist

    def allscannumbers(self):
        """
        Return a list of all scan numbers in the data directories
        """
        filelist = self.allscanfiles()
        return [fn.scanfile2number(file) for file in filelist if
                os.path.basename(file) == self._filename_format % fn.scanfile2number(file)]

    def getfile(self, scan_number):
        """
        Convert int scan number to file
        :param scan_number: int : scan number, scans < 1 will look for the latest scan
        :return: filename or '' if scan doesn't appear in directory
        """
        if os.path.isfile(scan_number):
            return scan_number

        if scan_number < 1:
            scan_number = self.latest() + scan_number

        for directory in self._data_directories:
            filename = os.path.join(directory, self._filename_format % scan_number)
            if os.path.isfile(filename):
                return filename
        raise Exception('Scan number: %s doesn\'t exist' % scan_number)

    def scan(self, scan_number_or_filename=0, **kwargs):
        """
        Generate Scan object for given scan using either scan number or filename.
        :param scan_number_or_filename: int or str file identifier
        :param kwargs: options to send to file loader
        :return: Scan object
        """

        try:
            filename = self.getfile(scan_number_or_filename)
        except TypeError:
            raise Exception('Scan(\'%s\') filename must be number or string' % scan_number_or_filename)

        options = self._options.copy()
        options.update(kwargs)

        if os.path.isfile(filename):
            return file_loader(filename, **options)
        raise Exception('Scan doesn\'t exist: %s' % filename)
    loadscan = readscan = scan

    def updating_scan(self, scan_number_or_filename=0):
        """
        Generate Scan object for given scan using either scan number or filename.
        Data in the scan object will update each time it is called. Useful for live scan data.
        :param scan_number_or_filename: int or str file identifier
        :return: Scan object
        """
        try:
            filename = self.getfile(scan_number_or_filename)
        except TypeError:
            raise Exception('Scan(\'%s\') filename must be number or string' % scan_number_or_filename)

        if os.path.isfile(filename):
            return file_loader(filename, reload=True, **self._options)
        raise Exception('Scan doesn\'t exist: %s' % filename)

    def scans(self, scan_numbers_or_filenames, variables=None, **kwargs):
        """
        Generate MultiScan object for given range of scans using either scan number or filename.
        :param scan_numbers_or_filenames: list of int scan numbers or str filenames
        :param variables: str or list of str names that vary in each scan
        :param kwargs: options to send to file loader
        :return: MultiScan object
        """
        scans = [self.scan(n, **kwargs) for n in scan_numbers_or_filenames]
        return MultiScan(scans, variables)
    loadscans = readscans = scans

    def scandata(self, scan_numbers, name):
        """
        Fast return of data from scan number(s)
        :param scan_numbers: int or list : scan numbers to get data
        :param name: str : name
        :return: data
        """
        scan_numbers = np.asarray(scan_numbers).reshape(-1)
        out = []
        for scn in scan_numbers:
            out += [self.scan(scn)(name)]
        if len(scan_numbers) == 1:
            return out[0]
        return out

    def printscan(self, scan_number=0):
        scan = self.scan(scan_number)
        print(scan)

    def printscans(self, scan_numbers, names='scan_command'):
        scan_numbers = np.asarray(scan_numbers).reshape(-1)
        out = ''
        for n in range(len(scan_numbers)):
            scan = self.scan(scan_numbers[n])
            data = ', '.join(scan.string(names).strip())
            out += '%s: %s\n' % (scan_numbers[n], data)
        print(out)
