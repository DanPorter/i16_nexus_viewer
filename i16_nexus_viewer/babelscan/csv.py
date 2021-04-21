"""
Functions for reading .csv files
"""

import numpy as np

from . import functions as fn
from .babelscan import Scan


"----------------------------LOAD FUNCTIONS---------------------------------"


def read_csv_file(filename):
    """
    Reads text file, assumes comma separated and comments defined by #
    :param filename: str path to file
    :return: headers, data: list of str, array
    """
    with open(filename) as f:
        lines = f.readlines()

    # find time_start of data
    for n, ln in enumerate(lines):
        values = ln.split(',')
        if len(values) < 2: continue
        value1 = values[0]
        if not value1:
            # line starts with ,
            value1 = values[1]
        try:
            float(value1)
            break
        except ValueError:
            continue

    # Headers
    try:
        header_line = lines[n-1].strip().strip('#')
        header = header_line.split(',')
    except (NameError, IndexError):
        raise Exception('%s contains no headers' % filename)

    # Data
    data = np.genfromtxt(lines[n:], delimiter=',')
    # Build dict
    # return {name: col for name, col in zip(header, data.T)}
    return header, data


"----------------------------------------------------------------------------------------------------------------------"
"---------------------------------------------- CsvScan -------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class CsvScan(Scan):
    """
    Scan for .csv files
    Reads data into babelscan class, storing data in the internal namespace
    Scan data and metadata can be requested using the the name of the dataset (e.g. 'eta')
    Usage:
        d = DatScan('file.csv')
        d('eta') >> finds data column or metadata called 'eta', returns the array
        d.axes() >> automatically finds the default xaxis, returns the array
        d.signal() >> automatically finds the default yaxis, returns the array
        d.image(idx) >> finds the image location if available and returns a detector image
    """
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.file = fn.file2name(filename)
        self.scan_number = fn.scanfile2number(filename)
        namespace = {
            'filename': filename,
            'filetitle': self.file,
            'scan_number': self.scan_number
        }
        alt_names = {
            # shortcut: name in file
            'scanno': 'scan_number',
            'cmd': 'scan_command',
        }
        super().__init__(namespace, alt_names, **kwargs)
        self._label_str.extend(['scanno', 'filetitle'])
        self.header, self.data = self._load_data()

    def reset(self):
        """Reset the namespace"""
        self._namespace = {
            'filename': self.filename,
            'scanno': self.scan_number
        }

    def __repr__(self):
        out = 'CsvScan(filename: %s, namespace: %d, associations: %d)'
        return out % (self.filename, len(self._namespace), len(self._alt_names))

    def _load_data(self, name):
        """
        Load data from hdf file
          Overloads Scan._load_data to read hdf file
          if 'name' not available, raises KeyError
        :param name: str name or address of data
        """
        header, data = read_csv_file(self.filename)
        dataobj = {name: col for name, col in zip(header, data.T)}
        self._namespace.update(dataobj)
        # Set axes, signal defaults
        self.add2namespace(header[0], other_names=self._axes_str[0])
        self.add2namespace(header[1], other_names=self._signal_str[0])
        super(CsvScan, self)._load_data(name)

