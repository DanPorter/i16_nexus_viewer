"""
Functions for reading .dat files
"""

import os
import numpy as np
from collections import OrderedDict

from . import functions as fn
from .babelscan import Scan
from .volume import ImageVolume


"----------------------------LOAD FUNCTIONS---------------------------------"


class Dict2Obj(OrderedDict):
    """
    Convert dictionary object to class instance
    """

    def __init__(self, dictvals, order=None):
        super(Dict2Obj, self).__init__()

        if order is None:
            order = dictvals.keys()

        for name in order:
            setattr(self, name, dictvals[name])
            self.update({name: dictvals[name]})


def read_dat_file(filename):
    """
    Reads #####.dat files from instrument, returns class instance containing all data
    Input:
      filename = string filename of data file
    Output:
      d = class instance with parameters associated to scanned values in the data file, plus:
         d.metadata - class containing all metadata from datafile
         d.keys() - returns all parameter names
         d.values() - returns all parameter values
         d.items() - returns parameter (name,value) tuples
    """
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    # Read metadata
    meta = OrderedDict()
    lineno = 0
    for ln in lines:
        lineno += 1
        if '&END' in ln: break
        ln = ln.strip(' ,\n')
        neq = ln.count('=')
        if neq == 1:
            'e.g. cmd = "scan x 1 10 1"'
            inlines = [ln]
        elif neq > 1:
            'e.g. SRSRUN=571664,SRSDAT=201624,SRSTIM=183757'
            inlines = ln.split(',')
        else:
            'e.g. <MetaDataAtStart>'
            continue

        for inln in inlines:
            vals = inln.split('=')
            try:
                meta[vals[0]] = eval(vals[1])
            except:
                meta[vals[0]] = vals[1]

    # Read Main data
    # previous loop ended at &END, now starting on list of names
    names = lines[lineno].split()
    # Load 2D arrays of scanned values
    vals = np.loadtxt(lines[lineno + 1:], ndmin=2)
    # Assign arrays to a dictionary
    main = OrderedDict()
    for name, value in zip(names, vals.T):
        main[name] = value

    # Convert to class instance
    d = Dict2Obj(main, order=names)
    d.metadata = Dict2Obj(meta)
    return d


"----------------------------------------------------------------------------------------------------------------------"
"---------------------------------------------- DatScan -------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class DatScan(Scan):
    """
    Scan for .dat files
    Reads data into babelscan class, storing data in the internal namespace
    Scan data and metadata can be requested using the the name of the dataset (e.g. 'eta')
    Usage:
        d = DatScan('file.dat')
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
            'scanno': ['scan_number'],
            'scan_command': ['cmd'],
            'energy': ['en'],
        }
        super(DatScan, self).__init__(namespace, alt_names, **kwargs)
        self._label_str.extend(['scanno', 'filetitle'])

    def reset(self):
        """Reset the namespace"""
        self._namespace = {
            'filename': self.filename,
            'scanno': self.scan_number
        }

    def __repr__(self):
        out = 'DatScan(filename: %s, namespace: %d, associations: %d)'
        return out % (self.filename, len(self._namespace), len(self._alt_names))

    def _load_data(self, name):
        """
        Load data from hdf file
          Overloads Scan._load_data to read hdf file
          if 'name' not available, raises KeyError
        :param name: str name or address of data
        """
        dataobj = read_dat_file(self.filename)
        self._namespace.update(dataobj.metadata)
        self._namespace.update(dataobj)
        if name in self._namespace:
            return
        if name in self._alt_names:
            for alt_name in self._alt_names[name]:
                if alt_name in self._namespace:
                    return
        super(DatScan, self)._load_data(name)
        
    def image(self, idx=None):
        """
        Load image from dat file image path tempate
        :param idx: int image number or 'sum'
        :return: numpy.array with ndim 2
        """
        volume = self.volume()
        if idx is None:
            idx = len(volume) // 2
        elif idx == 'sum':
            return np.sum(volume, axis=0)
        return volume[idx]

    def volume(self):
        """
        Load image as ImageVolume
        :return: ImageVolume
        """
        if self._volume:
            return self._volume
        path_spec = self._get_data("_path_template")  # image folder path tempalte e.g. folder/%d.tiff
        pointers = self._get_data('path')  # list of image numbers in dat files

        # add dat file path
        abs_filepath = os.path.dirname(self.filename)
        f = '/'.join(os.path.abspath(path_spec).replace('\\', '/').split('/')[-2:])
        path_spec = os.path.join(abs_filepath, f)

        filenames = [path_spec % pointer for pointer in pointers]
        self._volume = ImageVolume(filenames)
        return self._volume
