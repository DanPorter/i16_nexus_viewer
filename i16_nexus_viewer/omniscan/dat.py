"""
Functions for reading .dat files
"""

import numpy as np
from imageio import imread  # read Tiff images
from collections import OrderedDict

from . import functions as fn
from .omniscan import Scan


"----------------------------LOAD FUNCTIONS---------------------------------"


class Dict2Obj(OrderedDict):
    """
    Convert dictionary object to class instance
    """

    def __init__(self, dictvals, order=None):
        super().__init__()

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
    Reads data into omniscan class, storing data in the internal namespace
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
            'scanno': self.scan_number
        }
        alt_names = {
            # shortcut: name in file
            'scan_command': 'cmd',
            'energy': 'en',
        }
        super().__init__(namespace, alt_names, **kwargs)
        self._label_str.extend(['scanno', 'filetitle'])
        self.dat = self._load_data()

    def reset(self):
        """Reset the namespace"""
        self._namespace = {
            'filename': self.filename,
            'scanno': self.scan_number
        }

    def __repr__(self):
        out = 'DatScan(filename: %s, namespace: %d, associations: %d)'
        return out % (self.filename, len(self._namespace), len(self._other2name))

    def _load_data(self):
        dataobj = read_dat_file(self.filename)
        self._namespace.update(dataobj.metadata)
        self._namespace.update(dataobj)
        return dataobj

    def _get_data(self, name):
        if self._reload_mode:
            self.dat = self._load_data()
        return super()._get_data(name)

    def image(self, idx=None):
        """
        Load image from dat file image path tempate
        :param idx: int image number
        :return:
        """
        if self._image_name:
            image_name = self._image_name
        else:
            try:
                image_name = [s for s in self._namespace.keys() if "_path_template" in s][0]
                self._image_name = image_name
            except IndexError:
                raise KeyError('image path template not found in %r' % self)
        pointers = self._get_data('path')  # list of image numbers in dat files
        if idx is None:
            idx = len(pointers)
        image_pointer = self._namespace[image_name] % pointers[idx]
        return imread(image_pointer)
