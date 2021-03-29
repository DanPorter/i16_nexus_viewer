"""
Functions for reading .dat files
"""

import numpy as np
from collections import OrderedDict


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


def read_csv_file(filename):
    """
    Reads text file, assumes comma separated and comments defined by #
    :param filename: str path to file
    :return: headers, data: list of str, array
    """
    with open(filename) as f:
        lines = f.readlines()

    # find start of data
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
    #return {name: col for name, col in zip(header, data.T)}
    return header, data

