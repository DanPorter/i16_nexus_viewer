"""
Basic Functions for reading h5py/ nexus files

---Notes on hdf5 object and nexus---
nx objects read by h5py have several useful attributes and functions:
nx = h5py.File(filename, 'r')

nx.filename (File object only)
nx.file - points to File object
nx.file.filename
nx.get('/entry1')
nx.attrs
    nx.attrs.keys()
    nx.attrs.items()
nx.name - current path

hdf5 files are split into 3 parts: File, group, dataset:
    <HDF5 file "794940.nxs" (mode r+)>
    <HDF5 group "/entry1" (18 members)>
    <HDF5 group "/entry1/measurement" (23 members)>
    <HDF5 dataset "eta": shape (61,), type "<f8">

nx = h5py.File(filename, 'r')
file = nx
    file.filename
    file.get(address)
    file.attrs

group = nx['/entry1/measurement']
    group.file
    group.get
    group.attrs
    group.name

dataset = nx['/entry1/measurement/eta']
    dataset.file
    dataset.attrs
    dataset.name
    dataset.size
    dataset.shape
    dataset.value << depreciated, use dataset[()]

Further notes:
 - dataset.name returns the full location address, however it may be wrong if the file hasn't been set up properly.
 - dataset may return None in the case of dynamic hdf link if the link isn't esatblished.
"""

import sys, os
import numpy as np
import h5py
from imageio import imread  # read Tiff images


def load(filename):
    """Load a hdf5 or nexus file"""
    return h5py.File(filename, 'r')


def scanfile2number(filename):
    """
    Extracts the scan number from a .nxs filename
    :param filename: str : filename of .nxs file
    :return: int : scan number
    """
    return np.abs(np.int(os.path.split(filename)[-1][-10:-4]))


def hdf_tree(hdf_group):
    """
    Return str of the full tree of data in a hdf object
    :param hdf_group: hdf5 File or Group object
    :return:
    """
    outstr = '\nGroup: %s\n' % (hdf_group.name)
    for attr, val in hdf_group.attrs.items():
        outstr += '  attr: %s: %s\n' % (attr, val)
    try:
        for branch in hdf_group.keys():
            outstr += hdf_tree(hdf_group.get(branch))
        return outstr
    except AttributeError:
        name = os.path.basename(hdf_group.name)
        out = '  dataset: %s\n    %s, size: %s, shape: %s\n' % (name, hdf_group.name, hdf_group.size, hdf_group.shape)
        out += '    attrs: %s\n' % ','.join(hdf_group.attrs.keys())
        if len(hdf_group.shape) == 1 and hdf_group.size > 1:
            out += '    data: [%s ...%s... %s]\n' % (hdf_group[0], hdf_group.size, hdf_group[-1])
        elif len(hdf_group.shape) > 1:
            shape = hdf_group.shape
            amax = np.max(hdf_group)
            amin = np.min(hdf_group)
            mean = np.mean(hdf_group)
            out += '    data: %s, max: %.3g, min: %.3g, mean: %.3g\n' % (shape, amax, amin, mean)
        else:
            out += '    data: %s\n' % hdf_group[()]
        return out


def dataset_addresses(hdf_group, addresses='/', recursion_limit=100):
    """
    Return list of addresses of datasets, starting at each address
    :param hdf_group: hdf5 File or Group object
    :param addresses: list of str or str : start in this / these addresses
    :param recursion_limit: Limit on recursivley checking lower groups
    :return: list of str
    """
    addresses = np.asarray(addresses, dtype=str).reshape(-1)
    out = []
    for address in addresses:
        data = hdf_group.get(address)
        if data and type(data) is h5py.Dataset:
            out += [data.name]
        elif data and recursion_limit > 0:
            new_addresses = [data.get(d).name for d in data.keys()]
            out += dataset_addresses(hdf_group, new_addresses, recursion_limit-1)
    return out


def find_name(hdf_group, name, match_case=False, whole_word=False):
    """
    Find datasets using field name
    :param hdf_group: hdf5 File or Group object
    :param name: str : name to match in dataset field name
    :param match_case: if True, match case of name
    :param whole_word: if True, only return whole word matches
    :return: list of str addresses
    """
    out = []
    addresses = dataset_addresses(hdf_group)
    if not match_case: name = name.lower()
    for address in addresses:
        address_name = os.path.basename(address)
        if not match_case: address_name = address_name.lower()
        if whole_word and name == address_name:
            out += [hdf_group.get(address).name]
        elif not whole_word and name in address:
            out += [hdf_group.get(address).name]
    return out


def find_attr(hdf_group, attr='axes'):
    """
    Returns location of hdf attribute
    Workds recursively - starts at the top level and searches all lower hdf groups
    :param hdf_group: hdf5 File or Group object
    :param attr: str : attribute name to search for
    :return: str hdf address
    """
    if attr in hdf_group.attrs:
        attr_names = np.asarray(hdf_group.attrs[attr], dtype=str).reshape(-1)
        address = [hdf_group.get(ax).name for ax in attr_names]
        return address
    try:
        for branch in hdf_group.keys():
            address = find_attr(hdf_group.get(branch), attr)
            if address:
                return address
    except AttributeError:
        pass


def find_arrays(hdf_group):
    """
    Returns list of addresses of array data
    :param hdf_group: hdf5 File or Group object
    :return: list of str
    """
    addresses = dataset_addresses(hdf_group)  # returns dataset addresses
    return [hdf_group.get(address).name for address in addresses if hdf_group.get(address).size > 1]


def find_values(hdf_group):
    """
    Returns list of addresses of single value data
    :param hdf_group: hdf5 File or Group object
    :return: list of str
    """
    addresses = dataset_addresses(hdf_group)  # returns dataset addresses
    return [hdf_group.get(address).name for address in addresses if hdf_group.get(address).size == 1]


def find_image(hdf_group, address='/', multiple=False):
    """
    Return address of image data in hdf file
    Images can be stored as list of file directories when using tif file,
    or as a dynamic hdf link to a hdf file.

    :param hdf_group: hdf5 File or Group object
    :param address: initial hdf address to look for image data
    :param multiple: if True, return list of all addresses matching criteria
    :return: str or list of str
    """
    filepath = os.path.dirname(hdf_group.file.filename)
    addresses = dataset_addresses(hdf_group, address)
    all_addresses = []
    # First look for 2D image data
    for address in addresses:
        data = hdf_group.get(address)
        if not data or data.size == 1: continue
        if len(data.shape) > 1 and 'signal' in data.attrs:
            if multiple:
                all_addresses += [address]
            else:
                return address
    # Second look for image files
    for address in addresses:
        data = hdf_group.get(address)
        if 'signal' in data.attrs:  # not sure if this generally true, but seems to work for pilatus and bpm images
            if multiple:
                all_addresses += [address]
            else:
                return address
        """
        file = str(data[0])
        file = os.path.join(filepath, file)
        if os.path.isfile(file):
            if multiple:
                all_addresses += [address]
            else:
                return address
        """
    if multiple:
        return all_addresses
    else:
        return None


def image_data(hdf_group, index=None, address=None):
    """
    Return image data, if available
    if index=None, all images are combined, otherwise only a single frame at index is returned
    :param hdf_group: hdf5 File or Group object
    :param index: None or int : return a specific image
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :return: 2d array if index given, 3d array otherwise
    """
    if address is None:
        address = find_image(hdf_group)
    dataset = hdf_group.get(address)
    # if array - return array
    if len(dataset.shape) > 1:
        # array data
        if index is None:
            return np.asarray(dataset)
        else:
            return dataset[index]
    # if file - load file with imread, return array
    else:
        filepath = os.path.dirname(hdf_group.file.filename)
        try:
            filenames = [os.path.join(filepath, file) for file in dataset]
        except TypeError:
            # image_data filename is loaded in bytes format
            filenames = [os.path.join(filepath, file.decode()) for file in dataset]
        if index is None:
            # stitch images into volume
            image = imread(filenames[0])
            volume = np.zeros([len(filenames), image.shape[0], image.shape[1]])
            for n, filename in enumerate(filenames):
                volume[n] = imread(filename)
            return volume
        else:
            return imread(filenames[index])


"----------------------------------- HDF5 Class -----------------------------------"


class Hdf5Nexus(h5py.File):
    """
    Implementation of h5py.File, with additional nexus functions
    nx = Hdf5Nexus('/data/12345.nxs')

    Additional functions:
        nx.nx_dataset_addresses() - list of all hdf addresses for datasets
        nx.nx_tree_str() - string of internal data structure
        nx.nx_find_name('eta') - returns hdf address
        nx.nx_find_addresses( addresses=['/']) - returns list of addresses
        nx.nx_find_attr(attr='signal') - returns address with attribute
        nx.nx_find_image() - returns address of image data
        nx.nx_getdata(address) - returns numpy array of data at address
        nx.nx_array_data(n_points, addresses) - returns dict of n length arrays and dict of addresses
        nx.nx_value_data(addresses) - returns dict of values and dict of addresses
        nx.nx_str_data(addresses, format) - returns dict of string output and dict of addresses
        nx.nx_image_data(index, all) - returns 2/3D array of image data
    """
    def __init__(self, filename, mode='r', *args, **kwargs):
        super().__init__(filename, mode, *args, **kwargs)

    def nx_dataset_addresses(self):
        return dataset_addresses(self.get('/'))

    def nx_tree_str(self):
        return hdf_tree(self.get('/'))

    def nx_find_name(self, name, match_case=False, whole_word=False):
        """Return addresses with name"""
        return find_name(self, name, match_case, whole_word)

    def nx_find_attr(self, attr='axes'):
        """Returns location of hdf attribute "axes" """
        return find_attr(self, attr)

    def nx_find_image(self, address='/', multiple=False):
        """Return address of image data"""
        return find_image(self, address, multiple)

    def nx_getdata(self, address):
        """Return value within the dataset at address"""
        dataset = self.get(address)
        if dataset and type(dataset) is h5py.Dataset:
            return dataset[()]
        return None

    def nx_showattrs(self, address):
        """Return formatted string of attributes for hdf object at this address"""
        thing = self.get(address)
        out = '%s\n' % thing
        for key, value in thing.attrs.items():
            out += '%30s : %s\n' % (key, value)
        return out

    def nx_image_data(self, index=None, address=None):
        """
        Return image data, if available
        if index==None, all images are combined, otherwise only a single frame at index is returned
        """
        return image_data(self, index, address)

    def __repr__(self):
        return "Hdf5Nexus('%s')" % self.filename

    def __str__(self):
        return "%s\n%s" % (self.__repr__(), self.nx_tree_str())

