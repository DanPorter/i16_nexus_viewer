"""
Basic Functions for reading h5py/ nexus files

These functions always take the hdf filename and reload the file every time

HdfLoader class
    Collection of functions to operate on a hdf or nexus file
    Doesn't load the file into memory at any point but reads the file on every operation, making it very simple and
    reliable but quite slow.
    Useage:
      d = HdfLoader('12345.nxs')
      xdata, ydata = d('axes, signal')

    Behaviour:
     d['entry1/title'] - returns dataset object from file, using hdf address
     d('title') - searches file for address with 'title' in, returns data
     d('roi2_sum /Transmission') - searches file for address with roi2_sum and Transmission, evaluates operation
     print(d) - prints top of tree hiarachy

    Useful functions:
     d.axes()   Return axes (xaxis) label and data
     d.signal() Return signal (yaxis) label and data
     d.tree()   Returns full tree structure of file as str
     d.image_data()  Returns image data
     d.image_roi()  Returns sum and maxval of new region of interest from image
"""

import os
import numpy as np

from . import functions_nexus as fn


def hdf_dataset(filename, address):
    """Return dataset from a hdf file at given address, this leaves the file open."""
    return fn.load(filename).get(address)


def hdf_data(filename, addresses):
    """Return data from a hdf file at given address"""
    addresses = np.asarray(addresses, dtype=str).reshape(-1)
    out = []
    with fn.load(filename) as hdf:
        for address in addresses:
            dataset = hdf.get(address)
            out += [fn.dataset_data(dataset)]
    if len(addresses) == 1:
        return out[0]
    return out


def hdf_datetime(filename, address, input_format=None, output_format=None):
    """
    Read time stamps from hdf file at specific address
    If input is a string (or bytes), input_format is used to parse the string
    If input is a float, it is assumed to be a timestamp from the Unix Epoch (1970-01-01 00:00:00)

    Useful Format Specifiers (https://strftime.org/):
    %Y year         %m month      %d day      %H hours    %M minutes  %S seconds  %f microseconds
    %y year (short) %b month name %a day name %I 12-hour  %p AM or PM %z UTC offset

    :param filename: str hdf fileaname
    :param address: str hdf dataset address
    :param input_format: str datetime.strptime format specifier to parse dataset
    :param output_format: str datetime.strftime format specifier to generate output string (if None, returns datetime)
    """
    dataset = hdf_dataset(filename, address)
    return fn.dataset_datetime(dataset, input_format, output_format)


def hdf_dict(filename, addresses):
    """Loads data from each dataset and ands to a dict"""
    data = hdf_data(filename, addresses)
    names = [fn.address_name(address) for address in addresses]
    return dict(zip(names, data))


def hdf_data_strings(filename, addresses, output_format=None):
    """
    Return strings of data using output_format
    :param filename: str hdf fileaname
    :param addresses: list of str or str hdf dataset addresses
    :param output_format: str
    :return: str
    """
    with fn.load(filename) as hdf:
        out_str = fn.data_strings(hdf, addresses, output_format)
    return out_str


def hdf_tree(filename, address='/', recursion_limit=100):
    """Open hdf file and return tree string"""
    with fn.load(filename) as hdf:
        hdf_group = hdf.get(address)
        out = fn.tree(hdf_group, recursion_limit)
    return out


def hdf_addresses(filename, addresses='/', recursion_limit=100):
    """
    Return list of addresses of datasets, starting at each address
    :param filename: filename of hdf5 File
    :param addresses: list of str or str : start in this / these addresses
    :param recursion_limit: Limit on recursivley checking lower groups
    :return: list of str
    """
    with fn.load(filename) as hdf_group:
        out = fn.dataset_addresses(hdf_group, addresses, recursion_limit)
    return out


def hdf_find_name(filename, name, address='/', match_case=False, whole_word=False):
    """
    Find datasets using field name
    :param filename: str hdf5 File
    :param name: str : name to match in dataset field name
    :param address: str address of group to start in
    :param match_case: if True, match case of name
    :param whole_word: if True, only return whole word matches
    :return: list of str addresses
    """
    out = []
    addresses = hdf_addresses(filename, address)
    if not match_case: name = name.lower()
    for address in addresses:
        a_name = (fn.address_name(address) if whole_word else address)
        a_name = (a_name if match_case else a_name.lower())
        if whole_word and name == a_name:
            out += [address]
        elif not whole_word and name in a_name:
            out += [address]
    return out


def hdf_find_nxclass(filename, nxclass='NX_detector', address='/'):
    """
    Returns location of hdf group with attribute ['NX_class']== nxclass
    :param filename: str filename
    :param nxclass: str name of class attribute
    :param address: str address to start in
    :return:
    """
    with fn.load(filename) as hdf:
        hdf_group = hdf.get(address)
        out = fn.find_nxclass(hdf_group, nxclass)
    return out


def hdf_find_attr(filename, attr='axes', address='/'):
    """
    Returns location of hdf attribute
    Workds recursively - starts at the top level and searches all lower hdf groups
    :param filename: str filename
    :param attr: str : attribute name to search for
    :param address: str address to start in
    :return: str hdf address
    """
    with fn.load(filename) as hdf:
        hdf_group = hdf.get(address)
        out = fn.find_attr(hdf_group, attr)
    return out


def hdf_auto_xyaxis(filename, address='/'):
    """
    Find default axes, signal hdf addresses
    :param filename: str hdf filename
    :param address: str addtress to start in
    :return: xaxis_address, yaxis_address
    """
    with fn.load(filename) as hdf:
        hdf_group = hdf.get(address)
        xaxis, yaxis = fn.auto_xyaxis(hdf_group)
    return xaxis, yaxis


def hdf_names2data(filename, names, address='/'):
    """
    Return data using data names, names can be addresses or simpler, where find will be used
    :param filename: str hdf filename
    :param names: list
    :param address: str addtress to start in
    :return: dict
    """
    with fn.load(filename) as hdf:
        hdf_group = hdf.get(address)
        out = fn.names2data(hdf_group, names)
    return out


def hdf_names2string(filename, names, address='/', output_format=None):
    """
    Return string of data from names
    :param filename: str hdf filename
    :param names: list
    :param address: str addtress to start in
    :param output_format: str format or None
    :return: str
    """
    with fn.load(filename) as hdf:
        hdf_group = hdf.get(address)
        out = fn.names2string(hdf_group, names, output_format)
    return out


def hdf_names_operation(filename, operation, address='/'):
    """
    Interpret a string as a series of hdf addresses or dataset names, returning a evuatable string and dict of data.
      operation, data = names_operation(filename, operation)
    Example:
        operation, data = names_operation(filename, 'measurement/roi2_sum /Transmission')
        output = eval(operation, globals(), data)
    :param filename: str hdf filename
    :param operation: str operation e.g. 'measurement/roi2_sum /Transmission'
    :param address: str hdf group address
    :return operation: str updated operation string with addresses converted to names
    :return data: dict of names and data
    """
    with fn.load(filename) as hdf:
        hdf_group = hdf.get(address)
        operation, data_dict = fn.names_operation(hdf_group, operation)
    return operation, data_dict


def hdf_find_image(filename, address='/', multiple=False):
    """
    Return address of image data in hdf file
    Images can be stored as list of file directories when using tif file,
    or as a dynamic hdf link to a hdf file.

    :param filename: str hdf file
    :param address: initial hdf address to look for image data
    :param multiple: if True, return list of all addresses matching criteria
    :return: str or list of str
    """
    with fn.load(filename) as hdf:
        out = fn.find_image(hdf, address, multiple)
    return out


def hdf_image_data(filename, index=None, address=None):
    """
    Return image data, if available
    if index=None, all images are combined, otherwise only a single frame at index is returned
    :param filename: str hdf file
    :param index: None or int : return a specific image
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :return: 2d array if index given, 3d array otherwise
    """
    with fn.load(filename) as hdf:
        out = fn.image_data(hdf, index, address)
    return out


def hdf_image_roi(filename, address=None, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
    """
    Create new region of interest (roi) from image data, return roi volume
    :param filename: str hdf file
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :param cen_h: int centre of roi in dimension m (horizontal)
    :param cen_v: int centre of roi in dimension n (vertical)
    :param wid_h: int full width of roi in diemnsion m (horizontal)
    :param wid_v: int full width of roi in dimension n (vertical)
    :return: [n, wid_v, wid_h] array of roi
    """
    with fn.load(filename) as hdf:
        roi = fn.image_roi(hdf, address, cen_h, cen_v, wid_h, wid_v)
    return roi


def hdf_image_roi_sum(filename, address=None, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
    """
    Create new region of interest (roi) from image data and return sum and maxval
    :param filename: str hdf file
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :param cen_h: int centre of roi in dimension m (horizontal)
    :param cen_v: int centre of roi in dimension n (vertical)
    :param wid_h: int full width of roi in diemnsion m (horizontal)
    :param wid_v: int full width of roi in dimension n (vertical)
    :return: sum, maxval : [o] length arrays
    """
    with fn.load(filename) as hdf:
        roi_sum, roi_max = fn.image_roi_sum(hdf, address, cen_h, cen_v, wid_h, wid_v)
    return roi_sum, roi_max


"----------------------------------- HdfLoader Class -----------------------------------"


class HdfLoader:
    """
    HdfLoader class
    Collection of functions to operate on a hdf or nexus file
    Doesn't load the file into memory at any point but reads the file on every operation, making it very simple and
    reliable but quite slow.
    Useage:
      d = HdfLoader('12345.nxs')
      xdata, ydata = d('axes, signal')

    Behaviour:
     d['entry1/title'] - returns dataset object from file, using hdf address
     d('title') - searches file for address with 'title' in, returns data
     d('roi2_sum /Transmission') - searches file for address with roi2_sum and Transmission, evaluates operation
     print(d) - prints top of tree hiarachy

    Useful functions:
     d.axes()   Return axes (xaxis) label and data
     d.signal() Return signal (yaxis) label and data
     d.tree()   Returns full tree structure of file as str
     d.image_data()  Returns image data
     d.image_roi()  Returns sum and maxval of new region of interest from image

    Inputs:
    :param filename: str filename of the HDF5 (.hdf) or NeXus (.nxs) file
    """
    def __init__(self, filename):
        self.filename = filename
        self.basename = os.path.basename(filename)

    def __repr__(self):
        return 'HdfLoader(\'%s\')' % self.filename

    def __str__(self):
        out = fn.OUTPUT_FORMAT % ('filename', self.basename)
        axes, signal = hdf_auto_xyaxis(self.filename)
        out += fn.OUTPUT_FORMAT % ('axes', axes)
        out += fn.OUTPUT_FORMAT % ('signal', signal)
        top_addresses = self.addresses(recursion_limit=2)
        out += self.data_string(top_addresses)
        return out

    def __call__(self, *args, **kwargs):
        return self.eval_operation(*args, **kwargs)

    def __getitem__(self, item):
        return self.dataset(item)

    def load(self):
        """Load hdf or nxs file"""
        return fn.load(self.filename)

    def dataset(self, address):
        return hdf_dataset(self.filename, address)
    dataset.__doc__ = hdf_dataset.__doc__
    get = dataset

    def addresses(self, addresses='/', recursion_limit=100):
        return hdf_addresses(self.filename, addresses, recursion_limit)
    addresses.__doc__ = hdf_addresses.__doc__

    def data(self, addresses):
        return hdf_data(self.filename, addresses)
    data.__doc__ = hdf_data.__doc__

    def data_string(self, addresses, output_format=None):
        return hdf_data_strings(self.filename, addresses, output_format)
    data_string.__doc__ = hdf_data_strings.__doc__

    def data_dict(self, addresses):
        return hdf_dict(self.filename, addresses)
    data_dict.__doc__ = hdf_dict.__doc__

    def names2data(self, names, address='/'):
        return hdf_names2data(self.filename, names, address)
    names2data.__doc__ = hdf_names2data.__doc__

    def names2string(self, names, address='/', output_format=None):
        return hdf_names2string(self.filename, names, address, output_format)
    names2string.__doc__ = hdf_names2string.__doc__

    def tree(self, address='/', recursion_limit=100):
        return hdf_tree(self.filename, address, recursion_limit)
    tree.__doc__ = hdf_tree.__doc__

    def datetime(self, address, input_format="%Y-%m-%dT%H:%M:%S.%f%z", output_format=None):
        return hdf_datetime(self.filename, address, input_format, output_format)
    datetime.__doc__ = hdf_datetime.__doc__

    def find_name(self, name, address='/', match_case=False, whole_word=False):
        return hdf_find_name(self.filename, name, address, match_case, whole_word)
    find_name.__doc__ = hdf_find_name.__doc__

    def find_nxclass(self, nxclass='NX_detector', address='/'):
        return hdf_find_nxclass(self.filename, nxclass, address)
    find_nxclass.__doc__ = hdf_find_nxclass.__doc__

    def find_attr(self, attr='axes', address='/'):
        return hdf_find_attr(self.filename, attr, address)
    find_attr.__doc__ = hdf_find_attr.__doc__

    def find_image(self, address='/', multiple=False):
        return hdf_find_image(self.filename, address, multiple)
    find_image.__doc__ = hdf_find_image.__doc__

    def axes(self):
        """Return axes (xaxis) label and data"""
        address = self.find_attr('axes')
        if not address:
            return 'None', None
        else:
            address = address[0]
        name = fn.address_name(address)
        data = self.data(address)
        return name, data
    xaxis = axes

    def signal(self):
        """Return signal (yaxis) label and data"""
        address = self.find_attr('signal')
        if not address:
            return 'None', None
        else:
            address = address[0]
        name = fn.address_name(address)
        data = self.data(address)
        return name, data
    yaxis = signal

    def names_operation(self, operation, address='/'):
        return hdf_names_operation(self.filename, operation, address)
    names_operation.__doc__ = hdf_names_operation.__doc__

    def eval_operation(self, operation, address='/', namespace_dict=None):
        """Evaluate a string, names and addresses from the hdf file can be used"""
        operation, data = hdf_names_operation(self.filename, operation, address)
        if namespace_dict is None:
            namespace_dict = {}
        namespace_dict.update(data)
        return eval(operation, globals(), namespace_dict)

    def image_data(self, index=None, address=None):
        return hdf_image_data(self.filename, index, address)
    image_data.__doc__ = hdf_image_data.__doc__

    def image_roi(self, address=None, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
        return hdf_image_roi_sum(self.filename, address, cen_h, cen_v, wid_h, wid_v)
    image_data.__doc__ = hdf_image_data.__doc__

