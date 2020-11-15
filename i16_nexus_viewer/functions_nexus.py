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

import os
import datetime
import re
import numpy as np
import h5py
from imageio import imread  # read Tiff images


BYTES_DECODER = 'utf-8'
VALUE_FORMAT = '%.5g'
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
OUTPUT_FORMAT = '%20s = %s\n'


def scanfile2number(filename):
    """
    Extracts the scan number from a .nxs filename
    :param filename: str : filename of .nxs file
    :return: int : scan number
    """
    return np.abs(np.int(os.path.split(filename)[-1][-10:-4]))


def scannumber2file(number, name_format="%d.nxs"):
    """Convert number to scan file using format"""
    return name_format % number


def array_str(array):
    """
    Returns a short string with array information
    :param array: np array
    :return: str
    """
    shape = np.shape(array)
    amax = np.max(array)
    amin = np.min(array)
    amean = np.mean(array)
    out_str = "%s max: %4.5g, min: %4.5g, mean: %4.5g"
    return out_str % (shape, amax, amin, amean)


def load(filename):
    """Load a hdf5 or nexus file"""
    return h5py.File(filename, 'r')


def reload(hdf):
    """Reload a hdf file, hdf = reload(hdf)"""
    filename = hdf.filename
    return load(filename)


def address_name(address):
    """Convert hdf address to name"""
    return os.path.basename(address)


def dataset_data(dataset):
    """Get data from dataset"""
    data = np.asarray(dataset[()]).reshape(-1)
    if len(data) == 1: data = data[0]
    # Handle bytes strings to return string
    try:
        data = data.decode(BYTES_DECODER)
    except (UnicodeDecodeError, AttributeError):
        pass
    return data


def dataset_string(dataset):
    """Generate string from dataset"""
    data = dataset_data(dataset)
    try:
        # single value
        return VALUE_FORMAT % data
    except TypeError:
        pass
    try:
        # array
        return array_str(data)
    except TypeError:
        pass
    # probably a string
    return '%s' % data


def dataset_datetime(dataset, input_format=None, output_format=None):
    """
    Read time stamps from hdf file at specific address
    If input is a string (or bytes), input_format is used to parse the string
    If input is a float, it is assumed to be a timestamp from the Unix Epoch (1970-01-01 00:00:00)

    Useful Format Specifiers (https://strftime.org/):
    %Y year         %m month      %d day      %H hours    %M minutes  %S seconds  %f microseconds
    %y year (short) %b month name %a day name %I 12-hour  %p AM or PM %z UTC offset

    :param dataset: hdf dataset
    :param input_format: str datetime.strptime format specifier to parse dataset
    :param output_format: str datetime.strftime format specifier to generate output string (if None, returns datetime)
    """
    if input_format is None:
        input_format = DATE_FORMAT
    data = dataset_data(dataset)
    data = np.asarray(data, dtype=str).reshape(-1)
    try:
        # str date passed, e.g. start_time: '2020-10-22T09:33:11.894+01:00'
        dates = np.array([datetime.datetime.strptime(date, input_format) for date in data])
    except ValueError:
        # float timestamp passed, e.g. TimeFromEpoch: 1603355594.96
        dates = np.array([datetime.datetime.fromtimestamp(float(time)) for time in data])

    if output_format is None:
        if len(data) == 1:
            return dates[0]
        return dates
    if len(data) == 1:
        return dates[0].strftime(output_format)
    return [date.strftime(output_format) for date in dates]


def data_strings(hdf_group, addresses, output_format=None):
    """
    Return strings of data using output_format
    :param hdf_group: hdf5 File or Group object
    :param addresses: list of str or str hdf dataset addresses
    :param output_format: str
    :return: str
    """
    addresses = np.asarray(addresses, dtype=str).reshape(-1)
    if output_format is None:
        output_format = OUTPUT_FORMAT
    out_str = ''
    for address in addresses:
        name = address_name(address)
        dataset = hdf_group.get(address)
        if dataset:
            data_str = dataset_string(dataset)
        else:
            data_str = 'Not available'
        out_str += output_format % (name, data_str)
    return out_str


def tree(hdf_group, recursion_limit=100):
    """
    Return str of the full tree of data in a hdf object
    :param hdf_group: hdf5 File or Group object
    :param recursion_limit: int max number of levels
    :return:
    """
    if recursion_limit < 1: return ''
    outstr = '\nGroup: %s\n' % hdf_group.name
    for attr, val in hdf_group.attrs.items():
        outstr += '  attr: %s: %s\n' % (attr, val)
    try:
        for branch in hdf_group.keys():
            outstr += tree(hdf_group.get(branch), recursion_limit-1)
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


def hdf_addresses(filename, addresses='/', recursion_limit=100):
    """
    Return list of addresses of datasets, starting at each address
    :param filename: filename of hdf5 File
    :param addresses: list of str or str : start in this / these addresses
    :param recursion_limit: Limit on recursivley checking lower groups
    :return: list of str
    """
    with h5py.File(filename, 'r') as hdf_group:
        out = dataset_addresses(hdf_group, addresses, recursion_limit)
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
        a_name = (address_name(address) if whole_word else address)
        a_name = (a_name if match_case else a_name.lower())
        if whole_word and name == a_name:
            out += [hdf_group.get(address).name]
        elif not whole_word and name in a_name:
            out += [hdf_group.get(address).name]
    return out


def find_nxclass(hdf_group, nxclass='NX_detector'):
    """
    Returns location of hdf group with attribute ['NX_class']== nxclass
    :param hdf_group: hdf5 File or Group object
    :param nxclass: str
    :return: str hdf address
    """
    if 'NX_class' in hdf_group.attrs and hdf_group.attrs['NX_class'] == nxclass.encode():
        return hdf_group.name
    try:
        for branch in hdf_group.keys():
            address = find_nxclass(hdf_group.get(branch), nxclass)
            if address:
                return address
    except AttributeError:
        pass


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
    return []


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


def auto_xyaxis(hdf_group):
    """
    Find default axes, signal hdf addresses
    :param hdf_group: hdf5 File or Group object
    :return: xaxis_address, yaxis_address
    """
    xaxis = find_attr(hdf_group, 'axes')
    yaxis = find_attr(hdf_group, 'signal')
    if not xaxis: xaxis = [None]
    if not yaxis: yaxis = [None]
    return xaxis[0], yaxis[0]


def names2data(hdf_group, names):
    """
    Return data using data names, names can be addresses or simpler, where find will be used
    :param hdf_group: hdf5 File or Group object
    :param names: list
    :return: dict
    """
    out = {}
    for name in names:
        dataset = hdf_group.get(name)
        if dataset:
            out[name] = dataset_data(dataset)
        else:
            addresses = find_name(hdf_group, name, match_case=True)
            if len(addresses) == 0:
                addresses = find_name(hdf_group, name, match_case=False)
            if len(addresses) == 0:
                data = []
            else:
                dataset = hdf_group.get(addresses[0])
                data = dataset_data(dataset)
            out[name] = data
    return out


def names2string(hdf_group, names, output_format=None):
    """
    Return string of data from names
    :param hdf_group: hdf5 File or Group object
    :param names: list
    :param output_format: str format or None
    :return: dict
    """
    addresses = []
    for name in names:
        address = find_name(hdf_group, name)
        if address:
            addresses += [address[0]]
    return data_strings(hdf_group, addresses, output_format)


def names_operation(hdf_group, operation):
    """
    Interpret a string as a series of hdf addresses or dataset names, returning a evuatable string and dict of data.
      operation, data = names_operation(hdf_group, operation)
    Example:
        operation, data = names_operation(hdf, 'measurement/roi2_sum /Transmission')
        output = eval(operation, globals(), data)
    :param hdf_group: hdf5 File or Group object
    :param operation: str operation e.g. 'measurement/roi2_sum /Transmission'
    :return operation: str updated operation string with addresses converted to names
    :return data: dict of names and data
    """
    # First look for addresses in operation
    addresses = re.findall(r'\w*?/[\w/]*', operation)
    data_dict = {}
    for address in set(a.strip('/') for a in addresses):  # unique addresses with removed trailing /
        # check if full address
        dataset = hdf_group.get(address)
        if dataset:
            data = dataset_data(dataset)
            name = address_name(address)
            data_dict[name] = data
            operation = operation.replace(address, name)
        else:
            f_address = find_name(hdf_group, address)
            if len(f_address) > 0:
                dataset = hdf_group.get(f_address[0])
                data = dataset_data(dataset)
                name = address_name(f_address[0])
                data_dict[name] = data
                operation = operation.replace(address, name)

    # Determine data for other variables
    names = re.findall(r'[a-zA-Z]\w*', operation)
    for name in names:
        if name in data_dict.keys():
            continue
        elif name.lower() in ['xaxis', 'axes']:
            f_address = find_attr(hdf_group, 'axes')
        elif name.lower() in ['yaxis', 'signal']:
            f_address = find_attr(hdf_group, 'signal')
        else:
            f_address = find_name(hdf_group, name, match_case=False, whole_word=True)

        if len(f_address) > 0:
            dataset = hdf_group.get(f_address[0])
            data = dataset_data(dataset)
            f_name = address_name(f_address[0])
            data_dict[f_name] = data
            operation = operation.replace(name, f_name)
    return operation, data_dict


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
    # filepath = os.path.dirname(hdf_group.file.filename)
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


def image_size(hdf_group, address=None):
    """
    Returns image shape [scan len, vertical, horizontal]
    :param hdf_group: hdf5 File or Group object
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :return: (n,m,o) : len, vertical, horizontal
    """
    if address is None:
        address = find_image(hdf_group)
    dataset = hdf_group.get(address)
    # if array - return array shape
    if len(dataset.shape) > 1:
        return dataset.shape
    array_len = len(dataset)
    # Get 1st image
    image = image_data(hdf_group, 0, address)
    return (array_len, *image.shape)


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


def image_roi(hdf_group, address=None, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
    """
    Create new region of interest (roi) from image data, return roi volume
    :param hdf_group: hdf5 File or Group object
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :param cen_h: int centre of roi in dimension m (horizontal)
    :param cen_v: int centre of roi in dimension n (vertical)
    :param wid_h: int full width of roi in diemnsion m (horizontal)
    :param wid_v: int full width of roi in dimension n (vertical)
    :return: [n, wid_v, wid_h] array of roi
    """

    if address is None:
        address = find_image(hdf_group)

    shape = image_size(hdf_group, address)

    if cen_h is None:
        cen_h = shape[2] // 2
    if cen_v is None:
        cen_v = shape[1] // 2

    roi = np.zeros([shape[0], wid_v, wid_h])
    for n in range(shape[0]):
        image = image_data(hdf_group, n, address)
        roi[n] = image[cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]
    return roi


def image_roi_sum(hdf_group, address=None, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
    """
    Create new region of interest (roi) from image data and return sum and maxval
    :param hdf_group: hdf5 File or Group object
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :param cen_h: int centre of roi in dimension m (horizontal)
    :param cen_v: int centre of roi in dimension n (vertical)
    :param wid_h: int full width of roi in diemnsion m (horizontal)
    :param wid_v: int full width of roi in dimension n (vertical)
    :return: sum, maxval : [o] length arrays
    """

    if address is None:
        address = find_image(hdf_group)

    shape = image_size(hdf_group, address)

    if cen_h is None:
        cen_h = shape[2] // 2
    if cen_v is None:
        cen_v = shape[1] // 2

    roi_sum = np.zeros(shape[0])
    roi_max = np.zeros(shape[0])
    for n in range(shape[0]):
        image = image_data(hdf_group, n, address)
        roi = image[cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]
        roi_sum[n] = np.sum(roi)
        roi_max[n] = np.max(roi)
    return roi_sum, roi_max


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

    def nx_reload(self):
        """Closes the hdf file and re-opens"""
        filename = self.filename
        self.close()
        self.__init__(filename)

    def nx_refresh(self, dataset_address):
        """Refresh dataset"""
        dataset = self.get(dataset_address)
        dataset.refresh()

    def nx_dataset_addresses(self):
        return dataset_addresses(self.get('/'))

    def nx_tree_str(self):
        return tree(self.get('/'))

    def nx_find_name(self, name, match_case=False, whole_word=False):
        """Return addresses with name"""
        return find_name(self, name, match_case, whole_word)

    def nx_find_attr(self, attr='axes'):
        """Returns location of hdf attribute "axes" """
        return find_attr(self, attr)

    def nx_find_nxclass(self, nxclass='NXdetector'):
        """Returns location of hdf group with attr['NX_class']==nxclass"""
        return find_nxclass(self, nxclass)

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
