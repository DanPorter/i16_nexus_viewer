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
from pandas import DataFrame


BYTES_DECODER = 'utf-8'
VALUE_FUNCTION = np.mean  # lambda a: np.asarray(a).reshape(-1)[0]
VALUE_FORMAT = '%.5g'
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
OUTPUT_FORMAT = '%20s = %s'
TITLE_COMMAND = '#{entry_identifier} {title}'
LABEL_COMMAND = '#{entry_identifier}'

# Compile useful pattern strings
find_integer = re.compile(r'\d+')

# Scan default name/ addresses
DEFAULTS = {
    'filename_format': '%06d.nxs',
    'scan_length': 'scan_dimensions',
    'scan_number': 'entry_identifier',
    'start_time': 'start_time',
    'end_time': 'end_time',
    'title': 'title',
    'cmd': 'scan_command',
    'title_command': TITLE_COMMAND,
    'label_command': LABEL_COMMAND,
    'scan_addresses': 'measurement',
    'meta_addresses': 'before_scan',
    'show_metadata': '',
    'normalisation_names': ['sum', 'maxval', 'roi2_sum', 'roi1_sum'],
    'normalisation_command': '%s/Transmission/count_time',
    'error_names': ['sum', 'roi2_sum', 'roi1_sum'],
    'error_command': 'np.sqrt(%s+1)',
    'plot_errors': True
}

DEFAULTS_HELP = {
    'filename_format': 'name of scan files using scan number',
    'scan_length': 'nexus address or name of entry giving scan length or number of points',
    'scan_number': 'nexus address or name of entry giving scan number',
    'start_time': 'nexus address or name of entry giving time_start time',
    'end_time': 'nexus address or name of entry giving end time',
    'title': 'nexus address or name of entry giving scan title',
    'cmd': 'nexus address or name of entry giving scan command',
    'title_command': 'command string for scan title, values in {} will be evaluated',
    'label_command': 'command string for scan label, values in {} will be evaluated',
    'scan_addresses': 'nexus address or name of group where scan data is kept',
    'meta_addresses': 'nexus address or name of group where metadata is kept',
    'show_metadata': '',
    'normalisation_names': 'List of nexus addresses or names that will be automatically normalised',
    'normalisation_command': 'Default normalisation command, %s defines value to be normalised',
    'error_names': 'List of nexus addresses or names that will automatically have an error',
    'error_command': 'Default error command',
    'plot_errors': 'True/False, if True errors will be plotted'
}


def defaults_help(defaults_dict=None, help_dict=None):
    """
    Geenrate string that provides help for defaults dict
    :param defaults_dict: ditc
    :return: str
    """
    if defaults_dict is None:
        defaults_dict = DEFAULTS
    if help_dict is None:
        help_dict = DEFAULTS_HELP
    out = ' Defaults:'
    for key, item in defaults_dict.items():
        hp = help_dict[key] if key in help_dict else ''
        out += '%10s : %10s : %s,\n' % (key, item, hp)
    return out


"---------------------------BASIC FUNCTIONS---------------------------------"


def scanfile2number(filename):
    """
    Extracts the scan number from a .nxs filename
    :param filename: str : filename of .nxs file
    :return: int : scan number
    """
    nameext = os.path.split(filename)[-1]
    name = os.path.splitext(nameext)[0]
    numbers = find_integer.findall(name)
    if len(numbers) > 0:
        return np.int(numbers[-1])
    return 0


def scannumber2file(number, name_format="%d.nxs"):
    """Convert number to scan file using format"""
    return name_format % number


def shortstr(string):
    """
    Shorten string by removing long floats
    :param string: string, e.g. '#810002 scan eta 74.89533603616637 76.49533603616636 0.02 pil3_100k 1 roi2'
    :return: shorter string, e.g. '#810002 scan eta 74.895 76.495 0.02 pil3_100k 1 roi2'
    """
    #return re.sub(r'(\d\d\d)\d{4,}', r'\1', string)
    def subfun(m):
        return str(round(float(m.group()), 3))
    return re.sub(r'\d+\.\d{5,}', subfun, string)


def array_str(array):
    """
    Returns a short string with array information
    :param array: np array
    :return: str
    """
    shape = np.shape(array)
    try:
        amax = np.max(array)
        amin = np.min(array)
        amean = np.mean(array)
        out_str = "%s max: %4.5g, min: %4.5g, mean: %4.5g"
        return out_str % (shape, amax, amin, amean)
    except TypeError:
        # list of str
        array = np.asarray(array).reshape(-1)
        array_start = array[0]
        array_end = array[-1]
        out_str = "%s [%s, ..., %s]"
        return out_str % (shape, array_start, array_end)


def address_name(address):
    """Convert hdf address to name"""
    return os.path.basename(address)


def address_group(address, group_name):
    """
    Return part of address upto group_name
    :param address: str hdf address
    :param group_name: str name of group
    :return: reduced str
    """
    return re.findall(r'(.+?%s.*?)(?:\/|$)' % group_name, address, re.IGNORECASE)[0]


def axes_from_cmd(cmd):
    """
    Get axes name from command string
    :param cmd: str
    :return: str
    """
    cmd = cmd.split()
    axes = cmd[1]
    # These are specific to I16...
    if axes == 'hkl':
        if cmd[0] == 'scan':
            hstep, kstep, lstep = cmd[8:11]
        elif cmd[0] == 'scancn':
            hstep, kstep, lstep = cmd[2:5]
        else:
            raise Warning('Warning unknown type of hkl scan')

        if float(re.sub("[^0-9.]", "", hstep)) > 0.0:
            axes = 'h'
        elif float(re.sub("[^0-9.]", "", kstep)) > 0.0:
            axes = 'k'
        else:
            axes = 'l'
    elif axes == 'sr2':
        axes = 'azimuthal'  # 'phi' in pre-DiffCalc scans
    elif axes == 'th2th':
        axes = 'delta'
    elif axes == 'ppp_energy':
        axes = 'ppp_offset'
    return axes


def signal_from_cmd(cmd):
    """
    Get signal name from command string
    :param cmd: str
    :return: str
    """
    cmd_split = cmd.split()
    try:
        float(cmd_split[-1])
        signal = cmd_split[-2]
    except ValueError:
        signal = cmd_split[-1]
    # These are specific to I16...
    if signal == 't':
        signal = 'APD'
    elif 'roi' in signal:
        signal = signal + '_sum'
    elif 'pil100k' in cmd:
        signal = 'sum'
    elif 'pil2m' in cmd:
        signal = 'sum'
    elif 'merlin' in cmd:
        signal = 'sum'
    elif 'bpm' in cmd:
        signal = 'sum'
    elif 'QBPM6' in cmd:
        signal = 'C1'
    return signal


"----------------------------LOAD FUNCTIONS---------------------------------"


def load(filename):
    """Load a hdf5 or nexus file"""
    try:
        return h5py.File(filename, 'r')
    except OSError:
        raise Exception('File does not exist: %s' % filename)


def reload(hdf):
    """Reload a hdf file, hdf = reload(hdf)"""
    filename = hdf.filename
    return load(filename)


"--------------------------DATASET FUNCTIONS--------------------------------"


def is_dataset(dataset):
    """
    Check if input is a hdf dataset
     e.g. is_dataset(hdf_group.get(address))
    """
    return hasattr(dataset, 'size')


def is_group(dataset):
    """
    Check if input is a hdf group
    :param dataset:
    :return: True/ False
    """
    return hasattr(dataset, 'keys')


def dataset_name(dataset):
    """
    Return name of the dataset
    the name is the final part of the hdf dataset address
    equivalent to:
      dataset_name = dataset.name.split('/')[-1]
    Warning - dataset.name is not always stored as the correct value
    """
    return address_name(dataset.name)


def dataset_data(dataset):
    """Get data from dataset, return float, array or str"""
    # convert arrays of length 1 to values
    if not dataset:
        return None
    if dataset.size == 1 and len(dataset.shape) == 1:
        data = np.asarray(dataset)[0]
    else:
        data = dataset[()]
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
        # array
        if dataset.size > 1:
            return array_str(data)
    # probably a string
    return shortstr('%s' % data)


def dataset_value(dataset):
    """Return single value from dataset"""
    data = dataset_data(dataset)
    try:
        # numberic
        return VALUE_FUNCTION(data)
    except TypeError:
        # str
        return data


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
    :return datetime or list of datetime
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

    if output_format:
        if len(data) == 1:
            return dates[0].strftime(output_format)
        else:
            return [date.strftime(output_format) for date in dates]
    else:
        if len(data) == 1:
            return dates[0]
        return dates


def show_attrs(dataset):
    """Return formatted string of attributes for hdf object"""
    out = '%s with %d attrs\n' % (dataset, len(dataset.attrs))
    out += '%s\n' % dataset.name
    for key, value in dataset.attrs.items():
        out += '%30s : %s\n' % (key, value)
    return out


"-------------------------HDF ADDRESS FUNCTIONS-------------------------------"


def dataset_addresses(hdf_group, addresses='/', recursion_limit=100, get_size=None, get_ndim=None):
    """
    Return list of addresses of datasets, starting at each address
    :param hdf_group: hdf5 File or Group object
    :param addresses: list of str or str : time_start in this / these addresses
    :param recursion_limit: Limit on recursivley checking lower groups
    :param get_size: None or int, if int, return only datasets with matching size
    :param get_ndim: None or int, if int, return only datasets with matching ndim
    :return: list of str
    """
    addresses = np.asarray(addresses, dtype=str).reshape(-1)
    out = []
    for address in addresses:
        data = hdf_group.get(address)
        if data and is_dataset(data):
            # address is dataset
            if (get_size is None and get_ndim is None) or (get_size is not None and data.size == get_size) or (
                    get_ndim is not None and data.ndim == get_ndim):
                out += [address]
        elif data and recursion_limit > 0:
            # address is Group
            new_addresses = ['/'.join([address, d]).replace('//', '/') for d in data.keys()]
            out += dataset_addresses(hdf_group, new_addresses, recursion_limit - 1, get_size, get_ndim)
        elif recursion_limit > 0:
            # address is None, search for group address and iterate
            new_address = get_address(hdf_group, address, return_group=True)
            if new_address:
                out += dataset_addresses(hdf_group, new_address, recursion_limit - 1, get_size, get_ndim)
    return out


def find_name(name, address_list, match_case=False, whole_word=False):
    """
    Find datasets using field name
    :param name: str : name to match in dataset field name
    :param address_list: list of str: list of str to search in
    :param match_case: if True, match case of name
    :param whole_word: if True, only return whole word matches
    :return: list of str matching dataset addresses
    """
    out = []
    if not match_case: name = name.lower()
    for address in address_list:
        a_name = (address_name(address) if whole_word else address)
        a_name = (a_name if match_case else a_name.lower())
        if whole_word and name == a_name:
            out += [address]
        elif not whole_word and name in a_name:
            out += [address]
    return out


"----------------------ADDRESS DATASET FUNCTIONS------------------------------"


def get_address(hdf_group, name, address_list=None, return_group=False, ):
    """
    Return address of dataset that most closely matches str name
     if multiple addresses match, take the longest array
    :param hdf_group: hdf5 File or Group object
    :param name: str or list of str of dataset address or name
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :param return_group: Bool if True returns the group address rather than dataset address
    :return: str address of best match or list of str address with same length as name list
    """
    names = np.asarray(name, dtype=str).reshape(-1)
    if address_list is None:
        address_list = dataset_addresses(hdf_group)
    addresses = []
    for name in names:
        # address
        if is_dataset(hdf_group.get(name)):
            addresses += [name]
            continue
        elif return_group and is_group(hdf_group.get(name)):
            addresses += [name]
            continue

        # special names
        if name.lower() in ['xaxis', 'axes']:
            f_address = [auto_xyaxis(hdf_group)[0]]
        elif name.lower() in ['yaxis', 'signal']:
            f_address = [auto_xyaxis(hdf_group)[1]]
        else:
            # search tree
            f_address = find_name(name, address_list, match_case=True, whole_word=True)
        if len(f_address) == 0:
            f_address = find_name(name, address_list, match_case=False, whole_word=True)
        if len(f_address) == 0:
            f_address = find_name(name, address_list, match_case=False, whole_word=False)
        if len(f_address) == 0:
            addresses += [None]
            continue

        if return_group:
            f_address = address_group(f_address[0], name)
            addresses += [f_address]
            continue

        # select longest length dataset
        if len(f_address) > 1:
            datasets = [hdf_group.get(ad) for ad in f_address]
            max_len = np.argmax([ds.size for ds in datasets if is_dataset(ds)])
            addresses += [f_address[int(max_len)]]
        else:  # len address == 1
            addresses += f_address

    if len(names) == 1:
        return addresses[0]
    return addresses


def get_group_address(hdf_group, name, address_list=None):
    """
    Return address of hdf group that most closely matches str name
    Separates groups from datasets using keys attribute
    :param hdf_group: hdf5 File or Group object
    :param name: str or list of str of group address or name
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: str address of best match or list of str address with same length as name list
    """
    return get_address(hdf_group, name, address_list, return_group=True)


def get_datasets(hdf_group, names, address_list=None):
    """
    Return datasets using data names, names can be addresses or simpler, where find will be used
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: list of hdf datasets
    """
    names = np.asarray(names, dtype=str).reshape(-1)
    addresses = get_address(hdf_group, names, address_list)
    if len(names) == 1:
        if addresses:
            return [hdf_group.get(addresses)]
        return [None]
    return [hdf_group.get(a) if a else None for a in addresses]


def get_data(hdf_group, names, address_list=None):
    """
    Return data values using dataset names or addresses.
    If name is not an address, the longest dataset from find_name will be used
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: list of data, returning value, array or str
    """
    names = np.asarray(names, dtype=str).reshape(-1)
    datasets = get_datasets(hdf_group, names, address_list)
    if len(names) == 1:
        return dataset_data(datasets[0])
    return [dataset_data(dataset) for dataset in datasets]


def get_value(hdf_group, names, address_list=None):
    """
    Return single data values using dataset names or addresses.
    If name is not an address, the longest dataset from find_name will be used
    If the data is an array, the value will be treated using VALUE_FUNCTION
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: list of data, returning value, array or str
    """
    names = np.asarray(names, dtype=str).reshape(-1)
    datasets = get_datasets(hdf_group, names, address_list)
    if len(names) == 1:
        return dataset_value(datasets[0])
    return [dataset_value(dataset) for dataset in datasets]


def get_data_dict(hdf_group, names, address_list=None):
    """
    Return data using data names, names can be addresses or simpler, where find will be used
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: dict of data {name/address: data}
    """
    names = np.asarray(names, dtype=str).reshape(-1)
    data = get_data(hdf_group, names, address_list)
    out = dict(zip(names, data))
    return out


def get_data_array(hdf_group, names, data_length=None, address_list=None):
    """
    Return array of data using data names, names can be addresses or simpler, where find will be used
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :param data_length: None or int, defines width of array, if None uses the shortest array length
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: np.array
    """
    names = np.asarray(names, dtype=str).reshape(-1)
    data = get_data(hdf_group, names, address_list)
    if len(names) == 1:
        if data_length is None:
            data_length = len(data)
        return data[:data_length]

    if data_length is None:
        data_length = np.min([np.size(a) for a in data if np.size(a) > 1])
    return np.array([np.repeat(a, data_length) if np.size(a) == 1 else a[:data_length] for a in data], dtype=float)


def data_strings(hdf_group, names, output_format=None, address_list=None):
    """
    Return strings of data using output_format
    :param hdf_group: hdf5 File or Group object
    :param names: list of str or str hdf dataset addresses or names
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :param output_format: str
    :return: str
    """
    if output_format is None:
        output_format = OUTPUT_FORMAT

    names = np.asarray(names, dtype=str).reshape(-1)
    datasets = get_datasets(hdf_group, names, address_list)
    out = []
    for address, dataset in zip(names, datasets):
        name = address_name(address)
        if dataset:
            data_str = dataset_string(dataset)
        else:
            data_str = 'Not available'
        out += [output_format % (name, data_str)]
    if len(names) == 1:
        return out[0]
    return out


"-------------------------HDF GROUP DATA FUNCTIONS-------------------------------"


def get_group_data(hdf_group, names):
    """
    Return data values using hdf group names or addresses.
    If name is not an address, the first result of find_name will be used
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :return: dict of data, returning {name: value, array or str}
    """
    addresses = dataset_addresses(hdf_group, names)
    #datasets = get_datasets(hdf_group, addresses)  # this is slow to get datasets
    datasets = [hdf_group.get(address) for address in addresses]
    return {dataset_name(dataset): dataset_data(dataset) for dataset in datasets}


def get_group_values(hdf_group, names):
    """
    Return data values using hdf group names or addresses.
    If name is not an address, the first result of find_name will be used
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :return: dict of data
    """
    addresses = dataset_addresses(hdf_group, names)
    # datasets = get_datasets(hdf_group, addresses)  # this is slow to get datasets
    datasets = [hdf_group.get(address) for address in addresses]
    return {dataset_name(dataset): dataset_value(dataset) for dataset in datasets}


def get_group_string(hdf_group, names, output_format=None):
    """
    Return data string using hdf group names or addresses.
    If name is not an address, the first result of find_name will be used
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :param output_format: str
    :return: dict of data
    """
    if output_format is None:
        output_format = OUTPUT_FORMAT

    addresses = dataset_addresses(hdf_group, names)
    # datasets = get_datasets(hdf_group, addresses)  # this is slow to get datasets
    datasets = [hdf_group.get(address) for address in addresses]

    out = []
    for address, dataset in zip(addresses, datasets):
        name = address_name(address)
        data_str = dataset_string(dataset)
        out += [output_format % (name, data_str)]
    if len(names) == 1:
        return out[0]
    return out


def get_group_array(hdf_group, names, data_length=None):
    """
    Return array of data from hdf file
        expands group addresses and returns array of same length arrays
        addresses with array length 1 are expanded to the shortest length of other arrays or data_length if defined
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :param data_length: None or int, defines width of array
    :return: array with shape(len(names), data_length)
    """
    data = get_group_data(hdf_group, names)
    if data_length is None:
        data_length = np.min([np.size(a) for a in data.values() if np.size(a) > 1])
    try:
        return np.array([a for a in data.values()], dtype=float)[:, :data_length]
    except ValueError:
        pass
    return np.array([np.repeat(a, data_length) if np.size(a) == 1 else a[:data_length] for a in data.values()], dtype=float)


def get_dataframe(hdf_group, names):
    """
    Return Pandas DataFrame of data from hdf file
        expands group addresses and returns array of same length arrays
        addresses with array length 1 are expanded to the shortest length of other arrays or data_length if defined
    :param hdf_group: hdf5 File or Group object
    :param names: str or list of str of dataset names or addresses
    :return: DataFrame
    """
    return DataFrame(get_group_data(hdf_group, names))


"-------------------------HDF GROUP INFO FUNCTIONS-------------------------------"


def filetitle(hdf_group):
    """
    Return the filename without directory or extension
    :param hdf_group: hdf5 File or Group object
    :return: str
    """
    filename = hdf_group.file.filename
    filepath, filename = os.path.split(filename)
    ftitle, ext = os.path.splitext(filename)
    return ftitle


def scannumber(hdf_group):
    """
    Return the scan number of the file from the filename
    :param hdf_group: hdf5 File or Group object
    :return: int
    """
    return scanfile2number(hdf_group.file.filename)


def tree(hdf_group, detail=False, recursion_limit=100):
    """
    Return str of the full tree of data in a hdf object
    :param hdf_group: hdf5 File or Group object
    :param detail: False/ True - provide further information about each group and dataset
    :param recursion_limit: int max number of levels
    :return: str
    """
    if recursion_limit < 1: return ''
    outstr = '%s\n' % hdf_group.name
    if detail:
        for attr, val in hdf_group.attrs.items():
            outstr += '  @%s: %s\n' % (attr, val)
    try:
        for branch in hdf_group.keys():
            new_group = hdf_group.get(branch)
            if new_group:
                outstr += tree(new_group, detail, recursion_limit-1)
        return outstr
    except AttributeError:
        # doesn't have .keys(), hdf_group = dataset, should have .name, .size, .shape
        if detail:
            out = '  %s: %s\n' % (hdf_group.name, dataset_string(hdf_group))
            for attr, val in hdf_group.attrs.items():
                out += '    @%s: %s\n' % (attr, val)
        else:
            out = '  %s, size: %s, shape: %s\n' % (hdf_group.name, hdf_group.size, hdf_group.shape)
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


def auto_xyaxis(hdf_group, cmd_string=None, address_list=None):
    """
    Find default axes, signal hdf addresses
    :param hdf_group: hdf5 File or Group object
    :param cmd_string: str of command to take x,y axis from as backup
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: xaxis_address, yaxis_address
    """
    try:
        # try fast nexus compliant method
        xaxis, yaxis = nexus_xyaxis(hdf_group)
    except KeyError:
        xaxis = ''
        yaxis = ''
        if cmd_string:
            xname = axes_from_cmd(cmd_string)
            xaxis = get_address(hdf_group, xname, address_list)
            yname = signal_from_cmd(cmd_string)
            yaxis = get_address(hdf_group, yname, address_list)

        if not xaxis:
            try:
                xaxis = find_attr(hdf_group, 'axes')[0]
            except IndexError:
                raise KeyError('axes not found in hdf hierachy')
        if not yaxis:
            try:
                yaxis = find_attr(hdf_group, 'signal')[0]
            except IndexError:
                raise KeyError('signal not found in hdf hierachy')
    return xaxis, yaxis


def nexus_xyaxis(hdf_group):
    """
    Nexus compliant method of finding default plotting axes in hdf files
     - find "default" entry in top File group
     - find "default" data in entry
     - find "axes" attr in default data
     - find "signal" attr in default data
     - generate addresses of signal and axes
     if not nexus compliant, raises KeyError
    This method is very fast but only works on nexus compliant files
    :param hdf_group: hdf5 File
    :return axes_address, signal_address: str hdf addresses
    """
    # From: https://manual.nexusformat.org/examples/h5py/index.html
    nx_entry = hdf_group[hdf_group.attrs["default"]]
    nx_data = nx_entry[nx_entry.attrs["default"]]
    axes_list = np.asarray(nx_data.attrs["axes"], dtype=str).reshape(-1)
    signal_list = np.asarray(nx_data.attrs["signal"], dtype=str).reshape(-1)
    axes_address = nx_data[axes_list[0]].name
    signal_address = nx_data[signal_list[0]].name
    return axes_address, signal_address


def badnexus_xyaxis(hdf_group):
    """
    Non-Nexus compliant method of finding default plotting axes in hdf files
     - search hdf hierarchy for attrs "axes" and "signal"
     - generate address of signal and axes
    raises KeyError if axes or signal is not found
    This method can be quite slow but is will work on many old nexus files.
    :param hdf_group: hdf5 File or Group object
    :return axes_address, signal_address: str hdf addresses
    """
    axes_address = find_attr(hdf_group, 'axes')
    signal_address = find_attr(hdf_group, 'signal')
    if len(axes_address) == 0:
        raise KeyError('axes not found in hdf hierachy')
    if len(signal_address) == 0:
        raise KeyError('signal not found in hdf hierachy')
    return axes_address[0], signal_address[0]


def duration(hdf_group, start_address='start_time', end_address='end_time', address_list=None):
    """
    Determine the duration of a scan using the start_time and end_time datasets
    :param hdf_group: hdf5 File or Group object 
    :param start_address: address or name of time_start time dataset
    :param end_address: address or name of end time dataset
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: datetime.timedelta
    """
    st_dataset, et_dataset = get_datasets(hdf_group, [start_address, end_address], address_list)
    if st_dataset is None:
        raise Exception('%s doesn\'t exist' % start_address)
    start_time = dataset_datetime(st_dataset)
    if et_dataset is None:
        end_time = datetime.datetime.now()
    else:
        end_time = dataset_datetime(et_dataset)
    timedelta = end_time - start_time
    return timedelta


"------------------------------------ Operations ----------------------------------------------------"


def names_operation(hdf_group, operation, namespace_dict=None, associated_names=None, address_list=None):
    """
    Interpret a string as a series of hdf addresses or dataset names, returning a evuatable string and dict of data.
      operation, data = names_operation(hdf_group, operation)
    Example:
        operation, data = names_operation(hdf, 'measurement/roi2_sum /Transmission')
        output = eval(operation, globals(), data)

    The allowed and special names are tabulated below:
       operation      | Example       | Explanation
     /hdf/dataset/path| /entry1/title | quickly retreives dataset and returns data
      dataset_name    | title         | searches the file tree for a dataset with that name, returns first hit
      xaxis/axes      | axes          | determines the automatically assigned axes value
      yaxis/signal    | signal        | determines the automatically assigned signal value
      nroi            | nroi[31,31]   | creates a region of interest in the detector images

    Notes on regions of interest:
    Each 'nroi' in the operation will create a region of interest in the standard detector images and return the sum
    of that region of interest. It will be renamed 'nroiN_sum in the final operation and dict, where N is roi number.
      'nroi'      -   creates a region of interest in the detector centre with size 31x31
      'nroi[h,v]' -   creates a roi in the detector centre with size hxv, where h is horizontal, v is vertical
      'nroi[n,m,h,v] - create a roi with cen_h, cen_v, wid_h, wid_v = n, m, h, v

    :param hdf_group: hdf5 File or Group object
    :param operation: str operation e.g. 'measurement/roi2_sum /Transmission'
    :param namespace_dict: dictionary of available namespace
    :param associated_names: dictionary of associated names. Items should be keys in namespace_dict
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return operation: str updated operation string with addresses converted to names
    :return data: dict of names and data
    :return associated_names: dict of relationships between names and data
    """
    if namespace_dict is None:
        namespace_dict = {}
    if associated_names is None:
        associated_names = {}
    if address_list is None:
        address_list = dataset_addresses(hdf_group)

    # Check operation isn't already in namespace
    if operation in namespace_dict:
        return operation, namespace_dict, associated_names
    if operation in associated_names and associated_names[operation] in namespace_dict:
        return associated_names[operation], namespace_dict, associated_names

    # First look for addresses in operation to seperate addresses from divide operations
    addresses = re.findall(r'\w*?/[\w/]*', operation)
    for address in addresses:
        if address_name(address) in namespace_dict:
            associated_names[address] = address_name(address)
            operation = operation.replace(address, address_name(address))
            continue
        # check if full address
        dataset = hdf_group.get(address)
        if dataset:
            data = dataset_data(dataset)
            name = address_name(address)
            namespace_dict[name] = data
            operation = operation.replace(address, name)
        else:
            # Check if part of an address
            f_address = find_name(address, address_list)
            if len(f_address) > 0:
                dataset = hdf_group.get(f_address[0])
                data = dataset_data(dataset)
                name = address_name(f_address[0])
                associated_names[address] = name
                namespace_dict[name] = data
                operation = operation.replace(address, name)

    # Determine custom regions of interest 'nroi'
    # rois = re.findall(r'nroi[\[\d,\]]*', operation)
    rois = re.findall(r'nroi\[\d+,\d+,\d+,\d+\]|nroi\[\d+,\d+\]|nroi', operation)
    for roi_no, roi in enumerate(rois):
        roi_sum, roi_maxval = image_roi_op(hdf_group, roi)
        name_sum = 'nroi%d_sum' % (roi_no + 1)
        name_max = 'nroi%d_max' % (roi_no + 1)
        namespace_dict[name_sum] = roi_sum
        namespace_dict[name_max] = roi_maxval
        operation = operation.replace(roi, name_sum)

    # Determine data for other variables
    names = re.findall(r'[a-zA-Z]\w*', operation)
    for name in names:
        if name in namespace_dict:
            continue
        if name in associated_names and associated_names[name] in namespace_dict:
            operation = operation.replace(name, associated_names[name])
        elif name.lower() in ['filename', 'file']:
            namespace_dict[name] = filetitle(hdf_group)
            continue
        elif name.lower() in ['scanno', 'scannumber']:
            namespace_dict[name] = scannumber(hdf_group)
            continue
        else:
            f_address = get_address(hdf_group, name, address_list=address_list)
            if f_address:
                dataset = hdf_group.get(f_address)
                data = dataset_data(dataset)
                f_name = address_name(f_address)
                namespace_dict[f_name] = data
                if name != f_name:
                    associated_names[name] = f_name
                operation = operation.replace(name, f_name)
    return operation, namespace_dict, associated_names


def value_operation(hdf_group, operation, namespace_dict=None, associated_names=None, address_list=None):
    """
    Evaluate operation and return result
    :param hdf_group: hdf5 File or Group object
    :param operation: str operation e.g. 'measurement/roi2_sum /Transmission'
    :param namespace_dict: dict of additional values to add to eval namespace
    :param associated_names: dictionary of associated names. Items should be keys in namespace_dict
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return operation: str updated operation string with addresses converted to names
    :return result: result of evaluation
    """
    operation, namespace_dict, ndict = names_operation(hdf_group, operation, namespace_dict, associated_names, address_list)
    result = eval(operation, globals(), namespace_dict)
    return operation, result


def string_command(hdf_group, operation, namespace_dict=None, associated_names=None, address_list=None):
    """
    Generate string command including values from hdf file using values inside {} brackets
    e.g.
      operation = 'the energy is {energy} keV'
      out = string_operation(hdf, operation)
      # energy is found within hdf tree
      out = 'the energy is 3.00 keV'
    :param hdf_group: hdf5 File or Group object
    :param operation: str operation e.g. 'measurement/roi2_sum /Transmission'
    :param namespace_dict: dict of additional values to add to eval namespace
    :param associated_names: dictionary of associated names. Items should be keys in namespace_dict
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: str, dict, dict
    """
    # get values inside brackets
    ops = re.findall(r'\{(.+?)\}', operation)
    if namespace_dict is None:
        namespace_dict = {}
    for op in ops:
        op = op.split(':')[0]  # remove format specifier
        gen_op, namespace_dict, associated_names = names_operation(hdf_group, op, namespace_dict, associated_names, address_list)
        namespace_dict[gen_op] = eval(gen_op, globals(), namespace_dict)
        operation = operation.replace(op, gen_op)
    return operation, namespace_dict, associated_names


def string_operation(hdf_group, operation, namespace_dict=None, associated_names=None, address_list=None):
    """
    Generate string including values from hdf file using values inside {} brackets
    e.g.
      operation = 'the energy is {energy} keV'
      out = string_operation(hdf, operation)
      # energy is found within hdf tree
      out = 'the energy is 3.00 keV'
    :param hdf_group: hdf5 File or Group object
    :param operation: str operation e.g. 'measurement/roi2_sum /Transmission'
    :param namespace_dict: dict of additional values to add to eval namespace
    :param associated_names: dictionary of associated names. Items should be keys in namespace_dict
    :param address_list: list of str of dataset addresses (None to generate from hdf_group)
    :return: str
    """
    operation, out_dict, name_dict = string_command(hdf_group, operation, namespace_dict, associated_names, address_list)
    return operation.format(**out_dict)


"------------------------------------ IMAGE FUNCTIONS  ----------------------------------------------------"


def find_image(hdf_group, address_list=None, multiple=False):
    """
    Return address of image data in hdf file
    Images can be stored as list of file directories when using tif file,
    or as a dynamic hdf link to a hdf file.

    :param hdf_group: hdf5 File or Group object
    :param address_list: list of str: list of str to search in
    :param multiple: if True, return list of all addresses matching criteria
    :return: str or list of str
    """
    if address_list is None:
        address_list = dataset_addresses(hdf_group)
    all_addresses = []
    # First look for 2D image data
    for address in address_list:
        data = hdf_group.get(address)
        if not data or data.size == 1: continue
        if len(data.shape) > 1 and 'signal' in data.attrs:
            if multiple:
                all_addresses += [address]
            else:
                return address
    # Second look for image files
    for address in address_list:
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
    :param index: None or int : return a specific image, None uses the half way index
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :return: 2d array
    """
    if address is None:
        address = find_image(hdf_group)
    dataset = hdf_group.get(address)
    # if array - return array
    if len(dataset.shape) > 1:
        # array data
        if index is None:
            index = len(dataset) // 2
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
            index = len(filenames) // 2
        return imread(filenames[index])


def volume_data(hdf_group, address=None):
    """
    Return image volume, if available
    if index=None, all images are combined, otherwise only a single frame at index is returned
    :param hdf_group: hdf5 File or Group object
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :return: 3d array [npoints, pixel_i, pixel_j]
    """
    if address is None:
        address = find_image(hdf_group)
    dataset = hdf_group.get(address)

    # if array - return array
    if len(dataset.shape) > 1:
        # array data
        return np.array(dataset)

    # if file - load file with imread, return array
    else:
        filepath = os.path.dirname(hdf_group.file.filename)
        try:
            filenames = [os.path.join(filepath, file) for file in dataset]
        except TypeError:
            # image_data filename is loaded in bytes format
            filenames = [os.path.join(filepath, file.decode()) for file in dataset]
        # stitch images into volume
        image = imread(filenames[0])
        volume = np.zeros([len(filenames), image.shape[0], image.shape[1]])
        for n, filename in enumerate(filenames):
            volume[n] = imread(filename)
        return volume


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


def image_roi_op(hdf_group, operation, address=None):
    """
    Create new region of interest (roi) from image data and return sum and maxval
    The roi centre and size is defined by an operation:
      operation = 'nroi[210, 97, 75, 61]'
      'nroi'      -   creates a region of interest in the detector centre with size 31x31
      'nroi[h,v]' -   creates a roi in the detector centre with size hxv, where h is horizontal, v is vertical
      'nroi[m,n,h,v] - create a roi with cen_h, cen_v, wid_h, wid_v = n, m, h, v
    :param hdf_group: hdf5 File or Group object
    :param address: None or str : if not None, pointer to location of image data in hdf5
    :param operation: str : operation string
    :param cen_h: int centre of roi in dimension m (horizontal)
    :param cen_v: int centre of roi in dimension n (vertical)
    :param wid_h: int full width of roi in diemnsion m (horizontal)
    :param wid_v: int full width of roi in dimension n (vertical)
    :return: sum, maxval : [o] length arrays
    """
    vals = [int(val) for val in re.findall(r'\d+', operation)]
    nvals = len(vals)
    if nvals == 4:
        cen_h, cen_v, wid_h, wid_v = vals
    elif nvals == 2:
        cen_h, cen_v = None, None
        wid_h, wid_v = vals
    else:
        cen_h, cen_v = None, None
        wid_h, wid_v = 31, 31
    return image_roi_sum(hdf_group, address, cen_h, cen_v, wid_h, wid_v)


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
