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
import matplotlib.pyplot as plt

from . import functions_nexus as fn


"----------------------------------------------------------------------------------------------------------------------"
"-------------------------------------------- NexusLoader -------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class NexusLoader:
    """
    NexusLoader class
    Collection of functions to operate on a hdf or nexus file
    Doesn't load the file into memory at any point but reads the file on every operation, making it very simple and
    reliable but quite slow.
    Useage:
      d = NexusLoader('12345.nxs')
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
    def __init__(self, filename, address_list=None, updatemode=False):
        self.filename = filename
        self.basename = os.path.basename(filename)

        if address_list is None:
            self._address_list = self.addresses()
        else:
            self._address_list = address_list

        self._namespace = {}
        self._name_associations = {}
        self._updatemode = updatemode
        self._axes_address = None
        self._signal_address = None
        self._image_address = None
        self.fit_results = {}
        self._lmfit = None

    def __repr__(self):
        return 'NexusLoader(\'%s\')' % self.filename

    def __str__(self):
        out = '%s\n' % (fn.OUTPUT_FORMAT % ('filename', self.basename))
        axes, signal = self.auto_xyaxis()
        out += '%s\n' % (fn.OUTPUT_FORMAT % ('axes', axes))
        out += '%s\n' % (fn.OUTPUT_FORMAT % ('signal', signal))
        top_addresses = self.addresses(recursion_limit=2)
        s = '\n'.join('%20s : {%s}' % (fn.address_name(a), a) for a in top_addresses)
        out += self.string_operation(s)
        return out

    def __call__(self, *args, **kwargs):
        return self.eval_operation(*args, **kwargs)

    def __getitem__(self, item):
        hdf = fn.load(self.filename)
        group = hdf.get(item)
        if group:
            return group
        datasets = fn.get_datasets(hdf, item)
        if len(datasets) == 1:
            return datasets[0]
        return datasets

    def __add__(self, addee):
        """
        Add two scans together somehow
        """
        return NexusMultiLoader([self, addee])

    def load(self):
        """Load hdf or nxs file, return open hdf object"""
        return fn.load(self.filename)

    def dataset(self, address):
        """Return dataset from a hdf file at given address or search using a name, this leaves the file open."""
        hdf = fn.load(self.filename)
        datasets = fn.get_datasets(hdf, address, self._address_list)
        if len(datasets) == 1:
            return datasets[0]
        return datasets
    get = __getitem__

    def address(self, name):
        """Return address of dataset called name"""
        with fn.load(self.filename) as hdf:
            data = fn.get_address(hdf, name, self._address_list)
        return data

    def group_address(self, name):
        """Return address of hdf group using name"""
        with fn.load(self.filename) as hdf:
            data = fn.get_group_address(hdf, name, self._address_list)
        return data

    def addresses(self, address='/', recursion_limit=100, get_size=None, get_ndim=None):
        """
        Return list of addresses of datasets, starting at each group address
        :param address: list of str or str : start in this / these addresses of hdf groups
        :param recursion_limit: Limit on recursivley checking lower groups
        :param get_size: None or int, if int, return only datasets with matching size
        :param get_ndim: None or int, if int, return only datasets with matching ndim
        :return: list of str
        """
        with fn.load(self.filename) as hdf_group:
            out = fn.dataset_addresses(hdf_group, address, recursion_limit, get_size, get_ndim)
        return out

    # Change all these?
    def data(self, addresses):
        """Return data from a hdf file at given dataset address"""
        with fn.load(self.filename) as hdf:
            data = fn.get_data(hdf, addresses, self._address_list)
        return data

    def value(self, addresses):
        """Return single value data from a hdf file at given dataset address"""
        with fn.load(self.filename) as hdf:
            data = fn.get_value(hdf, addresses)
        return data

    def group_data(self, addresses):
        """Return data from a hdf file, expands group addresses to group constituents"""
        with fn.load(self.filename) as hdf:
            data = fn.get_group_data(hdf, addresses)
        return data

    def group_values(self, addresses):
        """Return dict of values from a hdf file, expands group addresses to group constituents"""
        with fn.load(self.filename) as hdf:
            data = fn.get_group_values(hdf, addresses)
        return data

    def group_string(self, addresses):
        """Return dict of values from a hdf file, expands group addresses to group constituents"""
        with fn.load(self.filename) as hdf:
            data = fn.get_group_string(hdf, addresses)
        return '\n'.join(data)

    def array_data(self, addresses, data_length=None):
        """
        Rerturn array of data from hdf file
        expands group addresses and returns array of same length arrays
        addresses with array length 1 are expanded to the shortest length of other arrays or data_length if defined
        """
        with fn.load(self.filename) as hdf:
            data = fn.get_group_array(hdf, addresses, data_length)
        return data

    def dataframe(self, addresses):
        """
        Rerturn Pandas DataFrame of data from hdf file
        expands group addresses and returns array of same length arrays
        addresses with array length 1 are expanded to the shortest length of other arrays or data_length if defined
        """
        with fn.load(self.filename) as hdf:
            data = fn.get_dataframe(hdf, addresses)
        return data

    def data_dict(self, addresses):
        """Loads data from each dataset and ands to a dict"""
        with fn.load(self.filename) as hdf:
            data = fn.get_data_dict(hdf, addresses)
        return data

    def datetime(self, address, input_format=None, output_format=None):
        """
        Read time stamps from hdf file at specific address
        If input is a string (or bytes), input_format is used to parse the string
        If input is a float, it is assumed to be a timestamp from the Unix Epoch (1970-01-01 00:00:00)

        Useful Format Specifiers (https://strftime.org/):
        %Y year         %m month      %d day      %H hours    %M minutes  %S seconds  %f microseconds
        %y year (short) %b month name %a day name %I 12-hour  %p AM or PM %z UTC offset

        :param address: str hdf dataset address
        :param input_format: str datetime.strptime format specifier to parse dataset
        :param output_format: str datetime.strftime format specifier to generate output string (if None, returns datetime)
        :return: datetime.datetime
        """
        with fn.load(self.filename) as hdf:
            datasets = fn.get_datasets(hdf, address, self._address_list)
            date = [fn.dataset_datetime(dataset, input_format, output_format) for dataset in datasets]
        if len(date) == 1:
            return date[0]
        return date

    "------------------------- String Generators -------------------------------------------"

    def tree(self, address='/', detail=False, recursion_limit=100):
        """Open hdf file and return tree string"""
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            out = fn.tree(hdf_group, detail, recursion_limit)
        return out

    "----------------------------- Search methods -------------------------------------------"

    def find_name(self, name, match_case=False, whole_word=False):
        """
        Find datasets using field name
        :param name: str : name to match in dataset field name
        :param match_case: if True, match case of name
        :param whole_word: if True, only return whole word matches
        :return: list of str addresses
        """
        results = fn.find_name(name, self._address_list, match_case, whole_word)
        if len(results) == 0:
            print('updating address list')
            self._address_list = self.addresses()
            results = fn.find_name(name, self._address_list, match_case, whole_word)
        return results

    def find_nxclass(self, nxclass='NX_detector', address='/'):
        """
        Returns location of hdf group with attribute ['NX_class']== nxclass
        :param nxclass: str name of class attribute
        :param address: str address to start in
        :return: str address
        """
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            out = fn.find_nxclass(hdf_group, nxclass)
        return out

    def find_attr(self, attr='axes', address='/'):
        """
        Returns location of hdf attribute
        Workds recursively - starts at the top level and searches all lower hdf groups
        :param attr: str : attribute name to search for
        :param address: str address to start in
        :return: str hdf address
        """
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            out = fn.find_attr(hdf_group, attr)
        return out

    def find_image(self, multiple=False):
        """
        Return address of image data in hdf file
        Images can be stored as list of file directories when using tif file,
        or as a dynamic hdf link to a hdf file.

        :param filename: str hdf file
        :param multiple: if True, return list of all addresses matching criteria
        :return: str or list of str
        """
        with fn.load(self.filename) as hdf:
            out = fn.find_image(hdf, self._address_list, multiple)
        return out

    "------------------------- Automatic Axes -------------------------------------------"

    def auto_xyaxis(self, address='/', cmd_string=None):
        """
        Find default axes, signal hdf addresses
        :param address: str addtress to start in
        :param cmd_string: str of command to take x,y axis from as backup
        :return: xaxis_address, yaxis_address
        """
        if self._axes_address and self._signal_address:
            return self._axes_address, self._signal_address
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            xaxis, yaxis = fn.auto_xyaxis(hdf_group, cmd_string, self._address_list)
        self._axes_address = xaxis
        self._signal_address = yaxis
        xname = fn.address_name(xaxis)
        yname = fn.address_name(yaxis)
        axis = {
            'axes': xname,
            'signal': yname,
            'xaxis': xname,
            'yaxis': yname,
            xaxis: xname,
            yaxis: yname,
        }
        self.add_to_associations(**axis)
        return xaxis, yaxis

    def axes(self):
        """Return axes (xaxis) label and data"""
        address, _ = self.auto_xyaxis()
        if not address:
            return 'None', None
        name = fn.address_name(address)
        data = self.data(address)
        return name, data
    xaxis = axes

    def signal(self):
        """Return signal (yaxis) label and data"""
        _, address = self.auto_xyaxis()
        if not address:
            return 'None', None
        name = fn.address_name(address)
        data = self.data(address)
        return name, data
    yaxis = signal

    def image_address(self):
        """Return address of image data"""
        if self._image_address is not None:
            return self._image_address
        address = self.find_image()
        if address:
            self._image_address = address
        else:
            raise Exception('Image address not found')
        return address

    "---------------------------- Operations -------------------------------------------"

    def get_namespace(self):
        """Returns the current operation namespace and name association dicts"""
        if self._updatemode:
            namespace = {}
        else:
            namespace = self._namespace
        associations = self._name_associations
        namespace.update(self.fit_results)
        return namespace, associations

    def update_namespace(self, updatemode=True):
        """Turns on updatemode, forcing new searches on each operation"""
        self._updatemode = updatemode
        if updatemode:
            self._namespace = {}
            self._address_list = self.addresses()

    def add_to_namespace(self, **kwargs):
        """
        Add values to the NexusLoader namespace
             self.add_to_namespace(cmd='scan x 1 2 1 pil 1')
        """
        self._namespace.update(kwargs)

    def add_to_associations(self, **kwargs):
        """
        Add address to the NexusLoader names associations
             self.add_to_associations(signal='/entry1/measurement/sum')
        """
        self._name_associations.update(kwargs)

    def show_namespace(self):
        """Return str of namespace etc"""
        out = '\nAddress list:\n  '
        out += '\n  '.join(self._address_list)
        out += '\nnamespace:\n  '
        out += '\n  '.join('%s : %s' % (nm, type(dt)) for nm, dt in self._namespace.items())
        out += '\nname associations:\n  '
        out += '\n  '.join('%s : %s' % (n1, n2) for n1, n2 in self._name_associations.items())
        out += '\n\n  axes = %s' % self._axes_address
        out += '\nsignal = %s' % self._signal_address
        out += '\n image = %s' % self._image_address
        return out

    def names_operation(self, operation, address='/'):
        """
        Interpret a string as a series of hdf addresses or dataset names, returning a evuatable string and dict of data.
          operation, data = self.names_operation(operation, hdf_address)
        Example:
            operation, data = names_operation('measurement/roi2_sum /Transmission')
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

        :param operation: str operation e.g. 'measurement/roi2_sum /Transmission'
        :param address: str address to start in
        :return operation: str updated operation string with addresses converted to names
        :return data: dict of names and data
        """
        namespace, associated_names = self.get_namespace()
        # Check operation isn't already in namespace
        if operation in namespace:
            return operation, namespace
        if operation in associated_names and associated_names[operation] in namespace:
            return associated_names[operation], namespace

        # Load file
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            operation, data_dict, ndict = fn.names_operation(
                hdf_group,
                operation,
                namespace,
                associated_names,
                self._address_list
            )
            self._namespace.update(data_dict)
            self._name_associations.update(ndict)
        return operation, data_dict

    def eval_operation(self, operation, address='/'):
        """Evaluate a string, names and addresses from the hdf file can be used"""
        operation, data = self.names_operation(operation, address)
        return eval(operation, globals(), data)

    def get_operation(self, operation, address='/'):
        """Return corrected operation string and evaluation result"""
        operation, data = self.names_operation(operation, address)
        result = eval(operation, globals(), data)
        if result is None:
            raise Exception('%s not available' % operation)
        return operation, result

    def string_operation(self, operation, address='/', shorten=True):
        """
        Return string with values from file
        e.g.
        operation = 'the title is {title}'
        out = self.string_operation(operation)
        out: 'the title is scan eta ...'
        :param operation: str with nexus dataset names in {brackets}
        :param address: address of hdf group to start in
        :param shorten: bool if True, automatically shorten long floats
        :return: str
        """
        namespace, associated_names = self.get_namespace()
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            out_str, data_dict, name_dict = fn.string_command(
                hdf_group, operation, namespace, associated_names, self._address_list)
            self._namespace.update(data_dict)
            self._name_associations.update(name_dict)
        if shorten:
            return fn.shortstr(out_str.format(**data_dict))
        return out_str.format(**data_dict)

    "------------------------- Detector Images -------------------------------------------"

    def image_data(self, index=None, address=None):
        """
        Return image data, if available
        if index=None, all images are combined, otherwise only a single frame at index is returned
        :param filename: str hdf file
        :param index: None or int : return a specific image
        :param address: None or str : if not None, pointer to location of image data in hdf5
        :return: 2d array if index given, 3d array otherwise
        """
        if address is None:
            address = self.image_address()
        with fn.load(self.filename) as hdf:
            out = fn.image_data(hdf, index, address)
        return out

    def volume_data(self, address=None):
        """
        Return image volume, if available
        if index=None, all images are combined, otherwise only a single frame at index is returned
        :param hdf_group: hdf5 File or Group object
        :param address: None or str : if not None, pointer to location of image data in hdf5
        :return: 3d array [npoints, pixel_i, pixel_j]
        """
        if address is None:
            address = self.image_address()
        with fn.load(self.filename) as hdf:
            out = fn.volume_data(hdf, address)
        return out

    def image_roi(self, address=None, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
        """
        Create new region of interest (roi) from image data, return roi volume
        :param address: None or str : if not None, pointer to location of image data in hdf5
        :param cen_h: int centre of roi in dimension m (horizontal)
        :param cen_v: int centre of roi in dimension n (vertical)
        :param wid_h: int full width of roi in diemnsion m (horizontal)
        :param wid_v: int full width of roi in dimension n (vertical)
        :return: [n, wid_v, wid_h] array of roi
        """
        if address is None:
            address = self.image_address()
        with fn.load(self.filename) as hdf:
            roi = fn.image_roi(hdf, address, cen_h, cen_v, wid_h, wid_v)
        return roi

    def image_roi_sum(self, address=None, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
        """
        Create new region of interest (roi) from image data and return sum and maxval
        :param address: None or str : if not None, pointer to location of image data in hdf5
        :param cen_h: int centre of roi in dimension m (horizontal)
        :param cen_v: int centre of roi in dimension n (vertical)
        :param wid_h: int full width of roi in diemnsion m (horizontal)
        :param wid_v: int full width of roi in dimension n (vertical)
        :return: sum, maxval : [o] length arrays
        """
        if address is None:
            address = self.image_address()
        with fn.load(self.filename) as hdf:
            roi_sum, roi_max = fn.image_roi_sum(hdf, address, cen_h, cen_v, wid_h, wid_v)
        return roi_sum, roi_max


"----------------------------------------------------------------------------------------------------------------------"
"--------------------------------------- NexusMultiLoader -------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class NexusMultiLoader:
    """
    NexusMultiLoader - container for multiple NexusLoader objects
    e.g.
    d1 = NexusLoader("file1.nxs")
    d2 = NexusLoader("file2.nxs")
    group = NexusMultiLoader([d1, d2])

    The same can be achieved by adding NexusLoader objects:
    group = d1 + d2

    Actions:
     group('command') >> returns list of NexusLoader('command') results for each object in the group
     group[n] >> returns NexusLoader n from list

    Additional parameters (can also be set by self.param(value)):
    :param loaders: list of NexusLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders
        self.filenames = [loader.filename for loader in self.loaders]
        self._first = loaders[0]

    def __repr__(self):
        return "NexusMultiLoader(%d files)" % len(self.loaders)

    def __str__(self):
        return '\n'.join(['%r' % loader for loader in self.loaders])

    def __call__(self, name):
        return [loader(name) for loader in self.loaders]

    def __add__(self, addee):
        return NexusMultiLoader(self.loaders + [addee])

    def __getitem__(self, key):
        #return [loader[key] for loader in self.loaders]
        return self.loaders.__getitem__(key)

    def data(self, name):
        """Return list of data for each stored NexusLoader"""
        return [loader.data(name) for loader in self.loaders]

    def value(self, name):
        """Return list of single values for each stored NexusLoader"""
        return [loader.value(name) for loader in self.loaders]

    def data_dict(self, name, label='filename'):
        """Return dict of data for each scan, each entry is named loader(label)"""
        return {loader(label): loader.data(name) for loader in self.loaders}

    def dataframe(self, name, label='filename'):
        """Return Pandas DataFrame of data in each loader"""
        data = self.data_dict(name, label)
        return fn.DataFrame(data)

    def new_roi(self, cen_h=None, cen_v=None, wid_h=31, wid_v=31, name=None):
        """Create new roi and add to the namespace"""

        if name is None:
            name = 'roi'

        if cen_h is None:
            cen_h = self._first.image_size()[1] // 2
        if cen_v is None:
            cen_v = self._first.image_size()[0] // 2

        for loader in self.loaders:
            loader.new_roi(cen_h, cen_v, wid_h, wid_v, name)

