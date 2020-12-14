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
from .fitting import peakfit


"----------------------------------- HdfLoader Class -----------------------------------"


class NexusReLoader:
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

        self.title_command = fn.TITLE_COMMAND
        self.label_command = fn.LABEL_COMMAND
        self.fit_results = {}
        self._lmfit = None

    def __repr__(self):
        return 'NexusReLoader(\'%s\')' % self.filename

    def __str__(self):
        out = '%s\n' % (fn.OUTPUT_FORMAT % ('filename', self.basename))
        axes, signal = self.auto_xyaxis()
        out += '%s\n' % (fn.OUTPUT_FORMAT % ('axes', axes))
        out += '%s\n' % (fn.OUTPUT_FORMAT % ('signal', signal))
        top_addresses = self.addresses(recursion_limit=2)
        out += '\n'.join(self.data_string(top_addresses))
        return out

    def __call__(self, *args, **kwargs):
        return self.eval_operation(*args, **kwargs)

    def __getitem__(self, item):
        return self.dataset(item)

    def __add__(self, addee):
        """
        Add two scans together somehow
        """
        return NexusMultiReLoader([self, addee])

    def load(self):
        """Load hdf or nxs file"""
        return fn.load(self.filename)

    def dataset(self, address):
        """Return dataset from a hdf file at given address, this leaves the file open."""
        with fn.load(self.filename) as hdf:
            dataset = fn.get_datasets(hdf, address)
            return dataset
    get = dataset

    def addresses(self, address='/', recursion_limit=100):
        """
        Return list of addresses of datasets, starting at each address
        :param address: list of str or str : start in this / these addresses
        :param recursion_limit: Limit on recursivley checking lower groups
        :return: list of str
        """
        with fn.load(self.filename) as hdf_group:
            out = fn.dataset_addresses(hdf_group, address, recursion_limit)
        return out

    def data(self, addresses):
        """Return data from a hdf file at given address"""
        with fn.load(self.filename) as hdf:
            data = fn.get_data(hdf, addresses)
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
            datasets = fn.get_datasets(hdf, address)
            date = [fn.dataset_datetime(dataset, input_format, output_format) for dataset in datasets]
        if len(date) == 1:
            return date[0]
        return date

    "------------------------- String Generators -------------------------------------------"

    def data_string(self, addresses, output_format=None):
        """
        Return strings of data using output_format
        :param filename: str hdf fileaname
        :param addresses: list of str or str hdf dataset addresses
        :param output_format: str
        :return: str
        """
        with fn.load(self.filename) as hdf:
            out_str = fn.data_strings(hdf, addresses, output_format)
        return out_str

    def tree(self, address='/', detail=False, recursion_limit=100):
        """Open hdf file and return tree string"""
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            out = fn.tree(hdf_group, detail, recursion_limit)
        return out

    "----------------------------- Search methods -------------------------------------------"

    def find_name(self, name, address='/', match_case=False, whole_word=False):
        """
        Find datasets using field name
        :param name: str : name to match in dataset field name
        :param address: str : address to start in
        :param match_case: if True, match case of name
        :param whole_word: if True, only return whole word matches
        :return: list of str addresses
        """
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            out = fn.find_name(hdf_group, name, match_case, whole_word)
        return out

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

    def find_image(self, address='/', multiple=False):
        """
        Return address of image data in hdf file
        Images can be stored as list of file directories when using tif file,
        or as a dynamic hdf link to a hdf file.

        :param filename: str hdf file
        :param address: initial hdf address to look for image data
        :param multiple: if True, return list of all addresses matching criteria
        :return: str or list of str
        """
        with fn.load(self.filename) as hdf:
            out = fn.find_image(hdf, address, multiple)
        return out

    "------------------------- Automatic Axes -------------------------------------------"

    def auto_xyaxis(self, address='/'):
        """
        Find default axes, signal hdf addresses
        :param address: str addtress to start in
        :return: xaxis_address, yaxis_address
        """
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            xaxis, yaxis = fn.auto_xyaxis(hdf_group)
        return xaxis, yaxis

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

    def duration(self, start_address='start_time', end_address='end_time'):
        """
        Determine the duration of a scan using the start_time and end_time datasets
        :param start_address: address or name of start time dataset
        :param end_address: address or name of end time dataset
        :return: datetime.timedelta
        """
        with fn.load(self.filename) as hdf:
            timedelta = fn.duration(hdf, start_address, end_address)
        return timedelta

    def scan_number(self):
        with fn.load(self.filename) as hdf:
            scanno = fn.scannumber(hdf)
        return scanno

    def title(self, title_command=None):
        """
        Generate scan title
        :param title_command:
        :return: str
        """
        if title_command is None:
            title_command = self.title_command
        return self.string_operation(title_command)

    def label(self, label_command=None):
        """
        Generate scan title
        :param label_command: str command e.g. '#{scanno}'
        :return: str
        """
        if label_command is None:
            label_command = self.label_command
        return self.string_operation(label_command)

    "---------------------------- Operations -------------------------------------------"

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
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            operation, data_dict = fn.names_operation(hdf_group, operation)
        return operation, data_dict

    def eval_operation(self, operation, address='/', namespace_dict=None):
        """Evaluate a string, names and addresses from the hdf file can be used"""
        operation, data = self.names_operation(operation, address)
        if namespace_dict is None:
            namespace_dict = {}
        namespace_dict.update(self.fit_results)
        namespace_dict.update(data)
        return eval(operation, globals(), namespace_dict)

    def get_operation(self, operation, address='/', namespace_dict=None):
        """Return corrected operation string and evaluation result"""
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            operation, result = fn.value_operation(hdf_group, operation, namespace_dict)
        return operation, result

    def string_operation(self, operation, address='/', namespace_dict=None):
        """Return formated string with values from file"""
        with fn.load(self.filename) as hdf:
            hdf_group = hdf.get(address)
            out_str = fn.string_operation(hdf_group, operation, namespace_dict)
        return out_str

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
        with fn.load(self.filename) as hdf:
            out = fn.image_data(hdf, index, address)
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
        with fn.load(self.filename) as hdf:
            roi_sum, roi_max = fn.image_roi_sum(hdf, address, cen_h, cen_v, wid_h, wid_v)
        return roi_sum, roi_max

    "------------------------------- Plot -------------------------------------------"

    def plot_line(self, axes, xaxis='axes', yaxis='signal', *args, label=None, **kwargs):
        """
        Plot line on given matplotlib axes subplot
        :param axes: matplotlib.axes subplot
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param label: str label for this line
        :param args: given directly to plt.plot(..., *args, **kwars)
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: lines object, output of plot
        """
        if axes is None:
            axes = plt.gca()
        xname, xdata = self.get_operation(xaxis)
        yname, ydata = self.get_operation(yaxis)

        if label is None:
            label = yname

        #if self._plot_errors:
        #    yerror = self._error(yaxis)
        #    lines = axes.errorbar(xdata, ydata, yerror, *args, label=label, **kwargs)
        #else:
        #    lines = axes.plot(xdata, ydata, *args, label=label, **kwargs)
        lines = axes.plot(xdata, ydata, *args, label=label, **kwargs)
        return lines

    def plot(self, xaxis='axes', yaxis='signal', *args, **kwargs):
        """
        Create new matplotlib figure and plot arrays
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param args: given directly to plt.plot(..., *args, **kwars)
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: lines object, output of plot
        """
        xname, xdata = self.get_operation(xaxis)
        list_yaxis = np.asarray(yaxis, dtype=str).reshape(-1)
        yname, ydata = self.get_operation(list_yaxis[0])

        fig, ax = plt.subplots(figsize=fn.FIG_SIZE, dpi=fn.FIG_DPI)
        lines = []
        for yaxis in list_yaxis:
            lines += self.plot_line(ax, xaxis, yaxis, *args, **kwargs)

        plt.title(self.title())
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.legend(list_yaxis)
        return lines

    def plot_image(self, index, image_address=None):
        """Plot detector image"""

        fig, ax = plt.subplots(figsize=fn.FIG_SIZE, dpi=fn.FIG_DPI)
        im = self.image_data(index, image_address)
        ax.pcolormesh(im, shading='gouraud')
        ax.invert_yaxis()

        plt.axis('image')
        plt.title(self.title())

    "------------------------------- fitting -------------------------------------------"

    def fit(self, xaxis='axes', yaxis='signal', yerrors=None, fit_type=None, print_result=True, plot_result=False):
        """
        Automatic fitting of scan

        Use LMFit
        Pass fit_type = LMFit model
        return LMFit output
        """

        xname, xdata = self.get_operation(xaxis)
        yname, ydata = self.get_operation(yaxis)
        if yerrors:
            error_name, error_data = self.get_operation(yerrors)


        # lmfit
        out = peakfit(xdata, ydata)

        self._lmfit = out
        fit_dict = {}
        for pname, param in out.params.items():
            ename = 'stderr_' + pname
            fit_dict[pname] = param.value
            fit_dict[ename] = param.stderr
        self.fit_results.update(fit_dict)

        if print_result:
            print(self.title())
            print(out.fit_report())
        if plot_result:
            fig, grid = out.plot()
            plt.suptitle(self.title(), fontsize=12)
            plt.subplots_adjust(top=0.85, left=0.15)
            ax1, ax2 = fig.axes
            ax2.set_xlabel(xname)
            ax2.set_ylabel(yname)
        return out

    def fit_result(self, parameter_name=None):
        """Returns parameter, error from the last run fit"""
        if self._lmfit is None:
            self.fit()
        if parameter_name is None:
            return self._lmfit
        param = self._lmfit.params[parameter_name]
        return param.value, param.stderr


class NexusMultiReLoader:
    """
    NexusMultiReLoader
    """
    def __init__(self, loaders):
        self.loaders = loaders
        self.filenames = [loader.filename for loader in self.loaders]
        self.scan_range = [loader.scan_number() for loader in self.loaders]
        self._first = loaders[0]
        self._variables = None

    def __repr__(self):
        return "NexusMultiReLoader(%d files)" % len(self.loaders)

    def __str__(self):
        return '\n'.join(['%r' % loader for loader in self.loaders])

    def __call__(self, name):
        return [loader(name) for loader in self.loaders]

    def __add__(self, addee):
        return NexusMultiReLoader(self.loaders + [addee])

    def __getitem__(self, key):
        return [loader[key] for loader in self.loaders]

    def set_variables(self, name):
        """Set default variable that changes with each scan"""
        self._variables = np.asarray(name, dtype=str).reshape(-1)

    def create_array(self, name):
        """Return numerical array from values"""
        return np.array([np.asarray(val) for val in self.__call__(name)])

    def create_matrix(self, scannable):
        """Return nxm matrix from scannables, where n is the number of scans and m is the shortest scan length"""
        # Get values
        values = self.__call__(scannable)
        # Determine shortest non-single array
        lenval = np.min([np.asarray(val).size for val in values if np.asarray(val).size > 1])
        # Create array
        return np.array([val[:lenval] for val in values if val.size > 1])

    def title(self):
        """Return plot title"""
        return '%s-%s' % (self.scan_range[0], self.scan_range[-1])

    def plot_line(self, axes=None, yaxis=None, xaxis=None, *args, **kwargs):
        """plot metadata"""
        if axes is None:
            axes = plt.gca()
        if xaxis is None and self._variables is None:
            xaxis = self.scan_range
        elif xaxis is None:
            xaxis = self._variables[0]
        if yaxis is None:
            yaxis = 'np.sum(signal)'

        xarray = self.create_array(xaxis)
        yarray = self.create_array(yaxis)
        lines = axes.plot(xarray, yarray, *args, **kwargs)
        return lines

    def plot_lines(self, axes=None, xaxis='axes', yaxis='signal', labels=None, *args, **kwargs):
        """Plot line on given axis"""
        if axes is None:
            axis = plt.gca()
        lines = []
        for loader in self.loaders:
            lines += loader.plot_line(axes, xaxis, yaxis, *args, label=labels, **kwargs)
        return lines

    def plot(self, xaxis='axes', yaxis='signal', labels=None, *args, **kwargs):
        """Create plot of scan data"""

        fig, ax = plt.subplots(figsize=fn.FIG_SIZE, dpi=fn.FIG_DPI)
        lines = self.plot_lines(ax, xaxis, yaxis, labels, *args, **kwargs)

        plt.title(self.title())
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.legend()

    def plot_value(self, yaxis=None, xaxis=None, *args, **kwargs):
        """Create plot of metadata"""

        if xaxis is None and self._variables is None:
            xaxis = 'scanno'
        elif xaxis is None:
            xaxis = self._variables[0]
        if yaxis is None:
            yaxis = 'np.sum(signal)'

        xarray = self.create_array(xaxis)
        yarray = self.create_array(yaxis)

        fig, ax = plt.subplots(figsize=fn.FIG_SIZE, dpi=fn.FIG_DPI)
        lines = ax.plot(xarray, yarray, *args, **kwargs)

        plt.title(self.title())
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)

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

    def fit(self, xaxis='axes', yaxis='signal', yerrors=None, fit_type=None, print_result=True, plot_result=False):
        """
        Automatic fitting of scans

        Use LMFit
        Pass fit_type = LMFit model
        return LMFit output
        """

        fit_result = []
        for loader in self.loaders:
            fit_result += [loader.fit(xaxis, yaxis, yerrors, fit_type, print_result, plot_result)]
        return fit_result

    def fit_result(self, parameter_name=None):
        """Returns parameter, error from the last run fit"""

        if parameter_name is None:
            return [loader.fit_result() for loader in self.loaders]

        values = np.zeros(len(self.loaders))
        errors = np.zeros(len(self.loaders))
        for n, scan in enumerate(self.loaders):
            values[n], errors[n] = scan.fit_result(parameter_name)
        return values, errors
