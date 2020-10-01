"""
Define the Scan object
"""

import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pandas import DataFrame

from .functions_general import array_str
from .functions_nexus import scanfile2number, dataset_addresses, Hdf5Nexus
from .nexus_decorator import DatasetPlus
from .fitting import peakfit


_namespace_modules = {
    'np': np,
    'sum': np.sum,
    'max': np.max,
    'mean': np.mean
}


class Scan(Hdf5Nexus):
    """
    Scan Class
    Loads a Nexus (.nxs) scan file and reads standard parameters into an internal namespace. Also contains useful
    automatic functions for images, regions of interest and fitting.
    Inherits Hdf5Nexus, h5py.File

    Usage:
      scan = Scan('/path/to/file/12345.nxs')
      # Functions
      xlabel, xarray = scan.xaxis() # return default x values, defined by 'axes' attr
      ylabel, yarray = scan.yaxis() # return default y values, defined by 'signal' attr
      scan.plot() # default plotting
      scan.fit() # plot with default fit
      print(scan) # scan information
      # namespace evaluation
      eta = scan('eta') # returns array eta (example) from /entry1/measurement/eta
      scan('np.max(roi2_sum)') # evaluates operations
    Inputs:
    :param filename: str : path to scan file
    :param string_address: str or list of str : addresses of string parameters to add to namespace
    :param array_address: str or list of str : addresses of array parameters to add to namespace
    :param value_address: str or list of str : addresses of string parameters to add to namespace
    """

    _xaxis = None
    _xaddress = None
    _yaxis = None
    _yaddress = None
    _label_format = '%s: %s'
    _image_address = None
    _image_size = None

    _normalisation_names = ['sum', 'maxval', 'roi2_sum', 'roi1_sum']
    _normalisation_command = '%s/Transmission/count_time'
    _error_names = [('sum', 'roi2_sum', 'roi1_sum')]
    _error_functions = ['np.sqrt(%s+1)']
    _duplicate_namespace_names = {
        'en': 'energy', # old, new
        'Ta': 'temp',
    }

    _plot_errors = False
    _plot_normalise = True

    _fit_model = None
    _fit_guess = None
    _fit_result = None

    def __init__(self, filename,
                 string_address='/entry1',
                 array_address='/entry1/measurement',
                 value_address='/entry1/before_scan'):
        super().__init__(filename, 'r')  # load thef file as h5py.File type with Hdf5Nexus extensions

        self.scan_number = scanfile2number(filename)

        self._string_address_list = dataset_addresses(self, string_address, 1)
        self.strings = [os.path.basename(address) for address in self._string_address_list]
        self._string_items = {}
        for address, name in zip(self._string_address_list, self.strings):
            self._string_items[name] = DatasetPlus(self, address)

        self._array_address_list = dataset_addresses(self, array_address)
        self.arrays = [os.path.basename(address) for address in self._array_address_list]
        self._array_items = {}
        for address, name in zip(self._array_address_list, self.arrays):
            self._array_items[name] = DatasetPlus(self, address)

        self._value_address_list = dataset_addresses(self, value_address)
        self.values = [os.path.basename(address) for address in self._value_address_list]
        self._value_items = {}
        for address, name in zip(self._value_address_list, self.values):
            self._value_items[name] = DatasetPlus(self, address)

        # Create namespace
        self._namespace = _namespace_modules.copy()
        self._namespace.update(self._string_items)
        self._namespace.update(self._value_items)
        self._namespace.update(self._array_items)
        # duplicate namespace values (en==energy)
        for old_name, new_name in self._duplicate_namespace_names.items():
            if old_name in self._namespace.keys():
                self._namespace[new_name] = self._namespace[old_name]
        # update attributes
        self.__dict__.update(self._string_items)
        self.__dict__.update(self._array_items)

    def __call__(self, source):
        if self._xaxis is None:
            self.xaxis()
        if self._yaxis is None:
            self.yaxis()
        return eval(source, self._namespace)

    def getvalue(self, source):
        """Return the actual value, rather than dataset"""
        value = self.__call__(source)
        if ',' in source:
            value = tuple(val[()] for val in value)
        elif isinstance(value, h5py.Dataset):
            value = value[()]
        return value

    def __repr__(self):
        return "Scan('%s')" % self.filename

    def __str__(self):
        """Return string of scan information, with config attributes"""
        fmt = '%20s : %s\n'
        out_str = '=====Scan(%s)=====\n' % self.scan_number
        out_str += fmt % ('scan number', self.scan_number)
        out_str += fmt % ('filename', self.filename)
        if self._namespace:
            # initialise x,y axis
            xdata = self.xaxis()
            ydata = self.yaxis()
            out_str += fmt % ('x-axis (axes)', self._xaxis)
            out_str += fmt % ('y-axis (signal)', self._yaxis)
            out_str += fmt % ('array length', len(xdata))
        out_str += '-String values-\n'
        for name, value in self._string_items.items():
            out_str += fmt % (name, value)
        return out_str

    def info(self):
        """Return longer string with full config attributres"""
        fmt = '%20s : %s\n'
        out_str = self.__str__()
        out_str += '-Arrays-\n'
        for name, value in self._array_items.items():
            out_str += fmt % (name, array_str(value[:]))
        out_str += '-Values-\n'
        for name, value in self._value_items.items():
            out_str += fmt % (name, value)
        return out_str

    def info_line(self, name=None):
        """Return single line string with scan details"""
        out = '%d axes=%s, signal=%s' % (self.scan_number, self.xaxis().basename, self.yaxis().basename)
        if name is not None:
            out += ', %s=%s' % (name, self.getvalue(name))
        return out

    def __add__(self, addee):
        """
        Add two scans together somehow
        """
        return MultiScans([self, addee])

    def who(self):
        """
        Return a string detailing the occupation of the namespace
        :return: str
        """
        out = ''
        for item, value in self._namespace.items():
            out += '%30s : %r\n' % (item, value)
        return out

    def add2namespace(self, data_dict):
        """
        Add additional parameters to scan namespace
        :param data_dict: dict
        :return: None
        """
        self._namespace.update(data_dict)

    def xaxis(self, name=None, address=None):
        """
        Return the dataset of the default axes value *the independent variable to plot on the abscissa*
        :param name: None or str : name of array from namespace
        :param address: None or str : dataset address in hdf5 file
        :return: DatasetPlus
        """
        if name and name in self.arrays:
            self._xaxis = name
            self._xaddress = self._array_items[name].address
        elif address:
            self._xaddress = address
            self._xaxis = os.path.basename(self._xaddress)
        if self._xaxis is None:
            axes = self.nx_find_attr('axes')
            if not axes or len(axes) < 1:
                axes = [self.get(self._array_address_list[0])]
                print('Warning: attribute "axes" not found, defaulting to %s' % axes[0])
            self._xaddress = axes[0]
            self._xaxis = os.path.basename(self._xaddress)
        dataset = DatasetPlus(self, self._xaddress)
        # add to namespace
        data_dict = {'xaxis': dataset,
                     'axes': dataset}
        self.add2namespace(data_dict)
        return dataset

    def yaxis(self, name=None, address=None):
        """
        Return the address of the default signal value *the dependent variable to plot on the ordinate*
        :param name: None or str : name of array from namespace
        :param address: None or str : dataset address in hdf5 file
        :return: DatasetPlus
        """
        if name and name in self.arrays:
            self._yaxis = name
            self._yaddress = self._array_items[name].address
        elif address:
            self._yaddress = address
            self._yaxis = os.path.basename(self._yaddress)
        if self._yaxis is None:
            signal = self.nx_find_attr('signal')
            if not signal or len(signal) < 1:
                signal = [self.get(self._array_address_list[-1])]
                print('Warning: attribute "signal" not found, defaulting to %s' % signal[0])
            self._yaddress = signal[0]
            self._yaxis = os.path.basename(self._yaddress)
        dataset = DatasetPlus(self, self._yaddress)
        # add to namespace
        data_dict = {'yaxis': dataset,
                     'signal': dataset}
        self.add2namespace(data_dict)
        return dataset

    def dataframe(self):
        """Return a pandas dataframe of the array data in this scan"""
        return DataFrame(self._array_items)

    def _error(self, name='signal'):
        """Return error on name if required"""
        for names, cmd in zip(self._error_names, self._error_functions):
            if name in names:
                err = cmd % name
                if self._plot_normalise:
                    err = self._normalise_name(err)
                return self.__call__(err)
        return 0 * self.__call__(name)

    def _normalise(self, name='signal'):
        """Return normalised value"""
        return self.__call__(self._normalise_name(name))

    def _normalise_name(self, name):
        """Return normalised value"""
        if name in self._normalisation_names:
            return self._normalisation_command % name
        return name

    def scan_title(self):
        """
        Return str scan_title for plot
        :return: str
        """
        if 'title' in self._string_items:
            return '%r\n%s' % (self, self.__call__('title'))
        return self.__repr__()

    def label(self, name, output_format=None):
        """
        Return formatted string of stored value. First attempts retreival from metadata namepace, then general namespace,
        then attemps to get value from nexus
        """
        value = self.__call__(name)
        if output_format is None:
            output_format = self._label_format
        return output_format % (name, value)

    def image_data(self, index=None, image_address=None):
        """Returns image data as 2D array"""
        if image_address:
            self._image_address = image_address
        if self._image_address is None:
            self._image_address = self.nx_find_image()
        return self.nx_image_data(index, self._image_address)

    def image_size(self, image_address=None):
        """Return size of a single data image"""
        if self._image_size:
            return self._image_size
        if image_address:
            self._image_address = image_address
        elif self._image_address is None:
            self._image_address = self.nx_find_image()
        self._image_size = self.image_data(0, self._image_address).shape
        return self._image_size

    def plot_options(self, normalise=None, errors=None):
        """Adjust plot options"""
        if normalise is not None:
            self._plot_normalise = normalise
        if errors is not None:
            self._plot_errors = errors

    def plotline(self, axis=None, xaxis='axes', yaxis='signal', *args, label=None, **kwargs):
        """Plot line on given axis"""
        # Tranpose arrays if data is multidimensional
        if axis is None:
            axis = plt.gca()
        xarray = np.array(self.__call__(xaxis)).T
        if self._plot_normalise:
            yarray = np.array(self._normalise(yaxis)).T
        else:
            yarray = np.array(self.__call__(yaxis)).T
        if label:
            label_str = self.label(label)
        else:
            label_str = yaxis

        if self._plot_errors:
            yerror = self._error(yaxis)
            lines = axis.errorbar(xarray, yarray, yerror, *args, label=label_str, **kwargs)
        else:
            lines = axis.plot(xarray, yarray, *args, label=label_str, **kwargs)
        return lines

    def plot(self, xaxis='axes', yaxis='signal', *args, **kwargs):
        """Create plot"""

        xdataset = self.__call__(xaxis)
        ydataset = self.__call__(yaxis)
        if hasattr(xdataset, 'basename'):
            xaxis = xdataset.basename
        if hasattr(ydataset, 'basename'):
            yaxis = ydataset.basename

        plt.figure()
        ax = plt.subplot(111)
        lines = self.plotline(ax, xaxis, yaxis, *args, **kwargs)

        if self._plot_normalise:
            yaxis = self._normalise_name(yaxis)

        plt.title(self.scan_title())
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.legend(yaxis.split(','))

    def plot_image(self, index, image_address=None):
        """Plot detector image"""

        plt.figure()
        ax = plt.subplot(111)

        im = self.image_data(index, image_address)
        ax.pcolormesh(im, shading='gouraud')

        plt.axis('image')
        plt.title(self.scan_title())

    def new_roi(self, cen_h=None, cen_v=None, wid_h=31, wid_v=31, name=None):
        """Create new roi and add to the namespace"""

        if name is None:
            name = 'roi'

        if cen_h is None:
            cen_h = self.image_size()[1] // 2
        if cen_v is None:
            cen_v = self.image_size()[0] // 2

        array_length = len(self.xaxis())

        roi_size = (cen_h, cen_v, wid_h, wid_v)
        roi_sum = np.zeros(array_length)
        roi_maxval = np.zeros(array_length)
        for n in range(array_length):
            image = self.image_data(n)
            roi = image[cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]
            roi_sum[n] = np.sum(roi)
            roi_maxval[n] = np.max(roi)

        name_size = '%s_size' % name
        name_sum = '%s_sum' % name
        name_maxval = '%s_maxval' % name
        n = 1
        while name_sum in self._namespace.keys():
            name_size = '%s%d_size' % (name, n)
            name_sum = '%s%d_sum' % (name, n)
            name_maxval = '%s%d_maxval' % (name, n)
            n += 1
        self._namespace[name_size] = roi_size
        self._namespace[name_sum] = roi_sum
        self._namespace[name_maxval] = roi_maxval
        print('Added new ROI to namespace: %s, %s and %s' % (name_size, name_sum, name_maxval))

    def fit(self, xaxis='axes', yaxis='signal', yerrors=None, fit_type=None, print_result=True, plot_result=False):
        """
        Automatic fitting of scan

        Use LMFit
        Pass fit_type = LMFit model
        return LMFit output
        """

        xdataset = self.__call__(xaxis)
        ydataset = self.__call__(yaxis)

        if hasattr(xdataset, 'basename'):
            xaxis = xdataset.basename
        if hasattr(ydataset, 'basename'):
            yaxis = ydataset.basename

        yerror = self._error(yaxis)
        if self._plot_normalise:
            yaxis = self._normalise_name(yaxis)
            ydataset = self.__call__(yaxis)

        # lmfit
        out = peakfit(xdataset, ydataset)

        # add to namespace
        self._fit_guess = None
        self._fit_model = None
        self._fit_result = out
        fit_dict = {}
        for pname, param in out.params.items():
            ename = 'stderr_' + pname
            fit_dict[pname] = param.value
            fit_dict[ename] = param.stderr
            self.add2namespace(fit_dict)

        if print_result:
            print(self.scan_title())
            print(out.fit_report())
        if plot_result:
            fig, grid = out.plot()
            plt.suptitle(self.scan_title(), fontsize=12)
            plt.subplots_adjust(top=0.85, left=0.15)
            ax1, ax2 = fig.axes
            ax2.set_xlabel(xaxis)
            ax2.set_ylabel(yaxis)
        return out

    def fit_result(self, parameter_name=None):
        """Returns parameter, error from the last run fit"""
        if self._fit_result is None:
            self.fit()
        if parameter_name is None:
            return self._fit_result
        param = self._fit_result.params[parameter_name]
        return param.value, param.stderr


class MultiScans:
    """
        MultiScan class, container for a number of Scan objects.
    """
    def __init__(self, scans, variable=None):
        self._scans = scans
        self._first_scan = scans[0]
        self._variable = variable

        self.scan_numbers = [scan.scan_number for scan in self._scans]
        self.scan_range = list(range(len(self._scans)))

        for n, scan in enumerate(scans):
            setattr(self, 'd%d' % n, scan)

    def __repr__(self):
        return "MultiScans(%s)" % self.scan_numbers

    def __str__(self):
        return '\n'.join(['%r' % scan for scan in self._scans])

    def __call__(self, name):
        out = []
        for scan in self._scans:
            out += [scan.__call__(name)]
        return out

    def __add__(self, addee):
        return MultiScans(self._scans + [addee])

    def __getitem__(self, key):
        out_list = []
        for scan in self._scans:
            out_list += [scan.get(key)]
        return out_list

    def get(self, key):
        return self.__getitem__(key)

    def set_variable(self, name):
        """Set default variable that changes with each scan"""
        self._variable = name

    def info(self, name=None):
        """Return str of name values"""
        if name is None and self._variable is None:
            return '\n'.join('%d' % scn for scn in self.scan_numbers)
        elif name is None:
            name = self._variable
        return '\n'.join(scan.info_line(name) for scan in self._scans)

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
        return '%s-%s' % (self.scan_numbers[0], self.scan_numbers[-1])

    def plotline(self, axis=None, xaxis=None, yaxis=None, *args, **kwargs):
        """plot metadata"""
        if axis is None:
            axis = plt.gca()
        if xaxis is None and self._variable is None:
            xaxis = self.scan_range
        elif xaxis is None:
            xaxis = self._variable
        if yaxis is None:
            yaxis = 'np.sum(%s)' % self._first_scan.yaxis()

        xarray = self.create_array(xaxis)
        yarray = self.create_array(yaxis)
        lines = axis.plot(xarray, yarray, *args, **kwargs)
        return lines

    def plotlines(self, axis=None, xaxis=None, yaxis=None, labels=None, *args, **kwargs):
        """Plot line on given axis"""
        if axis is None:
            axis = plt.gca()
        lines = []
        for scan in self._scans:
            lines += scan._plotline(axis, xaxis, yaxis, *args, label=labels, **kwargs)
        return lines

    def plot(self, xaxis=None, yaxis=None, labels=None, *args, **kwargs):
        """Create plot"""

        plt.figure()
        ax = plt.subplot(111)
        lines = self.plotlines(ax, xaxis, yaxis, labels, *args, **kwargs)

        if self._first_scan._plot_normalise:
            y = self._first_scan._normalise_name(yaxis)
        plt.title(self.title())
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)
        plt.legend()

    def plot_value(self, xaxis=None, yaxis=None, *args, **kwargs):
        """Create plot"""

        plt.figure()
        ax = plt.subplot(111)
        lines = self.plotline(ax, xaxis, yaxis, *args, **kwargs)

        plt.title(self.title())
        plt.xlabel(xaxis)
        plt.ylabel(yaxis)

    def new_roi(self, cen_h=None, cen_v=None, wid_h=31, wid_v=31, name=None):
        """Create new roi and add to the namespace"""

        if name is None:
            name = 'roi'

        if cen_h is None:
            cen_h = self._first_scan.image_size()[1] // 2
        if cen_v is None:
            cen_v = self._first_scan.image_size()[0] // 2

        for scan in self._scans:
            scan.new_roi(cen_h, cen_v, wid_h, wid_v, name)

    def fit(self, xaxis='axes', yaxis='signal', yerrors=None, fit_type=None, print_result=True, plot_result=False):
        """
        Automatic fitting of scans

        Use LMFit
        Pass fit_type = LMFit model
        return LMFit output
        """

        fit_result = []
        for scan in self._scans:
            fit_result += [scan.fit(xaxis, yaxis, yerrors, fit_type, print_result, plot_result)]
        return fit_result

    def fit_result(self, parameter_name=None):
        """Returns parameter, error from the last run fit"""

        if parameter_name is None:
            return [scan.fit_result() for scan in self._scans]

        values = np.zeros(len(self._scans))
        errors = np.zeros(len(self._scans))
        for n, scan in enumerate(self._scans):
            values[n], errors[n] = scan.fit_result(parameter_name)
        return values, errors
