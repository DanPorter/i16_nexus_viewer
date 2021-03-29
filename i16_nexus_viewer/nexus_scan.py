"""
Define the Scan object
"""

import numpy as np

from . import MATPLOTLIB_PLOTTING
from . import functions_nexus as fn
from .nexus_loader import NexusLoader, NexusMultiLoader
from .fitting import peakfit


"----------------------------------------------------------------------------------------------------------------------"
"------------------------------------------------ Scan ----------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class Scan(NexusLoader):
    """
    Scan Class
    Loads a Nexus (.nxs) scan file and reads standard parameters into an internal namespace. Also contains useful
    automatic functions for images, regions of interest and fitting.
    Inherits NexusLoader

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
    :param experiment: Experiment object
    """

    def __init__(self, filename, experiment=None, updatemode=False):
        self.experiment = experiment
        self.instrument = experiment.instrument if experiment else None
        address_list = experiment.dataset_addresses if experiment and not updatemode else None
        super().__init__(filename, address_list, updatemode)

        # scan defaults
        self.scanno = fn.scanfile2number(filename)
        self._scan_command = None
        self._default_address = fn.DEFAULTS
        if experiment:
            self._default_address.update(experiment._defaults)

        # scan command
        self.scan_command()
        # auto xy data
        self.auto_xyaxis()

    def __repr__(self):
        return 'Scan(%s, cmd=%s)' % (self.scanno, self.scan_command())

    def __str__(self):
        out = '%s\n' % (fn.OUTPUT_FORMAT % ('Scan Number', self.scanno))
        out += '%s\n' % (fn.OUTPUT_FORMAT % ('cmd', self.scan_command()))
        if self._axes_address is not None and self._signal_address is not None:
            axes, signal = self.auto_xyaxis()
            out += '%s\n' % (fn.OUTPUT_FORMAT % ('axes', axes))
            out += '%s\n' % (fn.OUTPUT_FORMAT % ('signal', signal))
        out += '%s\n' % (fn.OUTPUT_FORMAT % ('Duration', self.duration()))
        if self.experiment:
            data_addresses = self.addresses(self._default_address['scan_addresses'])
            s = '\n'.join('%20s : {%s}' % (fn.address_name(a), a) for a in data_addresses)
            out += self.string_operation(s)
        else:
            top_addresses = self.addresses(recursion_limit=2)
            s = '\n'.join('%20s : {%s}' % (fn.address_name(a), a) for a in top_addresses)
            out += self.string_operation(s)
        return out

    def __add__(self, addee):
        """
        Add two scans together somehow
        """
        return MultiScan([self, addee])

    def defaults(self, **kwargs):
        """Set or display the experiment default parameters"""
        if len(kwargs) == 0:
            df = self._default_address
            if self.experiment:
                dh = self.experiment._defaults_help
            else:
                dh = fn.DEFAULTS_HELP
            return fn.defaults_help(df, dh)
        self._default_address.update(kwargs)

    def scan_number(self):
        """Return the scan number from the current file"""
        with fn.load(self.filename) as hdf:
            try:
                scanno = fn.scannumber(hdf)
            except ValueError:
                scanno = 0
        return scanno

    def scan_command(self, command_address=None):
        """
        Return scan command
        :param command_address: None for automatic or str name or address
        :return: str
        """
        if command_address is None:
            if self._scan_command is not None:
                return self._scan_command
            command_address = self._default_address['cmd']
        command_command = "{%s}" % command_address
        cmd = self.string_operation(command_command, shorten=True)
        self._scan_command = cmd
        self.add_to_namespace(cmd=cmd)
        return cmd

    def duration(self, start_address=None, end_address=None):
        """
        Determine the duration of a scan using the start_time and end_time datasets
        :param start_address: address or name of start time dataset
        :param end_address: address or name of end time dataset
        :return: datetime.timedelta
        """
        if start_address is None:
            start_address = self._default_address['start_time']
        if end_address is None:
            end_address = self._default_address['end_time']
        with fn.load(self.filename) as hdf:
            timedelta = fn.duration(hdf, start_address, end_address, self._address_list)
        return timedelta

    def title(self, title_command=None):
        """
        Generate scan title
        :param title_command: str command e.g. 'title: {entry1/title}'
        :return: str
        """
        if title_command is None:
            title_command = self._default_address['title_command']
        return fn.shortstr(self.string_operation(title_command))

    def label(self, label_command=None):
        """
        Generate scan title
        :param label_command: str command e.g. '#{scanno}'
        :return: str
        """
        if label_command is None:
            label_command = self._default_address['label_command']
        return fn.shortstr(self.string_operation(label_command))

    def get_scan_addresses(self, scan_addresses=None):
        """
        Returns a list of addresses of scan values, each scan value is an array
        :param scan_addresses: str address of group or list of str addresses
        :return: list
        """
        if scan_addresses is None:
            scan_addresses = self._default_address['scan_addresses']
        return self.addresses(scan_addresses)

    def get_scan_names(self, scan_addresses=None):
        """
        Returns a list of names of scan values, each scan value is an array
        :param scan_addresses: str address of group or list of str addresses
        :return: list
        """
        addresses = self.get_scan_addresses(scan_addresses)
        return [fn.address_name(address) for address in addresses]

    def dataframe(self, scan_addresses=None):
        """
        returns Pandas.DataFrame of scan data
        :param scan_addresses: str address of group or list of str addresses
        :return: Pandas.DataFrame
        """
        if scan_addresses is None:
            scan_addresses = self._default_address['scan_addresses']
        # expand names and groups to dataset addresses
        return super().dataframe(scan_addresses)

    def arrays(self, scan_addresses=None, scan_length=None):
        """
        array of same length arrays of scan data
        :param scan_addresses: str address of group or list of str addresses
        :param scan_length: int or None length of data arrays
        :return: array
        """
        if scan_addresses is None:
            scan_addresses = self._default_address['scan_addresses']
        if scan_length is None:
            scan_length = self.data(self._default_address['scan_length'])
        # expand names and groups to dataset addresses
        return self.array_data(scan_addresses, scan_length)

    def metadata(self, name=None):
        """Return a value of the metadata or a metadata dict"""
        if name:
            return self.value(name)
        meta_address = self._default_address['meta_addresses']
        return self.group_values(meta_address)

    def normalise(self, name, norm_command=None):
        """Normalise named data"""
        if norm_command is None:
            norm_command = self._default_address['normalisation_command']
        operation = norm_command % name
        return self.get_operation(operation)

    def error(self, name, error_command=None):
        """Calcualte uncertainty on values"""
        if error_command is None:
            error_command = self._default_address['error_command']
        operation = error_command % name
        return self.get_operation(operation)

    def auto_data(self, name):
        """
        Automatically normalises and calculates error on data
        :param name: str dataset name or address
        :return: str(name), str(error_name), array(data), array(error)
        """
        op, data = self.get_operation(name)

        if op in self._default_address['error_names']:
            op_err, error = self.error(op)
        else:
            op_err, error = '', np.zeros(np.shape(data))

        if op in self._default_address['normalisation_names']:
            op, data = self.normalise(name)
            op_err, error = self.normalise(op_err)
        return op, op_err, data, error

    def auto_xyaxis(self, address='/', cmd_string=None):
        """
        Find default axes, signal hdf addresses using 'axes', 'signal' attributes, or from scan command
        :param address: str addtress to start in
        :param cmd_string: str of command to take x,y axis from as backup
        :return: xaxis_address, yaxis_address
        """
        # Note: this doesn't overwrite the use of the method in super().address
        if cmd_string is None:
            cmd_string = self.scan_command()
        return super().auto_xyaxis(address, cmd_string)

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
            #plt.suptitle(self.title(), fontsize=12)
            #plt.subplots_adjust(top=0.85, left=0.15)
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


"----------------------------------------------------------------------------------------------------------------------"
"---------------------------------------------- MultiScan -------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class MultiScan(NexusMultiLoader):
    """
    MultiScan - container for multiple Scan objects
    e.g.
    d1 = Scan("file1.nxs")
    d2 = Scan("file2.nxs")
    group = MultiScan([d1, d2])

    The same can be achieved by adding NexusLoader objects:
    group = d1 + d2

    Additional parameters (can also be set by self.param(value)):
    :param loaders: list of NexusLoader objects
    :param axes: default axis of each Loader
    :param signal: default signal of each loader
    :param variables: str or list of str values that change in each file
    :param shape: tuple shape of data, allowing for multiple dimesion arrays (nLoaders, scany, scanx)
    """
    def __init__(self, loaders, axes='axes', signal='signal', variables=None, shape=None):
        super().__init__(loaders)
        self._axes = axes
        self._signal = signal
        self._variables = None if variables is None else np.asarray(variables, dtype=str).reshape(-1)
        if shape is None:
            self._shape = (len(loaders),)
        else:
            self._shape = shape

    def __repr__(self):
        return "MultiScan(%d files)" % len(self.loaders)

    def __str__(self):
        variables = self.variables()
        if variables is not None:
            out = ''
            for loader in self.loaders:
                vstr = ', '.join('%s={%s}' % (v, v) for v in variables)
                out += '%r: %s\n' % (loader, loader.string_operation(vstr))
            return out
        return '\n'.join(['%r' % loader for loader in self.loaders])

    def __add__(self, addee):
        new_shape = (len(self.loaders) + 1,)
        return MultiScan(self.loaders + [addee], self._axes, self._signal, self._variables, new_shape)

    def __radd__(self, addee):
        new_shape = (len(self.loaders) + 1,)
        return MultiScan([addee] + self.loaders, self._axes, self._signal, self._variables, new_shape)

    def axes(self, name=None):
        """Set or return default axes name"""
        if name is None:
            return self._axes
        self._axes = name

    def signal(self, name=None):
        """Set or return default signal name"""
        if name is None:
            return self._signal
        self._signal = name

    def variables(self, names=None):
        """Set or return default variable that changes with each scan"""
        if names is None:
            return self._variables
        self._variables = np.asarray(names, dtype=str).reshape(-1)

    def add_variable(self, names):
        """Add a variable name to the list"""
        self._variables = np.append(self._variables, names)

    def shape(self, split_repeat=None):
        """Set or return expected shape of output array"""
        if split_repeat is None:
            return self._shape

        if split_repeat == 1:
            self._shape = (len(self.loaders), )
        else:
            self._shape = (len(self.loaders)//split_repeat, split_repeat)

    def array(self, name, data_length=None):
        """Return numerical array from values, defaults to the shortest length"""
        data = self.__call__(name)

        if data_length is None:
            data_length = self._shape[-1] if len(self._shape) > 1 else np.max([np.size(d) for d in data])

        # shorten or fill arrays
        out = np.nan * np.zeros([len(data), data_length])
        for n, d in enumerate(data):
            if np.size(d) == 1:
                out[n, :] = d
            elif np.size(d) >= data_length:
                out[n, :] = d[:data_length]
            else:
                out[n, :len(d)] = d
        return out

    def meshdata(self, xname, yname, signal, repeat=None):
        """Create meshgrid of 2 variables"""
        xdata = np.array(self.value(xname))
        ydata = np.array(self.value(yname))
        data = np.array(self.value(signal))

        if repeat is None:
            # Determine the repeat length of the scans
            delta_x = np.abs(np.diff(xdata))
            ch_idx_x = np.where(delta_x > delta_x.max() * 0.9)  # find biggest changes
            ch_delta_x = np.diff(ch_idx_x)
            rep_len_x = np.round(np.mean(ch_delta_x))
            delta_y = np.abs(np.diff(ydata))
            ch_idx_y = np.where(delta_y > delta_y.max() * 0.9)  # find biggest changes
            ch_delta_y = np.diff(ch_idx_y)
            rep_len_y = np.round(np.mean(ch_delta_y))
            print('Scans in {} are repeating every {} iterations'.format(xname, rep_len_x))
            print('Scans in {} are repeating every {} iterations'.format(yname, rep_len_y))
            repeat = int(max(rep_len_x, rep_len_y))

        xsquare = xdata[:repeat * (len(xdata) // repeat)].reshape(-1, repeat)
        ysquare = ydata[:repeat * (len(ydata) // repeat)].reshape(-1, repeat)
        dsquare = data[:repeat * (len(data) // repeat)].reshape(-1, repeat)
        return xsquare, ysquare, dsquare

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


"----------------------------------------------------------------------------------------------------------------------"
"---------------------------------------------- ADD PLOTTING ----------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


if MATPLOTLIB_PLOTTING:
    # Recast Scan, MultiScan with plotting
    from .plotting_matplotlib import ScanPlotsMatplotlib, MultiScanPlotsMatplotlib
    Scan = ScanPlotsMatplotlib
    MultiScan = MultiScanPlotsMatplotlib
