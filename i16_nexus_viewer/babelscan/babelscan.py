"""
babelscan object for holding many types of scan data
"""

import numpy as np
from . import functions as fn
from . import EVAL_MODE
from . import init_plot, init_peakfit


"----------------------------------------------------------------------------------------------------------------------"
"------------------------------------------------- Scan ---------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class Scan:
    """
    Scan class
    Contains a namespace of data associated with names and a seperate dictionary of name associations,
    allowing multiple names to reference the same data.
      namespace = {
        'data1': [1,2,3],
        'data2': [10,20,30]
      }
      alt_names = {
        'xaxis': 'data1',
        'yaxis': 'data2',
      }
      dh = Scan(namespace, alt_names)
      dh('xaxis') >> returns [1,2,3]

    :param namespace: dict : dict of names and data {name: data}
    :param alt_names: dict or None* : dict of alternative names to names in namespace
    :param kwargs: key-word-argments as options shwon below, keywords and argmuents will be added to the namespace.

    Options:
      reload - True/False*, if True, reload mode is activated, reloading data on each operation
      label_name - str, add a name to use to automatically find the label
      label_command - str, format specifier for label, e.g. '{scan_number}'
      title_name - str, add a name to use to automatically find the title
      title_command - str, format specifier for title, e.g. '#{scan_number} Energy={en:5.2f} keV'
      scan_command_name - str, add a name to use to automatically find the scan command
      start_time_name - str, add a name to use to automatically find the start_time
      end_time_name - str, add a name to use to automatically find the end_time
      axes_name - str, add a name to use to automatically find the axes (xaxis)
      signal_name - str, add a name to use to automatically find the signal (yaxis)
      image_name - str, add a name to use to automatically find the detector image
      str_list - list of str, list of names to display when print(self)
      signal_operation - str, operation to perform on signal, e.g. '/Transmission'
      error_function - func., operation to perform on signal to generate errors. e.g. np.sqrt
      debug - str or list of str, options for debugging, options:
        'namespace' - displays when items are added to the namespace

    Functions
    add2namespace(name, data=None, other_names=None, hdf_address=None)
        set data in namespace
    add2strlist(names)
        Add to list of names in str output
    address(name)
        Return hdf address of namespace name
    array(names, array_length=None)
        Return numpy array of data with same length
    axes()
        Return default axes (xaxis) data
    dataset(name)
        Return dataset object
    eval(operation)
        Evaluate operation using names in dataset or in associated names
    find_image(multiple=False)
        Return address of image data in hdf file
    get_plot_data(xname=None, yname=None, signal_op=None, error_op=None)
        Return xdata, ydata, yerror, xname, yname
    image(idx=None, image_address=None)
        Load image from hdf file, works with either image addresses or stored arrays
    image_roi(cen_h=None, cen_v=None, wid_h=31, wid_v=31)
        Create new region of interest from detector images
    image_roi_op(operation)
        Create new region of interest (roi) from image data and return sum and maxval
    image_roi_sum(cen_h=None, cen_v=None, wid_h=31, wid_v=31)
        Create new region of interest
    image_size()
        Returns the image size
    label(new_label=None)
        Set or Return the scan label. The label is a short identifier for the scan, such as scan number
    load()
        Open and return hdf.File object
    name(name)
        Return corrected name from namespace
    options(**kwargs)
        Set or display options
    reload_mode(mode=None)
        Turns on reload mode - reloads the dataset each time
    reset()
        Reset the namespace
    scan_command()
        Returns scan command
    show_namespace()
        return str of namespace
    signal()
        Return default signal (yaxis) data
    string(names, str_format=None)
        Return formated string of data
    string_format(operation)
        Process a string with format specified in {} brackets, values will be returned.
    title(new_title=None)
        Set or Return the title
    tree(detail=False, recursion_limit=100)
        Return str of the full tree of data in a hdf object
    value(names, array_function=None)
        Return single value of data
    """
    def __init__(self, namespace, alt_names=None, **kwargs):
        self._namespace = kwargs.copy()
        self._other2name = {}
        self._namespace.update(namespace)
        if alt_names is not None:
            self._other2name.update(alt_names)

        self._options = {}
        self._label_str = ['label']
        self._title_str = ['title', 'filename']
        self._scan_command_str = ['scan_command', 'cmd']
        self._start_time_str = ['start_time']
        self._end_time_str = ['end_time']
        self._exposure_time_str = ['count_time', 'counttime', 't']
        self._axes_str = ['axes', 'xaxis']
        self._signal_str = ['signal', 'yaxis']
        self._image_name = None
        self._image_size = None
        self._print_list = ['scan_command', 'axes', 'signal']
        self._reload_mode = False
        self._set_options(**kwargs)

    "------------------------------- Basic Operations -------------------------------------------"

    def reset(self):
        """Regenerate data lists"""
        self._namespace = {}

    def reload_mode(self, mode=None):
        """
        Turns on reload mode - reloads the dataset each time
        :param mode: Bool or None, True to turn on, None to return current mode
        :return: None or str
        """
        if mode is None:
            if self._reload_mode:
                return "Reload mode is ON"
            return "Reload mode is OFF"
        self._reload_mode = mode

    def add2namespace(self, name, data=None, other_names=None):
        """
        set data in namespace
        :param name: str name
        :param data: any or None, data to store in namespace (nothing stored if None)
        :param other_names: str, list of str or None - strings to associate with name, giving the same result
        :return: None
        """
        if data is not None:
            self._namespace[name] = data
            if 'debug' in self._options and 'namespace' in self._options['debug']:
                print('Add to namespace: %s: %s' % (name, fn.data_string(data)))
        if other_names is not None:
            other_names = np.asarray(other_names, dtype=str).reshape(-1)
            for other_name in other_names:
                self._other2name[other_name] = name
                if 'debug' in self._options and 'namespace' in self._options['debug']:
                    print('Add association: %s: %s' % (other_name, name))

    def show_namespace(self):
        """return str of namespace"""
        out = 'Namespace %r:\n' % self
        out += '%-20s %-60s | %s\n' % ('Name', 'Alternate Names', 'Data')
        for key, item in self._namespace.items():
            other_names = ', '.join(okey for okey, oitem in self._other2name.items() if oitem == key)
            out += '%-20s %-60s | %s\n' % (key, other_names, fn.data_string(item))
        return out

    def add2strlist(self, names):
        """Add to list of names in str output"""
        names = list(np.asarray(names, dtype=str).reshape(-1))
        self._print_list += names

    def options(self, **kwargs):
        """Set or display options"""
        if len(kwargs) == 0:
            # return options
            out = 'Options:\n'
            for key, item in self._options.items():
                out += '%20s : %s\n' % (key, item)
            return out
        self._set_options(**kwargs)

    def _set_options(self, **kwargs):
        """Set options"""
        self._options.update(kwargs)
        if 'reload' in kwargs:
            self._reload_mode = kwargs['reload']
        if 'label_name' in kwargs:
            self._label_str.insert(0, kwargs['label_name'])
        if 'title_name' in kwargs:
            self._title_str.insert(0, kwargs['title_name'])
        if 'scan_command_name' in kwargs:
            self._scan_command_str.insert(0, kwargs['scan_command_name'])
        if 'start_time_name' in kwargs:
            self._start_time_str.insert(0, kwargs['start_time_name'])
        if 'end_time_name' in kwargs:
            self._end_time_str.insert(0, kwargs['end_time_name'])
        if 'exposure_time_name' in kwargs:
            self._exposure_time_str.insert(0, kwargs['exposure_time_name'])
        if 'axes_name' in kwargs:
            self._axes_str.insert(0, kwargs['axes_name'])
        if 'signal_name' in kwargs:
            self._signal_str.insert(0, kwargs['signal_name'])
        if 'image_name' in kwargs:
            self._image_name.insert(0, kwargs['image_name'])
        if 'str_list' in kwargs:
            self.add2strlist(kwargs['str_list'])

    "------------------------------- class operations -------------------------------------------"

    def __repr__(self):
        return 'Scan(namespace: %d, alt_names: %d)' % (len(self._namespace), len(self._other2name))

    def __str__(self):
        out = self.__repr__()
        out += '\n' + '\n'.join(self.string(self._print_list))
        return out

    def __call__(self, name):
        return self.eval(name)

    def __getitem__(self, name):
        name, data = self._get_list_data(name)
        if len(data) == 1:
            return data[0]
        return data

    def __len__(self):
        return self.scan_length()

    def __add__(self, addee):
        """
        Add two scans together somehow
        """
        return MultiScan([self, addee])

    "------------------------------- data -------------------------------------------"

    def _get_data(self, name):
        """
        Get data from stored dicts
          This function may be overloaded in subclasses
        :param name: str, key or associated key in namespace
        :return: data from namespace dict
        """
        if name in self._namespace:
            return self._namespace[name]
        if name in self._other2name and self._other2name[name] in self._namespace:
            return self._namespace[self._other2name[name]]
        elif name in self._other2name:
            return self._get_data(self._other2name[name])
        # Check defaults
        if name in self._axes_str:
            return self.axes()
        if name in self._signal_str:
            return self.signal()
        # Check new region of interest
        if 'nroi' in name:
            roi_sum, roi_max = self.image_roi_op(name)
            return roi_sum
        # Start searching for alternate names
        if name.lower() in [key.lower() for key in self._namespace]:
            return self._namespace[name]
        if name.lower() in [key.lower() for key in self._other2name]:
            return self._namespace[self._other2name[name]]
        raise KeyError('\'%s\' not available' % name)

    def _get_name_data(self, name):
        """
        Get name and data from stored dicts
        :param name: str, key or associated key in namespace
        :return name, data: from namespace dict
        """
        data = self._get_data(name)
        if name in self._other2name:
            name = self._other2name[name]
        return name, data

    def _get_list_data(self, names):
        """
        Get data from stored dicts
        :param names: str or list of str, key or associated key in namespace
        :return: list of data from namespace dict
        """
        names = np.asarray(names, dtype=str).reshape(-1)
        data = []
        new_name = []
        for name in names:
            n, d = self._get_name_data(name)
            data += [d]
            new_name += [n]
        return new_name, data

    def array(self, names, array_length=None):
        """
        Return numpy array of data with same length
         data with length 1 will be cast over the full length
         data with length >1 and < array_length will be filled with nans
        :param names: str or list of str, key or associated key in namespace
        :param array_length: int or None, length of arrays returned
        :return: array(n,array_length) where n is the length of list names
        """
        names, data = self._get_list_data(names)
        if array_length is None:
            array_length = np.max([np.size(d) for d in data])
        out = np.nan * np.zeros(shape=(len(data), array_length))
        for n in range(len(data)):
            if np.size(data[n]) == 1:
                out[n, :] = data[n]
            else:
                out[n, :len(data[n])] = data[n]
        return out

    def value(self, names, array_function=None):
        """
        Return single value of data
        :param names: str or list of str, key or associated key in namespace
        :param array_function: function to return a single value from an array
        :return: value or list of values
        """
        names, data = self._get_list_data(names)
        if array_function is None:
            array_function = fn.VALUE_FUNCTION
        out = [array_function(val) for val in data]
        if len(out) == 1:
            return out[0]
        return out

    def name(self, name):
        """
        Return corrected name from namespace
        :param name: str or list of str
        :return: str or list of str
        """
        names = np.asarray(name, dtype=str).reshape(-1)
        out = [self._get_name_data(name)[0] for name in names]
        if len(out) == 1:
            return out[0]
        return out

    def string(self, names, str_format=None):
        """
        Return formated string of data
        :param names: str or list of str, key or associated key in namespace
        :param str_format: format to use, e.g. '%s:%s'
        :return: str or list of str
        """
        names, data = self._get_list_data(names)
        if str_format is None:
            str_format = fn.OUTPUT_FORMAT
        out = [str_format % (name, fn.data_string(val)) for name, val in zip(names, data)]
        if len(out) == 1:
            return out[0]
        return out

    def time(self, names, date_format=None):
        """
        Return datetime object from data name
        :param names: str or list of str, key or associated key in namespace
        :param date_format: str format used in datetime.strptime (see https://strftime.org/)
        :return: list of datetime ojbjects
        """
        names, data = self._get_list_data(names)
        return fn.data_datetime(data, date_format)

    "------------------------------- Operations -----------------------------------------"

    def _prep_operation(self, operation):
        """
        prepare operation string, replace names with names in namespace
        :param operation: str
        :return operation: str, names replaced to match namespace
        """
        # First look for addresses in operation to seperate addresses from divide operations
        # addresses = fn.re_address.findall(operation)

        # Determine custom regions of interest 'nroi'
        rois = fn.re_nroi.findall(operation)
        for name in rois:
            new_name, data = self._get_name_data(name)
            operation = operation.replace(name, new_name)

        # Determine data for other variables
        names = fn.re_varname.findall(operation)
        for name in names:
            new_name, data = self._get_name_data(name)
            if new_name != name:
                operation = operation.replace(name, new_name)
        return operation

    def _name_eval(self, operation):
        """
        Evaluate operation using names in dataset or in associated names
        :param operation: str
        :return: corrected operation, output of operation
        """
        if not EVAL_MODE:
            return self._get_name_data(operation)
        bad_names = ['import', 'os.', 'sys.']
        for bad in bad_names:
            if bad in operation:
                raise Exception('This operation is not allowed as it contains: "%s"' % bad)
        operation = self._prep_operation(operation)
        result = eval(operation, globals(), self._namespace)
        if operation in self._namespace or operation in self._other2name:
            return operation, result
        # add to namespace
        n = 1
        while 'operation%d' % n in self._namespace:
            n += 1
        self.add2namespace('operation%d' % n, result, operation)
        return operation, result

    def eval(self, operation):
        """
        Evaluate operation using names in dataset or in associated names
        :param operation: str
        :return: output of operation
        """
        _, out = self._name_eval(operation)
        return out

    def string_format(self, operation):
        """
        Process a string with format specified in {} brackets, values will be returned.
        e.g.
          operation = 'the energy is {energy} keV'
          out = string_command(operation)
          # energy is found within hdf tree
          out = 'the energy is 3.00 keV'
        :param operation: str format operation e.g. '#{scan_number}: {title}'
        :return: str
        """
        # get values inside brackets
        ops = fn.re_strop.findall(operation)
        format_namespace = {}
        for op in ops:
            op = op.split(':')[0]  # remove format specifier
            name, data = self._name_eval(op)
            try:
                value = fn.VALUE_FUNCTION(data)
            except TypeError:
                value = data
            format_namespace[name] = value
            operation = operation.replace(op, name)
        return operation.format(**format_namespace)

    def _get_error(self, name, operation=None):
        """
        Return uncertainty on data using operation
        :param operation: function to apply to signal, e.g. 'np.sqrt'
        :param operation: None* will default to zero, unless "error_function" in options
        :return: operation(array)
        """
        _, data = self._name_eval(name)
        if operation is None:
            if 'error_function' in self._options:
                operation = self._options['error_function']
            else:
                return np.zeros(np.shape(data))
        return operation(data)

    def _get_signal_operation(self, name, signal_op=None, error_op=None):
        """
        Return data after operation with error
        :param name: str name in namespace
        :param signal_op: operation to perform on name, e.g. '/Transmission'
        :param error_op: function to performon name, e.g. np.sqrt
        :return: signal_name, output, error arrays
        """
        name, data = self._name_eval(name)
        error = self._get_error(name, error_op)
        # add error array to namespace
        error_name = '%s_error' % name
        self.add2namespace(error_name, error)

        if signal_op is None:
            if 'signal_operation' in self._options:
                signal_op = self._options['signal_operation']
            else:
                return name, data, error
        # Create operations
        operation = name + signal_op
        operation_error = error_name + signal_op
        signal = self.eval(operation)
        error = self.eval(operation_error)
        return operation, signal, error

    "------------------------------- Defaults -------------------------------------------"

    def label(self, new_label=None):
        """
        Set or Return the scan label. The label is a short identifier for the scan, such as scan number
        :param new_label: str or None, if str sets the label as str, if None, returns automatic label
        :return: None or str
        """
        if new_label:
            self.add2namespace(self._label_str[0], new_label)
            return

        if 'label_command' in self._options:
            return self.string_format(self._options['label_command'])

        add2othernames = []
        for s in self._label_str:
            try:
                data = self._get_data(s)
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No label in %r' % self)

    def title(self, new_title=None):
        """
        Set or Return the title
        :param new_title: str or None, if str sets the title as str, if None, returns automatic title
        :return: None or str
        """
        if new_title:
            self.add2namespace(self._title_str[0], new_title)
            return

        if 'title_command' in self._options:
            return self.string_format(self._options['title_command'])

        add2othernames = []
        for s in self._title_str:
            try:
                data = self._get_data(s)
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No title in %r' % self)

    def scan_command(self):
        """
        Returns scan command
        :return: str
        """
        add2othernames = []
        for s in self._scan_command_str:
            try:
                data = self._get_data(s)
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No Scan Command in %r' % self)

    def scan_time(self):
        """
        Return scan start time
        :return: datetime
        """
        add2othernames = []
        for s in self._start_time_str:
            try:
                data = self.time(s)[0]
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No start time in %r' % self)

    def scan_finish(self):
        """
        Return scan end time
        :return: datetime
        """
        add2othernames = []
        for s in self._end_time_str:
            try:
                data = self.time(s)[0]
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        for s in self._start_time_str[::-1]:
            try:
                data = self.time(s)[-1]
                self.add2namespace(s, other_names=add2othernames)
                return data
            except KeyError:
                add2othernames += [s]
        raise Exception('No end time in %r' % self)

    def duration(self, start_time=None, end_time=None):
        """
        Calculate time difference between two times
        :param start_time: str name of date dataset or array of timestamps
        :param end_time: None or str name of date dataset
        :return: datetime.timedelta
        """
        if end_time is not None:
            end_time = self.time(end_time)[-1]
        if start_time is None:
            start_time = self.scan_time()
        else:
            lst = self.time(start_time)
            start_time = lst[0]
            if len(lst) > 1 and end_time is None:
                end_time = lst[-1]
        if end_time is None:
            end_time = self.scan_finish()
        return end_time - start_time

    def exposure_time(self):
        """Return the exposure time"""
        value = 1.0
        for s in self._exposure_time_str:
            try:
                value = self.value(s)
            except KeyError:
                pass
        self.add2namespace('exposure_time', value, other_names=self._exposure_time_str)
        return value

    def _find_defaults(self):
        """
        Find default axes and signal (x-axis/y-axis), adds to namespace
         This function may be overloaded in subclasses
        :return: axes_name, signal_name
        """
        scan_command = self.scan_command()
        # axes / x-axis
        axes_name = fn.axes_from_cmd(scan_command)
        axes_data = self._get_data(axes_name)
        self.add2namespace(axes_name, axes_data, self._axes_str)
        # signal / y-axis
        signal_name = fn.signal_from_cmd(scan_command)
        signal_data = self._get_data(signal_name)
        self.add2namespace(signal_name, signal_data, self._signal_str)
        return axes_name, signal_name

    def axes(self):
        """
        Return default axes (xaxis) data
        :return: array
        """
        add2othernames = []
        for name in self._axes_str:
            if name in self._namespace:
                self.add2namespace(name, other_names=add2othernames)
                return self._namespace[name]
            if name in self._other2name:
                return self._namespace[self._other2name[name]]
            add2othernames += [name]
        # axes not in namespace, get from scan command
        axes_name, signal_name = self._find_defaults()
        return self._get_data(axes_name)

    def signal(self):
        """
        Return default signal (yaxis) data
        :return: array
        """
        add2othernames = []
        for name in self._signal_str:
            if name in self._namespace:
                self.add2namespace(name, other_names=add2othernames)
                return self._namespace[name]
            if name in self._other2name:
                return self._namespace[self._other2name[name]]
            add2othernames += [name]
        # signal not in namespace, get from scan command
        axes_name, signal_name = self._find_defaults()
        return self._get_data(signal_name)

    def scan_length(self):
        """
        Return the number of points in the scan (length of 'axes')
        :return: int
        """
        return np.size(self.axes())

    def get_plot_data(self, xname=None, yname=None, signal_op=None, error_op=None):
        """
        Return xdata, ydata, yerror, xname, yname
         x, y, dy, xlabel, ylabel = scan.get_plot_data('axes', 'signal', '/Transmission', np.sqrt)

        :param xname: str name of value to use as x-axis
        :param yname: str name of value to use as y-axis
        :param signal_op: operation to perform on yaxis, e.g. '/Transmission'
        :param error_op: function to use on yaxis to generate error, e.g. np.sqrt
        :return xdata: array
        :return ydata: array
        :return yerror: array
        :return xname: str
        :return yname: str
        """
        if xname is None:
            xname = self._axes_str[0]
        if yname is None:
            yname = self._signal_str[0]
        xname, xdata = self._get_name_data(xname)
        yname, ydata, yerror = self._get_signal_operation(yname, signal_op, error_op)
        return xdata, ydata, yerror, xname, yname

    "------------------------------- images -------------------------------------------"

    def image(self, idx):
        """
        Return detector image
         Overloaded in subclasses, this version does nothing interesting
        :param idx: int index of image
        :return: 2d array
        """
        return np.zeros([100, 100])

    def image_size(self):
        """
        Returns the image size
        :return: tuple
        """
        if self._image_size:
            return self._image_size
        image = self.image(0)
        shape = np.shape(image)
        self._image_size = shape
        return shape

    def image_roi(self, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
        """
        Create new region of interest from detector images
        :param cen_h: int or None
        :param cen_v: int or None
        :param wid_h:  int or None
        :param wid_v:  int or None
        :return: l*v*h array
        """
        shape = self.image_size()
        scan_length = len(self.axes())

        if cen_h is None:
            cen_h = shape[1] // 2
        if cen_v is None:
            cen_v = shape[0] // 2

        roi = np.zeros([scan_length, wid_v, wid_h])
        for n in range(scan_length):
            image = self.image(n)
            roi[n] = image[cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]
        return roi

    def image_roi_sum(self, cen_h=None, cen_v=None, wid_h=31, wid_v=31):
        """
        Create new region of interest
        :param cen_h: int or None
        :param cen_v: int or None
        :param wid_h:  int or None
        :param wid_v:  int or None
        :return: roi_sum, roi_max
        """
        shape = self.image_size()
        scan_length = len(self.axes())

        if cen_h is None:
            cen_h = shape[1] // 2
        if cen_v is None:
            cen_v = shape[0] // 2

        roi_sum = np.zeros(scan_length)
        roi_max = np.zeros(scan_length)
        for n in range(scan_length):
            image = self.image(n)
            roi = image[cen_v - wid_v // 2:cen_v + wid_v // 2, cen_h - wid_h // 2:cen_h + wid_h // 2]
            roi_sum[n] = np.sum(roi)
            roi_max[n] = np.max(roi)
        # Add to namespace
        n = 1
        while 'nroi%d_sum' % n in self._namespace:
            n += 1
        full_name = 'nroi[%d,%d,%d,%d]' % (cen_h, cen_v, wid_h, wid_v)
        self.add2namespace('nroi%d_sum' % n, roi_sum, other_names=full_name)
        self.add2namespace('nroi%d_max' % n, roi_max)
        return roi_sum, roi_max

    def image_roi_op(self, operation):
        """
        Create new region of interest (roi) from image data and return sum and maxval
        The roi centre and size is defined by an operation:
          operation = 'nroi[210, 97, 75, 61]'
          'nroi'      -   creates a region of interest in the detector centre with size 31x31
          'nroi[h,v]' -   creates a roi in the detector centre with size hxv, where h is horizontal, v is vertical
          'nroi[m,n,h,v] - create a roi with cen_h, cen_v, wid_h, wid_v = n, m, h, v
        :param operation: str : operation string
        :return: sum, maxval : [o] length arrays
        """
        vals = [int(val) for val in fn.re.findall(r'\d+', operation)]
        nvals = len(vals)
        if nvals == 4:
            cen_h, cen_v, wid_h, wid_v = vals
        elif nvals == 2:
            cen_h, cen_v = None, None
            wid_h, wid_v = vals
        else:
            cen_h, cen_v = None, None
            wid_h, wid_v = 31, 31
        roi_sum, roi_max = self.image_roi_sum(cen_h, cen_v, wid_h, wid_v)
        # Add operation to associated namespace
        n = 1
        while 'nroi%d_sum' % n in self._namespace:
            n += 1
        n -= 1
        self.add2namespace('nroi%d_sum' % n, other_names=operation)
        return roi_sum, roi_max

    "------------------------------- fitting -------------------------------------------"

    def fit(self, xaxis='axes', yaxis='signal', fit_type=None, print_result=True, plot_result=False):
        """
        Automatic fitting of scan

        Use LMFit
        Pass fit_type = LMFit model
        return LMFit output
        """
        peakfit = init_peakfit()
        xdata, ydata, yerror, xname, yname = self.get_plot_data(xaxis, yaxis, None, None)

        # lmfit
        out = peakfit(xdata, ydata, yerror)

        self.add2namespace('lmfit', out, 'fit_result')
        fit_dict = {}
        for pname, param in out.params.items():
            ename = 'stderr_' + pname
            fit_dict[pname] = param.value
            fit_dict[ename] = param.stderr
        self._namespace.update(fit_dict)
        self.add2namespace('fit_%s' % yname, out.best_fit, other_names=['fit'])

        if print_result:
            print(self.title())
            print(out.fit_report())
        if plot_result:
            fig, grid = out.plot()
            # plt.suptitle(self.title(), fontsize=12)
            # plt.subplots_adjust(top=0.85, left=0.15)
            ax1, ax2 = fig.axes
            ax2.set_xlabel(xname)
            ax2.set_ylabel(yname)
        return out

    def fit_result(self, parameter_name=None):
        """
        Returns parameter, error from the last run fit
        :param parameter_name: str, name from last fit e.g. 'amplitude', or None to return lmfit object
        :param
        :return:
        """
        if 'lmfit' not in self._namespace:
            self.fit()
        lmfit = self._get_data('lmfit')
        if parameter_name is None:
            return lmfit
        param = lmfit.params[parameter_name]
        return param.value, param.stderr

    "------------------------------- plotting -------------------------------------------"

    def plotline(self, xaxis='axes', yaxis='signal', *args, **kwargs):
        """
        Plot scanned datasets on matplotlib axes subplot
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param args: given directly to plt.plot(..., *args, **kwars)
        :param axes: matplotlib.axes subplot, or None to use plt.gca()
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: list lines object, output of plot
        """
        pm = init_plot()
        xdata, ydata, yerror, xname, yname = self.get_plot_data(xaxis, yaxis, None, None)

        if 'label' not in kwargs:
            kwargs['label'] = self.label()
        axes = kwargs['axes'] if 'axes' in kwargs else None
        lines = pm.plot_line(axes, xdata, ydata, None, *args, **kwargs)
        return lines

    def plot(self, xaxis='axes', yaxis='signal', *args, **kwargs):
        """
        Create matplotlib figure with plot of the scan
        :param axes: matplotlib.axes subplot
        :param xaxis: str name or address of array to plot on x axis
        :param yaxis: str name or address of array to plot on y axis
        :param args: given directly to plt.plot(..., *args, **kwars)
        :param axes: matplotlib.axes subplot, or None to create a figure
        :param kwargs: given directly to plt.plot(..., *args, **kwars)
        :return: axes object
        """
        pm = init_plot()
        # Check for multiple inputs on yaxis
        ylist = np.asarray(yaxis, dtype=str).reshape(-1)

        # Create figure
        if 'axes' in kwargs:
            axes = kwargs['axes']
        else:
            axes = pm.create_axes(subplot=111)

        # x axis data
        xname, xdata = self._get_name_data(xaxis)

        # Add plots
        ynames = []
        for yaxis in ylist:
            yname, ydata = self._get_name_data(yaxis)
            ynames += [yname]
            pm.plot_line(axes, xdata, ydata, None, *args, label=yname, **kwargs)

        # Add labels
        ttl = self.title()
        ylabel = ', '.join(set(ynames))
        pm.labels(ttl, xname, ylabel, legend=True)
        return axes

    def plot_detector(self, index=None, xaxis='axes', axes=None, clim=None, cmap=None, colorbar=False, **kwargs):
        """
        Plot detector image in matplotlib figure
        :param index: int, detector image index, 0-length of scan, if None, use centre index
        :param xaxis: name or address of xaxis dataset
        :param axes: matplotlib axes to plot on (None to create figure)
        :param clim: [min, max] colormap cut-offs (None for auto)
        :param cmap: str colormap name (None for auto)
        :param colorbar: False/ True add colorbar to plot
        :param kwargs: additinoal arguments for plot_detector_image
        :return: axes object
        """
        pm = init_plot()
        # x axis data
        xname, xdata = self._get_name_data(xaxis)

        # image data
        if index is None:
            index = len(xdata) // 2
        im = self.image(index)

        # Create figure
        if axes is None:
            axes = pm.create_axes(subplot=111)
        pm.plot_detector_image(axes, im, **kwargs)

        # labels
        ttl = '%s\n%s [%d] = %s' % (self.title(), xname, index, xdata[index])
        pm.labels(ttl, colorbar=colorbar, colorbar_label='Detector', axes=axes)
        pm.colormap(clim, cmap, axes)
        return axes


"----------------------------------------------------------------------------------------------------------------------"
"--------------------------------------------- MultiScan ------------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class MultiScan:
    """
    Class for holding multiple DataHolders
    """
    def __init__(self, scan_list, variables=None):
        self._scan_list = []
        for scan in scan_list:
            if issubclass(type(scan), MultiScan):
                self._scan_list.extend(scan._scan_list)
            else:
                self._scan_list.append(scan)

        if variables is None:
            self._variables = []
        else:
            self._variables = list(np.asarray(variables, dtype=str).reshape(-1))

    def __repr__(self):
        return 'MultiScan(%d items)' % len(self._scan_list)

    def __str__(self):
        variables = self.string(self._variables)
        out = ''
        for n in range(len(self._scan_list)):
            out += '%s: %s\n' % (self._scan_list[n].label(), variables[n])
        return out

    def __add__(self, other):
        return MultiScan([self, other])

    def __call__(self, name):
        return [scan(name) for scan in self._scan_list]

    def __getitem__(self, item):
        return self._scan_list[item]

    def __len__(self):
        return len(self._scan_list)

    def add_variable(self, name):
        """
        Add variable
        :param name: name of variable parameter between scans
        :return:
        """
        self._variables.append(name)

    def _get_variable_data(self):
        """
        Return array of variable data such that
          data = self._get_variable_data()
          data[0] == data for self._variables[0]
        """
        return np.transpose(self.value(self._variables)).reshape(len(self._variables), -1)

    def _get_variable_string(self):
        """
        Return string of variable data
        :return: str
        """
        return '\n'.join(self.string(self._variables))

    def _get_name(self, name):
        """
        Return corrected name from first scan
        :param name: str
        :return: str
        """
        try:
            name = self._scan_list[0].name(name)
        except (IndexError, KeyError):
            pass
        return name

    def array(self, name, array_length=None):
        data = self.__call__(name)
        if array_length is None:
            array_length = np.max([np.size(d) for d in data])
        return np.array([scan.array(name, array_length)[0] for scan in self._scan_list])

    def value(self, name):
        return [scan.value(name) for scan in self._scan_list]

    def string(self, name):
        out = []
        for scan in self._scan_list:
            strlist = np.asarray(scan.string(name), dtype=str).reshape(-1)
            out += [', '.join(s.strip() for s in strlist)]
        return out

    def string_format(self, command):
        return [scan.string_format(command) for scan in self._scan_list]

    def griddata(self, axes=None, signal='signal', repeat_after=None):
        """
        Generate 2D square grid of single values for each scan
        Return x, y axis when taking single values from each scan
        :param axes: str or list of str, names of axes data
        :param signal: str name of signal data
        :param repeat_after: int or None, defines repeat length of data
        :return: xaxis, yaxis, zaxis square [n,m] arrays
        """
        if axes is None:
            axes = self._variables
        else:
            axes = np.asarray(axes).reshape(-1)

        if len(axes) > 0:
            xaxis = self.value(axes[0])
        else:
            xaxis = np.arange(len(self._scan_list))

        yaxis = self.value(axes[-1])
        zaxis = self.value(signal)
        xaxis, yaxis, zaxis = fn.square_array(xaxis, yaxis, zaxis, repeat_after)
        return xaxis, yaxis, zaxis

    def get_plot_variable(self, yname, variable=None):
        """
        Get plotting data for plotting data points from each scan
         x, y, xlabel, ylabel = scans.get_plot_variable('signal', 'scanno')
        e.g.
         for n in range(len(x)):
            plt.plot(x[n], y[n])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        :param yname: str name of value to use as y-axis
        :param variable: str, list of str or None for default list of scans variables
        :return xdata: list of arrays for each scan
        :return ydata: list of arrays for each scan
        :return yerror: list of arrays for each scan
        :return labels: list of str for each scan
        :return xlabel: str, axis label for x-axis
        :return ylabel: str, axis label for y-axis
        """
        if variable is None:
            variable = []
        variables = list(np.asarray(variable, dtype=str).reshape(-1)) + self._variables
        xlabel = variables[0]
        xdata = self.value(xlabel)
        ylabel = yname
        ydata = self.value(yname)
        return xdata, ydata, xlabel, ylabel

    def get_plot_lines(self, xname=None, yname=None, signal_op=None, error_op=None):
        """
        Get plotting data for plotting as series of lines
         x, y, dy, labels, xlabel, ylabel = scans.get_plot_lines('axes', 'signal', '/Transmission', np.sqrt)
        e.g.
         for n in range(len(x)):
            plt.errorbar(x[n], y[n], dy[n])
        plt.legend(labels)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        :param xname: str name of value to use as x-axis
        :param yname: str name of value to use as y-axis
        :param signal_op: operation to perform on yaxis, e.g. '/Transmission'
        :param error_op: function to use on yaxis to generate error, e.g. np.sqrt
        :return xdata: list of arrays for each scan
        :return ydata: list of arrays for each scan
        :return yerror: list of arrays for each scan
        :return labels: list of str for each scan
        :return xlabel: str, axis label for x-axis
        :return ylabel: str, axis label for y-axis
        """
        xdata = []
        ydata = []
        dydata = []
        xlabel, ylabel = xname, yname
        for scan in self._scan_list:
            x, y, dy, xlabel, ylabel = scan.get_plot_data(xname, yname, signal_op, error_op)
            xdata += [x]
            ydata += [y]
            dydata += [dy]
        labels = self.__str__().splitlines()
        return xdata, ydata, dydata, labels, xlabel, ylabel

    def get_plot_mesh(self, xname=None, yname=None, signal_op=None, error_op=None):
        """
        Return array data for plotting as mesh
         x, y, z, xlabel, ylabel, zlabel = scans.get_plot_mesh('axes', 'signal', '/Transmission', np.sqrt)
        e.g.
         plt.pcolormesh(x, y, z)
         plt.xlabel(xlabel)
         plt.ylabel(ylabel)

        :param xname: str name of value to use as x-axis
        :param yname: str name of value to use as y-axis
        :param signal_op: operation to perform on yaxis, e.g. '/Transmission'
        :param error_op: function to use on yaxis to generate error, e.g. np.sqrt
        :return xdata: list of arrays for each scan
        :return ydata: list of arrays for each scan
        :return yerror: list of arrays for each scan
        :return labels: list of str for each scan
        :return xlabel: str, axis label for x-axis
        :return ylabel: str, axis label for y-axis
        """
        xname = self._get_name(xname)
        yname = self._get_name(yname)
        pass


