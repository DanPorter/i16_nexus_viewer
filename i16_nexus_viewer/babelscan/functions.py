"""
Module of General functions
"""

import os
import re
import datetime
import numpy as np


BYTES_DECODER = 'utf-8'
VALUE_FUNCTION = np.mean  # lambda a: np.asarray(a).reshape(-1)[0]
VALUE_FORMAT = '%.5g'
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"
OUTPUT_FORMAT = '%20s = %s'

# Compile useful pattern strings
re_integer = re.compile(r'\d+')
re_address = re.compile(r'\w*?/[\w/]*')
re_nroi = re.compile(r'nroi.*\[[\d\s,]+\]')  # catch nroi[31, 31], nroi_bkg[243,97,31,41]
re_varname = re.compile(r'[a-zA-Z]\w*\[.+\]|[a-zA-Z]\w*')  # catch ab5, ab[31, 31]
re_strop = re.compile(r'\{(.+?)\}')


def file2name(filename):
    """
    Extracts filename from file
    :param filename: str
    :return: str
    """
    return os.path.basename(filename)


def scanfile2number(filename):
    """
    Extracts the scan number from a .nxs filename
    :param filename: str : filename of .nxs file
    :return: int : scan number
    """
    nameext = os.path.split(filename)[-1]
    name = os.path.splitext(nameext)[0]
    numbers = re_integer.findall(name)
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


def bytestr2str(string):
    """
    Convert bytestr or str to str
    :param string: Bytes or str
    :return: str
    """
    return np.asarray(string, dtype=str)[()]


def liststr(string):
    """
    Convert str or list of str to list of str
    :param string: str, byteString, list, array
    :return: list of str
    """
    return list(np.asarray(string, dtype=str).reshape(-1))


def data_string(data):
    """
    Return string depiction of data
    :param data: any
    :return: str
    """
    size = np.size(data)
    shape = np.shape(data)
    if size == 1:
        if issubclass(type(data), dict):
            return "%s" % data
        return shortstr(str(data))
    try:
        amax = np.max(data)
        amin = np.min(data)
        amean = np.mean(data)
        out_str = "%s max: %4.5g, min: %4.5g, mean: %4.5g"
        return out_str % (shape, amax, amin, amean)
    except TypeError:
        # list of str
        array = np.asarray(data).reshape(-1)
        array_start = array[0]
        array_end = array[-1]
        out_str = "%s [%s, ..., %s]"
        return out_str % (shape, array_start, array_end)


def data_datetime(data, date_format=None):
    """
    Convert date string to datetime object
      datetime_array = date_datetime('2020-10-22T09:33:11.894+01:00', "%Y-%m-%dT%H:%M:%S.%f%z")
     datetime_array[0] will give first time
     datetime_array[-1] will give last time
    :param data: str or list of str
    :param date_format: str format used in datetime.strptime (see https://strftime.org/)
    :return: list of datetime
    """
    if date_format is None:
        date_format = DATE_FORMAT

    data = liststr(data)
    try:
        # str date passed, e.g. start_time: '2020-10-22T09:33:11.894+01:00'
        dates = np.array([datetime.datetime.strptime(date, date_format) for date in data])
    except ValueError:
        # float timestamp passed, e.g. TimeFromEpoch: 1603355594.96
        dates = np.array([datetime.datetime.fromtimestamp(float(time)) for time in data])
    return dates


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


def axis_repeat(*args):
    """
    Determine the repeating pattern in a or several axes
    :param args: [n,] length array
    :return: repeat_length
    """

    rep_len = []
    for arg in args:
        delta = np.abs(np.diff(arg))
        ch_idx = np.append(-1, np.where(delta > delta.max() * 0.9))  # find biggest changes
        ch_delta = np.diff(ch_idx)
        rep_len += [np.round(np.mean(ch_delta))]
    return int(max(rep_len))


def square_array(xaxis, yaxis, zaxis=None, repeat_length=None):
    """
    Reshape 1D x/y axes into 2D axes, estimating the repeat length
    :param xaxis: [n,] list or array
    :param yaxis: [n,] list or array
    :param zaxis: [n,] list or array, or None
    :param repeat_length: int m, value to reshape array (None to determine automatically)
    :return: xaxis, yaxis, [zaxis] [n//m, m] array
    """
    xaxis = np.asarray(xaxis).reshape(-1)
    yaxis = np.asarray(yaxis).reshape(-1)

    # Determine the repeat length of the scans
    if repeat_length is None:
        repeat_length = axis_repeat(xaxis, yaxis)

    # Reshape into square arrays
    # If this is problematic, look at scipy.interpolate.griddata
    xaxis = xaxis[:repeat_length * (len(xaxis) // repeat_length)].reshape(-1, repeat_length)
    yaxis = yaxis[:repeat_length * (len(yaxis) // repeat_length)].reshape(-1, repeat_length)
    if zaxis is not None:
        zaxis = np.asarray(zaxis).reshape(-1)
        zaxis = zaxis[:repeat_length * (len(zaxis) // repeat_length)].reshape(-1, repeat_length)
        return xaxis, yaxis, zaxis
    return xaxis, yaxis


def time_difference(start_time, end_time=None):
    """
    Return time difference between first and last time
    :param start_time: str or list
    :param end_time: None or str
    :return: datetime.timedelta
    """
    start_time = data_datetime(start_time)[0]
    if end_time is None:
        end_time = data_datetime(start_time)[-1]
    else:
        end_time = data_datetime(end_time)[0]
    time_delta = end_time - start_time
    return time_delta

