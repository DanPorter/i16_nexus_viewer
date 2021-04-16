"""
I16 Parameter file
"""

import numpy as np
from i16_nexus_viewer.babelscan import Instrument

file_format = '%06d.nxs'
error_function = lambda x: np.mean(x+0.1)


# These will be added as **kwargs in each scan
options = {
    'label_command': '#{scan_number}',
    'title_command': '{FolderTitle} #{scan_number} {en:1.5g}keV {i16_temperature:.3g}K {i16_hkl}',
    'scan_command_name': 'scan_command',
    'exposure_time_name': ['measurement/count_time', 'measurement/counttime', 'measurement/Time', 'measurment/t'],
    'start_time_name': 'TimeSec',
    'end_time_name': 'TimeSec',
    'axes_name': 'axes',
    'signal_name': 'signal',
    'list_str': ['en', 'i16_temp', 'i16_hkl'],
    'signal_operation': '/Transmission/count_time/(rc/300.)',
    'error_function': error_function,
    'instrument': 'i16'
}


# these will be added to the namespace and alternative_names dict
default_names = {
    # name in namespace/ hdf address: [shortcuts],
    'incident_energy': ['i16_energy', 'en', 'Energy', 'energy'],
    'Ta': ['i16_temperature', 'Temperature', 'temp'],
    'delta_axis_offset': ['do'],
    'hkl': ['i16_hkl']
}


# These will be added to the scan namespace using scan.string_format
default_formats = {
    'ss': '[{s5xgap:4.2f},{s5xgap:5.2f}]',
    'ds': '[{s7xgap:4.2f},{s7xgap:5.2f}]',
    'hkl': '({h:.3g},{k:.3g},{l:.3g})',
    'euler': '{eta:.4g}, {chi:.4g}, {phi:.4g}, {mu:.4g}, {delta:.4g}, {gamma:.4g}',

}


# These operations will be run on loading the file, adding a name and value to the namespace
# these will make loading the file much slower
def count_time(scan):
    trial = ['entry1/measurement/count_time', 'entry1/measurement/counttime',
             'entry1/measurement/Time', 'entry1/measurement/t']

    count_time_address = None
    exposure = 1.0
    with scan.load() as hdf:
        for address in trial:
            if address in hdf:
                count_time_address = address
                exposure = hdf[address][0]
                break
    if count_time_address is None:
        print('Warning: Scan has no count time: %r' % scan)
    scan.add2namespace('count_time', exposure, trial, count_time_address)


functions = [count_time]


# Values that appear on print(scan)
list_str = ['scan_number', 'filename', 'scan_command', 'i16_energy', 'i16_temperature']


i16 = Instrument(
    name='i16',
    default_names=default_names,
    formats=default_formats,
    functions=functions,
    options=options,
    str_list=list_str,
    filename_format=file_format
)

