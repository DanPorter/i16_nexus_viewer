"""
I16 Parameter file
"""

import json
import numpy as np
from i16_nexus_viewer.babelscan import Instrument

file_format = '%06d.nxs'
def error_function(x): return np.mean(x + 0.1)


# Names of datasets that will appear on print(scan)
str_list = ['scan_number', 'filename', 'scan_command', 'axes', 'signal',
            'i16_energy', 'i16_temperature', 'i16_hkl', 'ss', 'ds']


# These will be added as **kwargs in each scan
options = {
    'filename_format': file_format,
    'label_command': '#{scan_number:1.0f}',
    'title_command': '{FolderTitle} #{scan_number} {i16_energy} {i16_temperature} {i16_hkl}\n{scan_command}\n'
                     'ss = {ss}, ds = {ds}',
    'scan_command_name': 'scan_command',
    'exposure_time_name': ['measurement/count_time', 'measurement/counttime', 'measurement/Time', 'measurment/t'],
    'start_time_name': ['start_time', 'TimeSec'],
    'end_time_name': ['end_time', 'TimeSec'],
    'axes_name': ['axes', 'xaxis'],
    'signal_name': ['signal', 'yaxis'],
    'str_list': str_list,
    'signal_operation': '/Transmission/count_time/(rc/300.)',
    'error_function': error_function,
    'instrument': 'i16'
}

# these will be added to alternative_names dict
default_names = {
    # name in namespace or hdf address: [shortcuts],
    'en': ['incident_energy', 'Energy', 'energy'],
    'Ta': ['Temperature', 'temp'],
    'Tb': ['Temperature', 'temp'],
    'Tgas': ['Temperature', 'temp'],
    'delta_axis_offset': ['do'],
    'hkl': ['i16_hkl'],
    'counttime': ['count_time'],
    'Time': ['count_time'],
    't': ['count_time'],
}

# These will be added to the scan namespace using scan.string_format
default_formats = {
    'i16_temperature': '{Temperature:.3g}K',
    'i16_energy': '{incident_energy:.5g} keV',
    'ss': '[{s5xgap:4.2f},{s5xgap:5.2f}]',
    'ds': '[{s7xgap:4.2f},{s7xgap:5.2f}]',
    'hkl': '({h:.3g},{k:.3g},{l:.3g})',
    'euler': '{eta:.4g}, {chi:.4g}, {phi:.4g}, {mu:.4g}, {delta:.4g}, {gamma:.4g}',
}

# These will be added to the default_value dict and will be called if the name or alt_names can't be found
# this should contain all requested names in default_formats
default_values = {
    'scan_command': 'scan x y',
    'count_time': 1.0,
    'Transmission': 1.0,
    'en': 0.0,
    'Temperature': 300.0,
    's5xgap': 0,
    's5ygap': 0,
    's7xgap': 0,
    's7ygap': 0,
    'h': 0,
    'k': 0,
    'l': 0,
    'eta': 0,
    'chi': 0,
    'phi': 0,
    'mu': 0,
    'delta': 0,
    'gamma': 0,
    'atest': 'itworks!'
}

config_file = {
    'name': 'i16',
    'default_names': default_names,
    'formats': default_formats,
    'default_values': default_values,
    'options': options
}

#with open('i16.config', 'w') as f:
#    json.dump(config_file, f, sort_keys=True, indent=4)

i16 = Instrument(
    name='i16',
    default_names=default_names,
    formats=default_formats,
    default_values=default_values,
    options=options
)
