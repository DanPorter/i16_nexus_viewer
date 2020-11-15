"""
i16_nexus_viewer

Load hdf5 data files from I16 or other beamlines, combined with automated fitting and plotting

Requires: numpy, matplotlib, tkinter, h5py, lmfit, imageio

Usage:
    ***In Python***
    from i16_nexus_viewer import i16
    exp = i16.experiment('data/directory')
    scan = exp.loadscan(12345)
    scan.plot()
    fit = scan.fit(plot_result=True)

Usage:
    ***From Terminal***
    cd /location/of/file
    ipython -i -m -matplotlib tk i16_nexus_viewer gui

By Dan Porter, PhD
Diamond
2019

Version 0.2.0
Last updated: 15/11/20

Version History:
22/09/20 0.1.0  Version History started.
15/11/20 0.2.0  nexus_loader added which reloads nexus file on every operation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, VoigtModel, LinearModel

from .functions_nexus import load, reload, Hdf5Nexus
from .nexus_loader import HdfLoader
from .nexus_scan import Scan, MultiScans
from .nexus_experiment import Beamline, Experiment
from .fitting import peakfit

from . import functions_general as fg

__version__ = "0.2.0"
__date__ = "15/11/2020"

# Set up Beamline i16
i16 = Beamline('i16')


def start_gui(experiment=None, config_filename=None):
    """Start tkinter GUI"""
    print('not implemented yet')
    pass
    #from .tkgui import ExperimentGui
    #ExperimentGui(experiment, config_filename)
