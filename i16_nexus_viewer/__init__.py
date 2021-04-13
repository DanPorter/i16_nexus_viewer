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

Version 0.5.0
Last updated: 13/04/21

Version History:
22/09/20 0.1.0  Version History started.
15/11/20 0.2.0  nexus_loader added which reloads nexus file on every operation
30/11/20 0.3.0  reformat functions_nexus and nexus_loader, add nroi
29/03/21 0.4.0  currently working using nexus ReLoader
13/04/21 0.5.0  Created omniscan scan wrapper, but nothing else works
"""

import os
import numpy as np
from lmfit.models import GaussianModel, VoigtModel, LinearModel

from .__settings__ import MATPLOTLIB_PLOTTING
from .functions_nexus import load, reload, Hdf5Nexus
from .nexus_loader import NexusLoader
from .nexus_scan import Scan, MultiScan
from .nexus_experiment import Beamline, Experiment
from .fitting import peakfit
from .omniscan import file_loader, FolderMonitor

from . import functions_general as fg

__version__ = "0.5.0"
__date__ = "13/04/2021"


# Set up Beamline i16
i16 = Beamline('i16')


def start_gui(experiment=None, config_filename=None):
    """Start tkinter GUI"""
    print('not implemented yet')
    pass
    #from .tkgui import ExperimentGui
    #ExperimentGui(experiment, config_filename)
