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

Version 0.6.1
Last updated: 20/04/21

Version History:
22/09/20 0.1.0  Version History started.
15/11/20 0.2.0  nexus_loader added which reloads nexus file on every operation
30/11/20 0.3.0  reformat functions_nexus and nexus_loader, add nroi
29/03/21 0.4.0  currently working using nexus ReLoader
13/04/21 0.5.0  Created babelscan scan wrapper, but nothing else works
16/04/21 0.6.0  Basic i16 specific file loading now works
20/04/21 0.6.1  Updates to babelscan, alternative names and default values added
"""

from .babelscan import file_loader, hdf_loader, FolderMonitor, Instrument
from .babelscan import plotting_matplotlib as pm  # Plotting functions
from .beamlines import beamlines
from . import functions_general as fg

__version__ = "0.6.1"
__date__ = "20/04/2021"


def experiment(data_folder, working_dir='.', **kwargs):
    return FolderMonitor(data_folder, working_dir, **kwargs)


# I16 Instrument
i16 = beamlines['i16']


def start_gui(experiment=None, config_filename=None):
    """Start tkinter GUI"""
    print('not implemented yet')
    pass
    #from .tkgui import ExperimentGui
    #ExperimentGui(experiment, config_filename)
