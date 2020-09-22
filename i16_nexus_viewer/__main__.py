"""
i16_nexus_viewer
Python package for loading nexus files

Usage:
    ***From Terminal***
    cd /location/of/file
    ipython -i -m --matplotlib tk i16_nexus_viewer
    >> exp = dnex.Experiment(['path/to/data'])
    >> scan = exp.loadscan(12345)
    >> print(scan.info())
    >> scan.plot()

For GUI use:
    ipython -m Dans_NexusLoader gui

By Dan Porter, PhD
Diamond
2020
"""
if __name__ == '__main__':

    import sys, os
    import numpy as np
    import matplotlib.pyplot as plt
    import i16_nexus_viewer as nex16

    print('\ni16_nexus_viewer version %s, %s\n By Dan Porter, Diamond Light Source Ltd.'%(nex16.__version__, nex16.__date__))
    print('See help(nex16.Experiment) for info, or nex16.start_gui() to get started!')

    for arg in sys.argv:
        if 'nxs' in arg.lower():
            dirname, filename = os.path.split(arg)
            exp = nex16.Experiment([dirname])
            print(exp)
            print('\n------')
            scan = exp.loadscan(filename=filename)
            print(scan)
            print('------')
        elif 'gui' in arg.lower():
            nex16.start_gui()

