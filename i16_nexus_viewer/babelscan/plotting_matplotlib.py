"""
Matplotlib plotting functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D plotting


DEFAULT_FONT = 'Times New Roman'
FIG_SIZE = [8, 6]
FIG_DPI = 100


# Setup matplotlib rc parameters
# These handle the default look of matplotlib plots
plt.rc('figure', figsize=FIG_SIZE, dpi=FIG_DPI, autolayout=False)
plt.rc('lines', marker='o', color='r', linewidth=2, markersize=6)
plt.rc('errorbar', capsize=2)
plt.rc('legend', loc='best', frameon=False, fontsize=16)
plt.rc('axes', linewidth=2, titleweight='bold', labelsize='large')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')
plt.rc('axes.formatter', limits=(-3, 3), offset_threshold=6)
# Note font values appear to only be set when plt.show is called
plt.rc('font', family='serif', style='normal', weight='bold', size=12, serif=['Times New Roman', 'Times', 'DejaVu Serif'])
#plt.rcParams["savefig.directory"] = os.path.dirname(__file__) # Default save directory for figures
#plt.rcdefaults()


'----------------------------Plot manipulation--------------------------'


def labels(ttl=None, xvar=None, yvar=None, zvar=None, legend=False,
           colorbar=False, colorbar_label=None,
           axes=None, size='Normal'):
    """
    Add formatted labels to current plot, also increases the tick size
    :param ttl: title
    :param xvar: x label
    :param yvar: y label
    :param zvar: z label (3D plots only)
    :param legend: False/ True, adds default legend to plot
    :param colorbar: False/ True, adds default colorbar to plot
    :param colorbar_label: adds label to colorbar
    :param axes: matplotlib axes to use, None for plt.gca()
    :param size: 'Normal' or 'Big'
    :param font: str font name, 'Times New Roman'
    :return: None
    """
    if axes is None:
        axes = plt.gca()

    if size.lower() in ['big', 'large', 'xxl', 'xl']:
        tik = 30
        tit = 32
        lab = 35
    else:
        # Normal
        tik = 16
        tit = 12
        lab = 14

    if ttl is not None:
        axes.set_title(ttl, fontsize=tit, fontweight='bold')

    if xvar is not None:
        axes.set_xlabel(xvar, fontsize=lab)

    if yvar is not None:
        axes.set_ylabel(yvar, fontsize=lab)

    if zvar is not None:
        # Don't think this works, use ax.set_zaxis
        axes.set_zlabel(zvar, fontsize=lab)

    if legend:
        axes.legend()

    if colorbar:
        mappables = axes.images + axes.collections
        cb = plt.colorbar(mappables[0], ax=axes)
        if colorbar_label:
            cb.set_ylabel(colorbar_label, fontsize=lab)


def colormap(clim=None, cmap=None, axes=None):
    """
    Set colour limits and colormap on axes
    :param clim: [min, max] color cut-offs
    :param cmap: str name of colormap
    :param axes: matplotlib axes or None for current axes
    :return: None
    """
    if axes is None:
        axes = plt.gca()

    # Get axes images
    mappables = axes.images + axes.collections
    for image in mappables:
        if cmap:
            image.set_cmap(plt.get_cmap(cmap))
        if clim:
            image.set_clim(clim)


def saveplot(name, dpi=None, figure_number=None):
    """
    Saves current figure as a png in the home directory
    :param name: filename, including or expluding directory and or extension
    :param dpi: image resolution, higher means larger image size, default=matplotlib default
    :param figure_number: figure number, default = plt.gcf()
    :return: None

    E.G.
    ---select figure to save by clicking on it---
    saveplot('test')
    E.G.
    saveplot('c:\somedir\apicture.jpg', dpi=600, figure=3)
    """
    if figure_number is None:
        gcf = plt.gcf()
    else:
        gcf = plt.figure(figure_number)

    filedir = os.path.dirname(name)
    file, ext = os.path.splitext(name)

    if filedir is None:
        filedir = os.path.expanduser('~')

    if len(ext) == 0:
        ext = '.png'

    savefile = os.path.join(filedir, file + ext)
    gcf.savefig(savefile, dpi=dpi)
    print('Saved Figure {} as {}'.format(gcf.number, savefile))


def create_axes(fig=None, subplot=111, *args, **kwargs):
    """
    Create new plot axis
    ax = create_axes(subplot=111)

    for 3D plot, use: create_axes(projection='3d')

    :param fig: matplotlib figure object, or None to create Figure
    :param subplot: subplot input
    :param *args, **kwargs: pass additional argments to fig.add_subplot
    :return: axes object
    """
    if fig is None:
        fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    ax = fig.add_subplot(subplot, *args, **kwargs)
    return ax


def plot_line(axes, xdata, ydata, yerrors=None, line_spec='-o', *args, **kwargs):
    """
    Plot line on given matplotlib axes subplot
    Uses matplotlib.plot or matplotlib.errorbar if yerrors is not None
    :param axes: matplotlib figure or subplot axes, None uses current axes
    :param xdata: array data on x axis
    :param ydata: array data on y axis
    :param yerrors: array errors on y axis (or None)
    :param line_spec: str matplotlib.plot line_spec
    :param args: additional arguments
    :param kwargs: additional arguments
    :return: output of plt.plot [line], or plt.errorbar [line, xerrors, yerrors]
    """
    if axes is None:
        axes = plt.gca()

    if yerrors is None:
        lines = axes.plot(xdata, ydata, line_spec, *args, **kwargs)
    else:
        lines = axes.errorbar(xdata, ydata, yerrors, *args, fmt=line_spec, **kwargs)
    return lines


def plot_detector_image(axes, image, clim=None, *args, **kwargs):
    """
    Plot detector image
    :param axes: matplotlib figure or subplot axes, None uses current axe
    :param image: 2d array image data
    :param clim: None or [min, max] values for color cutoff
    :param args: additional arguments for plt.pcolormesh
    :param kwargs: additional arguments for plt.pcolormesh
    :return: axes object
    """
    if axes is None:
        axes = plt.gca()

    if 'shading' not in kwargs.keys():
        kwargs['shading'] = 'gouraud'
    if clim:
        kwargs['vmin'] = clim[0]
        kwargs['vmax'] = clim[1]

    axes.pcolormesh(image, *args, **kwargs)
    axes.invert_yaxis()
    axes.axis('image')
    return axes


def plot_2d_surface(axes, image, xdata=None, ydata=None, clim=None, axlim='image', **kwargs):
    """
    Plot 2D data as colourmap surface
    :param axes: matplotlib figure or subplot axes, None uses current axe
    :param image: 2d array image data
    :param xdata: array data, 2d or 1d
    :param ydata: array data 2d or 1d
    :param clim: None or [min, max] values for color cutoff from plt.clim
    :param axlim: axis limits from plt.axis
    :param kwargs: additional arguments for plt.pcolormesh
    :return: output of plt.pcolormesh
    """
    if axes is None:
        axes = plt.gca()

    if 'shading' not in kwargs.keys():
        kwargs['shading'] = 'gouraud'
    if clim:
        kwargs['vmin'] = clim[0]
        kwargs['vmax'] = clim[1]
    if np.ndim(xdata) == 1 and np.ndim(ydata) == 1:
        ydata, xdata = np.meshgrid(ydata, xdata)

    if xdata:
        surface = axes.pcolormesh(xdata, ydata, image, **kwargs)
    else:
        surface = axes.pcolormesh(image, **kwargs)
    axes.axis(axlim)
    return surface


def plot_3d_surface(axes, image, xdata=None, ydata=None, samples=None, clim=None, axlim='auto', **kwargs):
    """
    Plot 2D image data as 3d surface
    :param axes: matplotlib figure or subplot axes, None uses current axe
    :param image: 2d array image data
    :param xdata: array data, 2d or 1d
    :param ydata: array data 2d or 1d
    :param samples: max number of points to take in each direction, by default does not downsample
    :param clim: None or [min, max] values for color cutoff from plt.clim
    :param axlim: axis limits from plt.axis
    :param kwargs: additional arguments for plt.plot_surface
    :return: output of plt.plot_surface
    """
    if axes is None:
        axes = plt.gca()

    if samples:
        kwargs['rcount'] = samples
        kwargs['ccount'] = samples
    else:
        # default in plot_surface is 50
        kwargs['rcount'],  kwargs['ccount'] = np.shape(image)
    if clim:
        kwargs['vmin'] = clim[0]
        kwargs['vmax'] = clim[1]

    if np.ndim(xdata) == 1 and np.ndim(ydata) == 1:
        ydata, xdata = np.meshgrid(ydata, xdata)

    if xdata:
        surface = axes.plot_surface(xdata, ydata, image, **kwargs)
    else:
        surface = axes.plot_surface(image, **kwargs)
    axes.axis(axlim)
    return surface

