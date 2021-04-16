"""
Fitting functions using lmfit
"""

import os, sys, time, glob
import numpy as np
from lmfit.models import GaussianModel, VoigtModel, LinearModel  # fitting models


def peakfit(xvals, yvals, yerrors=None):
    """
    Fit peak to scans
    """

    peak_mod = VoigtModel()
    # peak_mod = GaussianModel()
    bkg_mod = LinearModel()

    pars = peak_mod.guess(yvals, x=xvals)
    pars += bkg_mod.make_params(intercept=np.min(yvals), slope=0)
    # pars['gamma'].set(value=0.7, vary=True, expr='') # don't fix gamma

    mod = peak_mod + bkg_mod
    out = mod.fit(yvals, pars, x=xvals, weights=yerrors)

    return out