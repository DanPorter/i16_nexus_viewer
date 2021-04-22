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


"----------------------------------------------------------------------------------------------------------------------"
"------------------------------------------------ ScanFitManager ------------------------------------------------------"
"----------------------------------------------------------------------------------------------------------------------"


class ScanFitManager:
    """
    ScanFitManager
    :param scan: babelscan.Scan
    """

    def __init__(self, scan):
        self.scan = scan

    def __call__(self, *args, **kwargs):
        return self.fit(*args, **kwargs)

    def fit(self, xaxis='axes', yaxis='signal', fit_type=None, print_result=True, plot_result=False):
        """
        Automatic fitting of scan

        Use LMFit
        Pass fit_type = LMFit model
        return LMFit output
        """
        xdata, ydata, yerror, xname, yname = self.scan.get_plot_data(xaxis, yaxis, None, None)

        # lmfit
        out = peakfit(xdata, ydata, yerror)

        self.scan.add2namespace('lmfit', out, 'fit_result')
        fit_dict = {}
        for pname, param in out.params.items():
            ename = 'stderr_' + pname
            fit_dict[pname] = param.value
            fit_dict[ename] = param.stderr
        for name, value in fit_dict.items():
            self.scan.add2namespace(name, value)
        self.scan.add2namespace('fit', out.best_fit, other_names=['fit_%s' % yname])

        if print_result:
            print(self.scan.title())
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
        if 'lmfit' not in self.scan._namespace:
            self.fit()
        lmfit = self.scan('lmfit')
        if parameter_name is None:
            return lmfit
        param = lmfit.params[parameter_name]
        return param.value, param.stderr
