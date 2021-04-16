"""
Define the hdf5 Lazy loader class
"""

import os
import numpy as np
from h5py import Dataset


# Array methods to add to Dataset
METHODS = ['__str__']
# METHODS += ['__repr__', '__getitem__', '__index__']  # already in Dataset
METHODS += ['__abs__', '__add__', '__and__', '__concat__', '__contains__', '__delitem__', '__eq__', '__floordiv__',
            '__ge__', '__gt__', '__iadd__', '__iand__', '__iconcat__', '__ifloordiv__', '__ilshift__',
            '__imatmul__', '__imod__', '__imul__', '__inv__', '__invert__', '__ior__', '__ipow__',
            '__irshift__', '__isub__', '__itruediv__', '__ixor__', '__le__', '__lshift__', '__lt__', '__matmul__',
            '__mod__', '__mul__', '__ne__', '__neg__', '__not__', '__or__', '__pos__', '__pow__', '__rshift__',
            '__setitem__', '__spec__', '__sub__', '__truediv__', '__xor__']
METHODS += ['__radd__', '__rsub__', '__rmul__', '__rmatmul__', '__rtruediv__', '__rfloordiv__', '__rmod__',
            '__rdivmod__', '__rpow__', '__rlshift__', '__rrshift__', '__rand__', '__rxor__', '__ror__', ]


def dataset_decorator(cls):
    """
    Decorator to add additional functionality to Dataset, allowing it to behave as a numpy array
    The lazy nature of Dataset is kept and data is only retrieved when called for.
    :param cls: class
    :return: class
    """
    def add_method(method_name):
        def shadow_method(self, *args, **kwargs):
            return getattr(self.value(), method_name)(*args, **kwargs)
        setattr(cls, method_name, shadow_method)

    for method in METHODS:
        add_method(method)
    return cls


@dataset_decorator
class DatasetPlus(Dataset):
    """
    Shadow of h5py.Dataset type with added array methods
     - adds additional behaviours to h5py.Dataset including array addition and multiplication
     - interact with DatasetPlus like a numpy array
    The lazy nature of Dataset is kept and data is only retrieved when called for.

    Useage:
        data = DatasetPlus(hdf5_obj, address)
      hdf5_obj : A hdf5 File or Group object
      address : the str address of a dataset in this object
    """
    def __init__(self, hdf5_obj, dataset_address):
        self.File = hdf5_obj
        self.address = dataset_address
        self.basename = os.path.basename(dataset_address)
        dataset = hdf5_obj.get(dataset_address)
        super().__init__(dataset.id)

    def value(self):
        dataset = self.File.get(self.address)
        #dataset.refresh()
        return self.File.get(self.address)[()]
