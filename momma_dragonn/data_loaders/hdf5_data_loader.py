from __future__ import print_function
from .core import (BatchDataLoader_XYDictAPI, AtOnceDataLoader_XYDictAPI)
import h5py
import numpy as np
from avutils import util


class MultimodalBatchDataLoader(BatchDataLoader_XYDictAPI):

    def __init__(self, path_to_hdf5, **kwargs):
        self.path_to_hdf5 = path_to_hdf5
        self.f = h5py.File(self.path_to_hdf5)
        X = self.f['/X']
        Y = self.f['/Y']
        if ('weight' in self.f):
            weight = self.f['/weight']
        else:
            weight = {}
        super(MultimodalBatchDataLoader, self).__init__(
              X=X, Y=Y, weight=weight, **kwargs)
             
    def get_jsonable_object(self):
        the_dict = super(MultimodalBatchDataLoader, self).get_jsonable_object() 
        the_dict['path_to_hdf5'] = self.path_to_hdf5
        return the_dict 

 
class MultimodalAtOnceDataLoader(AtOnceDataLoader_XYDictAPI):

    def __init__(self, path_to_hdf5, **kwargs):
        self.path_to_hdf5 = path_to_hdf5
        self.f = h5py.File(self.path_to_hdf5)
        X = self.f['/X']
        Y = self.f['/Y']
        super(MultimodalAtOnceDataLoader, self).__init__(
            X_full=X, Y_full=Y, **kwargs)

    def get_jsonable_object(self):
        the_dict = super(MultimodalAtOnceDataLoader, self)\
                   .get_jsonable_object()
        the_dict['path_to_hdf5'] = self.path_to_hdf5
        return the_dict
