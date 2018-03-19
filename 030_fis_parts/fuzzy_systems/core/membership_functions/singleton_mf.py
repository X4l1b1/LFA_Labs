import numpy as np

from fuzzy_systems.core.membership_functions.free_shape_mf import FreeShapeMF

class SingletonMF(FreeShapeMF):
    def __init__(self, x):
        """
        TrapMF is a trapezoidal membership function that takes in parameters:
        - p0 as the starting point
        - p1 as the second point
        - p2 as the third point
        - p3 as the ending point
        """

        in_values = [x]
        mf_values = [1]
        super(SingletonMF, self).__init__(in_values, mf_values)

    def fuzzify(self, in_value):
        assert (in_value == self.in_values[0]), "Invalid in_value"
        return np.interp(in_value, self._in_values, self._mf_values)