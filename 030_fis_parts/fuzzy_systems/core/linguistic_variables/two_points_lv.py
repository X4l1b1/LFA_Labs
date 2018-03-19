from fuzzy_systems.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable
from fuzzy_systems.core.membership_functions.lin_piece_wise_mf import LinPWMF
import numpy as np


class TwoPointsPDLV(LinguisticVariable):
    """
    Syntactic sugar for simplified linguistic variable with only 2 points (p1 and
    p2) and fixed labels ("low", and "high").


      ^
      |
    1 |XXXXXXXXX                 XXXXXXXXXXX
      |        XX               XX
      |         XXX            XX
      |           XXX        XX
      |             XXX    XXX
      |               XX  XX
      |               XXXXX
      |             XXX    XXX
      |          XX           XX
      |        XX              XXX
    0 +------------------------------------>
              P<------ d ------>

    """

    def __init__(self, name, p, d):
        dico = {"low": p, 'high': d}
        super(ThreePointsLV, self).__init__(name,dico)