from typing import List

from fuzzy_systems.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable
from fuzzy_systems.core.membership_functions.lin_piece_wise_mf import LinPWMF
from fuzzy_systems.view.mf_viewer import MembershipFunctionViewer
from fuzzy_systems.view.viewer import Viewer


class LinguisticVariableViewer(Viewer):
    def __init__(self, lv, ax=None):
        """

        :type lv: LinguisticVariable
        """
        super(LinguisticVariableViewer, self).__init__(ax)
        self.__lv = lv

        self._viewers: List[MembershipFunctionViewer] = self.get_plot(self._ax)

    def fuzzify(self, in_value):
        [v.fuzzify(in_value) for v in self._viewers]
        return self

    def get_plot(self, ax):
        ax.set_title("MF: {}".format(self.__lv.name))
        ax.set_ylim([-0.1, 1.1])
        viewers = []
        for name in self.__lv.labels_name:
            mf = self.__lv[name]
            viewers.append(MembershipFunctionViewer(mf, label=name, ax=ax))
            # ax.set_title()
            # ax.scatter(mf.in_values, mf.mf_values, label=name)
        ax.legend(loc="best")
        return viewers

