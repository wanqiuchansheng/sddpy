from typing import List

from sddp.typedefinitions import *


class DefaultCutOracle(AbstractCutOracle):
    def __init__(self, cuts: List[Cut]=[]):
        self.cuts = cuts

    def storecut(self, m: 'SDDPModel', sp: 'Subproblem', cut: 'Cut'):
        self.cuts.append(cut)

    def validcuts(self):
        return self.cuts




