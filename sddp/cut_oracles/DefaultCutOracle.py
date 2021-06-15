#  Copyright 2017, Oscar Dowson
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################


from typing import List

from sddp.typedefinitions import *


class DefaultCutOracle(AbstractCutOracle):
    def __init__(self, cuts: List[Cut]=[]):
        self.cuts = cuts

    def storecut(self, m: 'SDDPModel', sp: 'Subproblem', cut: 'Cut'):
        self.cuts.append(cut)

    def validcuts(self):
        return self.cuts




