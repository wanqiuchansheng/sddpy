#  Copyright 2017, Oscar Dowson
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################


from typing import Any, Tuple, List, TypeVar, Generic
import numpy as np
from sddp.typedefinitions import AbstractCutOracle, Cut

T = TypeVar('T')


class DynamicOracle(AbstractCutOracle, Generic[T]):
    def storecut(self, m: 'SDDPModel', sp: 'Subproblem', cut: 'Cut', price: T):
        pass

    def validcuts(self) -> List[Tuple[Cut, T]]:
        pass

    def __init__(self):
        pass


class DefaultDynamicOracle(DynamicOracle[T]):
    def __init__(self, cuts=None):
        if cuts is None:
            cuts = []
        self.cuts = cuts  # type: List[Tuple[Cut,T]]

    def storecut(self, m: 'SDDPModel', sp: 'Subproblem', cut: 'Cut', price: T):
        self.cuts.append((cut, price))

    def validcuts(self):
        return self.cuts


class NanniciniOracle(DynamicOracle):
    def __init__(self,
                 rho: int,
                 cutsinmodel: int,
                 cuts: List[Tuple[Cut, Any]],
                 iterations_since_last_active: List[int]):
        self.iterations_since_last_active = iterations_since_last_active
        self.cuts = cuts
        self.cutsinmodel = cutsinmodel
        self.rho = rho

    def storecut(self, m: 'SDDPModel', sp: 'Subproblem', cut: 'Cut', price: T):
        self.cuts.append((cut, price))
        self.iterations_since_last_active.append(0)
        self.cutsinmodel += 1
        idxs = [x for x in self.iterations_since_last_active if x < self.rho]
        if len(idxs) <= 0:
            raise RuntimeError("No cuts in model used in the last %d iterations." % self.rho)
        idx = idxs[0]
        self.cutsinmodel = len(self.cuts) - idx + 1
        return self.cuts[idx:]

    def validcuts(self):
        p = reversed(np.argsort(self.iterations_since_last_active)).tolist()
        self.cuts = [self.cuts[i] for i in p]
        self.iterations_since_last_active = [self.iterations_since_last_active[i] for i in p]
