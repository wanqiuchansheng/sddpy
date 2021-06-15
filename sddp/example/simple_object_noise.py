#  Copyright 2017, Oscar Dowson, Zhao Zhipeng
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################


import unittest

from pyomotools.tools import cplex
from sddp.SDDP import createSDDPModel


from sddp.typedefinitions import *

CplexSolver = cplex() # type:CPLEXSHELL


def solve_model(noise_probability:List[float]):
    def build(sp: Subproblem, t: int, markov_state: int):
        model = sp.model
        model.x = Var(domain=NonNegativeReals)
        model.x0 = Var(domain=NonNegativeReals)
        model.xp = Param(initialize=1.5, mutable=True)
        sp.add_state(model.x, model.x0, model.xp)
        # 变量
        model.u = Var(bounds=(0, 1))
        # 约束
        sp.anonymous = Constraint(expr=model.x == model.x0 - model.u)
        if t == 0:
            sp.obj = model.u * 2
        else:
            sp.anonymous = Param(mutable=True)
            p = sp.anonymous
            sp.obj = p * model.u
            sp.setnoiseprobability(noise_probability)

    m=createSDDPModel(sense=Sense.Max,stages=2,objective_bound=5,)




class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
