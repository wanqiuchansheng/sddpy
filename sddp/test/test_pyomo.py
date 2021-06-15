#  Copyright 2017, Oscar Dowson, Zhao Zhipeng
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

import unittest
from pyomo.solvers.plugins.solvers.CPLEX import *
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core.base.var import *
from pyomo.core.base.param import SimpleParam
# from sddp.pyomo_tool import *
from pyomotools.tools import model_str


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.opt = SolverFactory('cplex',
                                 executable="/opt/ibm/ILOG/CPLEX_Studio128/cplex/bin/x86-64_linux/cplex")  # type:CPLEXSHELL
        self.model = ConcreteModel()

    def test_base(self):
        """
        min x**2
        """
        model = self.model
        model.x = Var(domain=Reals, bounds=(1, None))
        model.obj = Objective(expr=model.x ** 2, sense=minimize)
        result = self.opt.solve(model)
        self.assertEquals(value(model.x), 1)
        self.assertEquals(value(model.obj), 1)
        print(type(result))
        print(result.solver.status)
    def test_model_expresion(self):
        """
        min x**2
        """
        model = self.model
        model.x = Var(domain=Reals, bounds=(1, None))
        model.obj = Objective(expr=model.x ** 2, sense=minimize)
        result = self.opt.solve(model)
        print(model_str(model))




if __name__ == '__main__':
    unittest.main()
