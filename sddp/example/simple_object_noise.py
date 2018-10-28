import unittest

from sddp.SDDP import createSDDPModel
from sddp.typedefinitions import *

CplexSolver = SolverFactory('cplex',
                            executable="/opt/ibm/ILOG/CPLEX_Studio128/cplex/bin/x86-64_linux/cplex")  # type:CPLEXSHELL


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
