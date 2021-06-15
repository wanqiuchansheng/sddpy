#  Copyright 2017, Oscar Dowson
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################



from sddp.cut_oracles.DefaultCutOracle import DefaultCutOracle
from sddp.SDDP import createSDDPModel


from sddp.riskmeasures import Expectation
from sddp.typedefinitions import *

solver = SolverFactory('gurobi')


class Turbine:
    def __init__(self, flowknots: List[float], powerknots: List[float]):
        self.powerknots = powerknots
        self.flowknots = flowknots

    @property
    def nrange(self):
        return range(len(self.flowknots))


class Reservoir:
    def __init__(self, min, max, initial, turbine, spill_cost, inflows):
        self.inflows = inflows  # type:List[float]
        self.spill_cost = spill_cost  # type:float
        self.turbine = turbine  # type:Turbine
        self.initial = initial  # type:float
        self.max = max  # type:float
        self.min = min  # type:float


def hydrovalleymodel(
        riskmeasure=Expectation(),
        cutoracle=DefaultCutOracle(),
        hasstagewiseinflows: bool = True,
        hasmarkovprice: bool = True,
        sense=Sense.Max):
    valley_chain = [
        Reservoir(0, 200, 200, Turbine([50, 60, 70], [55, 65, 70]), 1000, [0, 20, 50]),
        Reservoir(0, 200, 200, Turbine([50, 60, 70], [55, 65, 70]), 1000, [0, 0, 20])
    ]

    def turbine(r: int):
        return valley_chain[r].turbine

    prices = [
        [1, 2, 0],
        [2, 1, 0],
        [3, 4, 0],
    ]
    if hasmarkovprice:
        transition = [
            [[1.0]],
            [[0.6, 0.4]],
            [[0.6, 0.4, 0.0], [0.3, 0.7, 0.0]]
        ]
    else:
        transition = [[[1]],
                      [[1]],
                      [[1]]]

    flipobj = 1 if sense == Sense.Max else -1
    N = len(valley_chain)

    def build(sp: Subproblem, stage: int, markov_state: int):
        model = sp.model
        model.N = list(range(N))

        # 设置状态
        model.reservoir = Var(model.N, bounds=lambda m, i: (valley_chain[i].min, valley_chain[i].max))
        model.reservoir0 = Var(model.N)
        model.rp = Param(model.N, initialize=lambda m, i: valley_chain[i].initial, mutable=True)
        # model.rc = Constraint(model.N, rule=lambda m, i: model.reservoir0[i] == model.rp[i])
        sp.add_state(model.reservoir, model.reservoir0, model.rp)

        #   Additional variables
        model.outflow = Var(model.N, domain=NonNegativeReals)
        model.spill = Var(model.N, domain=NonNegativeReals)
        model.inflow = Var(model.N, domain=NonNegativeReals)
        model.generation_quantity = Var(domain=NonNegativeReals)
        model.dispatch = Var(model.N, turbine(0).nrange, bounds=lambda m, r, level: (0, 1))

        # Constraints
        # 水量平衡方程

        def _bl(m, r):
            r = value(r)
            if r == 0:
                return m.reservoir[r] == m.reservoir0[r] + m.inflow[r] - m.outflow[r] - m.spill[r]
            else:
                return m.reservoir[r] == m.reservoir0[r] + m.inflow[r] - m.outflow[r] - m.spill[r] + m.spill[r - 1] + \
                       m.outflow[r - 1]

        sp.anonymous = Constraint(model.N, rule=_bl)
        sp.anonymous = Constraint(expr=sum(
            turbine(r).powerknots[level] * model.dispatch[r, level] for r in model.N for level in turbine(0).nrange) ==
                                       model.generation_quantity)
        sp.anonymous = Constraint(model.N, rule=lambda model, i: model.outflow[i] == sum(
            turbine(i).flowknots[level] * model.dispatch[i, level] for level in turbine(0).nrange
        ))
        sp.anonymous = Constraint(model.N,
                                  rule=lambda model, r: sum(
                                      model.dispatch[r, level] for level in turbine(r).nrange) <= 1)

        model.blocks = Block(model.N)
        for i in model.N:
            if hasstagewiseinflows and stage > 0:
                model.blocks[i].rainfall = Param(mutable=True)
                model.blocks[i].rf = Constraint(expr=model.inflow[i] <= model.blocks[i].rainfall)
                sp.add_noise_constraint(valley_chain[i].inflows, model.blocks[i].rainfall)
            else:
                model.blocks[i].rf = Constraint(expr=model.inflow[i] <= valley_chain[i].inflows[0])

        if hasmarkovprice:
            sp.obj = flipobj * (prices[stage][markov_state] * model.generation_quantity - sum(
                valley_chain[i].spill_cost * model.spill[i] for i in model.N))
        else:
            sp.obj=flipobj * (prices[stage][0] * model.generation_quantity - sum(
                valley_chain[i].spill_cost * model.spill[i] for i in model.N))

    m = createSDDPModel(build,
                        sense=sense,
                        stages=3,
                        objective_bound=flipobj * 1e6,
                        markov_transition=transition,
                        risk_measure=riskmeasure,
                        cut_oracle=cutoracle,
                        solver=solver
                        )
    # print(model_exp)
    return m
