#  Copyright 2017, Oscar Dowson, Zhao Zhipeng
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

import copy

from sddp.cut_oracles.DefaultCutOracle import DefaultCutOracle
from sddp.price_interpolation.discreate_distribution import *
from sddp.price_interpolation.price_interpolation import PriceInterpolationMethods


class StaticPriceInterpolation(PriceInterpolationMethods):
    def __init__(self,
                 initial_price,
                 location,
                 rib_locations: List,
                 variables: List[Var],
                 cutoracles,
                 noises: DiscreteDistribution,
                 objective,
                 dynamics,
                 bound: float):
        self.rib_locations = rib_locations
        self.location = location
        self.initial_price = initial_price
        self.variables = variables
        self.cutoracles = cutoracles  # type:List[AbstractCutOracle]
        self.noises = noises  # type:DiscreteDistribution
        self.objective = objective
        self.dynamics = dynamics
        self.bound = bound
        self._defalutcutoracle = None

    def set_default_cutoracle(self, cut_oracle):
        self._defalutcutoracle = cut_oracle

    def new_cutoracle_instence(self):
        return copy.deepcopy(self._defalutcutoracle)

    @staticmethod
    def create(cut_oracle=DefaultCutOracle(),
               dynamics=lambda p, w: p,
               initial_price: float = 0.0,
               rib_location=[0.0, 1.0],
               noise=DiscreteDistribution.create([0.0, 1.0])
               ):
        res = StaticPriceInterpolation(
            initial_price,
            initial_price,
            copy.deepcopy(rib_location),
            [],
            [],
            noise,
            None,  # TODO (p)->QuadExpr(p)
            dynamics,
            0.0
        )
        res.set_default_cutoracle(cut_oracle)
        return res

    def initializevaluefunction(self, sp: 'Subproblem', sense: Sense, bound: Reals) -> AbstractValueFunction:
        self.bound = bound
        for r in self.rib_locations:
            sp.anonymous = futureobjective(sense, bound)
            self.variables.append(sp.anonymous)
            self.cutoracles.append(self.new_cutoracle_instence())
        return self

    def interpolate(self) -> Expression:
        # y = AffExpr(0.0)  截距
        vf = self
        y = 0
        if len(vf.rib_locations) == 1:
            y.append(vf.variables[0])
        else:
            upper_idx = len(vf.rib_locations)
            for i in range(2, len(vf.rib_locations) + 1):
                if vf.location <= vf.rib_locations[i - 1]:
                    upper_idx = i
                    break
            lower_idx = upper_idx - 1
            lamb = (vf.location - vf.rib_locations[lower_idx - 1]) / (
                    vf.rib_locations[upper_idx - 1] - vf.rib_locations[lower_idx - 1])
            if lamb < -1e-6 or lamb > 1.0 + 1e-6:
                raise RuntimeError(
                    "The location %s is outside the interpolated region. lambda = %s" % (vf.location, lamb))
            y = y + vf.variables[lower_idx - 1] * (1 - lamb)
            y = y + vf.variables[upper_idx - 1] * lamb
        return y

    @property
    def cutoracle(self) -> AbstractCutOracle:
        return self.cutoracle


    def addcuttoPyomoModel(self, sp: Subproblem, theta: Var, cut: Cut):
        cut_expr = cut.intercept  # type: Expression
        for c, s in zip(cut.coefficients, self.states):
            cut_expr = c * s.variable + cut_expr
        if sp.sense == Sense.Min:
            sp.add_constraint(expr=theta >= cut_expr)
        else:
            sp.add_constraint(expr=theta <= cut_expr)

    def rebuildsubproblem(self, m: 'SDDPModel', sp: 'Subproblem'):
        vf = self
        # vf = sp.valueoracle  # type:# StaticPriceInterpolation
        sp.states.clear()
        sp.noises.clear()
        sp.reset_mod()

        vf.variables.clear()
        for r in vf.rib_locations:
            sp.anonymous = futureobjective(sp.sense, sp.problembound)
            vf.variables.append(sp.anonymous)

        m.build(sp, sp.stage, sp.markov_state)
        for i in range(len(vf.variables)):
            for cut in vf.cutoracles[i]:
                vf.addcuttoPyomoModel(sp, vf.variables[i], cut)
        m.stages[sp.stage].subproblems[sp.markov_state] = sp

    def updatevaluefunction(self, m: SDDPModel, settings: Settings, t: int, sp: Subproblem):
        vf = self  # vf = sp.valueoracle  # type:StaticPriceInterpolation
        for i, (rib, theta, cutoracle) in enumerate(zip(vf.rib_locations, vf.variables, vf.cutoracles)):
            cut = self.constructcut(m, sp, t, rib)  # 进行了计算
            # TODO write cut
            cutoracle.storecut(m, sp, cut)
            vf.addcuttoPyomoModel(sp, theta, cut)
            if settings.is_asyncronous:  # TODO
                pass