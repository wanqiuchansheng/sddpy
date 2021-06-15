#  Copyright 2017, Oscar Dowson, Zhao Zhipeng
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

from copy import deepcopy
from sddp.SDDP import solvesubproblem
from sddp.state import setstates
from sddp.cut_oracles.DefaultCutOracle import DefaultCutOracle
from .utilities import *

T = TypeVar('T')


class DefaultValueFunction(AbstractValueFunction[T]):
    def __init__(self, cutmanager: T = DefaultCutOracle()):
        self._theta = None  # value function 的值
        self._cutmanager = cutmanager

    @property
    def cutoracle(self) -> AbstractCutOracle:
        return self._cutmanager

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value

    def initializevaluefunction(self, sp: 'Subproblem', sense: Sense, bound: float):
        if sense == Sense.Min:
            self._theta = Var(bounds=(bound, None))
        else:
            self._theta = Var(bounds=(None, bound))
        sp.model.theta = self._theta


    @staticmethod
    def backwardpass(m: 'SDDPModel', setting: 'Settings'):
        for t in reversed(range(1, m.nstages)):  # 倒序
            m.storage.reset()
            for sp in m.stages[t].subproblems:
                setstates(m, sp)
                solvesubproblem(Direction.backwardpass, m, sp)
            for sp in m.stages[t - 1].subproblems:
                modifyvaluefunction(m, setting, sp)
        m.storage.reset()  # TODO 重新设置
        for sp in m.stages[0].subproblems:
            solvesubproblem(Direction.backwardpass, m, sp)
            # print(model_expression(sp.model))
        return float(np.dot(m.storage.objective, m.storage.probability))

    # @staticmethod
    def addcut(self, m: SDDPModel, sp: Subproblem, cut: Cut):
        # vf = sp.valueoracle  # type:DefaultValueFunction
        vf = self
        vf.cutoracle.storecut(m, sp, cut)
        vf.addcuttoPyomoModel(sp, cut)

    def addcuttoPyomoModel(self, sp: Subproblem, cut: Cut):
        cut_expr = cut.intercept  # type: Expression
        vf = self
        for c, s in zip(cut.coefficients, sp.states):
            cut_expr = c * s.variable + cut_expr
        if sp.sense == Sense.Min:
            sp.add_constraint(expr=vf.theta >= cut_expr)
        else:
            sp.add_constraint(expr=vf.theta <= cut_expr)

    def rebuildsubproblem(self, m: SDDPModel, sp: Subproblem):
        """
        重新构建子问题
        """
        vf = self  # type:DefaultValueFunction
        sp.states.clear()
        sp.noises.clear()
        sp.reset_mod()

        # if sp.sense == Sense.Max:
        #     sp.model.theta = Var(domain=Reals, bounds=(None, sp.problembound))
        # else:
        #     sp.model.theta = Var(domain=Reals, bounds=(sp.problembound, None))
        vf.theta = sp.model.theta
        m.build(sp, sp.stage, sp.markov_state)
        for cut in vf.cutoracle.validcuts():
            vf.addcuttoPyomoModel(sp, cut)
        m.stages[sp.stage].subproblems[sp.markov_state] = sp

    def setstageobjective(self, sp: 'Subproblem', obj: Expression):
        if sp.finalstage:
            sp._obj = Objective(expr=obj, sense=sp.sense.value)
        else:
            sp._obj = Objective(expr=obj + self.theta, sense=sp.sense.value)

    def getstageobjective(self, sp: 'Subproblem'):
        if sp.finalstage:
            return sp.getobjectivevalue()
        else:
            return sp.getobjectivevalue() - value(self.theta)


def modifyvaluefunction(m: SDDPModel, setting: Settings, sp: Subproblem):
    vf = sp.valueoracle  # type:DefaultValueFunction
    # vf = self
    # 此时m.storate 存储的是下一阶段所有的计算结果
    I = list(range(len(m.storage.objective)))
    current_transition = deepcopy(m.storage.probability.range(I))
    for i in I:
        # 原来存储的是noise的概率，乘以转移概率以后就是真实的概率
        m.storage.probability[i] *= m.stages[sp.stage + 1].transitionprobabilities[sp.markov_state][
            m.storage.markov[i]]
    modifiedprobability = sp.riskmeasure.modifyprobability(m.storage.modifiedprobability.range(I),
                                                           m.storage.probability.range(I),
                                                           m.storage.objective.range(I),
                                                           m, sp)
    m.storage.modifiedprobability.put_range(I, modifiedprobability)
    cut = constructcut(m, sp)
    # TODO asynchronous
    vf.addcut(m, sp, cut)
    m.storage.probability.put_range(I, current_transition)
