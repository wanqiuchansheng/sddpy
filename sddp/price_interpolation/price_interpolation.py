from typing import Callable

from sddp.price_interpolation.discreate_distribution import *
from sddp.price_interpolation.dynamic_price_interpolation_oracle import DynamicOracle
from sddp.state import setstates

T = TypeVar('T')
C = TypeVar('C', DynamicOracle)
T2 = TypeVar('T2')


class PriceInterpolationMethods(AbstractValueFunction):

    def __init__(self):
        self.objective = None  # type:Callable
        self.location = None,
        self.initial_price = None
        self.noises = None  # type:DiscreteDistribution

    def interpolate(self):
        raise NotImplementedError

    def updatevaluefunction(self, m: SDDPModel, settings: Settings, t: int, sp: Subproblem):
        raise NotImplementedError

    @staticmethod
    def backwardpass(m: SDDPModel, settings: Settings):
        for t in reversed(range(m.nstages - 1)):
            for sp in m.stages[t].subproblems:
                vf = sp.valueoracle  # type:PriceInterpolationMethods
                vf.updatevaluefunction(m, settings, t, sp)
        calculatefirststagebound(m)

    @staticmethod
    def solvesubproblem(m: SDDPModel, sp: Subproblem, solutionstore, dirction=Direction.forwardpass):
        if dirction != Direction.forwardpass:
            return
        vf = sp.valueoracle  # type:PriceInterpolationMethods
        if sp.stage == 0:
            vf.location = vf.initial_price
        p = vf.location
        w = samplepricenois(sp.stage, vf.noises, solutionstore)
        setobjective(sp, p, w)
        passpriceforward(m, sp)
        pyomoSolve(Direction.forwardpass, m, sp)

    @staticmethod
    def solvepricenoises(m: SDDPModel, sp: Subproblem, last_markov_state, price):
        vf = sp.valueoracle  # type:PriceInterpolationMethods
        markov_prob = m.stages[sp.stage].transitionprobabilities[last_markov_state, sp.markov_state]
        for price_noice in vf.noises.noises:
            setobjective(sp, price, price_noice.observation)
            vf.solvesubproblem(m, sp, markov_prob * price_noice.observation)

    @staticmethod
    def constructcut(m: SDDPModel, sp: Subproblem, t, price):
        m.storage.reset()
        for sp2 in m.stages[t + 1].subproblems:
            setstates(m, sp2)
            vf = sp2.valueoracle  # type:PriceInterpolationMethods
            vf.solvepricenoises(m, sp2, sp.markov_state, price)
        I = list(range(len(m.storage.objective)))
        modifiedprobability = sp.riskmeasure.modifyprobability(m.storage.modifiedprobability.range(I),
                                                               m.storage.probability.range(I),
                                                               m.storage.objective.range(I),
                                                               m, sp)
        m.storage.modifiedprobability.put_range(I, modifiedprobability)
        return m.constructcut(sp, m.storage)

    def setstageobjective(self,
                          sp: 'Subproblem', obj):
        self.objective = obj

    def getstageobjective(self,
                          sp: 'Subproblem'):
        if sp.finalstage:
            return sp.getobjectivevalue()
        else:
            return sp.getobjectivevalue() - value(self.interpolate())


def samplepricenois(stage: int, noises: DiscreteDistribution, solutionstore: Dict = None):
    if "pricenoise" in solutionstore:
        noiseidx = solutionstore["pricenoise"][stage]
        return noises[noiseidx].observation
    else:
        return noises.sample()


def calculatefirststagebound(m: SDDPModel):
    m.storage.reset()
    for sp in m.stages[0].subproblems:
        vf = sp.valueoracle  # type:PriceInterpolationMethods
        vf.solvepricenoises(m, sp, 0, vf.initial_price)
    return float(np.dot(m.storage.objective, m.storage.probability))


def setobjective(sp: Subproblem, price, noise):
    vf = sp.valueoracle  # type:StaticPriceInterpolation
    p = vf.dynamics(price, noise)
    vf.location = p  # 目前的价格

    # stage objective obj
    stageobj = vf.objective(p)

    # future objective
    future_value = vf.interpolate()
    if sp.finalstage:
        sp._obj = Objective(expr=stageobj, sense=sp.sense.value)
    else:
        sp._obj = Objective(expr=stageobj + future_value, sense=sp.sense.value)


def passpriceforward(m: SDDPModel, sp: Subproblem):
    stage = sp.stage
    if stage < m.nstages - 1:
        for sp2 in m.stages[stage + 1].subproblems:
            sp2.valueoracle.location = sp.valueoracle.location
