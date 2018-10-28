from typing import List, Generic, Callable
from sddp.price_interpolation.discreate_distribution import *
from sddp.price_interpolation.dynamic_price_interpolation_oracle import DynamicOracle
from sddp.price_interpolation.price_interpolation import PriceInterpolationMethods
from itertools import product

T = TypeVar('T')
C = TypeVar('C', DynamicOracle)
T2 = TypeVar('T2')

class DynamicPriceInterpolation(PriceInterpolationMethods, Generic[C, T, T2]):
    def __init__(self,
                 initial_price: T,
                 location: T,
                 minprice: T,
                 maxprice: T,
                 noises: DiscreteDistribution[T2],
                 objective: Callable,
                 dynamics: Callable,
                 mu: List[Var],
                 oracle: C,
                 lipschitz_constant: float,
                 bound: float):
        self.initial_price = initial_price
        self.location = location
        self.minprice = minprice
        self.maxprice = maxprice
        self.noises = noises  # type:DiscreteDistribution
        self.objective = objective
        self.dynamics = dynamics
        self.mu = mu  # type:List[Var]
        self.oracle = oracle  # type:C
        self.lipschitz_constant = lipschitz_constant  # type:float
        self.bound = bound  # type:float

    @staticmethod
    def create(
            dynamics=lambda p, w: p,
            initial_price: float = 0.0,
            min_price=0.0,
            max_price=1.0,
            noise=DiscreteDistribution.create([0.0, 1.0]),
            lipschitz_constant=1e6,
            cut_oracle=None):
        return DynamicPriceInterpolation(
            initial_price,
            initial_price,
            min_price,
            max_price,
            noise,
            None,  # TODO (p)->QuadExpr(p)
            dynamics,
            [],
            cut_oracle,
            lipschitz_constant,
            0.0
        )

    def interpolate(self, price=None, mu=None):  # TODO
        if mu is None:
            mu = self.mu
        if price is None:
            price = self.location
        return mu[0] + price * mu[1]  # TODO

    def addpricecut(self, sense: Sense, sp: Subproblem, price, affexpr):
        if sense == Sense.Max:
            sp.anonymous = Constraint(expr=self.interpolate(price, self.mu) <= affexpr)
        else:
            sp.anonymous = Constraint(expr=self.interpolate(price, self.mu) >= affexpr)

    def initializevaluefunction(self, sp: 'Subproblem', sense: Sense, bound: float):
        N = 1  # TODO 根据不同类型，取不同的值
        if isinstance(self.location, tuple):
            N = len(self.location)  # 对应 NTuple{N,T} 中的N
        self.bound = bound
        sp.anonymous = Var()
        self.mu.append(sp.anonymous)
        for i in range(N):
            sp.anonymous = Var(lb=-self.lipschitz_constant, ub=self.lipschitz_constant)
            self.mu.append(sp.anonymous)

        if 1 < N <= 4:
            for price in product(*zip(self.minprice, self.maxprice)):  # TODO
                self.addpricecut(sense, sp, price, bound)
        else:
            self.addpricecut(sense, sp, self.minprice, bound)
            self.addpricecut(sense, sp, self.maxprice, bound)
        return self

    def addcut(self, m: SDDPModel, sp: Subproblem, current_price, cut: Cut):
        vf = self
        vf.oracle.storecut(m, sp, cut, current_price)
        affexpr = cuttoaffexpr(sp, cut)
        self.addpricecut(sp.sense, sp, current_price, affexpr)

    def updatevaluefunction(self, m: SDDPModel, settings: Settings, t: int, sp: Subproblem):
        vf = self  # vf = sp.valueoracle  # type:StaticPriceInterpolation
        current_price = vf.location
        cut = constructcut(m, sp, t, current_price)
        self.addcut(m, sp, current_price, cut)
        # TODO 并行

    @property
    def cutoracle(self) -> AbstractCutOracle:
        return self.oracle

    def modifyvaluefunction(self, m: 'SDDPModel', setting: 'Settings', sp: 'Subproblem'):
        pass

    def rebuildsubproblem(self, m: 'SDDPModel', sp: 'Subproblem'):
        vf = self
        sp.states.clear()
        sp.noises.clear()
        sp.reset_mod()
        N = len(vf.mu) - 1
        self.mu.clear()
        self.initializevaluefunction(sp, sp.sense, vf.bound, N)
        m.build(sp, sp.stage, sp.markov_state)
        for cut in vf.cutoracle.validcuts():
            affexpr = cuttoaffexpr(sp, cut[0])
            self.addpricecut(sp.sense, sp, cut[1], affexpr)
        m.stages[sp.stage].subproblems[sp.markov_state] = sp