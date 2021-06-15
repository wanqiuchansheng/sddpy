#  Copyright 2017, Oscar Dowson
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################



from enum import Enum
from typing import List, TypeVar, Generic, Dict
from pyomo.core import ConcreteModel, Var, Constraint, Expression
from pyomo.core.base.constraint import _GeneralConstraintData, SimpleConstraint
from pyomo.core.base.var import _GeneralVarData, SimpleVar, IndexedVar
from pyomo.core.base.param import SimpleParam, _ParamData, _NotValid
# from pyomo.core.kernel import value
from pyomo.environ import *
from pyomo.solvers.plugins.solvers.CPLEX import *
from strgen import StringGenerator as SG

# from .riskmeasures import AbstractRiskMeasure
# from pyomo.util.modeling import unique_component_name
from pyomotools.tools import unique_component_name
from .base_utilities import sample

T = TypeVar('T')


class Direction(Enum):
    forwardpass = "forwardpass"
    backwardpass = "backwardpass"


class Staus(Enum):
    solving = "solving"
    converged = "converged"
    stalling_convergence = "stalling_convergence"
    time_limit = "time_limit"
    max_iterations = "max_iterations"


class Sense(Enum):
    Min = minimize
    Max = maximize


class SolveType(Enum):
    Asyncronous = "Asyncronous"
    Serial = "Serial"


# 抽象类
class AbstractCutOracle:
    def __init__(self):
        pass

    def storecut(self, m: 'SDDPModel', sp: 'Subproblem', cut: 'Cut'):
        raise NotImplementedError

    def validcuts(self):
        raise NotImplementedError


class AbstractValueFunction(Generic[T]):
    def __init__(self):
        pass

    @property
    def cutoracle(self) -> AbstractCutOracle:
        raise NotImplementedError

    def initializevaluefunction(self, sp: 'Subproblem', sense: Sense, bound: float):
        raise NotImplementedError

    @staticmethod
    def backwardpass(m: 'SDDPModel', settings: 'Settings'):
        raise NotImplementedError

    def rebuildsubproblem(self, m: 'SDDPModel', sp: 'Subproblem'):
        raise NotImplementedError

    def setstageobjective(self, sp: 'Subproblem', obj):
        """
        set stage objective,
        """
        raise NotImplementedError

    def getstageobjective(self, sp: 'Subproblem'):
        """
        get stage object
        """
        raise NotImplementedError


class AbstractRiskMeasure:
    def __init__(self):
        pass

    def modifyprobability(self, riskadjusted_distribution: List[float],
                          original_distribution: List[float], observations: List[float], m: 'SDDPModel',
                          sp: 'Subproblem'):
        raise NotImplementedError


class State:
    def __init__(self,
                 variable: _GeneralVarData,
                 variable_in: _GeneralVarData,
                 constraint: _GeneralConstraintData,
                 param: _ParamData):
        self.variable = variable
        self.variable_in = variable_in  # 时段处的状态值
        self.param = param
        self.constraint = constraint

    @property
    def model(self) -> ConcreteModel:
        return self.variable.model()

    # 设置时段初的状态的值，用于日后求对偶值
    def setvalue(self, v: float):
        self.param.value = v
        # self.constraint.set_value(expr=self.variable_in == v)

    @property
    def value(self):
        return value(self.variable)

    @property
    def dual(self):
        return self.model.dual.get(self.constraint)


class Cut:
    def __init__(self, intercept: float, coefficients: List[float]):
        self.coefficients = coefficients
        self.intercept = intercept


# parameter 需要设置成 mutable=true
class Noise:
    """
    均通过修改param_value，约束有可能是多个，params值能是一个，obj
    约束和 values 是对应的
    不同noise 对应的约束应该是相同的，但是values和
    len(values)==len(noise_param)
    """

    def __init__(self, has_objective_noise: bool = False, values: List[float] = None,
                 noise_params: List[_ParamData] = None, constraints: List = None, obj: Expression = None):
        if values is None: values = []
        if noise_params is None: noise_params = []

        self.constraints = constraints
        self.obj = obj
        self.has_objective_noise = has_objective_noise
        # ===============
        self.values = values
        self.params = noise_params
        self.obj_noise = []
        self.obj_param = []  # type:List[_ParamData]

    def add_obj_noise(self, nv: float, param: _ParamData):
        self.obj_noise.append(nv)
        self.obj_param.append(param)

    def works(self):
        """
        生效
        """
        for v, p in zip(self.values, self.params):
            p.value = v

        for v, p in zip(self.obj_noise, self.obj_param):
            p.value = v


# class ModelWrapper:
#     def __init__(self, model: ConcreteModel):
#         self.model = model
#         self.states = []  # type:List[Var]
#
#     def state(self, var: Var):
#         for index in var.index_set():
#             self.states.append(var[index])


class Subproblem:
    def __init__(self, finalstage=False, stage=1, markov_state=1, sense: Sense = Sense.Min,
                 bound: float = -1e6,
                 states: List[State] = None,
                 noises: List[Noise] = None,
                 value_function: AbstractValueFunction = None,
                 noiseprobability: List[float] = None,
                 risk_measure: 'AbstractRiskMeasure' = None
                 ):
        if states is None: states = []
        if noises is None: noises = []
        if noiseprobability is None: noiseprobability = []

        self.finalstage = finalstage
        self.stage = stage
        self.markov_state = markov_state
        self.problembound = bound
        self.sense = sense
        self.states = states  # type:List[State]
        self.valueoracle = value_function  # type:AbstractValueFunction
        self.noises = noises
        self.noiseprobability = noiseprobability
        self.riskmeasure = risk_measure
        self.solver = None  # type:CPLEXSHELL
        # model relevant
        self.model = None
        self.reset_mod()
        # temporary variable
        self._anonymous = None

    def add_constraint(self, expr: Expression):
        self.model.cuts.add(expr)

    def reset_mod(self):
        self.model = ConcreteModel()  # type:ConcreteModel
        ###########
        self.model.dual = Suffix(direction=Suffix.IMPORT_EXPORT)  # 需要获取对偶值
        self.model.cuts = ConstraintList()
        self.valueoracle.initializevaluefunction(self, self.sense, bound=self.problembound)

    @property
    def nstates(self):
        return len(self.states)

    def set_solver(self, solver):
        self.solver = solver

    # def get_dual(self)->List[float]:
    #     return [s.  for s in self.states]

    def setnoiseprobability(self, noise_probability:List[float]):
        self.noiseprobability = noise_probability

    @property
    def hasnoises(self):
        return len(self.noises) > 0

    def samplenoise(self, solutionstore=None):  # TODO
        if solutionstore is None:
            noiseidx = sample(self.noiseprobability)
            return noiseidx, self.noises[noiseidx]

    def setnoise(self, noise: Noise):
        """
        对约束和目标函数中的噪声项进行赋值
        """
        noise.works()

    def _get_obj_list(self):
        return [obj for obj in self.model.component_data_objects(Objective)]

    def _del_com(self, name: str):
        if hasattr(self.model, name):
            self.model.__delattr__(name)

    #
    # def setstageobjective(self, obj):
    #     self.valueoracle.setstageobjective(self, obj)
    #     # self._del_com("obj")
    #     # if self.finalstage:
    #     #     self.model.obj = Objective(expr=expr, sense=self.sense.value)
    #     # else:
    #     #     self.model.obj = Objective(expr=expr + self.valueoracle.theta, sense=self.sense.value)

    def getstageobjective(self):
        return self.valueoracle.getstageobjective(self)

    def getobjectivevalue(self):
        return value(self.model.obj)

    @property
    def obj(self):
        return self.model.obj

    @obj.setter
    def obj(self, value):
        self.valueoracle.setstageobjective(self, value)

    @property
    def _obj(self):
        """
        call by developer
        """
        return self.model.obj

    @_obj.setter
    def _obj(self, value):
        """
        重新设置目标函数
        """
        self._del_com("obj")
        self.model.obj = value

    # TODO 多维可能存在问题
    def add_state(self, state: SimpleVar, state0: SimpleVar, param: SimpleParam, cons: SimpleConstraint = None):
        if cons is None:
            cons = Constraint(param.index_set(), rule=lambda m, i: state0[i] == param[i])
            self.anonymous = cons
        for i in param.index_set():
            s = State(state[i], state0[i], cons[i], param[i])
            self.states.append(s)

    # def state(self, state, state0, param: SimpleParam):
    #     cons = Constraint(param.index_set(), rule=lambda m, i: state0[i] == param[i])
    #     self.anonymous = cons
    #     self.add_state(state, state0, cons)

    def add_noise_constraint(self, noises: List[float], param: _ParamData,
                             cons: SimpleConstraint or List[SimpleConstraint] = None):
        """
        只能值单个param，不能是带有index的param
        :param cons: 使用到这个param的约束，可以是单个约束也可以是一个List
        :return:
        """
        if len(self.noises) <= 0:
            for nv in noises:
                self.noises.append(Noise(has_objective_noise=False, values=[nv], noise_params=[param],
                                         constraints=[cons]))
        else:
            for i, nv in enumerate(noises):
                self.noises[i].params.append(param)
                self.noises[i].constraints.append(cons)
                self.noises[i].values.append(nv)

    def add_nosise_object(self, noises: List[float], param: _ParamData, obj: Expression):
        if len(self.noises) <= 0:
            for _ in noises:
                self.noises.append(Noise())
        for i, nv in enumerate(noises):
            self.noises[i].add_obj_noise(nv, param)
            self.noises[i].obj = obj

    # # delegate method
    # def rebuildsubproblem(self, m: "SDDPModel"):
    #     self.valueoracle.rebuildsubproblem(m, self)

    @property
    def anonymous(self):
        return self._anonymous

    @anonymous.setter
    def anonymous(self, value):
        """
        add anonymous component to model
        """
        random_name = SG("[\w]{3}").render()
        random_name = unique_component_name(self.model, random_name)
        setattr(self.model, random_name, value)
        self._anonymous = value


class Stage:
    def __init__(self, t: int = 1, subproblems: List[Subproblem] = None,
                 transitionprobabilities: List[List[float]] = None, state: List[float] = None):
        if subproblems is None: subproblems = []
        if transitionprobabilities is None: transitionprobabilities = []
        if state is None: state = []

        self._state = state  # type:List[float]
        self.transitionprobabilities = transitionprobabilities
        self.subproblems = subproblems
        self.t = t

    def savestates(self, sp: Subproblem):  # TODO 和SDDP.jl实现不同
        self._state = [s.value for s in sp.states]

    @property
    def state(self):
        return self._state

    def samplesubproblem(self, last_markov_state, solutionstore: Dict = None) -> (int, Subproblem):
        if solutionstore is None:
            newidx = sample(self.transitionprobabilities[last_markov_state])
            return newidx, self.subproblems[newidx]
        else:
            # TODO 仿照julia中的写
            pass

    @staticmethod
    def create(t: int, markov_transition=None) -> 'Stage':
        if markov_transition is None: markov_transition = []
        return Stage(t=t, transitionprobabilities=markov_transition)


import numpy as np


class CachedVector(Generic[T]):
    """
    泛型
    """

    def __init__(self, data: List = None, n: int = None):
        if data is None: data = []
        self.data = data
        self.n = n

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def range(self, index_list: List[int]):
        data = np.array(self.data)
        return data[index_list].tolist()

    def put_range(self, index_list: List[int], lst: List[T]):
        for index, datum in zip(index_list, lst):
            self.data[index] = datum

    def reset(self):
        self.n = 0
        self.data.clear()

    def append(self, v):
        self.data.append(v)

    @property
    def len(self):
        return len(self.data)


class Storage:
    def __init__(self, state: List[float],
                 noise: CachedVector[int],
                 markov: CachedVector[int],
                 duals: CachedVector[List[float]],
                 objective: CachedVector[float],
                 probability: CachedVector[float],
                 modifiedprobability: CachedVector[float]):
        """

        :param state:
        :param noise:
        :param markov:
        :param duals:
        :param objective:
        :param probability:
        :param modifiedprobability:
        """
        self.modifiedprobability = modifiedprobability
        self.probability = probability
        self.objective = objective
        self.duals = duals
        self.markov = markov
        self.noise = noise
        self.state = state
        self.n = 0

    def reset(self):
        """
        TODO
        """
        self.n = 0
        self.modifiedprobability.reset()
        self.probability.reset()
        self.objective.reset()
        self.duals.reset()
        self.markov.reset()
        self.noise.reset()
        # self.state.reset()

    @staticmethod
    def push(cached: CachedVector[T], ele: T):
        cached.data.append(ele)

    @staticmethod
    def create():
        return Storage([], CachedVector(), CachedVector(), CachedVector(),
                       CachedVector(), CachedVector(), CachedVector())


class SolutionLog:
    def __init__(self,
                 iteration: int, bound: float, lower_statistical_bound: float, upper_statistical_bound: float,
                 timecuts: float,
                 simulations: int, timesimulations: float, timetotal: float):
        self.simulations = simulations
        self.timecuts = timecuts
        self.bound = bound
        self.upper_statistical_bound = upper_statistical_bound
        self.lower_statistical_bound = lower_statistical_bound
        self.timesimulations = timesimulations
        self.timetotal = timetotal
        self.iteration = iteration

    def __str__(self):
        return str(self.__dict__)


class SDDPModel:
    def __init__(self, sense: Sense, build: [Subproblem, int, int],
                 stages: List[Stage] = None,
                 storage: Storage = None, log: List[SolutionLog] = None):
        if stages is None: stages = []
        if log is None: log = []
        if storage is None: storage = Storage.create()

        self.log = log  # typeList[SolutionLog]
        self.build = build
        self.storage = storage
        self.stages = stages
        self.sense = sense

    @property
    def nstages(self):
        return len(self.stages)

    def getbound(self):
        if len(self.log) > 0:
            return self.log[-1].bound
        else:
            raise RuntimeError("模型还没没有解决!")

    def constructcut(self, sp: 'Subproblem', storage: Storage = None):
        if storage is None:
            storage = self.storage
        m = self
        intercept = 0.0
        coefficients = [0.] * sp.nstates
        for i in range(storage.objective.len):
            intercept += storage.modifiedprobability[i] * (storage.objective[i] - float(np.dot(
                storage.duals[i], m.stages[sp.stage].state)
            ))
            # E[πᵀ]=a1π1ᵀ+a2π2ᵀ...anπnᵀ
            for j in range(sp.nstates):
                coefficients[j] += storage.modifiedprobability[i] * storage.duals[i][j]
        return Cut(intercept, coefficients)


class BoundStalling:
    def __init__(self, iterations: int = 0, rtol: float = 0, atol: float = 0):
        """

        :param iterations:  len(last_n)
        :param rtol:  相对误差， last_n-mean(last_n)
        :param atol:  绝对误差   last_n/mean(last_n)-1
        """
        self.iterations = iterations
        self.rtol = rtol
        self.atol = atol


class MonteCarloSimulation:
    def __init__(self, frequency: int = 0, steps: List[int] = [20], confidence: float = 0.95,
                 termination: bool = False):
        self.termination = termination
        self.confidence = confidence
        self.steps = steps
        self.frequency = frequency


class Settings:
    def __init__(self,
                 max_iterations: int = 0,
                 time_limit: float = 600,
                 simulation: MonteCarloSimulation = MonteCarloSimulation(),
                 bound_convergence: BoundStalling = BoundStalling(),
                 cut_selection_frequency: int = 0,
                 print_level: int = 0,
                 log_file: str = "",
                 reduce_memory_footprint: bool = False,
                 cut_output_file_handle=None,
                 is_asyncronous: bool = False):
        self.is_asyncronous = is_asyncronous
        self.reduce_memory_footprint = reduce_memory_footprint
        self.log_file = log_file
        self.print_level = print_level
        self.cut_selection_frequency = cut_selection_frequency
        self.bound_convergence = bound_convergence
        self.simulation = simulation
        self.time_limit = time_limit
        self.max_iterations = max_iterations
