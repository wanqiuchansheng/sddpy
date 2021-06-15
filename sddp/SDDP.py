#  Copyright 2017, Oscar Dowson, Zhao Zhipeng
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

import copy
# from sddp.defaultvaluefunction import DefaultValueFunction
from sddp.riskmeasures import Expectation
from sddp.state import setstates
from sddp.utilities import *
from sddp.cut_oracles.DefaultCutOracle import DefaultCutOracle
# from .state import *
from .typedefinitions import *
from sddp.print import *


def getel(x: float or List[float] or List[List[float]], t: int, i: int):
    dim = len(np.array(x).shape)
    if dim == 0:
        return x
    elif dim == 1:
        return x[t]
    elif dim == 2:
        return x[t][i]


# 可以向每个子问题设置自身特别的解决方法
def createSDDPModel(build_,  # Callable[[Subproblem, int, int]]
                    sense: Sense,
                    stages: int = 1,
                    objective_bound: float or List[float] or List[List[float]] = None,
                    markov_transition: List[List[List[float]]] = None,
                    risk_measure: AbstractRiskMeasure or List[AbstractRiskMeasure] or List[
                        List[AbstractRiskMeasure]] = Expectation(),
                    cut_oracle=DefaultCutOracle(),
                    solver=None,
                    value_function=None
                    # AbstractValueFunction or List[AbstractValueFunction] or List[List[AbstractValueFunction]]
                    ):
    # 默认的value_function
    if value_function is None:
        from sddp.defaultvaluefunction import DefaultValueFunction
        value_function = DefaultValueFunction(cut_oracle)  # 之后会进行深度复制

    if objective_bound is None: raise RuntimeError("You must specify the objective_bound keyword")

    m = SDDPModel(sense=sense,
                  build=build_)
    for t in range(stages):
        # 二维数组
        markov_transition_matrix = markov_transition[t]  # TODO 要进行转化
        stage = Stage.create(t=t, markov_transition=markov_transition_matrix)
        # 表示该时段markov状态数
        for i in range(len(markov_transition_matrix[0])):
            mod = Subproblem(
                finalstage=t == stages - 1,
                stage=t,
                markov_state=i,
                sense=sense,
                bound=getel(objective_bound, t, i),
                risk_measure=getel(risk_measure, t, i),
                value_function=copy.deepcopy(getel(value_function, t, i))
            )

            mod.set_solver(getel(solver, t, i))
            build_(mod, t, i)
            # 没有设置概率的话，认为是等概率选择
            if len(mod.noises) != len(mod.noiseprobability):
                mod.noiseprobability = [1 / len(mod.noises)] * len(mod.noises)
            stage.subproblems.append(mod)
        m.stages.append(stage)
    return m


def forwardpass(m: SDDPModel, setting: Settings, solutionstore=None) -> float:
    last_markov_state = 0
    noiseidx = 0
    obj = 0.0
    for t, stage in enumerate(m.stages):  # type:int,Stage
        last_markov_state, sp = stage.samplesubproblem(last_markov_state, solutionstore)
        if t > 0:
            setstates(m, sp)
        if sp.hasnoises:
            noiseidx, noise = sp.samplenoise()
            sp.setnoise(noise)
        solvesubproblem(Direction.forwardpass, m, sp)
        obj += sp.getstageobjective()
        stage.savestates(sp)
        # 保存中间过程，用于模拟 TODO
    return obj


def iteration_fun(m: SDDPModel, setting: Settings):
    """
    一正，一反为一次迭代
    """
    t = time.time()
    simulation_objective = forwardpass(m, setting)
    time_forwards = time.time() - t
    vf = m.stages[0].subproblems[0].valueoracle  # 默认采用第一个子问题的backwardpass方法
    objective_bound = vf.backwardpass(m, setting)  # 不同的value_function 使用的方法可能不同
    time_backwards = time.time() - time_forwards - t
    return objective_bound, time_backwards, simulation_objective, time_forwards


def solve(m: SDDPModel, settings: Settings = Settings()):
    status = Staus.solving
    time_simulating, time_cutting = 0.0, 0.0
    objectives = CachedVector([])
    nsimulations, iteration, keep_iterating = 0, 1, True
    start_time = time.time()
    while keep_iterating:
        # add cuts
        objective_bound, time_backwards, simulation_objective, time_forwards = iteration_fun(m, settings)  # 一次循环计算
        # update timers and bounds
        time_cutting += time_backwards + time_forwards  # 一次循环认为建立的一个cut
        lower, upper = simulation_objective, simulation_objective

        if applicable(iteration, settings.cut_selection_frequency):  # 对cut进行重新选择的频率
            if settings.print_level > 1:
                print("Running Cut Selection")
            rebuid(m)
        if applicable(iteration, settings.simulation.frequency):  # 进行模拟，采用置信区间判断收敛的频率
            t = time.time()
            if settings.print_level > 1:
                print("Running Monte-Carlo Simulation")
            simidx = 0
            objectives.reset()
            # 进行step步模拟计算，判断是否收敛
            for i in range(settings.simulation.steps[-1]):
                objectives.append(forwardpass(m, settings))
                nsimulations = +1
                # 从 min 一直到 max 都是在置信区间，才认为是收敛，只要有一个不是就不收敛
                if i == settings.simulation.steps[simidx] - 1:
                    lower, upper = confidenceinterval(objectives, settings.simulation.confidence)
                    if lower <= objective_bound <= upper:
                        if settings.simulation.termination and simidx == len(settings.simulation.steps) - 1:
                            status = Staus.converged
                            keep_iterating = False
                    else:
                        break
                    simidx += 1
            time_simulating += time.time() - t  # 模拟运行判断是否收敛的时间
        total_time = time.time() - start_time
        # 打印日志
        addsolutionlog(m, settings, iteration, objective_bound, lower, upper, time_cutting, nsimulations,
                       time_simulating, total_time, not applicable(iteration, settings.simulation.frequency))
        status, keep_iterating = testboundstall(m, settings, status, keep_iterating)
        # 最长运行时间
        if total_time > settings.time_limit:
            status = Staus.time_limit
            keep_iterating = False

        iteration += 1
        # 最大循环次数
        if iteration > settings.max_iterations:
            status = Staus.max_iterations
            keep_iterating = False
    return status


def rebuid(m: SDDPModel):
    for t, stage in enumerate(m.stages):
        if t == len(m.stages) - 1:
            continue
        for sp in stage.subproblems:
            sp.valueoracle.rebuildsubproblem(m, sp)


def addsolutionlog(m: SDDPModel, settings: Settings, iteration: int, objective: float, lower: float, upper: float,
                   cutting_time, simulations,
                   simulation_time, total_time, printsingle: bool):
    m.log.append(
        SolutionLog(iteration, objective, lower, upper, cutting_time, simulations, simulation_time, total_time))
    print_solutionLog(m.log[-1], printsingle, m.sense == Sense.Min)


# 相对误差停止准则
def testboundstall(m: SDDPModel, settings: Settings, status: Staus, keep_iteerating: bool):
    last_n_size = settings.bound_convergence.iterations
    if keep_iteerating:
        if settings.bound_convergence.iterations > 1 and len(m.log) >= last_n_size:
            last_n = np.array([l.bound for l in m.log[-last_n_size:]])
            mean = np.mean(last_n)
            if np.all(last_n - mean < settings.bound_convergence.atol) or np.all(
                    np.abs(last_n / mean - 1) < settings.bound_convergence.rtol):
                return Staus.stalling_convergence, False
    return status, keep_iteerating


def solvesubproblem(direction: Direction, m: SDDPModel, sp: Subproblem, incoming_probablility: float = 1.0):
    if direction == Direction.forwardpass:
        pyomoSolve(direction, m, sp)

    elif direction == Direction.backwardpass:
        if sp.hasnoises:
            for i in range(len(sp.noiseprobability)):
                sp.setnoise(sp.noises[i])
                pyomoSolve(direction, m, sp)
                Storage.push(m.storage.objective, sp.getobjectivevalue())
                Storage.push(m.storage.noise, i)
                Storage.push(m.storage.probability, incoming_probablility * sp.noiseprobability[i])
                Storage.push(m.storage.modifiedprobability, incoming_probablility * sp.noiseprobability[i])
                Storage.push(m.storage.markov, sp.markov_state)
                Storage.push(m.storage.duals, [s.dual for s in sp.states])  # 状态的对偶值，是其约束的对偶值
        else:
            pyomoSolve(direction, m, sp)
            Storage.push(m.storage.objective, sp.getobjectivevalue())
            Storage.push(m.storage.noise, 0)
            Storage.push(m.storage.probability, incoming_probablility)
            Storage.push(m.storage.modifiedprobability, incoming_probablility)
            Storage.push(m.storage.markov, sp.markov_state)
            Storage.push(m.storage.duals, [s.dual for s in sp.states])  # 状态的对偶值，是其约束的对偶值


def solve_default(m: SDDPModel,
                  iteration_limit: int = 1e9,
                  time_limit: float = float("inf"),
                  simulation=MonteCarloSimulation(
                      frequency=0,
                      steps=[20],
                      confidence=0.95,
                      termination=False
                  ),
                  bound_stalling=BoundStalling(
                      iterations=0,
                      rtol=0.0,
                      atol=0.0
                  ),
                  cut_selection_frequency: int = 0,
                  print_level: int = 1,
                  log_file: str = "",
                  solve_type=SolveType.Serial,
                  reduce_memory_footprint=False,
                  cut_output_file: str = ""
                  ):
    cut_output_file_handle = ""
    is_asyncronous = solve_type == SolveType.Asyncronous
    print("is_asyncronous=%s" % is_asyncronous)
    settings = Settings(
        iteration_limit,
        time_limit,
        simulation,
        bound_stalling,
        cut_selection_frequency,
        print_level,
        log_file,
        reduce_memory_footprint,
        cut_output_file_handle,
        is_asyncronous=is_asyncronous
    )
    printheader(m, solve_type.value)
    status = Staus.solving

    status = solve(m, settings)
    printfooter(m, settings, status.value, None)
    return status

    # print(status, "计算结束")
