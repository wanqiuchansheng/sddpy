"""
对基础类型的操作
"""
from .typedefinitions import *
import numpy as np
from scipy.stats import t, sem


def constructcut(m: SDDPModel, sp: Subproblem):
    # theta <=/>= E[ (y - πᵀx̄) + πᵀx ]
    intercept = 0.0
    coefficients = [0.] * sp.nstates
    for i in range(m.storage.objective.len):
        intercept += m.storage.modifiedprobability[i] * (m.storage.objective[i] - float(np.dot(
            m.storage.duals[i], m.stages[sp.stage].state)
        ))
        # E[πᵀ]=a1π1ᵀ+a2π2ᵀ...anπnᵀ
        for j in range(sp.nstates):
            coefficients[j] += m.storage.modifiedprobability[i] * m.storage.duals[i][j]
    return Cut(intercept, coefficients)


def applicable(iteration: int, frequency: int):
    """
    判断iteration 是否是整数倍:
    """
    return frequency > 0 and np.mod(iteration, frequency) == 0


def confidenceinterval(x: List[float], conf_level=0.95):
    """
    获得置信区间
    """
    a = 1.0 * np.array(x)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t._ppf((1 + conf_level) / 2., n - 1)
    return m - h, m + h


def isapprox(a, b, atol):
    """
    a,b 相对误差小于某值
    :param a:
    :param b:
    :param atol:
    :return:
    """
    return abs(a - b) / abs(b) <= atol


def futureobjective(sense: Sense, bound):
    if sense == Sense.Min:
        return Var(bounds=(bound, None))
    else:
        return Var(bounds=(None, bound))


def cuttoaffexpr(sp: Subproblem, cut: Cut) -> Expression:
    """
    generator cut expression
    """
    expr = cut.intercept
    for idx, coef in enumerate(cut.coefficients):
        x = x + coef * sp.states[idx].variable
    return expr


def pyomoSolve(direction: Direction, m: SDDPModel, sp: Subproblem):
    """
    TODO 前处理后处理
    """
    # print(model_expression(sp.model))
    res = sp.solver.solve(sp.model)  # type:SolverResults

    status = res.solver.status
    if res.solver.status == SolverStatus.ok:
        pass  # 可行解或者最优解
    else:
        print("%s solver.status=%s" % direction.value, status)


def dominates(sense: Sense, trial: float, incumbent: float):
    """
    如果 incumbent 比 trial 好 true
    """
    if sense == Sense.Min:
        return trial > incumbent
    else:
        return trial < incumbent
