# @version : python3.5
# @Time    : 2018/10/10 21:28
# @Author  : zzp
# @FileName: piecewise_nd.py
from functools import partial
from typing import List

from pyomo.core import Block, Var, Binary, NonNegativeReals, Constraint, Expression
from pyomo.core.base.var import SimpleVar
import scipy.spatial.qhull as qhull
import numpy as np
# from pyomo.core.kernel.component_piecewise.util import generate_gray_code

from pyomotools.tools import generate_points, generate_gray_code


def cc(m: Block, tri: qhull.Delaunay,
       values: List[float],
       input: List[SimpleVar] = None,
       output: SimpleVar = None,
       bound: str = 'eq', **kw):
    values = np.array(values).tolist()
    ndim = len(input)
    nsimplices = len(tri.simplices)
    npoints = len(tri.points)
    pointsT = list(zip(*tri.points))
    # create index objects
    dimensions = list(range(ndim))
    simplices = list(range(nsimplices))  # 跟单纯形 数量一致
    vertices = list(range(npoints))
    bound = bound.lower()

    m.lmbda = Var(vertices, domain=NonNegativeReals)  # 非负
    m.y = Var(simplices, domain=Binary)  # 二进制
    # m.y = Var(simplices, domain=NonNegativeReals, bounds=(0, 1))  # 二进制

    m.a0 = Constraint(dimensions, rule=lambda m, d: sum(m.lmbda[v] * pointsT[d][v] for v in vertices) == input[d])
    if bound == 'eq':
        m.a1 = Constraint(expr=output == sum(m.lmbda[v] * values[v] for v in vertices))
    elif bound == 'lb':
        m.a1 = Constraint(expr=output <= sum(m.lmbda[v] * values[v] for v in vertices))
    elif bound == 'ub':
        m.a1 = Constraint(expr=output >= sum(m.lmbda[v] * values[v] for v in vertices))
    else:
        raise RuntimeError("bound值错误！bound=" + bound)

    m.b = Constraint(expr=sum(m.lmbda[v] for v in vertices) == 1)

    # generate a map from vertex index to simplex index,
    # which avoids an n^2 lookup when generating the
    # constraint
    vertex_to_simplex = [[] for _ in vertices]
    for s, simplex in enumerate(tri.simplices):
        for v in simplex:
            vertex_to_simplex[v].append(s)
    m.c0 = Constraint(vertices, rule=lambda m, v: m.lmbda[v] <= sum(m.y[s] for s in vertex_to_simplex[v]))
    m.c1 = Constraint(expr=sum(m.y[s] for s in simplices) == 1)
    return m


# def generate_union_jack2(vars:List[SimpleVar],num=10):
#     """
#     生成米字形格子
#     :param vars:
#     :param num:
#     :return:
#     """
def log_lp(m: Block, tri: qhull.Delaunay,
           values: List[float],
           input: List[SimpleVar] = None,
           output: SimpleVar = None,
           bound: str = 'eq', **kw):
    num = kw["num"]
    values = np.array(values).tolist()
    ndim = len(input)
    npoints = len(tri.points)
    pointsT = list(zip(*tri.points))
    dims = list(range(ndim))
    vertices = list(range(npoints))
    bound = bound.lower()
    # 与 npoints 匹配的索引
    # vertices_idx = list(zip(*generate_points([list(range(num)) for _ in range(ndim)])))
    vertices_idx = generate_points([list(range(num)) for _ in range(ndim)]).tolist()
    if len(vertices_idx) != len(vertices):
        raise RuntimeError("生成的的tri需要是米字形！")
    # (9a)
    m.lmbda = Var(vertices, domain=NonNegativeReals)  # 非负
    m.a0 = Constraint(dims, rule=lambda m, d: sum(m.lmbda[v] * pointsT[d][v] for v in vertices) == input[d])
    m.a_sum = Expression(expr=sum(m.lmbda[v] * values[v] for v in vertices))
    if bound == 'eq':
        m.a1 = Constraint(expr=output == m.a_sum)
    elif bound == 'lb':
        m.a1 = Constraint(expr=output <= m.a_sum)
    elif bound == 'ub':
        m.a1 = Constraint(expr=output >= m.a_sum)
    else:
        raise RuntimeError("bound值错误！bound=" + bound)
    # (9b)
    m.b = Constraint(expr=sum(m.lmbda[v] for v in vertices) == 1)

    # (9c)

    # 约束a和b与cc方法是一样的
    K = num - 1  # K 必须是偶数
    N = list(range(1, ndim + 1))
    log2K = math.ceil(math.log2(K))
    L = list(range(1, log2K + 1))
    G = generate_gray_code(log2K)

    def O(l, b):
        # k == 0 或者 k == K 的意思是不能是第一个和最后一个，避免越界
        res = []
        for k in range(K + 1):
            if (k == 0 or G[k - 1][l - 1] == b) and (k == K or G[k][l - 1] == b):
                res.append(k)
        return res
        # return [k for k in range(K + 1) if (k == 0 or G[k][l] == b) and (k == K or G[k + 1][l] == b)]

    m.y_s1 = Var(N, L, domain=NonNegativeReals, bounds=(0, 1))  # 二进制
    m.c1_s1 = Constraint(N, L, rule=lambda m, s1, s2: sum(
        m.lmbda[v] for v, idx in zip(vertices, vertices_idx) if idx[s1 - 1] in O(s2, 1)
    ) <= m.y_s1[s1, s2])
    m.c2_s1 = Constraint(N, L, rule=lambda m, s1, s2: sum(
        m.lmbda[v] for v, idx in zip(vertices, vertices_idx) if idx[s1 - 1] in O(s2, 0)
    ) <= 1 - m.y_s1[s1, s2])
    S2 = [(s1, s2) for s1 in N for s2 in N if s1 < s2]
    S2_idx = list(range(len(S2)))
    m.y_s2 = Var(S2_idx, domain=NonNegativeReals, bounds=(0, 1))
    m.c1_s2 = Constraint(S2_idx, rule=lambda m, i: sum(
        m.lmbda[v] for v, idx in zip(vertices, vertices_idx) if
        idx[S2[i][0] - 1] % 2 == 0 and idx[S2[i][1] - 1] % 2 == 1
    ) <= m.y_s2[i])
    m.c2_s2 = Constraint(S2_idx, rule=lambda m, i: sum(
        m.lmbda[v] for v, idx in zip(vertices, vertices_idx) if
        idx[S2[i][0] - 1] % 2 == 1 and idx[S2[i][1] - 1] % 2 == 0
    ) <= 1 - m.y_s2[i])
    return m


def log(m: Block, tri: qhull.Delaunay,
        values: List[float],
        input: List[SimpleVar] = None,
        output: SimpleVar = None,
        bound: str = 'eq', **kw):
    """
    只适用于米字形分割方式
    :param num:
    :param m:
    :param tri:
    :param values:
    :param input:
    :param output:
    :param bound:
    :return:
    """
    num = kw["num"]
    values = np.array(values).tolist()
    ndim = len(input)
    npoints = len(tri.points)
    pointsT = list(zip(*tri.points))
    dims = list(range(ndim))
    vertices = list(range(npoints))
    bound = bound.lower()
    # 与 npoints 匹配的索引
    # vertices_idx = list(zip(*generate_points([list(range(num)) for _ in range(ndim)])))
    vertices_idx = generate_points([list(range(num)) for _ in range(ndim)]).tolist()
    if len(vertices_idx) != len(vertices):
        raise RuntimeError("生成的的tri需要是米字形！")
    # (9a)
    m.lmbda = Var(vertices, domain=NonNegativeReals)  # 非负
    m.a0 = Constraint(dims, rule=lambda m, d: sum(m.lmbda[v] * pointsT[d][v] for v in vertices) == input[d])
    m.a_sum = Expression(expr=sum(m.lmbda[v] * values[v] for v in vertices))
    if bound == 'eq':
        m.a1 = Constraint(expr=output == m.a_sum)
    elif bound == 'lb':
        m.a1 = Constraint(expr=output <= m.a_sum)
    elif bound == 'ub':
        m.a1 = Constraint(expr=output >= m.a_sum)
    else:
        raise RuntimeError("bound值错误！bound=" + bound)
    # (9b)
    m.b = Constraint(expr=sum(m.lmbda[v] for v in vertices) == 1)

    # (9c)

    # 约束a和b与cc方法是一样的
    K = num - 1  # K 必须是偶数
    N = list(range(1, ndim + 1))
    log2K = math.ceil(math.log2(K))
    L = list(range(1, log2K + 1))
    G = generate_gray_code(log2K)

    def O(l, b):
        # k == 0 或者 k == K 的意思是不能是第一个和最后一个，避免越界
        res = []
        for k in range(K + 1):
            if (k == 0 or G[k - 1][l - 1] == b) and (k == K or G[k][l - 1] == b):
                res.append(k)
        return res
        # return [k for k in range(K + 1) if (k == 0 or G[k][l] == b) and (k == K or G[k + 1][l] == b)]

    m.y_s1 = Var(N, L, domain=Binary)  # 二进制
    # m.y_s1 = Var(N, L, domain=NonNegativeReals,bounds=(0,1))  # 二进制
    m.c1_s1 = Constraint(N, L, rule=lambda m, s1, s2: sum(
        m.lmbda[v] for v, idx in zip(vertices, vertices_idx) if idx[s1 - 1] in O(s2, 1)
    ) <= m.y_s1[s1, s2])
    m.c2_s1 = Constraint(N, L, rule=lambda m, s1, s2: sum(
        m.lmbda[v] for v, idx in zip(vertices, vertices_idx) if idx[s1 - 1] in O(s2, 0)
    ) <= 1 - m.y_s1[s1, s2])
    S2 = [(s1, s2) for s1 in N for s2 in N if s1 < s2]
    S2_idx = list(range(len(S2)))
    m.y_s2 = Var(S2_idx, domain=Binary)
    # m.y_s2 = Var(S2_idx, domain=NonNegativeReals,bounds=(0,1))
    m.c1_s2 = Constraint(S2_idx, rule=lambda m, i: sum(
        m.lmbda[v] for v, idx in zip(vertices, vertices_idx) if
        idx[S2[i][0] - 1] % 2 == 0 and idx[S2[i][1] - 1] % 2 == 1
    ) <= m.y_s2[i])
    m.c2_s2 = Constraint(S2_idx, rule=lambda m, i: sum(
        m.lmbda[v] for v, idx in zip(vertices, vertices_idx) if
        idx[S2[i][0] - 1] % 2 == 1 and idx[S2[i][1] - 1] % 2 == 0
    ) <= 1 - m.y_s2[i])
    return m


import math


def dlog(m: Block, tri: qhull.Delaunay,
         values: List[float],
         input: List[SimpleVar] = None,
         output: SimpleVar = None,
         bound: str = 'eq', **kw):
    values = np.array(values).tolist()
    ndim = len(input)
    nsimplices = len(tri.simplices)
    npoints = len(tri.points)
    pointsT = list(zip(*tri.points))
    # create index objects
    dimensions = list(range(ndim))
    simplices = list(range(nsimplices))  # 跟单纯形 数量一致
    vertices = list(range(npoints))
    bound = bound.lower()
    L = int(math.ceil(math.log2(nsimplices)))
    L_Range = list(range(L))
    vp = [0, 1, 2]
    #
    m.lmbda = Var(simplices, vp, domain=NonNegativeReals)  # 非负
    m.a0 = Constraint(dimensions, rule=lambda m, d: sum(
        m.lmbda[s, v] * pointsT[d][tri.simplices[s][v]] for s in simplices for v in vp) == input[d])
    if bound == 'eq':
        m.a1 = Constraint(
            expr=output == sum(m.lmbda[s, v] * values[tri.simplices[s][v]] for s in simplices for v in vp))
    elif bound == 'lb':
        m.a1 = Constraint(
            expr=output <= sum(m.lmbda[s, v] * values[tri.simplices[s][v]] for s in simplices for v in vp))
    elif bound == 'ub':
        m.a1 = Constraint(
            expr=output >= sum(m.lmbda[s, v] * values[tri.simplices[s][v]] for s in simplices for v in vp))
    else:
        raise RuntimeError("bound值错误！bound=" + bound)

    m.b1 = Constraint(expr=sum(m.lmbda[s, v] for s in simplices for v in vp) == 1)

    m.y = Var(L_Range, domain=Binary)  # 二进制

    m.c0 = Constraint(L_Range,
                      rule=lambda m, l: sum(
                          m.lmbda[s, v] for s in simplices if bin(s)[2:].zfill(L)[l] == '1' for v in vp) <=
                                        m.y[l])
    m.c1 = Constraint(L_Range,
                      rule=lambda m, l: sum(
                          m.lmbda[s, v] for s in simplices if bin(s)[2:].zfill(L)[l] == '0' for v in vp) <=
                                        1 - m.y[l])
    return m


# def piecewise_ndlog()

def piecewise_nd(tri: qhull.Delaunay,
                 values: List[float],
                 input: List[SimpleVar] = None,
                 output: SimpleVar = None,
                 bound: str = 'eq',
                 repn: str = 'cc', parent=None, **kw):
    """
    添加多维线性插值，values 必须是float类型的浮点数，不能是np的
    tri (scipy.spatial.Delaunay): A triangulation over
            the discretized variable domain. Can be
            generated using a list of variables using the
            utility function :func:`util.generate_delaunay`.
            Required attributes:
              - points: An (npoints, D) shaped array listing
                the D-dimensional coordinates of the
                discretization points.
              - simplices: An (nsimplices, D+1) shaped array
                of integers specifying the D+1 indices of
                the points vector that define each simplex
                of the triangulation.
    values (numpy.array): An (npoints,) shaped array of
        the values of the piecewise function at each of
        coordinates in the triangulation points array.
    input: A D-length list of variables or expressions
        bound as the inputs of the piecewise function.
    output: The variable constrained to be the output of
        the piecewise linear function.
    bound (str): The type of bound to impose on the
        output expression. Can be one of:
              - 'lb': y <= f(x)
              - 'eq': y  = f(x)
              - 'ub': y >= f(x)
    repn (str): The type of piecewise representation to
        use. Can be one of:
            - 'cc': convex combination

    """
    if repn == "cc":
        pf = cc
    elif repn == "dlog":
        pf = dlog
    elif repn == "log":
        pf = log
    elif repn == "log_lp":
        pf = log_lp
    else:
        raise RuntimeError(f'{repn} 不支持！')

    _f = partial(pf, tri=tri, values=values, input=input, output=output, bound=bound, **kw)

    if parent is None:
        # raise RuntimeError("parent 不能为空，必须提前声明好，否则无法添加到model中")
        m = Block(rule=_f)
    else:
        m = parent
        _f(m)
    return m


def piecewise_nd_old(tri: qhull.Delaunay,
                     values: List[float],
                     input: List[SimpleVar] = None,
                     output: SimpleVar = None,
                     bound: str = 'eq',
                     repn: str = 'cc', parent=None):
    """
    添加多维线性插值，values 必须是float类型的浮点数，不能是np的
    tri (scipy.spatial.Delaunay): A triangulation over
            the discretized variable domain. Can be
            generated using a list of variables using the
            utility function :func:`util.generate_delaunay`.
            Required attributes:
              - points: An (npoints, D) shaped array listing
                the D-dimensional coordinates of the
                discretization points.
              - simplices: An (nsimplices, D+1) shaped array
                of integers specifying the D+1 indices of
                the points vector that define each simplex
                of the triangulation.
    values (numpy.array): An (npoints,) shaped array of
        the values of the piecewise function at each of
        coordinates in the triangulation points array.
    input: A D-length list of variables or expressions
        bound as the inputs of the piecewise function.
    output: The variable constrained to be the output of
        the piecewise linear function.
    bound (str): The type of bound to impose on the
        output expression. Can be one of:
              - 'lb': y <= f(x)
              - 'eq': y  = f(x)
              - 'ub': y >= f(x)
    repn (str): The type of piecewise representation to
        use. Can be one of:
            - 'cc': convex combination

    """
    values = np.array(values).tolist()
    ndim = len(input)
    nsimplices = len(tri.simplices)
    npoints = len(tri.points)
    pointsT = list(zip(*tri.points))
    # create index objects
    dimensions = list(range(ndim))
    simplices = list(range(nsimplices))  # 跟单纯形 数量一致
    vertices = list(range(npoints))
    bound = bound.lower()

    def _f(m: Block):
        m.lmbda = Var(vertices, domain=NonNegativeReals)  # 非负
        m.y = Var(simplices, domain=Binary)  # 二进制

        m.a0 = Constraint(dimensions, rule=lambda m, d: sum(m.lmbda[v] * pointsT[d][v] for v in vertices) == input[d])
        if bound == 'eq':
            m.a1 = Constraint(expr=output == sum(m.lmbda[v] * values[v] for v in vertices))
        elif bound == 'lb':
            m.a1 = Constraint(expr=output <= sum(m.lmbda[v] * values[v] for v in vertices))
        elif bound == 'ub':
            m.a1 = Constraint(expr=output >= sum(m.lmbda[v] * values[v] for v in vertices))
        else:
            raise RuntimeError("bound值错误！bound=" + bound)

        m.b = Constraint(expr=sum(m.lmbda[v] for v in vertices) == 1)

        # generate a map from vertex index to simplex index,
        # which avoids an n^2 lookup when generating the
        # constraint
        vertex_to_simplex = [[] for _ in vertices]
        for s, simplex in enumerate(tri.simplices):
            for v in simplex:
                vertex_to_simplex[v].append(s)
        m.c0 = Constraint(vertices, rule=lambda m, v: m.lmbda[v] <= sum(m.y[s] for s in vertex_to_simplex[v]))
        m.c1 = Constraint(expr=sum(m.y[s] for s in simplices) == 1)
        return m

    if parent is None:
        # raise RuntimeError("parent 不能为空，必须提前声明好，否则无法添加到model中")
        m = Block(rule=_f)
    else:
        m = parent
        _f(m)
    return m
    # 生成变量
