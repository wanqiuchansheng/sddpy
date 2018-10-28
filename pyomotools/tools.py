from random import random
from typing import List

from pyomo.environ import *
from pyomo.solvers.plugins.solvers.CPLEX import *
from pyomo.solvers.plugins.solvers.IPOPT import *
from pyomo.solvers.plugins.solvers.GUROBI import *
import numpy as np
import scipy


def model_str(model: ConcreteModel):
    print("===========obj==============")
    for obj in model.component_data_objects(Objective, active=True):
        print(obj.expr)
    print("========constraint==========")
    for cons in model.component_data_objects(Constraint, active=True):
        try:
            print("%s=%s" % (cons.name, cons.expr))
        except:
            print("%s=错误" % cons.name)

    print("=========variables==========")
    print(["%s lb=%s ub=%s" % (v.name, v.lb, v.ub) for v in model.component_data_objects(Var)])
    for v in model.component_data_objects(Var):
        try:
            print("%s=%f" % (v.name, value(v)))
        except:
            print("%s=错误" % v.name)

    print("==========params============")
    print(["%s=%f" % (v.name, value(v)) for v in model.component_data_objects(Param)])


def randint(a, b):
    """Our implementation of random.randint.

    The Python random.randint is not consistent between python versions
    and produces a series that is different in 3.x than 2.x.  So that we
    can support deterministic testing (i.e., setting the random.seed and
    expecting the same sequence), we will implement a simple, but stable
    version of randint()."""
    return int((b - a + 1) * random())


def unique_component_name(instance, name):
    # test if this name already exists in model. If not, we're good.
    # Else, we add random numbers until it doesn't
    if instance.component(name) is None:
        return name
    name += '_%d' % (randint(0, 9),)
    while True:
        if instance.component(name) is None:
            return name
        else:
            name += str(randint(0, 9))


def generate_gray_code(nbits):
    """Generates a Gray code of nbits as list of lists"""
    bitset = [0 for i in xrange(nbits)]
    # important that we copy bitset each time
    graycode = [list(bitset)]

    for i in xrange(2, (1 << nbits) + 1):
        if i % 2:
            for j in xrange(-1, -nbits, -1):
                if bitset[j]:
                    bitset[j - 1] = bitset[j - 1] ^ 1
                    break
        else:
            bitset[-1] = bitset[-1] ^ 1
        # important that we copy bitset each time
        graycode.append(list(bitset))

    return graycode


def generate_points(linegrids: List[List[float or int]]):
    # 根据坐标自动生成多维的点
    points = np.vstack(np.meshgrid(*linegrids)). \
        reshape(len(linegrids), -1).T
    return points


def generate_delaunay(variables, num=10, **kwds):
    """
    Generate a Delaunay triangulation of the D-dimensional
    bounded variable domain given a list of D variables.

    Requires numpy and scipy.

    Args:
        variables: A list of variables, each having a finite
            upper and lower bound.
        num (int): The number of grid points to generate for
            each variable (default=10).
        **kwds: All additional keywords are passed to the
          scipy.spatial.Delaunay constructor.

    Returns:
        A scipy.spatial.Delaunay object.
    """
    linegrids = []
    for v in variables:
        if v.has_lb() and v.has_ub():
            linegrids.append(np.linspace(v.lb, v.ub, num))
        else:
            raise ValueError(
                "Variable %s does not have a "
                "finite lower and upper bound.")
    # generates a meshgrid and then flattens and transposes
    # the meshgrid into an (npoints, D) shaped array of
    # coordinates
    points = generate_points(linegrids)
    return scipy.spatial.Delaunay(points, **kwds)


def cplex() -> CPLEXSHELL:
    """
    整数,线性,二次规划求解器
    """
    return SolverFactory('cplex',
                         executable="/opt/ibm/ILOG/CPLEX_Studio128/cplex/bin/x86-64_linux/cplex")  # type:CPLEXSHELL


def gurobi() -> GUROBISHELL:
    """
    线性,整数规划求解器
    """
    return SolverFactory('gurobi')  # type:CPLEXSHELL


def gurobi_python() -> GUROBISHELL:
    """
    线性,整数规划求解器
    """
    return SolverFactory('gurobi_persistent')  # type:CPLEXSHELL

def ipopt() -> IPOPT:
    """
    非线性规划求解器
    """
    return SolverFactory('ipopt')


def glpk():
    """
    线性,整数规划求解器
    """
    return SolverFactory('glpk')
