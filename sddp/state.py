# @version : python3.5
# @Time    : 2018/7/2 9:34
# @Author  : zzp
# @FileName: state.py
# from sddp.typedefinitions import *
from sddp.typedefinitions import SDDPModel, Subproblem


def setstates(m: SDDPModel, sp: Subproblem):
    """
    将上一阶段算出的结果作为当前时段的初始值
    """
    s = m.stages[sp.stage - 1]
    # TODO SDDP.jl号称是数值问题，
    # https://github.com/odow/SDDP.jl/issues/6#issuecomment-343022931
    for st, v in zip(sp.states, s.state):  # type:State,float
        lb = st.variable.lb
        up = st.variable.ub
        if v < lb:
            st.setvalue(lb)
        elif v > up:
            st.setvalue(v)
        else:
            st.setvalue(v)


