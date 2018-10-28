# @version : python3.5
# @Time    : 2018/7/5 11:11
# @Author  : zzp
# @FileName: print.py

from sddp.typedefinitions import SDDPModel, Settings, SolutionLog
# from prettytable import PrettyTable


def atol(x, y): abs(x - y)


def rtol(x, y):
    ar = abs(x - y)
    br = (1 + abs(y))
    return ar / br


def humanize(value: int) -> str:
    if 1000 > value > -1000:
        return "%5d" % value
    else:
        return "%5.1f" % value


def printheader(m: SDDPModel, solve_type="todo"):
    n = m.nstages
    print("""
-------------------------------------------------------------------------------
                                  SDDP
-------------------------------------------------------------------------------
    Solver:
        %s
    Model:
        Stages:         %d
        States:         %d
        Subproblems:    %d
        Value Function: ===
-------------------------------------------------------------------------------
    """ % (solve_type,
           m.nstages,
           m.stages[0].subproblems[0].nstates,
           sum(len(s.subproblems) for s in m.stages)
           ))
    print("              Objective              |  Cut  Passes    Simulations   Total    ")
    print("     Simulation       Bound   % Gap  |   #     Time     #    Time    Time     ")
    print("-------------------------------------------------------------------------------")


def print_solutionLog(l: SolutionLog, printmean: bool = False, is_min=True):
    if printmean:
        bound_string = "     " + "%8.3f" % (0.5 * (l.lower_statistical_bound + l.upper_statistical_bound)) + "     "
        rtol_string = "      "
    else:
        bound_string = "%8.3f" % (l.lower_statistical_bound) + "  " + "%8.3f" % (l.upper_statistical_bound)
        if is_min:
            tt = rtol(l.lower_statistical_bound, l.bound)
            tol = 100 * tt
        else:
            tol = -100 * rtol(l.upper_statistical_bound, l.bound)
        rtol_string = " %5.1f" % tol

    res_str = "%s %8.3f %s   | %s %8.1f %s %s %8.1f" % (bound_string,
                                                        l.bound,
                                                        rtol_string,
                                                        humanize(l.iteration),
                                                        l.timecuts,
                                                        humanize(l.simulations),
                                                        humanize(l.timesimulations),
                                                        l.timetotal
                                                        # humanize(l.timetotal)
                                                        )
    print(res_str)


def printfooter(m: SDDPModel, settings: Settings, status, timer):
    print("-------------------------------------------------------------------------------")
    # if settings.print_level > 1:
    #     print_timer(io, timer, title="Timing statistics")
    #     print(io, "\n")
    # end
    print("""    Other Statistics:
        Iterations:         %d
        Termination Status: %s
===============================================================================""" % (m.log[-1].iteration, status))
