
from sddp.typedefinitions import *
from sddp.SDDP import *

CplexSolver = SolverFactory('cplex',
                            executable="/opt/ibm/ILOG/CPLEX_Studio128/cplex/bin/x86-64_linux/cplex")  # type:CPLEXSHELL


def newsvendormodel(oracle=DefaultCutOracle(), riskmeasure=Expectation()):
    Demand = [
        [10.0, 15.0],
        [12.0, 20.0],
        [8.0, 20.0]
    ]

    # Markov state purchase prices
    PurchasePrice = [5.0, 8.0]
    RetailPrice = 7.0
    # 一个，两个，两个
    Transition = [
        [[1.0]],
        [[0.6, 0.4]],
        [[0.3, 0.7], [0.3, 0.7]]
    ]

    def build(sp: Subproblem, stage: int, markov_state: int):
        model = sp.model
        # state
        model.stock = Var(bounds=(0, 100))
        model.stock0 = Var()
        model.initP = Param(initialize=5, mutable=True)
        model.stock0c = Constraint(expr=model.stock0 == model.initP)
        sp.add_state(model.stock, model.stock0, model.initP, model.stock0c)
        # other variables
        model.buy = Var(domain=NonNegativeReals)
        model.sell = Var(domain=NonNegativeReals)
        # Constrains
        model.D = Param(initialize=0, mutable=True)
        model.cs0 = Constraint(expr=model.sell <= model.D)
        model.cs1 = Constraint(expr=model.sell >= 0.5 * model.D)
        # model.cs = ConstraintList()
        # model.cs.add(Constraint(expr=model.sell <= model.D_noise))
        # model.cs.add(Constraint(expr=model.sell >= 0.5 * model.D_noise))
        # =添加噪音
        sp.add_noise_constraint(Demand[stage], model.D, [model.cs0, model.cs1])  # 添加噪音

        model.dc = Constraint(expr=model.stock == model.stock0 + model.buy - model.sell)
        # model.obj=Objective(expr=-model.sell * RetailPrice + model.buy * PurchasePrice[markov_state], sense=minimize)

        sp.obj = -model.sell * RetailPrice + model.buy * PurchasePrice[markov_state]

    m = createSDDPModel(build,
                        sense=Sense.Min,
                        stages=3,
                        objective_bound=-1000,
                        markov_transition=Transition,
                        solver=CplexSolver,
                        cut_oracle=oracle,
                        risk_measure=riskmeasure
                        )

    return m
