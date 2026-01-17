import gurobipy as gp
import math
import numpy as np

def coniMan(Binter, Z, Qmin, Qmax, M0min,
            M0max=None,
            K0min=None, K0max=None,
            Kbounds = 100,
            Mbounds = 100,
            Nsol: int = 100,
            timelimit: float=None,
            verbosity: int = 0):
    
    # helper variables
    h11 = Binter.shape[0]
    assert Binter.shape[1] == h11-1
    e0  = np.array([1]+[0]*(h11-1))
    
    # define the model
    model = gp.Model("coniBox")
    model.setParam('OutputFlag', verbosity >= 1)
    if timelimit is not None:
        model.setParam('TimeLimit', timelimit)  # 60 seconds

    # variables
    # =========
    # physically relevant variables
    # -----------------------------
    # (define K := (Z@Binter@c + alpha*e0) / K_scale)
    # (equiv: K*K_scale - alpha*e0 = Z@Binter@c)
    M       = [model.addVar(lb=-Mbounds, ub=Mbounds, vtype=gp.GRB.INTEGER) for i in range(h11)]
    K       = [model.addVar(lb=-Kbounds, ub=Kbounds, vtype=gp.GRB.INTEGER) for i in range(h11)]
    K_scale = model.addVar(lb=1, ub=float('inf'), vtype=gp.GRB.INTEGER)

    # internal variables
    # ------------------
    c        = [model.addVar(lb=-float('inf'), ub=float('inf'), vtype=gp.GRB.INTEGER) for i in range(h11-1)]
    alpha    = model.addVar(lb=-float('inf'), ub=float('inf'), vtype=gp.GRB.INTEGER)
    
    # constraints
    # ===========
    # M constraints
    # -------------
    # enforce M == Binter@c
    for i in range(h11):
        model.addConstr(
            M[i] == gp.quicksum([Binter[i,j] * c[j] for j in range(h11-1)])
        )

    # M0 constraints
    if M0min is not None: model.addConstr(M[0] >= M0min)
    if M0max is not None: model.addConstr(M[0] <= M0max)
    
    # K constraints
    # -------------
    # enforce K == Z@Binter@c
    for i in range(h11):
        if i == 0:
            alphaterm = alpha
        else:
            alphaterm = 0
    
        model.addQConstr(
            K[i]*K_scale - alphaterm,
            gp.GRB.EQUAL,
            gp.quicksum([(Z@Binter)[i,j] * c[j] for j in range(h11-1)])
        )

    # K0 constraints
    if K0min is not None: model.addConstr(K[0]+alpha >= K0min)
    if K0max is not None: model.addConstr(K[0]+alpha <= K0max)

    # tadpole constraints
    # -------------------
    model.addQConstr(K[0]*M[0] + gp.quicksum([K[i]*M[i] for i in range(1,h11)]) <= -Qmin)
    model.addQConstr(K[0]*M[0] + gp.quicksum([K[i]*M[i] for i in range(1,h11)]) >= -Qmax)

    # optimize for all solutions
    # ==========================
    model.setParam('PoolSearchMode', 2)
    model.setParam('PoolSolutions', Nsol)

    model.optimize()

    # read the solutions
    Ks = []
    Ms = []
    for i in range(model.SolCount):
        model.setParam('SolutionNumber', i)

        Kval = [K[i].xn for i in range(h11)]
        Mval = [M[i].xn for i in range(h11)]

        Ks.append(np.rint(Kval).astype(int))
        Ms.append(np.rint(Mval).astype(int))

    return np.vstack(Ks), np.vstack(Ms)
