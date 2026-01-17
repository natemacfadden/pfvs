import gurobipy as gp
import math
import numpy as np

def coniMan(Binter, Z, Qmin, Qmax, M0min, M0max=None, K0min=None, K0max=None,
            Nsol: int = 100,
            timelimit: float=None,
            verbosity: int = 0):

    # helper variables
    h11 = Binter.shape[0]
    e0  = np.array([1]+[0]*(h11-1))
    
    # define the model
    model = gp.Model("coniBox")
    model.setParam('OutputFlag', verbosity >= 1)
    if timelimit is not None:
        model.setParam('TimeLimit', timelimit)  # 60 seconds

    # variables
    c_alpha = model.addMVar(
        (Binter.shape[1]+1,),
        lb=-float('inf'), ub=float('inf'),
        vtype=gp.GRB.INTEGER)

    projc = np.hstack([
        np.identity(Binter.shape[1],dtype=int),
        np.zeros((Binter.shape[1],1),dtype=int)
    ])
    
    # constraints
    # -----------
    # tadpole constraints
    topleft  = -Binter.T@Z@Binter
    topright = (-Binter.T@e0/2).reshape(-1,1)
    botleft  = (-e0.T@Binter/2).reshape(1,-1)
    tmp = np.block([[topleft, topright],
                    [botleft, 0]])
    model.addMQConstr(Q=tmp, c=None, sense='>', rhs=Qmin-1, xQ_L=c_alpha, xQ_R=c_alpha)
    model.addMQConstr(Q=tmp, c=None, sense='<', rhs=Qmax+1, xQ_L=c_alpha, xQ_R=c_alpha)

    # coni-M constraints
    model.addConstr(e0.T @ Binter @ projc@c_alpha >= M0min)
    if M0max is not None:
        model.addConstr(e0.T @ Binter @ projc@c_alpha <= M0max)

    # coni-K constraints
    K0_extractor = np.hstack([e0.T @ Z@Binter, [1]])
    if K0min is not None:
        model.addConstr(K0_extractor @ c_alpha >= K0min) # e0.T @ Z@Binter @ c + alpha >= K0min
    if K0max is not None:
        model.addConstr(K0_extractor @ c_alpha <= K0max) # e0.T @ Z@Binter @ c + alpha <= K0max

    # Kprime constraints
    #-self.K[0] + (self.M@self._cydata.kappa_cob@self.p)[0]
    #-K0_extractor + 
    
    #model.addMQConstr(
    #    Q = -Binter.T @ Z @ Binter,
    #    c = -

    # optimize for all solutions
    model.setParam('PoolSearchMode', 2)
    model.setParam('PoolSolutions', Nsol)

    model.optimize()

    # read the solutions
    sols = []
    for i in range(model.SolCount):
        model.setParam('SolutionNumber', i)

        sols.append(np.rint(c_alpha.xn).astype(int))

    return np.vstack(sols)