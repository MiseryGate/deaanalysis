import numpy as np
from scipy.optimize import linprog

def dea_with_slacks(X, Y, orientation="input", rts="CRS"):
    """
    DEA model with slacks (input excess and output shortfalls).
    
    Parameters
    ----------
    X : np.ndarray
        Inputs matrix (n x m), where n = number of DMUs, m = inputs.
    Y : np.ndarray
        Outputs matrix (n x s), where s = outputs.
    orientation : str
        "input" or "output" orientation.
    rts : str
        "CRS" for constant returns to scale, "VRS" for variable returns to scale.
        
    Returns
    -------
    scores : list
        Efficiency scores.
    lambdas_all : list
        Lambda weights for peers.
    slacks_in_all : list
        Input excesses.
    slacks_out_all : list
        Output shortfalls.
    """

    n, m = X.shape
    _, s = Y.shape

    scores = []
    lambdas_all = []
    slacks_in_all = []
    slacks_out_all = []

    for k in range(n):
        total_vars = n + 1 + m + s  # λ + θ/φ + s- + s+
        c = np.zeros(total_vars)

        if orientation == "input":
            c[n] = 1.0  # minimize θ
        else:
            c[n] = -1.0  # maximize φ (linprog minimizes → negative)

        A_ub, b_ub = [], []

        # Input constraints
        for i in range(m):
            row = np.zeros(total_vars)
            row[:n] = X[:, i]
            row[n] = -X[k, i]  # -θ * x_ki
            row[n + 1 + i] = 1.0  # s-
            A_ub.append(row)
            b_ub.append(0.0)

        # Output constraints
        for r in range(s):
            row = np.zeros(total_vars)
            row[:n] = -Y[:, r]
            row[n] = Y[k, r]  # φ * y_kr
            row[n + 1 + m + r] = 1.0  # s+
            A_ub.append(row)
            b_ub.append(0.0)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # VRS constraint: sum λ = 1
        A_eq, b_eq = None, None
        if rts == "VRS":
            A_eq = np.zeros((1, total_vars))
            A_eq[0, :n] = 1.0
            b_eq = np.array([1.0])

        bounds = [(0, None)] * total_vars  # all variables ≥ 0

        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")

        if res.success:
            lambdas = res.x[:n]
            theta_phi = res.x[n]
            slack_in = res.x[n + 1:n + 1 + m]
            slack_out = res.x[n + 1 + m:]

            if orientation == "input":
                eff = theta_phi
            else:
                eff = 1.0 / theta_phi if theta_phi > 1e-6 else np.inf
        else:
            lambdas = np.zeros(n)
            slack_in = np.zeros(m)
            slack_out = np.zeros(s)
            eff = np.nan

        scores.append(float(eff))
        lambdas_all.append(lambdas.tolist())
        slacks_in_all.append(slack_in.tolist())
        slacks_out_all.append(slack_out.tolist())

    return scores, lambdas_all, slacks_in_all, slacks_out_all