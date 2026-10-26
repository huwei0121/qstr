import sys

import numpy as np

from ibcdfo.pounders.bqmin import bqmin
from ibcdfo.pounders.checkinputss import checkinputss
from ibcdfo.pounders.prepare_outputs_before_return import prepare_outputs_before_return
from lagrange_utils import formquad_lagrange
from lagrange_utils import model_improvement

def _default_model_par_values(n):
    par = np.zeros(3)
    par[0] = np.maximum(10.0, np.sqrt(n))
    par[1] = par[0]
    par[2] = 0.001

    return par


def _default_model_np_max(n):
    return int((n + 1) * (n + 2) /2)#2 * n + 1


def _default_prior():
    Prior = {}
    Prior["nfs"] = 0
    Prior["X_init"] = []
    Prior["F_init"] = []
    Prior["xk_in"] = 0

    return Prior

def get_min_sample_delta(L, H, valid, epsilon, r):

    if valid:
        Lnew = np.max(np.linalg.eigvals(H))
        L = np.maximum(Lnew, 1.0)
        print("Lnew: ", L)
    min_sample_delta = np.sqrt(2 * r * epsilon / L)
    return min_sample_delta, L


def does_noise_dominate(Mind, hF, xk_in, epsilon):

    return np.all(np.abs(hF[Mind] - hF[xk_in]) < epsilon)



def anatra(Ffun, X_0, n, nf_max, g_tol, delta_0, m, Low, Upp, eps_bound=None, Prior=None, Options=None, Model=None):
    """
    Much of this code is modified from our code, POUNDers, which is more actively maintained at
    https://github.com/POptUS/IBCDFO

    This software comes with no warranty, is not bug-free, and is not for
    industrial use or public distribution.
    Direct requests can go to mmenickelly@anl.gov.

    --INPUTS-----------------------------------------------------------------
    Ffun    [f h] Function handle so that Ffun(x) evaluates F (@calfun)
    X_0     [dbl] [1-by-n] Initial point (zeros(1,n))
    n       [int] Dimension (number of continuous variables)
    nf_max  [int] Maximum number of function evaluations (>n+1) (100)
    g_tol   [dbl] Tolerance for the 2-norm of the model gradient (1e-4)
    delta_0 [dbl] Positive initial trust region radius (.1)
    m       [int] Number of components returned from Ffun
    Low     [dbl] [1-by-n] Vector of lower bounds (-Inf(1,n))
    Upp     [dbl] [1-by-n] Vector of upper bounds (Inf(1,n))
    eps_bound [dbl] An estimate of an upper bound on absolute noise of hfun(Ffun(x))

    Prior   [dict] of past evaluations of values Ffun with keys:
        X_init  [dbl] [nfs-by-n] Set of initial points
        F_init  [dbl] [nfs-by-m] Set of values for points in X_init
        xk_in   [int] Index in X_init for initial starting point
        nfs     [int] Number of function values in F_init known in advance

    Options [dict] of options to the method
        printf   [int] 0 No printing to screen (default)
                       1 Debugging level of output to screen
                       2 More verbose screen output
        spsolver       [int] Trust-region subproblem solver flag (2)
        hfun           [f h] Function handle for mapping output from F
        combinemodels  [f h] Function handle for combining models of F

    Model   [dict] of options for model building
        np_max  [int] Maximum number of interpolation points (>n+1) (2*n+1)
        Par     [1-by-4] list for formquad


    --OUTPUTS----------------------------------------------------------------
    X       [dbl] [nf_max+nfs-by-n] Locations of evaluated points
    F       [dbl] [nf_max+nfs-by-m] Ffun values of evaluated points in X
    hF      [dbl] [nf_max+nfs-by-1] Composed values h(Ffun) for evaluated points in X
    flag    [dbl] Termination criteria flag:
                  = 0 normal termination because of grad,
                  > 0 exceeded nf_max evals,   flag = norm of grad at final X
                  = -1 if input was fatally incorrect (error message shown)
                  = -2 if a valid model produced X[nf] == X[xk_in] or (mdec == 0, hF[nf] == hF[xk_in])
                  = -3 error if a NaN was encountered
                  = -4 error in TRSP Solver
                  = -5 unable to get model improvement with current parameters
    xk_in    [int] Index of point in X representing approximate minimizer
    """
    if Options is None:
        Options = {}

    if Model is None:
        Model = {}
        Model["Par"] = _default_model_par_values(n)
        Model["np_max"] = _default_model_np_max(n)
    else:
        if "Par" not in Model:
            Model["Par"] = _default_model_par_values(n)
        if "np_max" not in Model:
            Model["np_max"] = _default_model_np_max(n)

    if Prior is None:
        Prior = _default_prior()
    else:
        key_list = ["nfs", "X_init", "F_init", "xk_in"]
        assert set(Prior.keys()) == set(key_list), f"Prior keys must be {key_list}"
        Prior["X_init"] = np.atleast_2d(Prior["X_init"])
        if Prior["X_init"].ndim == 2 and Prior["X_init"].shape[1] == 1:
            Prior["X_init"] = Prior["X_init"].T

    nfs = Prior["nfs"]
    delta = delta_0
    spsolver = Options.get("spsolver", 2)
    delta_max = Options.get("delta_max", min(0.5 * np.min(Upp - Low), (10**3) * delta))
    delta_min = Options.get("delta_min", min(delta * (10**-13), g_tol / 10))
    min_sample_delta = delta
    gamma_dec = Options.get("gamma_dec", 0.5)
    gamma_inc = Options.get("gamma_inc", 2)
    eta_1 = Options.get("eta1", 0.25)
    eta_2 = 0.75
    printf = Options.get("printf", 0)
    delta_inact = Options.get("delta_inact", 0.75)

    if "hfun" in Options:
        hfun = Options["hfun"]
        combinemodels = Options["combinemodels"]
    else:
        hfun = lambda F: np.sum(F**2)
        from ibcdfo.pounders.general_h_funs import leastsquares as combinemodels

    # choose your spsolver
    if spsolver == 2:
        try:
            from minqsw import minqsw
        except ModuleNotFoundError as e:
            print(e)
            sys.exit("Ensure a python implementation of MINQ is available. For example, clone https://github.com/POptUS/minq and add minq/py/minq5 to the PYTHONPATH environment variable")

    [flag, X_0, _, F_init, Low, Upp, xk_in] = checkinputss(Ffun, X_0, n, Model["np_max"], nf_max, g_tol, delta_0, Prior["nfs"], m, Prior["X_init"], Prior["F_init"], Prior["xk_in"], Low, Upp)
    if flag == -1:
        X = []
        F = []
        hF = []
        return X, F, hF, flag, xk_in
    eps = np.finfo(float).eps  # Define machine epsilon
    if printf:
        print("  nf   epsilon   delta    fl  np       f0           g0       ierror")
        progstr = "%4i %9.2e %9.2e %2i %3i  %11.5e %12.4e %11.3e\n"  # Line-by-line
    if Prior["nfs"] == 0:
        X = np.vstack((X_0, np.zeros((nf_max - 1, n))))
        F = np.zeros((nf_max, m))
        hF = np.zeros(nf_max)
        nf = 0  # in Matlab this is 1
        mean, var = Ffun(X[nf])
        F_0 = np.atleast_2d(mean)
       
        if eps_bound is not None:
            epsilon = eps_bound
            r = 1.0
        else:
            epsilon = np.sqrt(var)
            r = 2.0 # this should be exposed as an optional input!!!

        if F_0.shape[1] != m:
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -1)
            return X, F, hF, flag, xk_in
        F[nf] = F_0
        if np.any(np.isnan(F[nf])):
            X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
            return X, F, hF, flag, xk_in
        if printf:
            print("%4i    Initial point  %11.5e\n" % (nf, hfun(F[nf, :])))
    else:
        X = np.vstack((Prior["X_init"], np.zeros((nf_max, n))))
        F = np.vstack((Prior["F_init"], np.zeros((nf_max, m))))
        hF = np.zeros(nf_max + nfs)
        nf = nfs - 1
        nf_max = nf_max + nfs
    for i in range(nf + 1):
        hF[i] = hfun(F[i])
    Res = np.zeros(np.shape(F))
    Cres = F[xk_in]
    ng = np.nan  # Needed for early termination, e.g., if a model is never built

    Mind = np.arange(nf)  # Initial Mind
    L = 1.0 # initial guess of Lipschitz constant, should be exposed as a param eventually
    best_in = xk_in

    while nf + 1 < nf_max:
        #  1a. Compute the interpolation set.
        Res[: nf + 1, :] = (F[: nf + 1, :] - Cres)
        Mdir, mp, valid, Cres, Gres, Hres, Mind = formquad_lagrange(X[0: nf + 1, :], Res[0: nf + 1, :],
                                                                   max(delta, min_sample_delta), xk_in, Model["np_max"],
                                                                   Model["Par"], 0, Mind)

        if mp < n + 1:
            for i in range(min(Mdir.shape[0], nf_max - (nf + 1))): #range(int(min(n + 1 - mp, nf_max - (nf + 1)))):
                #Mdiri, _ = bmpts(X[xk_in], np.atleast_2d(Mdir[i, :]), Low, Upp, delta, Model["Par"][2])
                Mdiri = Mdir[i, :]
                nf += 1
                X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdiri))
                F[nf], _ = Ffun(X[nf])
                if np.any(np.isnan(F[nf])):
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                    return X, F, hF, flag, xk_in
                hF[nf] = hfun(F[nf])
                if printf:
                    print("%4i   Geometry point  %11.5e\n" % (nf, hF[nf]))
                Res[nf, :] = (F[nf, :] - Cres)
            if nf + 1 >= nf_max:
                break
            _, mp, valid, Cres, Gres, Hres, Mind = formquad_lagrange(X[0: nf + 1, :], Res[0: nf + 1, :],
                                                                       max(delta, min_sample_delta), xk_in,
                                                                       Model["np_max"],
                                                                       Model["Par"], 0, Mind)
            if mp < n + 1:
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -5)
                return X, F, hF, flag, xk_in
        #  1b. Update the quadratic model
        c = hF[xk_in]
        G, H = combinemodels(Cres, Gres, Hres)
        min_sample_delta, L = get_min_sample_delta(L, H, valid, epsilon, r)
        ind_Lownotbinding = (X[xk_in] > Low) * (G.T > 0)
        ind_Uppnotbinding = (X[xk_in] < Upp) * (G.T < 0)
        ng = np.linalg.norm(G * (ind_Lownotbinding + ind_Uppnotbinding).T, 2)
        if printf:
            IERR = np.zeros(len(Mind))
            for i in range(len(Mind)):
                D = X[Mind[i]] - X[xk_in]
                IERR[i] = (c - hF[Mind[i]]) + [D @ (G + 0.5 * H @ D)]
            if np.any(hF[Mind] == 0.0):
                ierror = np.nan
            else:
                ierror = np.linalg.norm(IERR / np.abs(hF[Mind]), np.inf)
            print(progstr % (nf, epsilon, delta, valid, mp, hF[xk_in], ng, ierror))
            if printf >= 2:
                jerr = np.zeros((len(Mind), m))
                for i in range(len(Mind)):
                    D = X[Mind[i]] - X[xk_in]
                    for j in range(m):
                        jerr[i, j] = (Cres[j] - F[Mind[i], j]) + D @ (Gres[:, j] + 0.5 * Hres[:, :, j] @ D)
                print(jerr)

        # 3. Solve the subproblem min{G.T * s + 0.5 * s.T * H * s : Lows <= s <= Upps }
        Lows = np.maximum(Low - X[xk_in], -delta * np.ones((np.shape(Low))))
        Upps = np.minimum(Upp - X[xk_in], delta * np.ones((np.shape(Upp))))
        if spsolver == 1:  # Stefan's crappy 10line solver
            [Xsp, mdec] = bqmin(H, G, Lows, Upps)
        elif spsolver == 2:  # Arnold Neumaier's minq5
            [Xsp, mdec, minq_err, _] = minqsw(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
            if minq_err < 0:
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -4)
                return X, F, hF, flag, xk_in
        # elif spsolver == 3:  # Arnold Neumaier's minq8
        #     [Xsp, mdec, minq_err, _] = minq8(0, G, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
        #     assert minq_err >= 0, "Input error in minq"
        Xsp = Xsp.squeeze()
        step_norm = np.linalg.norm(Xsp, np.inf)

        # 4. Evaluate the function at the old and new point
        if (step_norm >= 0.01 * delta or valid) and not (mdec == 0 and not valid):
            Xsp = np.minimum(Upp, np.maximum(Low, X[xk_in] + Xsp))  # Temp safeguard; note Xsp is not a step anymore

            # Project if we're within machine precision
            for i in range(n):  # This will need to be cleaned up eventually
                if (Upp[i] - Xsp[i] < eps * abs(Upp[i])) and (Upp[i] > Xsp[i] and G[i] >= 0):
                    Xsp[i] = Upp[i]
                    print("eps project!")
                elif (Xsp[i] - Low[i] < eps * abs(Low[i])) and (Low[i] < Xsp[i] and G[i] >= 0):
                    Xsp[i] = Low[i]
                    print("eps project!")
            
            nf += 1
            X[nf] = Xsp
            F[nf], new_variance = Ffun(X[nf])
            if len(Mind) < Model["np_max"]:
                Mind = np.append(Mind, nf)  # try to add new point to the next model
            else:
                oldest = int(np.argmin(Mind))
                Mind[oldest] = nf

            if np.any(np.isnan(F[nf])):
                X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                return X, F, hF, flag, xk_in
            hF[nf] = hfun(F[nf])

            if mdec != 0:
                rho = (hF[nf] - hF[xk_in] - 2 * r * epsilon) / mdec  # Katya and Albert rho test
            else:
                if hF[nf] == hF[xk_in]:
                    X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -2)
                    return X, F, hF, flag, xk_in
                else:
                    rho = np.inf * np.sign(hF[nf] - hF[xk_in])

            # 4a. Update the center
            if (rho >= eta_1):
                # Update model to reflect new center
                xk_in = nf  # Change current center
                if eps_bound is None:
                    epsilon = np.sqrt(new_variance) 

            # 4b. Update the trust-region radius:
            if (rho >= eta_1):
                if step_norm > delta_inact * delta:
                    delta = min(delta * gamma_inc, delta_max)
            elif valid:
                delta = max(delta * gamma_dec, delta_min)

        else:  # Don't evaluate f at Xsp
            rho = -1  # Force yourself to do a model-improving point
            if printf:
                print("Warning: skipping sp soln!-----------")
        
        # 5a. Do we need to reset?
        best_in = int(np.argmin(hF[0:nf + 1]))
        if hF[best_in] < hF[xk_in] - r * epsilon:
            xk_in = best_in
            valid = False

        # 5b. Improve model if necessary
        if not valid and (nf + 1 < nf_max) and (rho < eta_1):  # Implies xk_in, delta unchanged
            # Need to check because model may be valid after Xsp evaluation
            _, _, valid, _, _, _, Mind = formquad_lagrange(X[: nf + 1, :], F[: nf + 1, :],
                                                        max(delta, min_sample_delta), xk_in,
                                                        Model["np_max"],
                                                        Model["Par"], 1, Mind)
            if not valid:
                Mdir, Mind, _ = model_improvement(X[: nf + 1, :], max(delta, min_sample_delta), xk_in, Model["Par"], Mind)
                for j in range(min(np.shape(Mdir)[0], nf_max - nf - 1)):
                    nf += 1
                    X[nf] = np.minimum(Upp, np.maximum(Low, X[xk_in] + Mdir[j]))
                    F[nf], _ = Ffun(X[nf])
                    if np.any(np.isnan(F[nf])):
                        X, F, hF, flag = prepare_outputs_before_return(X, F, hF, nf, -3)
                        return X, F, hF, flag, xk_in
                    hF[nf] = hfun(F[nf])
                    if printf:
                        print("%4i   Model point     %11.5e\n" % (nf, hF[nf]))

    if printf:
        print("Number of function evals exceeded")
    flag = ng
    return X, F, hF, flag, xk_in
