import sys
import numpy as np
from ibcdfo.pounders.formquad import formquad
import ipdb

try:
    from minqsw import minqsw
except ModuleNotFoundError as e:
    print(e)
    sys.exit("Ensure a python implementation of MINQ is available. For example, clone https://github.com/POptUS/minq and add minq/py/minq5 to the PYTHONPATH environment variable")


def complete_nonpoised_set(Y, basis, x0, delta):
    # Interpret the first row of Y as the center of the set
    p_init, _ = Y.shape
    n = len(x0)
    Y = Y - np.tile(x0, (p_init, 1))

    M = compute_vandermonde(Y, np.zeros(n), basis)
    p_init, p = M.shape
    L = np.eye(p)
    flag = False

    for i in range(p):
        # Step 1: Move a point up to normalize
        if i < p_init:
            ML = M @ L
            liyi, ji = np.abs(ML[:, i]).max(), np.abs(ML[:, i]).argmax()
            if ML[ji, i] == 0:
                flag = True
            elif ji != i:
                # Swap the two rows of Y
                Yji = Y[ji, :].copy()
                Y[ji, :] = Y[i, :].copy()
                Y[i, :] = Yji
        if i >= p_init or flag:
            c, g, H = get_lagrange_polynomiali(L, i, n, basis)
            Lows = -1.0 * delta * np.ones(n)
            Upps = delta * np.ones(n)
            s1, val1, minq_err1, _ = minqsw(0, -1.0 * g, -1.0 * H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
            s2, val2, minq_err2, _ = minqsw(0, g, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))

            if abs(c - val1) > abs(c + val2):
                Y = np.vstack((Y, s1.T))
                liyi = c - val1
            else:
                Y = np.vstack((Y, s2.T))
                liyi = c + val2
            flag = False
        M = compute_vandermonde(Y, np.zeros(n), basis)

        # Normalize
        L[:, i] = L[:, i] / liyi

        # Orthogonalize
        for j in range(p):
            if i != j:
                ljyi = np.dot(M[i, :], L[:, j])
                L[:, j] = L[:, j] - ljyi * L[:, i]

    return Y, L

def get_lagrange_polynomiali(L, i, n, basis):
    Li = L[:, i]
    c = Li[0]
    g = Li[1:(n + 1)]
    H = np.zeros((n, n))
    if basis == 'quadratic':
        cont = n + 1
        for j in range(n):
            H[j:n, j] = Li[cont:cont + n - (j - 1)]
            cont = cont + n - (j - 1)
        H = H + H.T - np.diag(np.diag(H))
    return c, g, H

def compute_vandermonde(Y, x0, basis):
    p_curr, s = Y.shape
    Y = Y - np.tile(x0, (p_curr, 1))

    if basis == 'quadratic':
        phi_Q = []
        for i in range(p_curr):
            y = Y[i, :]
            aux_H = np.outer(y, y) - 0.5 * np.diag(y**2)
            aux = []
            for j in range(s):
                aux = np.append(aux, aux_H[j:s, j])
            phi_Q.append(aux)
        M = np.column_stack((np.ones(p_curr), Y, phi_Q))
    elif basis == 'linear':
        M = np.column_stack((np.ones(p_curr), Y))
    return M


def compute_model_from_scratch(Y, FY, x0, F0, delta):
    m, n = Y.shape
    Y_shifted = np.real((Y - np.tile(x0, (m, 1))) / delta)
    F_shifted = FY - F0

    A = 0.5 * (np.dot(Y_shifted, Y_shifted.T) ** 2)

    W = np.block([[A, np.ones((m, 1)), Y_shifted],
                  [np.ones((1, m)), np.zeros((1, n + 1))],
                  [Y_shifted.T, np.zeros((n, n + 1))]])

    b = np.concatenate((F_shifted, np.zeros(n + 1)))

    tol_svd = np.finfo(float).eps ** 5
    U, S, V = np.linalg.svd(W)

    indices = np.where(S < tol_svd)
    S[indices] = tol_svd
    Sinv = np.diag(1.0 / S)
    invW = V.T @ Sinv @ U.T
    alpha = np.real(invW @ b)

    c = alpha[m]
    g = alpha[m + 1:m + n + 1]
    H = np.zeros((n, n))
    for k in range(m):
        H += alpha[k] * np.outer(Y_shifted[k, :], Y_shifted[k, :])

    # Rescale
    g = g / delta
    H = H / (delta ** 2)

    return c, g, H, invW


def compute_lambda_mfn(invW, Y, x0, delta):
    m, n = Y.shape
    Y_shifted = (Y - np.tile(x0, (m, 1))) / delta

    Lambda_vec = np.zeros(m)
    suggestions = np.zeros((m, n))

    for j in range(m):
        c, g, H = get_mfn_lagrange_polynomialj(invW, Y_shifted, j)
        g = g / delta
        H = H / (delta ** 2)
        Lows = -1.0 * delta * np.ones(n)
        Upps = delta * np.ones(n)
        s1, val1, minq_err1, _ = minqsw(0, -1.0 * g, -1.0 * H, Lows.T, Upps.T, 0, np.zeros((n, 1)))
        s2, val2, minq_err2, _ = minqsw(0, g, H, Lows.T, Upps.T, 0, np.zeros((n, 1)))

        if abs(c - val1) > abs(c + val2):
            Lambda_vec[j] = abs(c - val1)
            suggestions[j, :] = s1.flatten()
        else:
            Lambda_vec[j] = abs(c + val2)
            suggestions[j, :] = s2.flatten()

    try:
        Lambda = np.max(Lambda_vec)
        suggestion_ind = np.argmax(Lambda_vec)
        suggestion = x0 + suggestions[suggestion_ind, :]
    except:
        print("Y: ", Y)
        Lambda = np.inf
        suggestion_ind = 0
        suggestion = x0
    
    return Lambda, suggestion_ind, suggestion



def get_mfn_lagrange_polynomialj(invW, Y, j):
    m, n = Y.shape
    c = invW[m, j]
    g = invW[m + 1:m + n + 1, j]
    H = np.zeros((n, n))

    for k in range(m):
        H = H + invW[k, j] * np.outer(Y[k, :], Y[k, :])

    return c, g, H

# one of the two methods actually seen by anatra:
def formquad_lagrange(X, F, delta, xkin, npmax, Pars, vf, Mind):
    # Internal parameters:
    nf, n = X.shape
    m = F.shape[1]
    valid = False
    C = np.zeros(m)
    G = np.zeros((n, m))
    H = np.zeros((n, n, m))

    try:
        Mind = Mind.astype(int)
    except:
        Mind = np.array(Mind).astype(int)

    # Does Mind contain a sufficiently affinely independent set of points?
    Pars2 = np.copy(Pars)
    Pars2[1] = Pars2[0]
    Pars2 = np.append(Pars2, Pars[2])
    mp = len(Mind)
    
    D = np.zeros((nf, n))  # Scaled displacements
    Nd = np.zeros(nf)
    for i in range(nf - 1, -1, -1):
        D[i, :] = X[i, :] - X[xkin, :]
        Nd[i] = np.linalg.norm(D[i, :])
    
    #if mp < n + 1:
    #    aff = False
    #else:
    #    try:
    #        _, _, aff, _, _, _ = formquad(X[Mind, :], F[Mind, :], delta, Mind.tolist().index(xkin), npmax, Pars2, 1)
    ##    except:
    #        Mind = np.append(Mind, xkin)
    #        _, _, aff, _, _, _ = formquad(X[Mind, :], F[Mind, :], delta, Mind.tolist().index(xkin), npmax, Pars2, 1)
    
    if mp < n + 1: #not aff:
        # find n+1 sufficiently affinely independent points
        Mdir, mp, _, _, _, Mind = formquad(X[:nf, :], F[:nf, :], delta, xkin, n + 2, Pars2, 0)
        Mdir = delta * Mdir
        if mp < n + 1:  # cannot continue until we get more points
            if vf:
                Mdir = []
                return Mdir, mp, valid, C, G, H, Mind
            return Mdir, mp, valid, C, G, H, Mind
    
    # Check the validity of the point set indexed by Mind
    # Note that for geometry-checking purposes, the rhs (FY) DOES NOT MATTER:
    
    _, _, _, invW = compute_model_from_scratch(D[Mind, :], np.zeros(len(Mind)), np.zeros(n), 0, delta)
    Lambda, _, _ = compute_lambda_mfn(invW, D[Mind, :], np.zeros(n), delta)
    Mdir = []
    if Lambda < Pars[1]:  # small threshold
        valid = True
    if vf:
        return np.atleast_2d(Mdir), mp, valid, C, G, H, Mind
    # We entered formquad_lagrange to get an actual model:
    for k in range(m):
        c, g, h, _ = compute_model_from_scratch(X[Mind, :], F[Mind, k], X[xkin, :], F[xkin, k], delta)
        C[k] = c
        G[:, k] = g
        H[:, :, k] = h

    return np.atleast_2d(Mdir), mp, valid, C, G, H, Mind

# the other method actually seen by anatra:
def model_improvement(X, delta, xkin, Pars, Mind):
    # Internal parameters:
    nf, n = X.shape

    mp = len(Mind)

    # Precompute the scaled displacements
    D = np.zeros((mp, n))  # Scaled displacements
    Nd = np.zeros(mp)
    for i in range(mp - 1, -1, -1):
        D[i, :] = X[Mind[i], :] - X[xkin, :]
        Nd[i] = np.linalg.norm(D[i, :])
    
    keepers = np.where(Nd <= Pars[0] * delta)[0]
    try:
        keepers = keepers.astype(int)
        Mind = Mind.astype(int)
    except:
        keepers = np.array(keepers).astype(int)
        Mind = np.array(Mind).astype(int)
    Mind = Mind[keepers]
    mp = len(Mind)
    D = D[keepers, :]
    if D.size == 0:
        print("whoops, D is empty?")
        return [], Mind, False
    _, _, _, W = compute_model_from_scratch(D, np.zeros(mp), np.zeros(n), 0.0, delta)
    Lambda, suggestion_ind, suggestion = compute_lambda_mfn(W, D, np.zeros(n), delta)

    Mdir = []
    count = 0
    successful = False
    N = 1
    while Lambda > Pars[1] and count < N:
        D[suggestion_ind, :] = suggestion
        Mdir.append(suggestion)
        Mind[suggestion_ind] = nf + count

        _, _, _, W = compute_model_from_scratch(D, np.zeros(mp), np.zeros(n), 0.0, delta)
        Lambda, suggestion_ind, suggestion = compute_lambda_mfn(W, D, np.zeros(n), delta)

        count += 1

    if count <= N:
        successful = True

    return Mdir, Mind, successful




