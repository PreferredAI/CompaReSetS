import numpy as np
from scipy.optimize import nnls

class Result(object):
    __doc__ = 'Result object for storing input and output data for omp.  When called from \n    `omp`, runtime parameters are passed as keyword arguments and stored in the \n    `params` dictionary.\n    Attributes:\n        X:  Predictor array after (optional) standardization.\n        y:  Response array after (optional) standarization.\n        ypred:  Predicted response.\n        residual:  Residual vector.\n        coef:  Solution coefficients.\n        active:  Indices of the active (non-zero) coefficient set.\n        err:  Relative error per iteration.\n        params:  Dictionary of runtime parameters passed as keyword args.   \n    '

    def __init__(self, **kwargs):
        self.X = None
        self.y = None
        self.ypred = None
        self.residual = None
        self.coef = None
        self.active = None
        self.err = None
        self.params = {}
        for key, val in kwargs.items():
            self.params[key] = val

    def update(self, coef, active, err, residual, ypred):
        """Update the solution attributes.
        """
        self.coef = coef
        self.active = active
        self.err = err
        self.residual = residual
        self.ypred = ypred


def omp(X, y, nonneg=True, ncoef=None, maxit=200, tol=0.001, ztol=1e-12, verbose=False):
    """Compute sparse orthogonal matching pursuit solution with unconstrained
    or non-negative coefficients.
    
    Args:
        X: Dictionary array of size n_samples x n_features. 
        y: Reponse array of size n_samples x 1.
        nonneg: Enforce non-negative coefficients.
        ncoef: Max number of coefficients.  Set to n_features/2 by default.
        tol: Convergence tolerance.  If relative error is less than
            tol * ||y||_2, exit.
        ztol: Residual covariance threshold.  If all coefficients are less 
            than ztol * ||y||_2, exit.
        verbose: Boolean, print some info at each iteration.
        
    Returns:
        result:  Result object.  See Result.__doc__
    """

    def norm2(x):
        return np.linalg.norm(x) / np.sqrt(len(x))

    result = Result(nnoneg=nonneg, ncoef=ncoef, maxit=maxit, tol=tol,
      ztol=ztol)
    if verbose:
        print(result.params)
    if type(X) is not np.ndarray:
        X = np.array(X)
    if type(y) is not np.ndarray:
        y = np.array(y)
    if X.shape[0] != len(y):
        return result
    result.y = y
    result.X = X
    if np.ndim(y) > 1:
        y = np.reshape(y, (len(y),))
    if ncoef is None:
        ncoef = int(X.shape[1] / 2)
    X_transpose = X.T
    active = []
    coef = np.zeros((X.shape[1]), dtype=float)
    residual = y
    ypred = np.zeros((y.shape), dtype=float)
    ynorm = norm2(y)
    err = np.zeros(maxit, dtype=float)
    if ynorm < tol:
        result.update(coef, active, err[0], residual, ypred)
        return result
    tol = tol * ynorm
    ztol = ztol * ynorm
    if verbose:
        print('\nIteration, relative error, number of non-zeros')
    for it in range(maxit):
        rcov = np.dot(X_transpose, residual)
        if nonneg:
            i = np.argmax(rcov)
            rc = rcov[i]
        else:
            i = np.argmax(np.abs(rcov))
            rc = np.abs(rcov[i])
        if rc < ztol:
            break
        else:
            if i not in active:
                active.append(i)
            if nonneg:
                coefi, _ = nnls(X[:, active], y)
            else:
                coefi, _, _, _ = np.linalg.lstsq(X[:, active], y)
            coef[active] = coefi
            residual = y - np.dot(X[:, active], coefi)
            ypred = y - residual
            err[it] = norm2(residual) / ynorm
            if verbose:
                print('{}, {}, {}'.format(it, err[it], len(active)))
        if err[it] < tol:
            break
        if len(active) >= ncoef:
            break
        if it == maxit - 1:
            break

    result.update(coef, active, err[:it + 1], residual, ypred)
    return result

