"""
UC-SV (Unobserved Components Stochastic Volatility) model for inflation forecasting.
Based on Stock and Watson (2007) model.
"""

import numpy as np
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import calculate_errors


def gamrand(alpha, lambda_val):
    """
    Generate gamma random variable using rejection sampling.
    """
    if alpha > 1:
        d = alpha - 1/3
        c = 1 / np.sqrt(9 * d)
        
        while True:
            Z = np.random.randn()
            if Z > -1/c:
                V = (1 + c * Z) ** 3
                U = np.random.rand()
                if np.log(U) <= 0.5 * Z**2 + d - d*V + d*np.log(V):
                    return d * V / lambda_val
    else:
        x = gamrand(alpha + 1, lambda_val)
        return x * np.random.rand() ** (1/alpha)


def SVRW(ystar, h, omega2h, Vh):
    """
    Stochastic Volatility with Random Walk.
    Sample h from its conditional distribution.
    """
    T = len(h)
    
    # Parameters for the Gaussian mixture (7-component)
    pi = np.array([0.0073, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.2575])
    mui = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704
    sig2i = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])
    sigi = np.sqrt(sig2i)
    
    # Sample S from 7-point discrete distribution
    q = np.zeros((T, 7))
    for k in range(7):
        q[:, k] = pi[k] * stats.norm.pdf(ystar, loc=h + mui[k], scale=sigi[k])
    
    q = q / q.sum(axis=1, keepdims=True)
    
    # Sample S
    S = np.zeros(T, dtype=int)
    for t in range(T):
        S[t] = np.random.choice(7, p=q[t, :])
    
    # Sample h
    H = np.eye(T) - np.diag(np.ones(T-1), -1)
    invOmegah = np.diag(np.concatenate([[1/Vh], np.ones(T-1)/omega2h]))
    
    d = mui[S]
    invSigystar = np.diag(1/sig2i[S])
    
    Kh = H.T @ invOmegah @ H + invSigystar
    
    # Cholesky decomposition
    try:
        Ch = np.linalg.cholesky(Kh)
        hhat = np.linalg.solve(Kh, invSigystar @ (ystar - d))
        h_new = hhat + np.linalg.solve(Ch.T, np.random.randn(T))
    except:
        h_new = h  # Keep old value if numerical issues
    
    return h_new, S


def ucsv(y, display=False, nloop=4000, burnin=1000):
    """
    UC-SV model: Unobserved Components with Stochastic Volatility.
    
    Parameters:
    -----------
    y : ndarray
        Time series data
    display : bool
        Whether to plot results
    nloop : int
        Number of MCMC iterations
    burnin : int
        Number of burn-in iterations
    
    Returns:
    --------
    dict : Dictionary with 'tauhat', 'hhat', 'store_tau', 'store_h'
    """
    T = len(y)
    
    # Prior hyperparameters
    Vtau = 0.12
    Vh = 0.12
    atau = 10
    ltau = 0.04 * 9
    ah = 10
    lh = 0.03 * 9
    
    # Initialize
    omega2tau = ltau / (atau - 1)
    omega2h = lh / (ah - 1)
    h = np.log(np.var(y) * 0.8) * np.ones(T)
    
    # Storage
    store_omega2tau = np.zeros(nloop - burnin)
    store_omega2h = np.zeros(nloop - burnin)
    store_tau = np.zeros((nloop - burnin, T))
    store_h = np.zeros((nloop - burnin, T))
    
    # Precompute H matrix
    H = np.eye(T) - np.diag(np.ones(T-1), -1)
    
    # Posterior parameters
    newatau = (T - 1) / 2 + atau
    newah = (T - 1) / 2 + ah
    
    for loop in range(nloop):
        # Sample tau
        invOmegatau = np.diag(np.concatenate([[1/Vtau], np.ones(T-1)/omega2tau]))
        invSigy = np.diag(np.exp(-h))
        Ktau = H.T @ invOmegatau @ H + invSigy
        
        try:
            Ctau = np.linalg.cholesky(Ktau)
            tauhat = np.linalg.solve(Ktau, invSigy @ y)
            tau = tauhat + np.linalg.solve(Ctau.T, np.random.randn(T))
        except:
            tau = np.mean(y) * np.ones(T)
        
        # Sample h
        ystar = np.log((y - tau)**2 + 0.0001)
        h, _ = SVRW(ystar, h, omega2h, Vh)
        
        # Sample omega2tau
        tau_diff = tau[1:] - tau[:-1]
        newltau = ltau + np.sum(tau_diff**2) / 2
        omega2tau = 1 / gamrand(newatau, newltau)
        
        # Sample omega2h
        h_diff = h[1:] - h[:-1]
        newlh = lh + np.sum(h_diff**2) / 2
        omega2h = 1 / gamrand(newah, newlh)
        
        # Store after burn-in
        if loop >= burnin:
            i = loop - burnin
            store_tau[i, :] = tau
            store_h[i, :] = h
            store_omega2tau[i] = omega2tau
            store_omega2h[i] = omega2h
    
    tauhat = np.mean(store_tau, axis=0)
    hhat = np.mean(store_h, axis=0)
    
    if display:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(y, 'b-', label='Data')
        plt.plot(tauhat, 'r-', label='Trend (tau)')
        plt.legend()
        plt.show()
    
    return {'tauhat': tauhat, 'hhat': hhat, 'store_tau': store_tau, 'store_h': store_h}


def ucsv_rw(Y, npred, h_horizons=None):
    """
    Rolling window UC-SV forecasting.
    
    Parameters:
    -----------
    Y : ndarray
        Time series data (1D)
    npred : int
        Number of predictions
    h_horizons : list
        Forecast horizons (default: 1 to 12)
    
    Returns:
    --------
    ndarray : Predictions matrix (npred x len(h_horizons))
    """
    if h_horizons is None:
        h_horizons = list(range(1, 13))
    
    Y = np.array(Y).flatten()
    
    z = npred + len(h_horizons) - 1
    save_p = np.full(z, np.nan)
    
    for i in range(z, 0, -1):
        # Window of data
        y_window = Y[(z - i):(len(Y) - i)]
        
        # Fit UC-SV model
        result = ucsv(y_window, display=False, nloop=2000, burnin=500)
        
        # Use last value of tau as forecast
        save_p[z - i] = result['tauhat'][-1]
        
        print(f"iteration {z - i + 1}")
    
    # Organize predictions by horizon
    pr = np.full((npred, len(h_horizons)), np.nan)
    for i, h in enumerate(h_horizons):
        pr[:, i] = save_p[i:i+npred]
    
    return pr


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    y = np.cumsum(np.random.randn(200) * 0.1) + np.random.randn(200) * 0.5
    
    result = ucsv(y, display=False, nloop=1000, burnin=200)
    print(f"Trend mean: {np.mean(result['tauhat']):.4f}")
    print(f"Volatility mean: {np.mean(np.exp(result['hhat']/2)):.4f}")
