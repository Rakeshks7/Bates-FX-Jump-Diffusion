import numpy as np
import matplotlib.pyplot as plt

def bates_model_mc(S0, T, r, v0, kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j, N, M):
    """
    Monte Carlo simulation for the Bates model (Heston Stochastic Volatility + Merton Jump-Diffusion).
    
    Parameters:
    S0       : Initial spot price (e.g., FX rate)
    T        : Time to maturity (in years)
    r        : Risk-free rate (or interest rate differential r_d - r_f for FX)
    v0       : Initial variance
    kappa    : Mean reversion rate of variance
    theta    : Long-term mean variance
    sigma    : Volatility of variance (vol of vol)
    rho      : Correlation between asset and variance Brownian motions
    lambda_j : Jump intensity (expected number of jumps per year)
    mu_j     : Mean of jump size (log-normal)
    sigma_j  : Volatility of jump size
    N        : Number of time steps
    M        : Number of simulated paths
    """
    dt = T / N

    S = np.zeros((N + 1, M))
    v = np.zeros((N + 1, M))
    S[0] = S0
    v[0] = v0

    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    
    for t in range(1, N + 1):
        Z1 = np.random.standard_normal(M)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(M)
        
        dW1 = np.sqrt(dt) * Z1
        dW2 = np.sqrt(dt) * Z2

        n_jumps = np.random.poisson(lambda_j * dt, M)

        jump_multiplier = np.zeros(M)
        for i in range(M):
            if n_jumps[i] > 0:
                # Sum the normal random variables if multiple jumps occur in dt
                jump_multiplier[i] = np.sum(np.random.normal(mu_j, sigma_j, n_jumps[i]))

        v_prev = np.maximum(v[t-1], 0)
        v[t] = v_prev + kappa * (theta - v_prev) * dt + sigma * np.sqrt(v_prev) * dW2
        v[t] = np.maximum(v[t], 0)

        drift = (r - lambda_j * k - 0.5 * v_prev) * dt
        diffusion = np.sqrt(v_prev) * dW1
        
        S[t] = S[t-1] * np.exp(drift + diffusion + jump_multiplier)
        
    return S, v

if __name__ == "__main__":
    S0 = 1.0        
    T = 1.0         
    r = 0.05        
    v0 = 0.04       
    kappa = 2.0     
    theta = 0.04    
    sigma = 0.2     
    rho = -0.5      
    lambda_j = 3.0  
    mu_j = -0.05    
    sigma_j = 0.1   
    
    N = 252         
    M = 1000        

    S_paths, v_paths = bates_model_mc(S0, T, r, v0, kappa, theta, sigma, rho, lambda_j, mu_j, sigma_j, N, M)

    plt.figure(figsize=(10, 6))
    plt.plot(S_paths[:, :50], lw=1.5, alpha=0.8)
    plt.title('Bates Model Monte Carlo: FX Spot Paths with Jump-Diffusion')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Simulated Spot Price')
    plt.grid(True, alpha=0.3)
    plt.show()

    K = 1.0
    payoffs = np.maximum(S_paths[-1] - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoffs)
    print(f"Estimated European Call Option Price: {call_price:.4f}")