# Bates Model FX Pricing Engine: Jump-Diffusion & Stochastic Volatility

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

## Overview
Standard continuous-time models like Garman-Kohlhagen (an extension of Black-Scholes for FX) assume frictionless liquidity and continuous price paths. In a fragmented macroeconomic environment subject to geopolitical shocks, capital controls, and clearing sanctions, these models severely underprice tail risk. 

This repository implements a **Bates Model Monte Carlo simulation** to accurately price FX derivatives in discontinuous markets. By combining the Heston model's stochastic volatility with Merton's jump-diffusion, this engine captures both volatility clustering and sudden liquidity gaps.



## Mathematical Framework
The core engine simulates the following system of Stochastic Differential Equations (SDEs) under the risk-neutral measure:

**1. Spot Exchange Rate Dynamics (Jump-Diffusion):**
$$\frac{dS_t}{S_t} = (r_d - r_f - \lambda \bar{k})dt + \sqrt{V_t}dW_t^{(1)} + k dq_t$$

**2. Variance Dynamics (Mean-Reverting Stochastic Volatility):**
$$dV_t = \kappa(\theta - V_t)dt + \sigma \sqrt{V_t}dW_t^{(2)}$$

Where:
* $dW_t^{(1)}$ and $dW_t^{(2)}$ are Wiener processes with correlation $\rho$.
* $dq_t$ is a Poisson jump process with intensity $\lambda$.
* $k$ is the random percentage jump size, distributed log-normally.
* The term $- \lambda \bar{k} dt$ is the martingale compensator to ensure risk-neutrality.

## Features
* **Euler-Maruyama Discretization:** Robust numerical simulation of the Bates SDEs.
* **Full Truncation Scheme:** Prevents the Feller condition from failing during extreme volatility spikes, ensuring variance remains non-negative.
* **Vectorized Poisson Jumps:** Highly optimized array operations for rolling discontinuous jump probabilities across thousands of simulated paths.
* **Geopolitical Calibration:** Architecture allows for jump intensity ($\lambda$) to be calibrated against alternative data (e.g., sovereign credit default swaps, sanction indices).

## Disclaimer

For Educational and Research Purposes Only. This software is provided "as is" and is not intended as financial advice, trading signals, or a production-ready risk management system. Do not use this code to allocate real capital. The author is not responsible for any financial losses incurred from the use of this repository.