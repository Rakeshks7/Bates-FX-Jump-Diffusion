import numpy as np

class BatesFXPricer:
    def __init__(self, S_paths, T, r):
        """
        Initializes the pricer with the generated Monte Carlo paths.
        
        :param S_paths: 2D numpy array of simulated asset paths (Time x Paths)
        :param T: Time to maturity (in years)
        :param r: Risk-free rate (or FX interest rate differential)
        """
        self.S_paths = S_paths
        self.T = T
        self.r = r
        self.discount_factor = np.exp(-self.r * self.T)
        self.terminal_prices = self.S_paths[-1]

    def european_call(self, K):
        payoffs = np.maximum(self.terminal_prices - K, 0)
        return self.discount_factor * np.mean(payoffs)

    def european_put(self, K):
        payoffs = np.maximum(K - self.terminal_prices, 0)
        return self.discount_factor * np.mean(payoffs)

    def up_and_out_call(self, K, B):
        """
        Prices a Discrete Up-and-Out Barrier Call Option.
        Very common in FX markets for hedging specific geopolitical risk levels.
        
        :param K: Strike price
        :param B: Barrier level (Knock-out price)
        """
        hit_barrier = np.any(self.S_paths >= B, axis=0)

        payoffs = np.maximum(self.terminal_prices - K, 0)

        payoffs[hit_barrier] = 0.0
        
        return self.discount_factor * np.mean(payoffs)

if __name__ == "__main__":
    dummy_paths = np.random.normal(1.0, 0.2, (252, 1000)) 
    
    pricer = BatesFXPricer(dummy_paths, T=1.0, r=0.05)
    print(f"Call Price: {pricer.european_call(K=1.0):.4f}")
    print(f"U&O Barrier Price: {pricer.up_and_out_call(K=1.0, B=1.3):.4f}")