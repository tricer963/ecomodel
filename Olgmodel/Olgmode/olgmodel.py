import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

class OLGModel:
    def __init__(self, α=0.4, β=0.9, θ=0.5):
        # 定義模型參數,α=資本份額,β=折現因子,θ=風險偏好係數
        self.params = namedtuple('Model', ['α', 'β', 'θ'])(α, β, θ)
        self.history_R = []  # 紀錄歷史均衡利率
        self.history_K = []  # 紀錄歷史均衡資本

    def crra(self, c):
        # 計算CRRA效用函數
        return c**(1 - self.params.θ) / (1 - self.params.θ)

    def capital_demand(self, R):
        # 計算資本需求函數
        return (self.params.α/R)**(1/(1-self.params.α))

    def savings_crra(self, w, R):
        # 計算資本供給函數
        return w / (1 + self.params.β**(-1/self.params.θ) * R**((self.params.θ-1)/self.params.θ))

    def find_equilibrium(self, w, tolerance=0.0001):
        # 尋找短期均衡利率和均衡資本
        R_vals = np.linspace(0.1, 1)
        
        fig, ax = plt.subplots()
        ax.plot(R_vals, self.capital_demand(R_vals), label="aggregate demand")
        ax.plot(R_vals, self.savings_crra(w, R_vals), label="aggregate supply")

        equilibrium_R = 0
        equilibrium_k = 0
        for R_val in np.arange(0.0001, 0.9999, 0.0001):
            demand_value = self.capital_demand(R_val)
            supply_value = self.savings_crra(w, R_val)

            if np.abs(demand_value - supply_value) <= tolerance:
                equilibrium_R = R_val
                equilibrium_k = demand_value
                print("均衡利率:", equilibrium_R, "均衡資本:", equilibrium_k)
                break

        if equilibrium_R is not None:
            ax.plot(equilibrium_R, equilibrium_k, 'ro', label="equilibrium")
            ax.annotate(r'equilibrium',
                        xy=(equilibrium_R, equilibrium_k),
                        xycoords='data',
                        xytext=(0, 20),
                        textcoords='offset points',
                        fontsize=12,
                        arrowprops=dict(arrowstyle="->"))

        ax.set_xlabel("$R_{t+1}$")
        ax.set_ylabel("$k_{t+1}$")
        ax.legend()
        plt.show()

        self.history_R.append(equilibrium_R)
        self.history_K.append(equilibrium_k)

        return equilibrium_R, equilibrium_k
    

