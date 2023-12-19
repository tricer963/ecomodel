import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class OLGModel_steadystate:
    def __init__(self, α=0.4, β=0.9, θ=1):
         # 定義模型參數,α=資本份額,β=折現因子,θ=風險偏好係數
        self.α = α
        self.β = β
        self.θ = θ
        self.k_star = []

    def f(self, k_prime, k):
        z = (1 - self.α) * k**self.α
        a = self.α**(1 - 1/self.θ)
        b = k_prime**((self.α * self.θ - self.α + 1) / self.θ)
        p = k_prime + k_prime * self.β**(-1/self.θ) * a * b
        return p - z

    def k_update(self, k):
        return optimize.newton(lambda k_prime: self.f(k_prime, k), 0.1)

    def h(self, k_star):
        z = (1 - self.α) * k_star**self.α
        R1 = self.α**(1 - 1/self.θ)
        R2 = k_star**((self.α * self.θ - self.α + 1) / self.θ)
        p = k_star + k_star * self.β**(-1/self.θ) * R1 * R2
        return p - z

    def plot_dynamics(self, kmin, kmax, x=1000):
        k_grid = np.linspace(kmin, kmax, x)
        k_grid_next = np.empty_like(k_grid)

        # 計算下一期資本存量
        for i in range(x):
            k_grid_next[i] = self.k_update(k_grid[i])

        # 繪製圖形
        fig, ax = plt.subplots(figsize=(6, 6))
        ymin, ymax = np.min(k_grid_next), np.max(k_grid_next)

        # 繪製 k_{t+1} vs. k_t
        ax.plot(k_grid, k_grid_next,  lw=2, alpha=0.6, label='$g$')
        # 繪製 45 度線
        ax.plot(k_grid, k_grid, 'k-', lw=1, alpha=0.7, label='$45^{\circ}$')

        # 標記均衡點處
        k_star = optimize.newton(self.h, 0.2)
        self.k_star.append(k_star)
        ax.scatter(k_star, k_star, color='red', marker='o', label=' $k*$')
        arrow_text = "k*"
        ax.annotate(arrow_text,
                    xy=(k_star, k_star),
                    xycoords='data',
                    xytext=(0, 30),  # 箭頭的位置
                    textcoords='offset points',
                    fontsize=12,
                    color='blue',  # 箭頭的顏色
                    arrowprops=dict(arrowstyle="->", color='RED'))
        # 圖例和坐標軸
        ax.legend(loc='upper left', frameon=False, fontsize=12)
        ax.set_xlabel('$k_t$', fontsize=12)
        ax.set_ylabel('$k_{t+1}$', fontsize=12)

        plt.show()

        print(f"k* = {k_star}")
        print(f"y* = {(self.α*k_star)}")
