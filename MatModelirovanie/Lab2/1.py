import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt

class AdvectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Уравнение переноса")
        self.root.geometry("340x650")

        # Параметры задачи
        self.T = 100.0      # время, с
        self.u = 0.5        # скорость, м/с
        self.L = 100.0      # длина, м
        self.h = 1.0       # шаг по пространству h, м
        self.sigma = 0.05    # число Куранта

        self.setup_ui()
        self.update_params()   # сразу посчитаем τ и Nt

    # ---------- Аналитическое задание треугольника ----------
    def triangle(self, x):
        """Прямоугольный треугольник."""
        x = np.asarray(x, dtype=float)
        C = np.zeros_like(x)
        mask = (x >= 10.0) & (x <= 30.0)
        C[mask] = (x[mask] - 10.0) / 20.0
        return C

    def exact_solution(self, x, t):
        """Аналитическое решение: C(x, t) = triangle(x - u*t)."""
        return self.triangle(x - self.u * t)

    # ---------- Интерфейс ----------
    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Параметры задачи",
                  font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=5)

        self.entry_T     = self.create_entry(frame, "T (время, с):", str(self.T))
        self.entry_u     = self.create_entry(frame, "u (скорость, м/с):", str(self.u))
        self.entry_h    = self.create_entry(frame, "h (шаг по x, м):", str(self.h))
        self.entry_sigma = self.create_entry(frame, "σ (число Куранта):", str(self.sigma))

        self.sigma_label = ttk.Label(frame, text="", font=("Arial", 10, "bold"))
        self.sigma_label.pack(anchor=tk.W, pady=5)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(frame, text="Выбор схемы:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=2)
        ttk.Button(frame, text="1. Левый уголок",
                   command=lambda: self.plot_scheme("upwind")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="2. Центральная",
                   command=lambda: self.plot_scheme("central")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="3. Кабаре",
                   command=lambda: self.plot_scheme("cabaret")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="4. Комбинированная",
                   command=lambda: self.plot_scheme("combined")).pack(fill=tk.X, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(frame, text="Начальное условие",
                   command=self.plot_initial).pack(fill=tk.X, pady=5)

    def create_entry(self, parent, label_text, default_val):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=22).pack(side=tk.LEFT)
        entry = ttk.Entry(frame, width=10)
        entry.insert(0, default_val)
        entry.pack(side=tk.RIGHT)
        return entry

    def update_params(self):
        try:
            self.T = float(self.entry_T.get())
            self.u = float(self.entry_u.get())
            self.h = float(self.entry_h.get())
            self.sigma = float(self.entry_sigma.get())

            # Проверка устойчивости
            if self.sigma <= 0 or self.sigma > 1:
                self.sigma_label.config(
                    text=f"σ = {self.sigma:.2f} (нарушено CFL!)",
                    foreground="red")
                return

            # Шаг по времени вычисляется автоматически
            self.τ = self.sigma * self.h / self.u

            self.Nx = int(self.L / self.h) + 1
            self.Nt = int(self.T / self.τ) + 1
            self.x = np.linspace(0, self.L, self.Nx)
        except ValueError:
            pass

    # ---------- Численные схемы ----------
    def solve_upwind(self):
        C = np.zeros((self.Nt, self.Nx))
        C[0, :] = self.triangle(self.x)
        sigma = self.sigma
        for n in range(0, self.Nt - 1):
            C[n+1, 1:] = C[n, 1:] - sigma * (C[n, 1:] - C[n, :-1])
            C[n+1, 0] = 0.0
        return C

    def solve_central(self):
        C = np.zeros((self.Nt, self.Nx))
        C[0, :] = self.triangle(self.x)
        sigma = self.sigma
        for n in range(0, self.Nt - 1):
            C[n+1, 1:-1] = C[n, 1:-1] - 0.5 * sigma * (C[n, 2:] - C[n, :-2])
            C[n+1, 0] = 0.0
            C[n+1, -1] = C[n+1, -2]
        return C

    def solve_cabaret(self):
        C = np.zeros((self.Nt, self.Nx))
        C[0, :] = self.triangle(self.x)
        sigma = self.sigma
        C[1, 1:] = C[0, 1:] - sigma * (C[0, 1:] - C[0, :-1])
        C[1, 0] = 0.0
        for n in range(1, self.Nt - 1):
            C[n+1, 1:] = (1 - 2*sigma)*C[n, 1:] + (2*sigma - 1)*C[n, :-1] + C[n-1, :-1]
            C[n+1, 0] = 0.0
        return C

    def solve_combined(self):
        C = np.zeros((self.Nt, self.Nx))
        C[0, :] = self.triangle(self.x)
        sigma = self.sigma
        C[1, 1:] = C[0, 1:] - sigma * (C[0, 1:] - C[0, :-1])
        C[1, 0] = 0.0
        for n in range(1, self.Nt - 1):
            C[n+1, 1:-1] = ((1 - sigma) * C[n, 1:-1] -
                            (sigma / 4) * C[n, 2:] +
                            (5 * sigma / 4 - 0.5) * C[n, :-2] +
                            0.5 * C[n-1, :-2])
            C[n+1, 0] = 0.0
            C[n+1, -1] = C[n+1, -2]
        return C

    # ---------- Отрисовка графиков ----------
    def plot_initial(self):
        """Треугольник на мелкой сетке — независимо от h."""
        self.update_params()
        x_fine = np.linspace(0, self.L, 2000)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x_fine, self.triangle(x_fine), 'k-', linewidth=2.5,
                label="Начальное условие")
        ax.set_xlabel("x (м)", fontsize=12)
        ax.set_ylabel("C(x, 0)", fontsize=12)
        ax.set_title("Начальное условие", fontsize=13)
        ax.set_ylim(-0.1, 1.2)
        ax.set_xlim(0, self.L)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)
        fig.tight_layout()
        plt.show()

    def plot_scheme(self, scheme_name):
        self.update_params()

        if scheme_name == "upwind":
            C = self.solve_upwind()
            title = "Схема 'Левый уголок'"
            color = 'b'
        elif scheme_name == "central":
            C = self.solve_central()
            title = "Центральная разностная схема"
            color = 'r'
        elif scheme_name == "cabaret":
            C = self.solve_cabaret()
            title = "Схема 'Кабаре'"
            color = 'g'
        elif scheme_name == "combined":
            C = self.solve_combined()
            title = "Комбинированная схема (Центральная + Кабаре)"
            color = 'm'
        else:
            return

        # Аналитическое решение на мелкой сетке
        x_fine = np.linspace(0, self.L, 2000)
        C_exact = self.exact_solution(x_fine, self.T)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(self.x, C[-1, :], color + 'o-', markersize=7,
                linewidth=1.5, label=f"{title}")
        ax.plot(x_fine, C_exact, 'k--', linewidth=2.5,
                label=f"Аналитическое решение")

        ax.set_xlabel("x (м)", fontsize=12)
        ax.set_ylabel("C(x, t)", fontsize=12)
        ax.set_title(f"{title}\nT = {self.T:g} с,  τ = {self.τ:g} с, h = {self.h:g} м,  "
                     f"u = {self.u:g} м/с,  σ = {self.sigma:.2f}",
                     fontsize=12)
        ax.set_ylim(-0.15, 1.3)
        ax.set_xlim(0, self.L)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend(fontsize=11, loc='upper left')
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvectionApp(root)
    root.mainloop()