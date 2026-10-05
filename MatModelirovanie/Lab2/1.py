import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt

class AdvectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Уравнение переноса: 4 разностные схемы")
        self.root.geometry("320x600")

        # Параметры задачи
        self.T = 100.0
        self.dt = 10.0
        self.u = 0.5
        self.L = 100.0
        self.dx = 10.0

        self.Nx = int(self.L / self.dx) + 1
        self.Nt = int(self.T / self.dt) + 1
        self.x = np.linspace(0, self.L, self.Nx)
        self.sigma = self.u * self.dt / self.dx

        self.setup_ui()

    # ---------- Аналитическое задание треугольника ----------
    def triangle(self, x):
        """Прямоугольный треугольник: (10,0) -> (30,1) -> (30,0)."""
        x = np.asarray(x, dtype=float)
        C = np.zeros_like(x)
        mask = (x >= 10.0) & (x <= 30.0)
        C[mask] = (x[mask] - 10.0) / 20.0
        return C

    def exact_solution(self, x, t):
        """Точное решение: C(x, t) = triangle(x - u*t)."""
        return self.triangle(x - self.u * t)

    # ---------- Интерфейс ----------
    def setup_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Параметры задачи", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=5)

        self.entry_T = self.create_entry(frame, "T (время, с):", str(self.T))
        self.entry_dt = self.create_entry(frame, "dt (шаг по времени, с):", str(self.dt))
        self.entry_u = self.create_entry(frame, "u (скорость, м/с):", str(self.u))
        self.entry_dx = self.create_entry(frame, "dx (шаг по пространству, м):", str(self.dx))

        self.sigma_label = ttk.Label(frame, text=f"Число Куранта: {self.sigma:.2f}",
                                     font=("Arial", 10, "bold"))
        self.sigma_label.pack(anchor=tk.W, pady=5)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(frame, text="Выбор схемы:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=2)
        ttk.Button(frame, text="1. Левый уголок (Upwind)",
                   command=lambda: self.plot_scheme("upwind")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="2. Центральная",
                   command=lambda: self.plot_scheme("central")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="3. Кабаре (Cabaret)",
                   command=lambda: self.plot_scheme("cabaret")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="4. Комбинированная",
                   command=lambda: self.plot_scheme("combined")).pack(fill=tk.X, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(frame, text="Сравнить все схемы",
                   command=self.plot_all).pack(fill=tk.X, pady=5)
        ttk.Button(frame, text="Начальное условие",
                   command=self.plot_initial).pack(fill=tk.X, pady=5)
        ttk.Button(frame, text="Профиль при t = T (срезы)",
                   command=self.plot_slices).pack(fill=tk.X, pady=5)

        desc_text = tk.Text(frame, height=14, width=36, wrap=tk.WORD, font=("Consolas", 9))
        desc_text.insert(tk.END,
            "Уравнение переноса:\n∂C/∂t + u ∂C/∂x = 0\n\n"
            "Начальное условие:\nпрямоугольный треугольник\n(10,0) → (30,1) → (30,0)\n\n"
            "1. Левый уголок:\nC_i^{n+1}=C_i^n-σ(C_i^n-C_{i-1}^n)\n\n"
            "2. Центральная:\nC_i^{n+1}=C_i^n-σ/2(C_{i+1}^n-C_{i-1}^n)\n\n"
            "3. Кабаре:\nC_i^{n+1}=(1-2σ)C_i^n+(2σ-1)C_{i-1}^n+C_{i-1}^{n-1}\n\n"
            "4. Комбинированная:\nC_i^{n+1}=(1-σ)C_i^n-σ/4·C_{i+1}^n+(5σ/4-0.5)C_{i-1}^n+0.5C_{i-1}^{n-1}")
        desc_text.config(state=tk.DISABLED)
        desc_text.pack(fill=tk.X, pady=5)

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
            self.dt = float(self.entry_dt.get())
            self.u = float(self.entry_u.get())
            self.dx = float(self.entry_dx.get())

            self.Nx = int(self.L / self.dx) + 1
            self.Nt = int(self.T / self.dt) + 1
            self.x = np.linspace(0, self.L, self.Nx)
            self.sigma = self.u * self.dt / self.dx

            if self.sigma > 1:
                self.sigma_label.config(text=f"Число Куранта: {self.sigma:.2f} (неустойчиво!)",
                                        foreground="red")
            else:
                self.sigma_label.config(text=f"Число Куранта: {self.sigma:.2f}",
                                        foreground="black")
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

    # ---------- Отрисовка (отдельные окна matplotlib) ----------
    def plot_initial(self):
        """Треугольник на мелкой сетке — независимо от dx."""
        self.update_params()
        x_fine = np.linspace(0, self.L, 2000)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x_fine, self.triangle(x_fine), 'k-', linewidth=2.5,
                label="Начальное условие (t=0)")
        ax.set_xlabel("x (м)", fontsize=12)
        ax.set_ylabel("C(x, 0)", fontsize=12)
        ax.set_title("Начальное условие — прямоугольный треугольник", fontsize=13)
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
            title = "Схема 'Левый уголок' (Upwind)"
            color = 'b'
        elif scheme_name == "central":
            C = self.solve_central()
            title = "Центральная разностная схема"
            color = 'r'
        elif scheme_name == "cabaret":
            C = self.solve_cabaret()
            title = "Схема 'Кабаре' (Cabaret)"
            color = 'g'
        elif scheme_name == "combined":
            C = self.solve_combined()
            title = "Комбинированная схема (Центральная + Кабаре)"
            color = 'm'
        else:
            return

        # Точное решение на мелкой сетке
        x_fine = np.linspace(0, self.L, 2000)
        C_exact = self.exact_solution(x_fine, self.T)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(self.x, C[-1, :], color + 'o-', markersize=7,
                linewidth=1.5, label=f"{title} (t = {self.T:g} с)")
        ax.plot(x_fine, C_exact, 'k--', linewidth=2.5,
                label=f"Точное решение (t = {self.T:g} с)")

        ax.set_xlabel("x (м)", fontsize=12)
        ax.set_ylabel("C(x, t)", fontsize=12)
        ax.set_title(f"{title}\nT = {self.T:g} с,  dt = {self.dt:g} с,  dx = {self.dx:g} м,  "
                     f"u = {self.u:g} м/с,  σ = {self.sigma:.2f}",
                     fontsize=12)
        ax.set_ylim(-0.15, 1.3)
        ax.set_xlim(0, self.L)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend(fontsize=11, loc='upper left')
        fig.tight_layout()
        plt.show()

    def plot_all(self):
        self.update_params()

        C_up = self.solve_upwind()
        C_ce = self.solve_central()
        C_ca = self.solve_cabaret()
        C_co = self.solve_combined()

        x_fine = np.linspace(0, self.L, 2000)
        C_exact = self.exact_solution(x_fine, self.T)

        schemes = [
            (C_up, 'b', "Левый уголок"),
            (C_ce, 'r', "Центральная"),
            (C_ca, 'g', "Кабаре"),
            (C_co, 'm', "Комбинированная")
        ]

        fig, axs = plt.subplots(2, 2, figsize=(14, 9))

        for ax, (C, color, title) in zip(axs.flat, schemes):
            ax.plot(self.x, C[-1, :], color + 'o-', markersize=6,
                    linewidth=1.5, label=f"{title}")
            ax.plot(x_fine, C_exact, 'k--', linewidth=2.5, label="Точное")
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("x (м)", fontsize=11)
            ax.set_ylabel("C(x, t = T)", fontsize=11)
            ax.set_ylim(-0.15, 1.3)
            ax.set_xlim(0, self.L)
            ax.grid(True, which='both', linestyle='--', alpha=0.6)
            ax.legend(fontsize=10)

        fig.suptitle(f"Сравнение 4 схем (T = {self.T:g} с, dt = {self.dt:g} с, "
                     f"dx = {self.dx:g} м, σ = {self.sigma:.2f})",
                     fontsize=13, y=1.00)
        fig.tight_layout()
        plt.show()

    def plot_slices(self):
        """Срезы в несколько моментов времени для одной (показательной) схемы."""
        self.update_params()

        C_ca = self.solve_cabaret()
        C_up = self.solve_upwind()

        times_idx = np.linspace(0, self.Nt - 1, 6, dtype=int)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))

        x_fine = np.linspace(0, self.L, 2000)

        for ax, C, title in [(axs[0], C_up, "Левый уголок"),
                             (axs[1], C_ca, "Кабаре")]:
            for k, idx in enumerate(times_idx):
                t_val = idx * self.dt
                ax.plot(self.x, C[idx, :], 'o-', markersize=4,
                        alpha=0.4 + 0.6 * k / len(times_idx),
                        label=f"t = {t_val:g} с")
                ax.plot(x_fine, self.exact_solution(x_fine, t_val),
                        'k--', linewidth=1, alpha=0.4 + 0.6 * k / len(times_idx))
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("x (м)", fontsize=11)
            ax.set_ylabel("C(x, t)", fontsize=11)
            ax.set_ylim(-0.15, 1.3)
            ax.set_xlim(0, self.L)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=9, loc='upper right')

        fig.suptitle("Эволюция решения во времени (пунктир — точное решение)",
                     fontsize=13)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = AdvectionApp(root)
    root.mainloop()