import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------------------------------------------------------
# Правая часть ДУ
# ------------------------------------------------------------
def f(t, y):
    return -y**2 / (t**2 + 1)

# ------------------------------------------------------------
# Аналитическое решение
# ------------------------------------------------------------
def exact_solution(t):
    return 1.0 / (np.arctan(t) + 1.0/16.0)

# ------------------------------------------------------------
# Явный метод Эйлера
# ------------------------------------------------------------
def euler_explicit(t0, y0, T, h):
    n_steps = int(np.ceil((T - t0) / h))
    t = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    t[0] = t0
    y[0] = y0
    
    for i in range(n_steps):
        t[i+1] = t[i] + h
        y[i+1] = y[i] + h * f(t[i], y[i])
    return t, y

# ------------------------------------------------------------
# Неявный метод Эйлера (метод Ньютона)
# ------------------------------------------------------------
def euler_implicit(t0, y0, T, h):
    n_steps = int(np.ceil((T - t0) / h))
    t = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    t[0] = t0
    y[0] = y0
    
    for i in range(n_steps):
        t[i+1] = t[i] + h
        z = y[i]  # начальное приближение
        for _ in range(100):
            F = z - y[i] - h * f(t[i+1], z)
            dF = 1 - h * (-2*z / (t[i+1]**2 + 1))
            dz = F / dF
            z -= dz
            if abs(dz) < 1e-12:
                break
        y[i+1] = z
    return t, y

# ------------------------------------------------------------
# θ - метод (θ = 0.5 - метод трапеций)
# ------------------------------------------------------------
def theta_method(t0, y0, T, h, theta=0.5):
    n_steps = int(np.ceil((T - t0) / h))
    t = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    t[0] = t0
    y[0] = y0
    
    for i in range(n_steps):
        t[i+1] = t[i] + h
        f_old = f(t[i], y[i])
        z = y[i]  # начальное приближение
        for _ in range(100):
            f_new = f(t[i+1], z)
            F = z - y[i] - h * ((1-theta)*f_old + theta*f_new)
            dF = 1 - h * theta * (-2*z / (t[i+1]**2 + 1))
            dz = F / dF
            z -= dz
            if abs(dz) < 1e-12:
                break
        y[i+1] = z
    return t, y

# ------------------------------------------------------------
# Графический интерфейс
# ------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Решение задачи Коши численными методами")
        root.geometry("900x700")
        
        # Параметры
        param_frame = ttk.Frame(root, padding=10)
        param_frame.pack(fill=tk.X)
        
        ttk.Label(param_frame, text="Конечное время T:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.T_entry = ttk.Entry(param_frame, width=10)
        self.T_entry.grid(row=0, column=1, padx=5, pady=5)
        self.T_entry.insert(0, "5.0")
        
        ttk.Label(param_frame, text="Шаг h:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.h_entry = ttk.Entry(param_frame, width=10)
        self.h_entry.grid(row=0, column=3, padx=5, pady=5)
        self.h_entry.insert(0, "0.1")
        
        self.calc_button = ttk.Button(param_frame, text="Вычислить", command=self.calculate)
        self.calc_button.grid(row=0, column=4, padx=20, pady=5)
        
        # График
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Текстовая область
        self.text_output = tk.Text(root, height=12, wrap=tk.NONE)
        self.text_output.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # Начальная отрисовка
        self.ax.set_xlabel('t')
        self.ax.set_ylabel('y(t)')
        self.ax.grid(True)
        self.canvas.draw()
    
    def calculate(self):
        try:
            T = float(self.T_entry.get())
            h = float(self.h_entry.get())
            if T <= 0 or h <= 0:
                raise ValueError("T и h должны быть положительными")
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Проверьте значения:\n{e}")
            return
        
        t0, y0 = 0.0, 16.0
        
        # Вычисление решений
        t_exact = np.arange(t0, T + h/2, h)
        y_exact = exact_solution(t_exact)
        
        t_euler_expl, y_euler_expl = euler_explicit(t0, y0, T, h)
        t_euler_impl, y_euler_impl = euler_implicit(t0, y0, T, h)
        t_theta, y_theta = theta_method(t0, y0, T, h, theta=0.5)
        
        # Проверка на отрицательные значения
        warning = ""
        if np.any(y_euler_expl < 0):
            warning += "Внимание: явный метод Эйлера дал отрицательные значения – шаг слишком велик, метод неустойчив.\n"
        if np.any(np.isnan(y_theta)):
            warning += "Внимание: в методе трапеций возникли NaN – возможно, шаг слишком велик.\n"
        
        # Очистка графика
        self.ax.clear()
        self.ax.set_xlabel('t')
        self.ax.set_ylabel('y(t)')
        self.ax.grid(True)
        
        # Ограничение оси Y
        y_all = np.concatenate([y_euler_expl, y_euler_impl, y_theta])
        y_all = y_all[np.isfinite(y_all)]
        if len(y_all) > 0:
            y_min = min(0, np.min(y_all))
            y_max = max(y0, np.max(y_all))
            self.ax.set_ylim([-0.1*y_max, 1.1*y_max])
        else:
            self.ax.set_ylim([-0.1*y0, 1.1*y0])
        
        # Построение графиков
        self.ax.plot(t_exact, y_exact, 'k-', linewidth=2, label='Точное решение')
        self.ax.plot(t_euler_expl, y_euler_expl, 'ro--', markersize=3, label='Явный Эйлер')
        self.ax.plot(t_euler_impl, y_euler_impl, 'bs--', markersize=3, label='Неявный Эйлер')
        self.ax.plot(t_theta, y_theta, 'g^--', markersize=3, label='Метод трапеций (θ=0.5)')
        self.ax.legend()
        self.canvas.draw()
        
        # Таблица (вывод каждого ~0.5 по t)
        step_display = max(1, int(0.5 / h))
        indices = list(range(0, len(t_exact), step_display))
        if indices[-1] != len(t_exact)-1:
            indices.append(len(t_exact)-1)
        
        # Вычисление максимальных относительных погрешностей в процентах
        # Используем только те точки, где точное решение != 0 и все значения конечны
        mask = (np.abs(y_exact) > 1e-12) & np.isfinite(y_euler_expl) & np.isfinite(y_euler_impl) & np.isfinite(y_theta)
        
        if np.any(mask):
            # Относительная погрешность: |(y_числ - y_точн) / y_точн| * 100%
            rel_err_expl = np.abs((y_euler_expl[mask] - y_exact[mask]) / y_exact[mask]) * 100
            rel_err_impl = np.abs((y_euler_impl[mask] - y_exact[mask]) / y_exact[mask]) * 100
            rel_err_theta = np.abs((y_theta[mask] - y_exact[mask]) / y_exact[mask]) * 100
            
            max_rel_err_expl = np.max(rel_err_expl)
            max_rel_err_impl = np.max(rel_err_impl)
            max_rel_err_theta = np.max(rel_err_theta)
        else:
            max_rel_err_expl = max_rel_err_impl = max_rel_err_theta = np.nan
        
        # Заполнение текстового поля
        self.text_output.delete(1.0, tk.END)
        if warning:
            self.text_output.insert(tk.END, warning + "\n")
        
        header = f"{'t':>8} | {'Точное':>12} | {'Явный Эйлер':>12} | {'Неявный Эйлер':>12} | {'Трапеции':>12}\n"
        self.text_output.insert(tk.END, header)
        self.text_output.insert(tk.END, "-"*70 + "\n")
        
        for i in indices:
            line = (f"{t_exact[i]:8.3f} | {y_exact[i]:12.6f} | "
                    f"{y_euler_expl[i]:12.6f} | {y_euler_impl[i]:12.6f} | "
                    f"{y_theta[i]:12.6f}\n")
            self.text_output.insert(tk.END, line)
        
        self.text_output.insert(tk.END, "\nМаксимальные относительные погрешности:\n")
        self.text_output.insert(tk.END, f"Явный Эйлер:    {max_rel_err_expl:.4f}%\n")
        self.text_output.insert(tk.END, f"Неявный Эйлер:  {max_rel_err_impl:.4f}%\n")
        self.text_output.insert(tk.END, f"Трапеции (θ=0.5): {max_rel_err_theta:.4f}%\n")

# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()