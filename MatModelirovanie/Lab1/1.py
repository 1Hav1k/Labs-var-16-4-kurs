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
        z = y[i]
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
# θ-метод (θ = 0.5 — метод с весами)
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
        z = y[i]
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
        root.geometry("1000x820")

        # ---- Панель параметров ----
        param_frame = ttk.LabelFrame(root, text="Параметры интегрирования", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)

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

        # ---- Панель чекбоксов ----
        check_frame = ttk.LabelFrame(root, text="Отображение графиков", padding=10)
        check_frame.pack(fill=tk.X, padx=10, pady=5)

        self.show_exact = tk.BooleanVar(value=True)
        self.show_expl  = tk.BooleanVar(value=True)
        self.show_impl  = tk.BooleanVar(value=True)
        self.show_theta = tk.BooleanVar(value=True)

        ttk.Checkbutton(check_frame, text="Точное решение",
                        variable=self.show_exact,
                        command=self.redraw).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(check_frame, text="Явный Эйлер",
                        variable=self.show_expl,
                        command=self.redraw).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(check_frame, text="Неявный Эйлер",
                        variable=self.show_impl,
                        command=self.redraw).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(check_frame, text="Метод с весами (θ=0.5)",
                        variable=self.show_theta,
                        command=self.redraw).pack(side=tk.LEFT, padx=10)

        # ---- График ----
        self.fig = Figure(figsize=(9, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ---- Текстовая область с полосой прокрутки ----
        text_frame = ttk.Frame(root)
        text_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)

        self.text_output = tk.Text(text_frame, height=14, wrap=tk.NONE,
                                   font=("Courier New", 9))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical",
                                  command=self.text_output.yview)
        self.text_output.configure(yscrollcommand=scrollbar.set)

        self.text_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.data = None

        self.ax.set_xlabel('t')
        self.ax.set_ylabel('y(t)')
        self.ax.grid(True)
        self.canvas.draw()

    def set_all(self, value: bool):
        self.show_exact.set(value)
        self.show_expl.set(value)
        self.show_impl.set(value)
        self.show_theta.set(value)
        self.redraw()

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

        t_exact = np.arange(t0, T + h/2, h)
        y_exact = exact_solution(t_exact)

        t_euler_expl, y_euler_expl = euler_explicit(t0, y0, T, h)
        t_euler_impl, y_euler_impl = euler_implicit(t0, y0, T, h)
        t_theta, y_theta = theta_method(t0, y0, T, h, theta=0.5)

        self.data = {
            "t_exact": t_exact, "y_exact": y_exact,
            "t_expl": t_euler_expl, "y_expl": y_euler_expl,
            "t_impl": t_euler_impl, "y_impl": y_euler_impl,
            "t_theta": t_theta, "y_theta": y_theta,
            "h": h,
        }

        self.redraw()
        self.update_table()

    def redraw(self):
        self.ax.clear()
        self.ax.set_xlabel('t')
        self.ax.set_ylabel('y(t)')
        self.ax.grid(True)

        if self.data is None:
            self.canvas.draw()
            return

        d = self.data
        ys = []
        if self.show_exact.get(): ys.append(d["y_exact"])
        if self.show_expl.get():  ys.append(d["y_expl"])
        if self.show_impl.get():  ys.append(d["y_impl"])
        if self.show_theta.get(): ys.append(d["y_theta"])

        if not ys:
            self.ax.text(0.5, 0.5, "Все графики отключены",
                         transform=self.ax.transAxes,
                         ha='center', va='center', fontsize=14, color='gray')
            self.canvas.draw()
            return

        y_all = np.concatenate(ys)
        y_all = y_all[np.isfinite(y_all)]
        if len(y_all) > 0:
            y_max = max(16.0, np.max(y_all))
            self.ax.set_ylim([-0.1*y_max, 1.1*y_max])

        if self.show_exact.get():
            self.ax.plot(d["t_exact"], d["y_exact"], 'k-',
                         linewidth=2, label='Точное решение')
        if self.show_expl.get():
            self.ax.plot(d["t_expl"], d["y_expl"], 'ro--',
                         markersize=3, label='Явный Эйлер')
        if self.show_impl.get():
            self.ax.plot(d["t_impl"], d["y_impl"], 'bs--',
                         markersize=3, label='Неявный Эйлер')
        if self.show_theta.get():
            self.ax.plot(d["t_theta"], d["y_theta"], 'g^--',
                         markersize=3, label='Метод с весами')

        self.ax.legend()
        self.canvas.draw()

    def update_table(self):
        """Таблица выводится С ВЫБРАННЫМ ШАГОМ h (каждый узел сетки)."""
        if self.data is None:
            return
        d = self.data
        t_exact = d["t_exact"]
        y_exact = d["y_exact"]
        y_expl  = d["y_expl"]
        y_impl  = d["y_impl"]
        y_theta = d["y_theta"]

        warning = ""
        if np.any(y_expl < 0):
            warning += ("Внимание: явный метод Эйлера дал отрицательные значения – "
                        "шаг слишком велик, метод неустойчив.\n")
        if np.any(np.isnan(y_theta)):
            warning += "Внимание: в методе с весами возникли NaN – возможно, шаг слишком велик.\n"

        mask = (np.abs(y_exact) > 1e-12) & np.isfinite(y_expl) & np.isfinite(y_impl) & np.isfinite(y_theta)
        if np.any(mask):
            max_rel_expl  = np.max(np.abs((y_expl[mask]  - y_exact[mask]) / y_exact[mask]) * 100)
            max_rel_impl  = np.max(np.abs((y_impl[mask]  - y_exact[mask]) / y_exact[mask]) * 100)
            max_rel_theta = np.max(np.abs((y_theta[mask] - y_exact[mask]) / y_exact[mask]) * 100)
        else:
            max_rel_expl = max_rel_impl = max_rel_theta = np.nan

        self.text_output.delete(1.0, tk.END)
        if warning:
            self.text_output.insert(tk.END, warning + "\n")

        header = (f"{'t':>8} | {'Точное':>12} | {'Явный Эйлер':>13} | "
                  f"{'Неявный Эйлер':>14} | {'С весами':>12}\n")
        self.text_output.insert(tk.END, header)
        self.text_output.insert(tk.END, "-" * 76 + "\n")

        # ВЫВОД ВСЕХ УЗЛОВ СЕТКИ (шаг = заданному h)
        for i in range(len(t_exact)):
            line = (f"{t_exact[i]:8.4f} | {y_exact[i]:12.6f} | "
                    f"{y_expl[i]:13.6f} | {y_impl[i]:14.6f} | "
                    f"{y_theta[i]:12.6f}\n")
            self.text_output.insert(tk.END, line)

        self.text_output.insert(tk.END, "\nМаксимальные относительные погрешности (в %):\n")
        self.text_output.insert(tk.END, f"Явный Эйлер:      {max_rel_expl:.4f}%\n")
        self.text_output.insert(tk.END, f"Неявный Эйлер:    {max_rel_impl:.4f}%\n")
        self.text_output.insert(tk.END, f"С весами (θ=0.5): {max_rel_theta:.4f}%\n")

# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()