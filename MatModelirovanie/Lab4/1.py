"""
Лабораторная работа №4
«Задача диффузии-конвекции»
Дисциплина «Математическое и имитационное моделирование»

Уравнение:  dC/dt + u dC/dx = d/dx(mu dC/dx) + f,   0 < x < L,  0 < t <= T
Начальное условие: треугольный импульс C0(x)
Граничные условия: C(0, t) = C_left,   dC/dx(L, t) = p - q*C(L, t)

Численный метод: интегро-интерполяционный метод (метод баланса),
весовая схема с параметром sigma, решение СЛАУ методом прогонки.
"""

import math

import numpy as np
from matplotlib.figure import Figure

try:  # графический интерфейс (tkinter входит в стандартную поставку Python)
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:  # позволяет использовать расчётную часть без GUI
    tk = None


# ======================================================================
#                         РАСЧЁТНАЯ ЧАСТЬ
# ======================================================================

def initial_condition(x, x1=10.0, x2=20.0, x3=30.0, height=1.0):
    """Начальное распределение: треугольный импульс на [x1, x3] с вершиной в x2."""
    x = np.asarray(x, dtype=float)
    c0 = np.zeros_like(x)
    left = (x >= x1) & (x <= x2)
    right = (x > x2) & (x <= x3)
    c0[left] = height * (x[left] - x1) / (x2 - x1)
    c0[right] = height * (x3 - x[right]) / (x3 - x2)
    return c0


def progonka(a, b, d, F, alpha1=0.0, beta1=0.0):
    """Метод прогонки для трёхдиагональной системы
        C_0 = alpha1 * C_1 + beta1                          (левое ГУ)
        -a_i C_{i-1} + d_i C_i - b_i C_{i+1} = F_i,  i = 1..N-1
        -a_N C_{N-1} + d_N C_N = F_N                        (правое ГУ, b_N = 0)
    Прямой ход:  alpha_{i+1} = b_i / (d_i - a_i alpha_i),
                 beta_{i+1}  = (F_i + a_i beta_i) / (d_i - a_i alpha_i)
    Обратный ход: C_i = alpha_{i+1} C_{i+1} + beta_{i+1}
    """
    n = len(F) - 1
    alpha = np.zeros(n + 2)
    beta = np.zeros(n + 2)
    alpha[1], beta[1] = alpha1, beta1
    for i in range(1, n + 1):
        den = d[i] - a[i] * alpha[i]
        alpha[i + 1] = b[i] / den
        beta[i + 1] = (F[i] + a[i] * beta[i]) / den
    c = np.zeros(n + 1)
    c[n] = beta[n + 1]                      # т.к. b_N = 0, alpha_{N+1} = 0
    for i in range(n - 1, -1, -1):
        c[i] = alpha[i + 1] * c[i + 1] + beta[i + 1]
    return c


def solve(L=100.0, T=50.0, u=0.5, mu=1.0, f=0.0, h=0.5, tau=0.1, sigma=1.0,
          init=(10.0, 20.0, 30.0, 1.0), c_left=0.0, p=1.0, q=1.0,
          bc_order=2, max_layers=1001):
    """Решение задачи диффузии-конвекции по весовой схеме.

    bc_order = 2 — правое ГУ аппроксимируется методом баланса на полуячейке
                   [x_{N-1/2}, x_N] (второй порядок);
    bc_order = 1 — простейшая аппроксимация (C_N - C_{N-1})/h = p - q C_N.
    Возвращает x, массив моментов времени и матрицу C[слой, узел].
    """
    N = max(2, int(round(L / h)))
    h = L / N
    M = max(1, int(round(T / tau)))
    tau = T / M
    x = np.linspace(0.0, L, N + 1)
    c = initial_condition(x, *init)
    c[0] = c_left

    # Коэффициенты для внутренних узлов (u, mu постоянны, поэтому
    # u_{i+1/2} = u_{i-1/2} = u, mu_{i+1/2} = mu_{i-1/2} = mu)
    A = u / (2 * h) + mu / h ** 2           # a_i / sigma
    B = -u / (2 * h) + mu / h ** 2          # b_i / sigma
    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    d = np.zeros(N + 1)
    a[1:N] = sigma * A
    b[1:N] = sigma * B
    d[1:N] = 1.0 / tau + a[1:N] + b[1:N]

    # Правая граница
    if bc_order == 2:
        Ar = 2 * mu / h ** 2 + u / h
        a[N] = sigma * Ar
        d[N] = 1.0 / tau + sigma * (Ar + 2 * mu * q / h)
    else:
        a[N] = 1.0 / h
        d[N] = 1.0 / h + q

    stride = max(1, int(math.ceil(M / (max_layers - 1))))
    times = [0.0]
    history = [c.copy()]
    F = np.zeros(N + 1)
    for n in range(1, M + 1):
        # правая часть для внутренних узлов
        conv = u * (c[2:] - c[:-2]) / (2 * h)
        diff = mu * (c[2:] - 2 * c[1:-1] + c[:-2]) / h ** 2
        F[1:N] = c[1:N] / tau - (1 - sigma) * (conv - diff) + f
        # правая часть для граничного узла x_N
        if bc_order == 2:
            F[N] = (c[N] / tau + 2 * mu * p / h + f
                    - (1 - sigma) * (Ar * (c[N] - c[N - 1]) + 2 * mu * q / h * c[N]))
        else:
            F[N] = p
        c = progonka(a, b, d, F, alpha1=0.0, beta1=c_left)
        if not np.all(np.isfinite(c)) or np.max(np.abs(c)) > 1e6:
            times.append(n * tau)
            history.append(np.clip(np.nan_to_num(c, nan=1e6), -1e6, 1e6))
            break                           # схема «разошлась»
        if n % stride == 0 or n == M:
            times.append(n * tau)
            history.append(c.copy())
    return x, np.array(times), np.array(history)


def _ramp_conv(z, s):
    """Свёртка функции (z)_+ с нормальным ядром дисперсии s^2: z*Phi(z/s) + s*phi(z/s)."""
    erf = np.vectorize(math.erf)
    zs = z / s
    Phi = 0.5 * (1.0 + erf(zs / math.sqrt(2.0)))
    phi = np.exp(-0.5 * zs ** 2) / math.sqrt(2.0 * math.pi)
    return z * Phi + s * phi


def analytic_free(x, t, u=0.5, mu=1.0, f=0.0, init=(10.0, 20.0, 30.0, 1.0)):
    """Точное решение на бесконечной прямой (без влияния границ):
    импульс переносится со скоростью u и размывается гауссовым ядром
    с дисперсией 2*mu*t. Для треугольника решение выражается через erf."""
    x1, x2, x3, H = init
    x = np.asarray(x, dtype=float)
    if t <= 0:
        return initial_condition(x, *init)
    s = math.sqrt(2.0 * mu * t)
    y = x - u * t
    k1, k3 = 1.0 / (x2 - x1), 1.0 / (x3 - x2)
    return H * (k1 * _ramp_conv(y - x1, s) - (k1 + k3) * _ramp_conv(y - x2, s)
                + k3 * _ramp_conv(y - x3, s)) + f * t


def stability_info(u, mu, h, tau, sigma):
    """Сеточные параметры и вывод об устойчивости (анализ фон Неймана)."""
    cur = abs(u) * tau / h                  # число Куранта
    bnum = mu * tau / h ** 2                # диффузионное число
    pe = abs(u) * h / mu if mu > 0 else float('inf')   # сеточное число Пекле
    lines = [f"Число Куранта  a = u·τ/h = {cur:.4g}",
             f"Диффузионное число  b = μ·τ/h² = {bnum:.4g}",
             f"Сеточное число Пекле  Pe = u·h/μ = {pe:.4g}"]
    if sigma >= 0.5:
        lines.append("σ ≥ 0.5: схема безусловно устойчива.")
    else:
        ok = bnum <= 0.5 / (1 - 2 * sigma)
        lines.append("Условие b ≤ 1/(2(1−2σ)): " + ("выполнено" if ok else "НАРУШЕНО — возможна неустойчивость!"))
    if pe > 2:
        lines.append("Pe > 2: возможны нефизичные осцилляции решения.")
    return "\n".join(lines)


def error_vs_time(x, times, hist, u, mu, f, init, x_max):
    """Максимальное отклонение от точного решения на отрезке [0, x_max]."""
    mask = x <= x_max
    return np.array([np.max(np.abs(c[mask] - analytic_free(x[mask], t, u, mu, f, init)))
                     for t, c in zip(times, hist)])


# ======================================================================
#                      ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ======================================================================

PARAMS = [  # (ключ, подпись, значение по умолчанию, группа)
    ("L", "Длина области L, м", 100.0, "Параметры задачи"),
    ("T", "Время моделирования T, с", 50.0, "Параметры задачи"),
    ("u", "Скорость конвекции u, м/с", 0.5, "Параметры задачи"),
    ("mu", "Коэф. диффузии μ, м²/с", 1.0, "Параметры задачи"),
    ("f", "Источник f", 0.0, "Параметры задачи"),
    ("x1", "Начало импульса x₁, м", 10.0, "Начальное условие (треугольник)"),
    ("x2", "Вершина импульса x₂, м", 20.0, "Начальное условие (треугольник)"),
    ("x3", "Конец импульса x₃, м", 30.0, "Начальное условие (треугольник)"),
    ("H", "Высота импульса", 1.0, "Начальное условие (треугольник)"),
    ("cl", "C(0, t) =", 0.0, "Граничные условия: ∂C/∂x(L) = p − q·C"),
    ("p", "p =", 1.0, "Граничные условия: ∂C/∂x(L) = p − q·C"),
    ("q", "q =", 1.0, "Граничные условия: ∂C/∂x(L) = p − q·C"),
    ("h", "Шаг по пространству h, м", 0.5, "Сетка и схема"),
    ("tau", "Шаг по времени τ, с", 0.1, "Сетка и схема"),
    ("sigma", "Вес схемы σ (0…1)", 1.0, "Сетка и схема"),
    ("dt_out", "Шаг вывода профилей, с", 10.0, "Сетка и схема"),
]


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ЛР №4. Задача диффузии-конвекции")
        self.root.geometry("1280x780")
        self.vars = {}
        self.result = None
        self.compare = None
        self._build_controls()
        self._build_tabs()
        self.calculate()

    # ---------------- интерфейс ----------------
    def _build_controls(self):
        panel = ttk.Frame(self.root, padding=6)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        groups = {}
        for key, label, default, group in PARAMS:
            if group not in groups:
                groups[group] = ttk.LabelFrame(panel, text=group, padding=4)
                groups[group].pack(fill=tk.X, pady=3)
            row = ttk.Frame(groups[group])
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=26).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            ttk.Entry(row, textvariable=var, width=9).pack(side=tk.RIGHT)
            self.vars[key] = var

        row = ttk.Frame(groups["Сетка и схема"])
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text="Аппроксимация ГУ при x = L").pack(side=tk.LEFT)
        self.bc_var = tk.StringVar(value="баланс, O(h²)")
        ttk.Combobox(row, textvariable=self.bc_var, width=13, state="readonly",
                     values=["баланс, O(h²)", "простая, O(h)"]).pack(side=tk.RIGHT)

        ttk.Button(panel, text="Рассчитать", command=self.calculate).pack(fill=tk.X, pady=(8, 2))
        ttk.Button(panel, text="Сравнить схемы σ = 0; 0.5; 1",
                   command=self.compare_schemes).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Сохранить графики…", command=self.save_figures).pack(fill=tk.X, pady=2)

        self.info = tk.Text(panel, width=44, height=12, font=("Consolas", 9), wrap=tk.WORD)
        self.info.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _add_tab(self, title):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        NavigationToolbar2Tk(canvas, frame).update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return frame, fig, canvas

    def _build_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.tabs = {}
        for name in ["Профили C(x,t)", "Тепловая карта", "Динамика", "C на правой границе",
                     "Сравнение схем"]:
            self.tabs[name] = self._add_tab(name)
        frame = self.tabs["Динамика"][0]
        self.slider = tk.Scale(frame, from_=0, to=1, orient=tk.HORIZONTAL, showvalue=False,
                               label="Момент времени", command=self._on_slider)
        self.slider.pack(fill=tk.X, side=tk.BOTTOM)

    # ---------------- чтение параметров ----------------
    def _params(self):
        p = {}
        for key, label, _, _ in PARAMS:
            try:
                p[key] = float(self.vars[key].get().replace(",", "."))
            except ValueError:
                raise ValueError(f"Некорректное значение: «{label}»")
        if p["L"] <= 0 or p["T"] <= 0 or p["h"] <= 0 or p["tau"] <= 0 or p["mu"] < 0:
            raise ValueError("L, T, h, τ должны быть положительными, μ ≥ 0")
        if not 0 <= p["sigma"] <= 1:
            raise ValueError("Вес σ должен лежать в отрезке [0, 1]")
        if not p["x1"] < p["x2"] < p["x3"]:
            raise ValueError("Должно выполняться x₁ < x₂ < x₃")
        if p["L"] / p["h"] > 5000 or p["T"] / p["tau"] > 200000:
            raise ValueError("Слишком мелкая сетка (N ≤ 5000, T/τ ≤ 200000)")
        p["bc_order"] = 2 if self.bc_var.get().startswith("баланс") else 1
        return p

    def _run(self, p, sigma=None):
        return solve(p["L"], p["T"], p["u"], p["mu"], p["f"], p["h"], p["tau"],
                     p["sigma"] if sigma is None else sigma,
                     (p["x1"], p["x2"], p["x3"], p["H"]), p["cl"], p["p"], p["q"],
                     p["bc_order"])

    # ---------------- расчёт ----------------
    def calculate(self):
        try:
            p = self._params()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        x, times, hist = self._run(p)
        self.root.config(cursor="")
        self.result = (p, x, times, hist)
        self._plot_profiles()
        self._plot_heatmap()
        self._plot_boundary()
        self.slider.config(to=len(times) - 1)
        self.slider.set(len(times) - 1)
        self._on_slider(len(times) - 1)
        self._show_info()

    def _show_info(self):
        p, x, times, hist = self.result
        init = (p["x1"], p["x2"], p["x3"], p["H"])
        c_end = hist[-1]
        x_cut = 0.8 * p["L"]
        inner = x <= x_cut
        i_max = int(np.argmax(np.where(inner, c_end, -np.inf)))
        err = np.max(np.abs(c_end[inner] - analytic_free(x[inner], times[-1], p["u"], p["mu"],
                                                         p["f"], init)))
        lines = [stability_info(p["u"], p["mu"], p["h"], p["tau"], p["sigma"]), "",
                 f"Узлов N+1 = {len(x)}, слоёв по времени = {int(round(p['T'] / p['tau']))}"]
        if times[-1] < p["T"] - 1e-9:
            lines.append(f"!!! Решение разрушилось при t = {times[-1]:.3g} с")
        else:
            c_inf = p["p"] / (p["q"] + p["u"] / p["mu"]) if p["mu"] > 0 else float("nan")
            lines += [f"t = {times[-1]:.4g} с:",
                      f"  максимум импульса C = {c_end[i_max]:.4f} при x = {x[i_max]:.2f} м",
                      f"  ожидаемое положение вершины x₂ + u·T = {p['x2'] + p['u'] * p['T']:.2f} м",
                      f"  C(L) = {c_end[-1]:.4f} (стационар p/(q + u/μ) = {c_inf:.4f})",
                      f"  max|C − C_точн| при x ≤ {x_cut:g} м: {err:.2e}"]
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, "\n".join(lines))

    # ---------------- графики ----------------
    def _plot_profiles(self):
        p, x, times, hist = self.result
        _, fig, canvas = self.tabs["Профили C(x,t)"]
        fig.clear()
        ax = fig.add_subplot(111)
        dt_out = p["dt_out"] if p["dt_out"] > 0 else p["T"]
        for t_out in np.arange(0.0, times[-1] + 1e-9, dt_out):
            k = int(np.argmin(np.abs(times - t_out)))
            ax.plot(x, hist[k], label=f"t = {times[k]:.4g} с")
        ax.set_title(f"Профили концентрации C(x, t), σ = {p['sigma']:g}")
        ax.set_xlabel("x, м")
        ax.set_ylabel("C")
        ax.grid(True, alpha=0.4)
        ax.legend()
        canvas.draw()

    def _plot_heatmap(self):
        p, x, times, hist = self.result
        _, fig, canvas = self.tabs["Тепловая карта"]
        fig.clear()
        ax = fig.add_subplot(111)
        vmax = None
        if times[-1] < p["T"] - 1e-9:      # схема разошлась — ограничиваем шкалу
            vmax = 2 * max(p["H"], abs(p["cl"]), 1.0)
        im = ax.imshow(hist.T, aspect="auto", origin="lower", cmap="viridis",
                       extent=[times[0], times[-1], x[0], x[-1]],
                       vmin=None if vmax is None else -vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, label="Концентрация C")
        ax.plot(times, p["x2"] + p["u"] * times, "w--", lw=1, label="x₂ + u·t")
        ax.set_ylim(x[0], x[-1])
        ax.set_title(f"Эволюция поля концентрации, σ = {p['sigma']:g}")
        ax.set_xlabel("Время t, с")
        ax.set_ylabel("Координата x, м")
        ax.legend(loc="lower right")
        canvas.draw()

    def _plot_boundary(self):
        p, x, times, hist = self.result
        _, fig, canvas = self.tabs["C на правой границе"]
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(times, hist[:, -1], label="C(L, t) — численно")
        if p["mu"] > 0:
            c_inf = p["p"] / (p["q"] + p["u"] / p["mu"])
            ax.axhline(c_inf, color="k", ls="--", lw=1, label=f"стационар p/(q+u/μ) = {c_inf:.4f}")
        ax.set_title("Концентрация на правой границе x = L")
        ax.set_xlabel("t, с")
        ax.set_ylabel("C(L, t)")
        ax.grid(True, alpha=0.4)
        ax.legend()
        canvas.draw()

    def _on_slider(self, value):
        if self.result is None:
            return
        p, x, times, hist = self.result
        k = int(float(value))
        init = (p["x1"], p["x2"], p["x3"], p["H"])
        _, fig, canvas = self.tabs["Динамика"]
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(x, hist[k], lw=2, label=f"численное, σ = {p['sigma']:g}")
        ax.plot(x, analytic_free(x, times[k], p["u"], p["mu"], p["f"], init), "r--",
                lw=1.2, label="точное (без учёта границ)")
        top = max(p["H"], float(np.max(hist[0]))) * 1.15
        ax.set_ylim(-0.1 * top, top)
        ax.set_title(f"t = {times[k]:.4g} с")
        ax.set_xlabel("x, м")
        ax.set_ylabel("C")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper left")
        canvas.draw()

    def compare_schemes(self):
        try:
            p = self._params()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        init = (p["x1"], p["x2"], p["x3"], p["H"])
        _, fig, canvas = self.tabs["Сравнение схем"]
        fig.clear()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        x_cut = 0.8 * p["L"]
        for s, style, color in [(0.0, ":", "tab:blue"), (0.5, "-", "tab:orange"),
                                (1.0, "--", "tab:green")]:
            x, times, hist = self._run(p, sigma=s)
            if times[-1] < p["T"] - 1e-9:
                ax1.plot([], [], style, color=color, label=f"σ = {s:g}: неустойчива")
                continue
            ax1.plot(x, hist[-1], style, color=color, lw=1.5, label=f"σ = {s:g}")
            err = error_vs_time(x, times, hist, p["u"], p["mu"], p["f"], init, x_cut)
            ax2.semilogy(times[1:], err[1:], style, color=color, label=f"σ = {s:g}")
        ax1.plot(x, analytic_free(x, p["T"], p["u"], p["mu"], p["f"], init), "k", lw=0.8,
                 label="точное (без учёта границ)")
        ax1.set_title(f"Профили при t = {p['T']:g} с  (h = {p['h']:g}, τ = {p['tau']:g})")
        ax1.set_xlabel("x, м")
        ax1.set_ylabel("C")
        ax1.grid(True, alpha=0.4)
        ax1.legend()
        ax2.set_title(f"Погрешность max|C − C_точн| при x ≤ {x_cut:g} м")
        ax2.set_xlabel("t, с")
        ax2.grid(True, which="both", alpha=0.4)
        ax2.legend()
        fig.tight_layout()
        canvas.draw()
        self.notebook.select(self.tabs["Сравнение схем"][0])

    def save_figures(self):
        folder = filedialog.askdirectory(title="Папка для сохранения графиков")
        if not folder:
            return
        for i, (name, (_, fig, _)) in enumerate(self.tabs.items(), start=1):
            fig.savefig(f"{folder}/fig{i}.png", dpi=150, bbox_inches="tight")
        messagebox.showinfo("Готово", f"Графики сохранены в папку:\n{folder}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if tk is None:
        raise SystemExit("Для запуска интерфейса нужен модуль tkinter")
    App().run()