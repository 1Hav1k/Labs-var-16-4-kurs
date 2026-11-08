import csv
import os
import re

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# =============================================================================
#                         ИСХОДНЫЕ ДАННЫЕ ПО УМОЛЧАНИЮ
# =============================================================================
DEFAULTS = {
    "vertices": "30, 0; 40, 1; 60, 0",   # вершины треугольника (x, T)
    "a": "1",                             # коэффициент температуропроводности
    "l": "100",                           # длина отрезка
    "tmax": "100",                        # время моделирования t, с
    "h": "10",                            # шаг по пространству
    "r": "0.4",                           # безразмерный параметр r = a*tau/h^2
    "sigma": "0.5",                       # вес схемы (0.5 — Кранк–Николсон)
    "K": "200",                           # число членов ряда Фурье
    "times": "0; 100",                    # моменты вывода (пусто — 0 и t)
    "dx_out": "10",                       # шаг вывода по x в таблицах
}

METHOD_KEYS = ("analytic", "explicit", "implicit", "weighted")
METHOD_NAMES = {
    "analytic": "Аналитическое решение",
    "explicit": "Явная схема",
    "implicit": "Неявная схема",
    "weighted": "Схема с весами",
}

# Цвета: методы — фиксированный порядок + разные типы линий; время — один оттенок
METHOD_COLORS = {
    "analytic": "#0b0b0b",
    "explicit": "#2a78d6",
    "implicit": "#eb6834",
    "weighted": "#1baf7a",
}
METHOD_STYLES = {
    "analytic": dict(ls="-", lw=2.4),
    "explicit": dict(ls="--", lw=1.8),
    "implicit": dict(ls="-.", lw=1.8),
    "weighted": dict(ls=":", lw=2.2),
}
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


# =============================================================================
#                           ЧИСЛЕННАЯ ЧАСТЬ
# =============================================================================
def initial_profile(x, vertices):
    """Начальное условие T0(x): ломаная по вершинам, вне её — ноль."""
    xs = [p[0] for p in vertices]
    ys = [p[1] for p in vertices]
    return np.interp(x, xs, ys, left=0.0, right=0.0)


def make_grid(l, h, tmax, a, r):
    """
    Сетка: x_i = i*h (i = 0..N-1);  tau = r*h^2/a;  t_n = n*tau (n = 0..M).
    M = ceil(t_кон / tau); если t_кон не кратно tau, последний шаг укорачивается
    так, чтобы t_M = t_кон. Возвращает x, t, tau и массив rs = a*tau_n/h^2
    для каждого шага (все равны r, кроме, возможно, последнего).
    """
    N = int(round(l / h)) + 1
    tau = r * h ** 2 / a
    M = max(1, int(np.ceil(tmax / tau - 1e-9)))
    x = np.linspace(0.0, l, N)
    t = np.minimum(tau * np.arange(M + 1), tmax)
    t[-1] = tmax
    rs = a * np.diff(t) / h ** 2
    return x, t, tau, rs


# ---------------------------- 1. Аналитическое -------------------------------
def fourier_coefficients(vertices, l, K):
    """
    C_k = 2/l * ∫_0^l T0(x) sin(pi k x / l) dx  — точное значение.
    Для ломаной двукратное интегрирование по частям даёт (mu_k = pi k / l):
        C_k = 2/(l mu_k^2) * Σ_m s_m [sin(mu_k x_{m+1}) - sin(mu_k x_m)],
    где s_m — наклон m-го звена. Для треугольника (30,0)-(40,1)-(60,0):
        C_k = 2/(l mu_k^2) * (0.15 sin 40mu_k - 0.1 sin 30mu_k - 0.05 sin 60mu_k).
    """
    k = np.arange(1, K + 1)
    mu = np.pi * k / l
    total = np.zeros(K)
    for (x1, y1), (x2, y2) in zip(vertices[:-1], vertices[1:]):
        s = (y2 - y1) / (x2 - x1)
        total += s * (np.sin(mu * x2) - np.sin(mu * x1))
    return mu, 2.0 / (l * mu ** 2) * total


def solve_analytic(x, t, a, l, vertices, K):
    """T(x,t) = Σ_{k=1}^{K} C_k exp(-a (pi k / l)^2 t) sin(pi k x / l)."""
    mu, C = fourier_coefficients(vertices, l, K)
    S = np.sin(np.outer(mu, x))
    E = np.exp(-a * np.outer(t, mu ** 2))
    return (E * C) @ S


# ----------------------------- 2. Явная схема --------------------------------
def solve_explicit(T0v, rs):
    """
    T_i^{n+1} = T_i^n + r (T_{i+1}^n - 2 T_i^n + T_{i-1}^n),   r = a*tau/h^2,
    i = 1, ..., N-2;  T_0^{n+1} = T_{N-1}^{n+1} = 0.
    Устойчива при r <= 1/2.  rs — значения r на каждом шаге.
    """
    M = len(rs)
    N = len(T0v)
    U = np.zeros((M + 1, N))
    U[0] = T0v
    U[0, 0] = U[0, -1] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(M):
            r = rs[n]
            U[n + 1, 1:-1] = U[n, 1:-1] + r * (U[n, 2:] - 2 * U[n, 1:-1] + U[n, :-2])
            U[n + 1, 0] = 0.0
            U[n + 1, -1] = 0.0
    return U

# ---------------------------- 3. Неявная схема -------------------------------
def solve_implicit(T0v, rs):
    """
    -r T_{i-1}^{n+1} + (1 + 2r) T_i^{n+1} - r T_{i+1}^{n+1} = T_i^n
    A_i = B_i = r,  C_i = 1 + 2r,  F_i = T_i^n.  Абсолютно устойчива.
    """
    M = len(rs)
    N = len(T0v)
    U = np.zeros((M + 1, N))
    U[0] = T0v
    U[0, 0] = U[0, -1] = 0.0
    cache = {}
    for n in range(M):
        r = rs[n]
        A = B = r
        C = 1.0 + 2.0 * r
        alpha, gamma = _cached_coefficients(cache, A, B, C, N)
        F = U[n].copy()
        U[n + 1] = progonka(alpha, gamma, A, F, 0.0, 0.0)
    return U
 
 
# ---------------------------- 4. Схема с весами ------------------------------
def solve_weighted(T0v, rs, sigma):
    """
    -sigma r T_{i-1}^{n+1} + (1 + 2 sigma r) T_i^{n+1} - sigma r T_{i+1}^{n+1} =
        = T_i^n + (1 - sigma) r (T_{i+1}^n - 2 T_i^n + T_{i-1}^n)
    A_i = B_i = sigma r,  C_i = 1 + 2 sigma r,
    F_i = T_i^n + (1 - sigma) r (T_{i+1}^n - 2 T_i^n + T_{i-1}^n).
    """
    M = len(rs)
    N = len(T0v)
    U = np.zeros((M + 1, N))
    U[0] = T0v
    U[0, 0] = U[0, -1] = 0.0
    cache = {}
    for n in range(M):
        r = rs[n]
        A = B = sigma * r
        C = 1.0 + 2.0 * sigma * r
        alpha, gamma = _cached_coefficients(cache, A, B, C, N)
        T = U[n]
        F = T.copy()
        F[1:-1] = T[1:-1] + (1.0 - sigma) * r * (T[2:] - 2 * T[1:-1] + T[:-2])
        U[n + 1] = progonka(alpha, gamma, A, F, 0.0, 0.0)
    return U

# ---------------------------- Метод прогонки ---------------------------------
def progonka_coefficients(A, B, C, N):
    """
    Прямой ход прогонки, часть 1 — то, что НЕ зависит от правой части F.
    Матрица системы  -A T_{i-1} + C T_i - B T_{i+1} = F_i  одна и та же на всех
    временных слоях (при постоянном r), поэтому alpha_i и множители
        gamma_i = 1 / (C - A alpha_{i-1}),   alpha_i = B * gamma_i,   i = 1..N-2,
    вычисляются ОДИН раз до начала расчёта по времени (alpha_0 = 0).
    """
    alpha = np.zeros(N)
    gamma = np.zeros(N)
    for i in range(1, N - 1):
        gamma[i] = 1.0 / (C - A * alpha[i - 1])
        alpha[i] = B * gamma[i]
    return alpha, gamma
 
 
def progonka(alpha, gamma, A, F, T_left, T_right):
    """
    Решение системы  -A_i T_{i-1} + C_i T_i - B_i T_{i+1} = F_i,  i = 1..N-2,
    при известных T_0 = T_left, T_{N-1} = T_right, с заранее вычисленными
    alpha_i и gamma_i (см. progonka_coefficients). На каждом слое — 5 операций на узел:
 
    Прямой ход:  beta_0 = T_0;  i = 1..N-2:
        beta_i = (F_i + A_i beta_{i-1}) * gamma_i          — 3 операции (×, +, ×)
    Обратный ход: T_{N-1} = T_right;  i = N-2..1:
        T_i = alpha_i T_{i+1} + beta_i                      — 2 операции (×, +)
    """
    N = len(F)
    beta = np.zeros(N)
    beta[0] = T_left
    for i in range(1, N - 1):
        beta[i] = (F[i] + A * beta[i - 1]) * gamma[i]
    T = np.zeros(N)
    T[0] = T_left
    T[N - 1] = T_right
    for i in range(N - 2, 0, -1):
        T[i] = alpha[i] * T[i + 1] + beta[i]
    return T
 
 
def _cached_coefficients(cache, A, B, C, N):
    """Прогоночные коэффициенты для данной матрицы считаются один раз и запоминаются
    (при укороченном последнем шаге r' != r нужен второй набор)."""
    key = (round(A, 14), round(B, 14), round(C, 14))
    if key not in cache:
        cache[key] = progonka_coefficients(A, B, C, N)
    return cache[key]

# =============================================================================
#                                ГРАФИКИ
# =============================================================================
def _style_axes(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK_2, labelsize=9)


def time_colors(count):
    if count == 1:
        return [BLUE_RAMP[-1]]
    idx = np.linspace(0, len(BLUE_RAMP) - 1, count)
    return [BLUE_RAMP[int(round(v))] for v in idx]


def _is_unstable(U, vmax):
    with np.errstate(invalid="ignore"):
        return (not np.all(np.isfinite(U))) or np.nanmax(np.abs(U)) > 2 * vmax


def plot_initial(fig, p, x, T0v):
    fig.clear()
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.fill_between(x, T0v, color="#cde2fb", alpha=0.6, lw=0)
    ax.plot(x, T0v, color=BLUE_RAMP[4], lw=2.2)
    vx = [v[0] for v in p["vertices"]]
    vy = [v[1] for v in p["vertices"]]
    ax.plot(vx, vy, "o", ms=8, color=BLUE_RAMP[4], mec=SURFACE, mew=2, zorder=5)
    top = max(vy) if max(vy) > 0 else 1
    for xv, yv in p["vertices"]:
        ax.annotate(f"({xv:g}; {yv:g})", (xv, yv), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, color=INK)
    ax.set_xlim(0, p["l"])
    ax.set_ylim(-0.05 * top, 1.25 * top)
    ax.set_xlabel("x", color=INK_2)
    ax.set_ylabel("T(x, 0)", color=INK_2)
    ax.set_title(f"Начальное распределение температуры на отрезке [0; {p['l']:g}]",
                 color=INK, fontsize=12)
    ax.text(0.01, 0.97, f"Граничные условия: T(0, t) = 0,  T({p['l']:g}, t) = 0\n"
                        f"a = {p['a']:g},  время моделирования t = {p['tmax']:g} с",
            transform=ax.transAxes, va="top", fontsize=9, color=INK_2)


def _marker_style(x):
    """На грубой сетке узлы отмечаются маркерами."""
    return dict(marker="o", ms=5) if len(x) <= 51 else {}


def plot_method(fig, key, p, x, t, U, out_idx, fine=None):
    """fine = (x_fine, {n: T(x_fine, t_n)}) — аналитическое решение на мелкой сетке."""
    fig.clear()
    ax = fig.add_subplot(111)
    _style_axes(ax)
    vmax = max(v[1] for v in p["vertices"]) or 1.0
    with np.errstate(invalid="ignore", over="ignore"):
        for c, n in zip(time_colors(len(out_idx)), out_idx):
            if key == "analytic" and fine is not None:
                ax.plot(fine[0], fine[1][n], color=c, lw=1.8, label=f"t = {t[n]:g} с")
                if len(x) <= 51:
                    ax.plot(x, U[n], "o", color=c, ms=5)
            else:
                ax.plot(x, U[n], color=c, lw=1.8, label=f"t = {t[n]:g} с", **_marker_style(x))
    unstable = _is_unstable(U, vmax)
    if unstable:
        ax.set_ylim(-0.6 * vmax, 1.6 * vmax)
        ax.text(0.02, 0.96, f"Схема неустойчива: r = {p['r']:g} > 0.5", transform=ax.transAxes,
                ha="left", va="top", color="#d03b3b", fontsize=10, fontweight="bold", zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="#d03b3b"))
    ax.set_xlim(0, p["l"])
    ax.set_xlabel("x", color=INK_2)
    ax.set_ylabel("T(x, t)", color=INK_2)
    extra = f",  σ = {p['sigma']:g}" if key == "weighted" else ""
    ax.set_title(f"{METHOD_NAMES[key]}: распределение T(x, t)   "
                 f"(h = {p['h']:g},  r = {p['r']:g},  τ = {p['tau']:g}{extra})",
                 color=INK, fontsize=11)
    leg = ax.legend(fontsize=9, frameon=unstable, loc="upper right",
                    facecolor=SURFACE, framealpha=0.95)
    leg.set_zorder(10)


def plot_comparison(fig, p, x, t, results, n_sel, fine=None):
    fig.clear()
    ax = fig.add_subplot(111)
    _style_axes(ax)
    vmax = max(v[1] for v in p["vertices"]) or 1.0
    # начальное условие (треугольник) — серым фоном
    xf = fine[0] if fine is not None else x
    T0v = initial_profile(xf, p["vertices"])
    ax.fill_between(xf, T0v, color="#f0efec", lw=0, zorder=1)
    ax.plot(xf, T0v, color="#898781", lw=1.6, zorder=2, label="Начальное условие T(x, 0)")
    for key in METHOD_KEYS:
        with np.errstate(invalid="ignore", over="ignore"):
            if key == "analytic" and fine is not None and n_sel in fine[1]:
                ax.plot(xf, fine[1][n_sel], color=METHOD_COLORS[key], zorder=3,
                        label=METHOD_NAMES[key], **METHOD_STYLES[key])
            else:
                st = dict(METHOD_STYLES[key])
                if len(x) <= 51:
                    st.update(marker={"explicit": "o", "implicit": "s", "weighted": "^"}[key], ms=6)
                ax.plot(x, results[key][n_sel], color=METHOD_COLORS[key], zorder=3,
                        label=METHOD_NAMES[key], **st)
    ax.set_xlim(0, p["l"])
    if _is_unstable(results["explicit"], vmax):
        ax.set_ylim(-0.6 * vmax, 1.6 * vmax)
    ax.set_xlabel("x", color=INK_2)
    ax.set_ylabel("T(x, t)", color=INK_2)
    ax.set_title(f"Сравнение методов при t = {t[n_sel]:g} с   "
                 f"(h = {p['h']:g},  r = {p['r']:g},  τ = {p['tau']:g},  σ = {p['sigma']:g})",
                 color=INK, fontsize=11)
    leg = ax.legend(fontsize=9, loc="upper right", facecolor=SURFACE, framealpha=0.95,
                    edgecolor=GRID)
    leg.set_zorder(10)


# =============================================================================
#                           ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# =============================================================================
def parse_number(s, name):
    try:
        return float(s.strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"Поле «{name}»: «{s}» не является числом")


def parse_list(s, name):
    parts = [w for w in re.split(r"[;\s]+", s.strip()) if w]
    return [parse_number(w, name) for w in parts]


def parse_vertices(s):
    pts = []
    for chunk in s.split(";"):
        chunk = chunk.strip().strip("()")
        if not chunk:
            continue
        nums = [w for w in re.split(r"[,\s]+", chunk) if w]
        if len(nums) != 2:
            raise ValueError("Вершины задаются парами «x, T», разделёнными «;» "
                             "(дробная часть — через точку), например: 30, 0; 40, 1; 60, 0")
        pts.append((float(nums[0]), float(nums[1])))
    if len(pts) < 3:
        raise ValueError("Нужно не меньше трёх вершин")
    xs = [v[0] for v in pts]
    if any(x2 <= x1 for x1, x2 in zip(xs[:-1], xs[1:])):
        raise ValueError("Абсциссы вершин должны строго возрастать")
    if abs(pts[0][1]) > 1e-12 or abs(pts[-1][1]) > 1e-12:
        raise ValueError("Первая и последняя вершины должны лежать на оси x (T = 0)")
    return pts


class HeatApp:
    def __init__(self, root):
        self.root = root
        root.title("Лабораторная работа №3 — Решение задачи теплопроводности численными методами")
        root.geometry("1400x880")
        root.minsize(1100, 700)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=22, font=("Consolas", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

        self.vars = {k: tk.StringVar(value=v) for k, v in DEFAULTS.items()}
        self.results = None
        self.figs = {}
        self.canvases = {}
        self.tables = {}

        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=(0, 8, 8, 8))
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self._build_params(left)
        self._build_tabs(right)
        root.after(100, self.calculate)

    # ------------------------------------------------------------------ панель
    def _build_params(self, parent):
        box = ttk.LabelFrame(parent, text="Исходные данные", padding=8)
        box.pack(fill=tk.X)
        rows = [
            ("vertices", "Вершины T0(x)  (x, T; ...)"),
            ("a", "Коэффициент a"),
            ("l", "Длина отрезка l"),
            ("tmax", "Время моделирования t, с"),
            ("h", "Шаг по пространству h"),
            ("r", "Параметр r = aτ/h²"),
            ("sigma", "Вес σ (схема с весами)"),
            ("K", "Членов ряда Фурье K"),
        ]
        for r, (key, label) in enumerate(rows):
            ttk.Label(box, text=label).grid(row=2 * r, column=0, sticky="w", pady=(4, 0))
            e = ttk.Entry(box, textvariable=self.vars[key], width=30)
            e.grid(row=2 * r + 1, column=0, sticky="we")
            e.bind("<Return>", lambda ev: self.calculate())

        out = ttk.LabelFrame(parent, text="Вывод результатов", padding=8)
        out.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(out, text="Моменты времени через «;»\n(пусто — 0 и t)").grid(
            row=0, column=0, sticky="w")
        e = ttk.Entry(out, textvariable=self.vars["times"], width=30)
        e.grid(row=1, column=0, sticky="we")
        e.bind("<Return>", lambda ev: self.calculate())
        ttk.Label(out, text="Шаг вывода по x в таблицах").grid(row=2, column=0, sticky="w", pady=(4, 0))
        e = ttk.Entry(out, textvariable=self.vars["dx_out"], width=30)
        e.grid(row=3, column=0, sticky="we")
        e.bind("<Return>", lambda ev: self.calculate())

        btns = ttk.Frame(parent)
        btns.pack(fill=tk.X, pady=10)
        ttk.Button(btns, text="Рассчитать", style="Accent.TButton",
                   command=self.calculate).pack(fill=tk.X)
        ttk.Button(btns, text="Сохранить графики и таблицы…",
                   command=self.save_results).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btns, text="Сбросить к исходным данным",
                   command=self.reset_defaults).pack(fill=tk.X, pady=(6, 0))

        info = ttk.LabelFrame(parent, text="Параметры сетки", padding=8)
        info.pack(fill=tk.X)
        self.info_label = tk.Label(info, text="", justify=tk.LEFT, anchor="w",
                                   font=("Consolas", 9))
        self.info_label.pack(fill=tk.X)
        self.stab_label = tk.Label(info, text="", justify=tk.LEFT, anchor="w",
                                   font=("Segoe UI", 9, "bold"), wraplength=260)
        self.stab_label.pack(fill=tk.X, pady=(6, 0))

    # ------------------------------------------------------------------ вкладки
    def _make_figure(self, parent, key, height=4.2):
        fig = Figure(figsize=(10, height), dpi=100, layout="constrained", facecolor=SURFACE)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        toolbar = NavigationToolbar2Tk(canvas, parent, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.figs[key] = fig
        self.canvases[key] = canvas
        return fig

    def _make_table(self, parent, key):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(frame, show="headings")
        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="we")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.tables[key] = tree
        return tree

    def _split_tab(self, title):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=title)
        return tab

    def _plot_and_table(self, tab, key, table_title):
        pw = ttk.PanedWindow(tab, orient=tk.VERTICAL)
        pw.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(pw)
        bottom = ttk.Frame(pw)
        pw.add(top, weight=3)
        pw.add(bottom, weight=2)
        self._make_figure(top, key)
        ttk.Label(bottom, text=table_title, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(4, 2))
        self._make_table(bottom, key)

    def _build_tabs(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.pack(fill=tk.BOTH, expand=True)

        tab = self._split_tab("Исходные данные")
        self._make_figure(tab, "initial", height=5)

        for key in METHOD_KEYS:
            tab = self._split_tab(METHOD_NAMES[key])
            title = "Таблица значений T(x_i, t_n)"
            if key != "analytic":
                title += ("   (последняя строка — максимальная погрешность "
                          "max|T − T_аналит| относительно аналитического решения)")
            self._plot_and_table(tab, key, title)

        tab = self._split_tab("Сравнение методов")
        ctrl = ttk.Frame(tab)
        ctrl.pack(fill=tk.X, pady=4)
        ttk.Label(ctrl, text="Момент времени для сравнения, с:").pack(side=tk.LEFT, padx=(4, 4))
        self.cmp_time = tk.StringVar()
        self.cmp_combo = ttk.Combobox(ctrl, textvariable=self.cmp_time, width=12, state="readonly")
        self.cmp_combo.pack(side=tk.LEFT)
        self.cmp_combo.bind("<<ComboboxSelected>>", lambda e: self.update_comparison())
        self._plot_and_table(tab, "compare",
                             "Значения температуры в точке максимума начального профиля")

    # --------------------------------------------------------------- расчёт
    def read_params(self):
        v = {k: var.get() for k, var in self.vars.items()}
        p = {
            "vertices": parse_vertices(v["vertices"]),
            "a": parse_number(v["a"], "Коэффициент a"),
            "l": parse_number(v["l"], "Длина отрезка l"),
            "tmax": parse_number(v["tmax"], "Время моделирования t"),
            "h": parse_number(v["h"], "Шаг по пространству h"),
            "r": parse_number(v["r"], "Параметр r"),
            "sigma": parse_number(v["sigma"], "Вес σ"),
            "K": int(parse_number(v["K"], "Членов ряда Фурье")),
            "times": parse_list(v["times"], "Моменты времени"),
            "dx_out": parse_number(v["dx_out"], "Шаг вывода по x"),
        }
        for key, name in (("a", "a"), ("l", "l"), ("tmax", "t"), ("h", "h"),
                          ("r", "r"), ("dx_out", "шаг вывода по x")):
            if p[key] <= 0:
                raise ValueError(f"Параметр {name} должен быть положительным")
        if p["K"] < 1:
            raise ValueError("Число членов ряда должно быть не меньше 1")
        if not 0 <= p["sigma"] <= 1:
            raise ValueError("Вес σ должен лежать в пределах [0; 1]")
        N = p["l"] / p["h"]
        if abs(N - round(N)) > 1e-6 * max(1, N) or round(N) < 2:
            raise ValueError("Шаг h должен укладываться в отрезок l целое число раз")
        if p["vertices"][0][0] < 0 or p["vertices"][-1][0] > p["l"]:
            raise ValueError("Вершины T0(x) должны лежать внутри отрезка [0; l]")
        if any(tt < 0 or tt > p["tmax"] * (1 + 1e-9) for tt in p["times"]):
            raise ValueError("Моменты времени должны лежать в пределах [0; t]")
        return p

    def calculate(self):
        try:
            p = self.read_params()
        except ValueError as e:
            messagebox.showerror("Ошибка в исходных данных", str(e))
            return
        x, t, tau, rs = make_grid(p["l"], p["h"], p["tmax"], p["a"], p["r"])
        p["tau"] = tau
        N, M = len(x), len(t) - 1
        if N * M > 4_000_000:
            if not messagebox.askyesno(
                    "Большая сетка",
                    f"Сетка {N} × {M + 1} узлов: расчёт может занять заметное время. Продолжить?"):
                return
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            T0v = initial_profile(x, p["vertices"])
            res = {
                "analytic": solve_analytic(x, t, p["a"], p["l"], p["vertices"], p["K"]),
                "explicit": solve_explicit(T0v, rs),
                "implicit": solve_implicit(T0v, rs),
                "weighted": solve_weighted(T0v, rs, p["sigma"]),
            }
        finally:
            self.root.config(cursor="")

        # моменты вывода: заданные пользователем или 0 и t
        times = p["times"] or [0.0, p["tmax"]]
        out_idx = sorted({int(np.argmin(np.abs(t - tt))) for tt in times})
        # аналитическое решение на мелкой сетке — для гладких графиков
        x_fine = np.linspace(0.0, p["l"], 1001)
        Uf = solve_analytic(x_fine, t[out_idx], p["a"], p["l"], p["vertices"], p["K"])
        fine = (x_fine, {n: Uf[j] for j, n in enumerate(out_idx)})
        self.results = dict(p=p, x=x, t=t, T0=T0v, res=res, out_idx=out_idx, fine=fine)

        r = p["r"]
        lines = [f"τ = r·h²/a        = {tau:.6g}",
                 f"N (узлов по x)   = {N}",
                 f"M (шагов по t)   = {M}",
                 f"σ·r              = {p['sigma'] * r:.6g}"]
        if abs(rs[-1] - r) > 1e-9 * r:
            lines.append(f"t не кратно τ: последний\nшаг τ' = {t[-1] - t[-2]:.6g} с, "
                         f"r' = {rs[-1]:.4g}")
        self.info_label.config(text="\n".join(lines))
        if r <= 0.5 + 1e-12:
            self.stab_label.config(fg="#006300", text="✔ Явная схема устойчива: r ≤ 0.5")
        else:
            self.stab_label.config(fg="#d03b3b",
                                   text=f"✖ Явная схема НЕУСТОЙЧИВА: r = {r:g} > 0.5. "
                                        f"Неявная схема и схема с весами (σ ≥ 0.5) устойчивы.")

        self.update_plots()
        self.update_tables()
        self.cmp_combo["values"] = [f"{t[n]:g}" for n in out_idx]
        self.cmp_time.set(f"{t[out_idx[-1]]:g}")
        self.update_comparison()

    # --------------------------------------------------------------- вывод
    def update_plots(self):
        r = self.results
        plot_initial(self.figs["initial"], r["p"], r["x"], r["T0"])
        self.canvases["initial"].draw_idle()
        for key in METHOD_KEYS:
            plot_method(self.figs[key], key, r["p"], r["x"], r["t"], r["res"][key], r["out_idx"],
                        r["fine"])
            self.canvases[key].draw_idle()

    def table_data(self, key):
        """Строки: узлы x с шагом вывода; столбцы: выбранные моменты времени."""
        r = self.results
        x, t = r["x"], r["t"]
        step = max(1, int(round(r["p"]["dx_out"] / r["p"]["h"])))
        rows_idx = list(range(0, len(x), step))
        if rows_idx[-1] != len(x) - 1:
            rows_idx.append(len(x) - 1)
        header = ["x"] + [f"t = {t[n]:g}" for n in r["out_idx"]]
        rows = [[x[i]] + [r["res"][key][n, i] for n in r["out_idx"]] for i in rows_idx]
        if key != "analytic":
            ref = r["res"]["analytic"]
            with np.errstate(invalid="ignore", over="ignore"):
                rows.append(["max|Δ|"] + [np.max(np.abs(r["res"][key][n] - ref[n]))
                                          for n in r["out_idx"]])
        return header, rows

    def compare_data(self):
        r = self.results
        t = r["t"]
        xpk = max(r["p"]["vertices"], key=lambda v: v[1])[0]
        ipk = int(np.argmin(np.abs(r["x"] - xpk)))
        xp = f"{r['x'][ipk]:g}"
        header = ["t, с", f"T({xp}) аналит.", f"T({xp}) явная", f"T({xp}) неявная",
                  f"T({xp}) с весами"]
        rows = []
        with np.errstate(invalid="ignore", over="ignore"):
            for n in r["out_idx"]:
                row = [t[n]]
                for key in METHOD_KEYS:
                    row.append(r["res"][key][n, ipk])
                rows.append(row)
        return header, rows

    @staticmethod
    def _fmt(v, first=False):
        if isinstance(v, str):
            return v
        if first:
            return f"{v:g}"
        if not np.isfinite(v):
            return "переполн." if np.isinf(v) else "нет числа"
        if v != 0 and (abs(v) < 1e-4 or abs(v) >= 1e5):
            return f"{v:.4e}"
        return f"{v:.6f}"

    def _fill_tree(self, tree, header, rows):
        tree.delete(*tree.get_children())
        cols = [f"c{j}" for j in range(len(header))]
        tree["columns"] = cols
        for j, (c, hname) in enumerate(zip(cols, header)):
            tree.heading(c, text=hname)
            tree.column(c, width=70 if j == 0 else 118, anchor="center", stretch=True)
        tree.tag_configure("err", background="#f0efec", font=("Consolas", 9, "bold"))
        for row in rows:
            tags = ("err",) if isinstance(row[0], str) else ()
            tree.insert("", tk.END, values=[self._fmt(v, j == 0) for j, v in enumerate(row)],
                        tags=tags)

    def update_tables(self):
        for key in METHOD_KEYS:
            self._fill_tree(self.tables[key], *self.table_data(key))
        self._fill_tree(self.tables["compare"], *self.compare_data())

    def update_comparison(self):
        if not self.results:
            return
        r = self.results
        try:
            tsel = float(self.cmp_time.get())
        except ValueError:
            tsel = r["t"][r["out_idx"][0]]
        n = int(np.argmin(np.abs(r["t"] - tsel)))
        plot_comparison(self.figs["compare"], r["p"], r["x"], r["t"], r["res"], n, r["fine"])
        self.canvases["compare"].draw_idle()

    # ------------------------------------------------------------ сохранение
    def save_results(self):
        if not self.results:
            return
        folder = filedialog.askdirectory(title="Папка для сохранения результатов")
        if not folder:
            return
        names = {"initial": "1_Исходные_данные", "analytic": "2_Аналитическое_решение",
                 "explicit": "3_Явная_схема", "implicit": "4_Неявная_схема",
                 "weighted": "5_Схема_с_весами", "compare": "6_Сравнение_методов"}
        try:
            for key, fname in names.items():
                self.figs[key].savefig(os.path.join(folder, fname + ".png"), dpi=200,
                                       facecolor=SURFACE)
            for key in (*METHOD_KEYS, "compare"):
                header, rows = self.compare_data() if key == "compare" else self.table_data(key)
                with open(os.path.join(folder, names[key] + ".csv"), "w", newline="",
                          encoding="utf-8-sig") as f:
                    w = csv.writer(f, delimiter=";")
                    w.writerow(header)
                    for row in rows:
                        w.writerow([self._fmt(v, j == 0).replace(".", ",")
                                    for j, v in enumerate(row)])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Ошибка сохранения", str(e))
            return
        messagebox.showinfo("Сохранено", f"Графики (PNG) и таблицы (CSV) сохранены в папку:\n{folder}")

    def reset_defaults(self):
        for k, v in DEFAULTS.items():
            self.vars[k].set(v)
        self.calculate()


def main():
    root = tk.Tk()
    HeatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()