"""
Лабораторна робота: Чисельне розв'язання ОДР
=============================================
Рівняння:  dy/dx = f(x, y) = sin(x) - y²
Відрізок:  [0, 1]
Початкова умова: y(0) = 0

Примітка: рівняння Ріккаті — точного аналітичного розв'язку
в елементарних функціях немає. Як "точний" використовується
еталонний розв'язок РК4 з кроком h_ref = 10^-6.
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def out(filename):
    return os.path.join(OUTPUT_DIR, filename)

# ─────────────────────────────────────────────────
# Задані параметри
# ─────────────────────────────────────────────────
def f(x, y):
    """Права частина ОДР: f(x,y) = sin(x) - y²"""
    return math.sin(x) - y ** 2

a, b    = 0.0, 1.0
y0      = 0.0
H_ADAMS = 0.1       # крок Адамса (Ч.1)
H_RK    = 1e-2      # крок РК4   (Ч.2)
H_REF   = 1e-6      # крок еталонного розв'язку
EPS     = 1e-5      # точність автовибору кроку


# ─────────────────────────────────────────────────
# Еталонний розв'язок (РК4, h=1e-6)
# ─────────────────────────────────────────────────
def build_reference():
    """РК4 з дуже малим кроком — використовується як 'точний' розв'язок."""
    print("  Будую еталонний розв'язок (РК4, h=1e-6)...", end=" ", flush=True)
    h = H_REF
    x, y = a, y0
    ref = {round(x, 8): y}
    while x < b - h * 0.5:
        k1 = h * f(x,         y)
        k2 = h * f(x + h/2,   y + k1/2)
        k3 = h * f(x + h/2,   y + k2/2)
        k4 = h * f(x + h,     y + k3)
        y  = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x  = x + h
        # Зберігаємо тільки кратні 1e-4, щоб не забивати пам'ять
        if abs(round(x * 10000) - x * 10000) < 1e-4:
            ref[round(x, 4)] = y
    print("готово.")
    return ref

def ref_value(ref, x):
    """Найближче значення еталону до точки x."""
    key = min(ref.keys(), key=lambda k: abs(k - x))
    return ref[key]


# ═══════════════════════════════════════════════════════════════
# Базовий крок РК4
# ═══════════════════════════════════════════════════════════════
def rk4_step(x, y, h):
    k1 = h * f(x,         y)
    k2 = h * f(x + h/2,   y + k1/2)
    k3 = h * f(x + h/2,   y + k2/2)
    k4 = h * f(x + h,     y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6


# ═══════════════════════════════════════════════════════════════
# Ч.1  МЕТОД ПРОГНОЗУ ТА КОРЕКЦІЇ АДАМСА 2-го ПОРЯДКУ
# ═══════════════════════════════════════════════════════════════
def adams2(a, b, y0, h):
    """
    Прогноз:   y^pr  = y_n + h/2*(3*f_n - f_{n-1})
    Корекція:  y^cor = y_n + h/2*(f(x_{n+1}, y^pr) + f_n)
    """
    xs    = [a]
    ys    = [y0]
    ys_pr = [None]

    # Стартовий крок через РК4
    y1 = rk4_step(a, y0, h)
    xs.append(a + h);  ys.append(y1);  ys_pr.append(None)

    n = 1
    while xs[-1] < b - h * 1e-9:
        xn, yn = xs[n], ys[n]
        fn     = f(xn, yn)
        fn_1   = f(xs[n-1], ys[n-1])
        xn1    = xn + h

        y_pr  = yn + h/2 * (3*fn - fn_1)
        y_cor = yn + h/2 * (f(xn1, y_pr) + fn)

        xs.append(xn1);  ys.append(y_cor);  ys_pr.append(y_pr)
        n += 1

    return xs, ys, ys_pr


def adams2_auto(a, b, y0, eps):
    """Адамс 2-го порядку з автоматичним вибором кроку."""
    h     = (b - a) / 10
    h_min = 1e-8
    h_max = (b - a) / 2

    xs = [a];  ys = [y0];  hs = []
    y1 = rk4_step(a, y0, h)
    xs.append(a + h);  ys.append(y1);  hs.append(h)

    n = 1
    while xs[-1] < b - h * 1e-9:
        xn, yn = xs[n], ys[n]
        fn     = f(xn, yn)
        fn_1   = f(xs[n-1], ys[n-1])

        if xn + h > b + 1e-12:
            h = b - xn

        xn1   = xn + h
        y_pr  = yn + h/2 * (3*fn - fn_1)
        y_cor = yn + h/2 * (f(xn1, y_pr) + fn)
        err   = abs(y_cor - y_pr)

        if err > eps and h > h_min * 2:
            h = max(h / 2, h_min)
            continue

        xs.append(xn1);  ys.append(y_cor);  hs.append(h)
        n += 1

        if err < eps / 4 and h < h_max:
            h = min(h * 2, h_max)
            if xs[-1] + h > b:
                h = b - xs[-1]

    return xs, ys, hs


# ═══════════════════════════════════════════════════════════════
# Ч.2  МЕТОД РУНГЕ-КУТТА 4-го ПОРЯДКУ
# ═══════════════════════════════════════════════════════════════
def runge_kutta4(a, b, y0, h):
    xs = [a];  ys = [y0]
    x, y = a, y0
    while x < b - h * 1e-9:
        if x + h > b + 1e-12:
            h = b - x
        y = rk4_step(x, y, h)
        x = x + h
        xs.append(x);  ys.append(y)
    return xs, ys


def runge_kutta4_auto(a, b, y0, eps):
    """РК4 з автовибором кроку за правилом Рунге: Φ = |y^(h/2) - y^h| / 15."""
    h     = (b - a) / 10
    h_min = 1e-8

    xs = [a];  ys = [y0];  hs = []
    x, y = a, y0

    while x < b - h * 1e-9:
        if x + h > b + 1e-12:
            h = b - x

        y_h   = rk4_step(x,       y,     h)
        y_mid = rk4_step(x,       y,     h/2)
        y_h2  = rk4_step(x + h/2, y_mid, h/2)

        phi = abs(y_h2 - y_h) / 15

        if phi > eps and h > h_min * 2:
            h /= 2
            continue

        x = x + h;  y = y_h2
        xs.append(x);  ys.append(y);  hs.append(h)

        if phi < eps / 32:
            h = min(h * 2, (b - a) / 2)

    return xs, ys, hs


# ═══════════════════════════════════════════════════════════════
# ЧАСТИНА 1: пункти 1–5
# ═══════════════════════════════════════════════════════════════
def part1_plots(ref):
    print("\n" + "="*60)
    print("Ч.1  МЕТОД АДАМСА 2-го ПОРЯДКУ")
    print("="*60)

    h = H_ADAMS
    xs, ys, ys_pr = adams2(a, b, y0, h)
    ye = [ref_value(ref, x) for x in xs]

    # Пункт 1
    print(f"\nПУНКТ 1: Еталонний розв'язок y'=sin(x)-y², y(0)=0")
    print(f"  {'x':>6}  {'y_ref':>14}")
    for x, yr in zip(xs, ye):
        print(f"  {x:6.3f}  {yr:14.10f}")

    # Пункт 2
    print(f"\nПУНКТ 2: Метод Адамса 2-го порядку (h={h})")
    print(f"  {'x':>6}  {'y_cor':>14}  {'y_ref':>14}  {'|err|':>12}")
    for x, yc, yr in zip(xs, ys, ye):
        print(f"  {x:6.3f}  {yc:14.10f}  {yr:14.10f}  {abs(yc-yr):12.2e}")

    # Пункт 3: локальна похибка через еталон
    phi_exact = [abs(yc - yr) for yc, yr in zip(ys, ye)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, phi_exact, 'b.-', markersize=7,
            label=r'$\varphi_n = |y_n - y_{ref}(x_n)|$')
    ax.set_xlabel('x');  ax.set_ylabel('Похибка')
    ax.set_title(f"П.3: Локальна похибка Адамса (еталонний розв'язок), h={h}")
    ax.grid(True, alpha=0.4);  ax.legend()
    fig.tight_layout()
    fig.savefig(out('adams_error_exact.png'), dpi=120)
    plt.close(fig)
    print(f"\nПУНКТ 3: Збережено: adams_error_exact.png")

    # Пункт 4: оцінка через (y^cor - y^pr)
    phi_est = [abs(ys[i] - ys_pr[i]) for i in range(2, len(xs))]
    xs_est  = xs[2:]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs_est, phi_est, 'r.-', markersize=7,
            label=r'$|y_{n+1}^{cor} - y_{n+1}^{pr}|$')
    ax.set_xlabel('x');  ax.set_ylabel('Оцінка похибки')
    ax.set_title(f'П.4: Оцінка локальної похибки Адамса, h={h}')
    ax.grid(True, alpha=0.4);  ax.legend()
    fig.tight_layout()
    fig.savefig(out('adams_error_est.png'), dpi=120)
    plt.close(fig)

    max_phi_exact = max(phi_exact[2:]) if len(phi_exact) > 2 else 1e-30
    max_phi_est   = max(phi_est) if phi_est else 0
    ratio = max_phi_est / max_phi_exact if max_phi_exact > 0 else float('inf')
    print(f"ПУНКТ 4: Збережено: adams_error_est.png")
    print(f"  max|ref err|   = {max_phi_exact:.3e}")
    print(f"  max|cor - pr|  = {max_phi_est:.3e}")
    print(f"  Відношення     = {ratio:.2f}  "
          f"({'оптимально' if 0.01 < ratio < 100 else 'крок варто скоригувати'})")

    # Пункт 5: автовибір кроку
    xs_auto, ys_auto, hs_auto = adams2_auto(a, b, y0, EPS)
    ye_auto = [ref_value(ref, x) for x in xs_auto]
    xs_mid  = [(xs_auto[i] + xs_auto[i+1]) / 2 for i in range(len(hs_auto))]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(xs_mid, hs_auto, where='mid', color='green', linewidth=1.5, label='h(x)')
    ax.set_xlabel('x');  ax.set_ylabel('Крок h')
    ax.set_title(f'П.5: Автоматичний вибір кроку (Адамс, ε={EPS})')
    ax.grid(True, alpha=0.4);  ax.legend()
    fig.tight_layout()
    fig.savefig(out('adams_auto_step.png'), dpi=120)
    plt.close(fig)

    max_err_auto = max(abs(yc - yr) for yc, yr in zip(ys_auto, ye_auto))
    print(f"\nПУНКТ 5: Збережено: adams_auto_step.png")
    print(f"  Кроків: {len(hs_auto)},  мін h: {min(hs_auto):.2e},  макс h: {max(hs_auto):.2e}")
    print(f"  Макс похибка з автокроком: {max_err_auto:.3e}")


# ═══════════════════════════════════════════════════════════════
# ЧАСТИНА 2: пункти 6–9
# ═══════════════════════════════════════════════════════════════
def part2_plots(ref):
    print("\n" + "="*60)
    print("Ч.2  МЕТОД РУНГЕ-КУТТА 4-го ПОРЯДКУ")
    print("="*60)

    h = H_RK
    xs, ys = runge_kutta4(a, b, y0, h)
    ye = [ref_value(ref, x) for x in xs]

    # Пункт 6
    print(f"\nПУНКТ 6: Рунге-Кутта 4-го порядку (h={h})")
    print(f"  {'x':>6}  {'y_RK4':>14}  {'y_ref':>14}  {'|err|':>12}")
    step_p = max(1, len(xs) // 10)
    for i in range(0, len(xs), step_p):
        print(f"  {xs[i]:6.3f}  {ys[i]:14.10f}  {ye[i]:14.10f}  {abs(ys[i]-ye[i]):12.2e}")

    # Пункт 7: локальна похибка РК4
    phi_rk = [abs(yc - yr) for yc, yr in zip(ys, ye)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, phi_rk, 'b-', linewidth=1.4,
            label=r'$\varphi_n = |y_n - y_{ref}(x_n)|$')
    ax.set_xlabel('x');  ax.set_ylabel('Похибка')
    ax.set_title(f"П.7: Локальна похибка РК4 (еталонний розв'язок), h={h}")
    ax.grid(True, alpha=0.4);  ax.legend()
    fig.tight_layout()
    fig.savefig(out('rk4_error_exact.png'), dpi=120)
    plt.close(fig)
    print(f"\nПУНКТ 7: Збережено: rk4_error_exact.png")

    print(f"\n  Залежність макс похибки від h (РК4, порядок збіжності ~4):")
    print(f"  {'h':>10}  {'max|err|':>14}  {'порядок':>10}")
    prev_err = None
    for hh in [0.1, 0.05, 0.01, 0.005, 0.001]:
        xs_t, ys_t = runge_kutta4(a, b, y0, hh)
        ye_t = [ref_value(ref, x) for x in xs_t]
        err_t = max(abs(yc - yr) for yc, yr in zip(ys_t, ye_t))
        if prev_err and err_t > 0:
            order = f"{math.log(prev_err/err_t)/math.log(2):.2f}"
        else:
            order = "  —"
        print(f"  {hh:10.4f}  {err_t:14.3e}  {order:>10}")
        prev_err = err_t

    # Пункт 8: оцінка за Рунге Φ = 1/15 |y^(h/2) - y^h|
    xs2, ys2 = runge_kutta4(a, b, y0, h / 2)
    phi_runge = []
    xs_runge  = []
    for i in range(len(xs)):
        j = 2 * i
        if j < len(xs2):
            phi_runge.append(abs(ys2[j] - ys[i]) / 15)
            xs_runge.append(xs[i])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs_runge, phi_runge, 'r-', linewidth=1.4,
            label=r'$\Phi_n = \frac{1}{15}|y_n^{h/2} - y_n^h|$')
    ax.set_xlabel('x');  ax.set_ylabel('Оцінка похибки')
    ax.set_title(f'П.8: Оцінка похибки РК4 за Рунге, h={h}')
    ax.grid(True, alpha=0.4);  ax.legend()
    fig.tight_layout()
    fig.savefig(out('rk4_error_runge.png'), dpi=120)
    plt.close(fig)

    max_phi_runge = max(phi_runge) if phi_runge else 0
    max_phi_exact = max(phi_rk)
    print(f"\nПУНКТ 8: Збережено: rk4_error_runge.png")
    print(f"  max|Φ_Runge|   = {max_phi_runge:.3e}")
    print(f"  max|ref err|   = {max_phi_exact:.3e}")
    if max_phi_runge > 0:
        h_opt = h * (EPS / max_phi_runge) ** 0.25
        print(f"  Рекомендований крок для ε={EPS}: h_opt ≈ {h_opt:.5f}")

    # Пункт 9: автовибір кроку РК4
    xs_a, ys_a, hs_a = runge_kutta4_auto(a, b, y0, EPS)
    ye_a   = [ref_value(ref, x) for x in xs_a]
    xs_mid = [(xs_a[i] + xs_a[i+1]) / 2 for i in range(len(hs_a))]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(xs_mid, hs_a, where='mid', color='purple', linewidth=1.5, label='h(x)')
    ax.set_xlabel('x');  ax.set_ylabel('Крок h')
    ax.set_title(f'П.9: Автоматичний вибір кроку (РК4, ε={EPS})')
    ax.grid(True, alpha=0.4);  ax.legend()
    fig.tight_layout()
    fig.savefig(out('rk4_auto_step.png'), dpi=120)
    plt.close(fig)

    max_err_a = max(abs(yc - yr) for yc, yr in zip(ys_a, ye_a))
    print(f"\nПУНКТ 9: Збережено: rk4_auto_step.png")
    print(f"  Кроків: {len(hs_a)},  мін h: {min(hs_a):.2e},  макс h: {max(hs_a):.2e}")
    print(f"  Макс похибка з автокроком: {max_err_a:.3e}")

    return xs, ys


# ═══════════════════════════════════════════════════════════════
# ЗВЕДЕНИЙ ГРАФІК
# ═══════════════════════════════════════════════════════════════
def summary_plot(ref, xs_rk, ys_rk):
    xs_ad, ys_ad, _ = adams2(a, b, y0, H_ADAMS)

    x_ref = sorted([k for k in ref.keys() if a - 1e-9 <= k <= b + 1e-9])
    y_ref = [ref[k] for k in x_ref]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_ref,  y_ref,  'k-',  linewidth=2,   label="Еталон (РК4, h=1e-6)")
    ax.plot(xs_ad,  ys_ad,  'b.-', markersize=8,  label=f'Адамс 2-го порядку (h={H_ADAMS})')
    ax.plot(xs_rk,  ys_rk,  'r--', linewidth=1.4, label=f'Рунге-Кутта 4-го порядку (h={H_RK})')
    ax.set_xlabel('x');  ax.set_ylabel('y')
    ax.set_title("Порівняння: Адамс vs РК4 vs Еталон\ny' = sin(x) − y²,  y(0) = 0")
    ax.grid(True, alpha=0.4);  ax.legend()
    fig.tight_layout()
    fig.savefig(out('summary.png'), dpi=120)
    plt.close(fig)
    print(f"\n  Зведений графік збережено: summary.png")


# ═══════════════════════════════════════════════════════════════
# ГОЛОВНА ПРОГРАМА
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  ОДР dy/dx = sin(x) - y²,  [0,1],  y(0)=0")
    print("  (рівняння Ріккаті; еталон — РК4 з h=1e-6)")
    print("=" * 60)

    ref = build_reference()

    part1_plots(ref)
    xs_rk, ys_rk = part2_plots(ref)
    summary_plot(ref, xs_rk, ys_rk)

    print("\n" + "="*60)
    print("  Збережені файли:")
    print("  adams_error_exact.png — п.3: похибка Адамса (еталон)")
    print("  adams_error_est.png   — п.4: оцінка похибки Адамса")
    print("  adams_auto_step.png   — п.5: автокрок Адамса h(x)")
    print("  rk4_error_exact.png   — п.7: похибка РК4 (еталон)")
    print("  rk4_error_runge.png   — п.8: оцінка похибки за Рунге")
    print("  rk4_auto_step.png     — п.9: автокрок РК4 h(x)")
    print("  summary.png           — зведений графік")
    print("="*60)