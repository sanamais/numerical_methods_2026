"""
Лабораторна робота: Метод Хука-Джівса для розв'язання систем нелінійних рівнянь
=================================================================================
Система нелінійних рівнянь (m=2):
    f1(x1, x2) = x1^2 + x2^2 - 4 = 0
    f2(x1, x2) = x1*x2 - 1 = 0

Цільова функція (метод найменших квадратів):
    Φ(X) = f1(x1,x2)^2 + f2(x1,x2)^2

Мінімум Φ(X) = 0 відповідає розв'язку системи.
"""

import math
import os
import matplotlib.pyplot as plt
import numpy as np

# Папка для збереження результатів — поруч зі скриптом
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def out(filename):
    """Повертає повний шлях до файлу результатів."""
    return os.path.join(OUTPUT_DIR, filename)

# ─────────────────────────────────────────────────
# 1. Система нелінійних рівнянь та цільова функція
# ─────────────────────────────────────────────────

def f1(x1, x2):
    return x1**2 + x2**2 - 4

def f2(x1, x2):
    return x1 * x2 - 1

def Phi(X):
    """Цільова функція (сума квадратів нев'язок)."""
    x1, x2 = X
    return f1(x1, x2)**2 + f2(x1, x2)**2


# ─────────────────────────────────────────────────
# 2. Метод Хука-Джівса
# ─────────────────────────────────────────────────

def hooke_jeeves(func, X0, delta, q=0.5, p=2.0, eps1=1e-4, eps2=1e-6, max_iter=10000):
    """
    Метод Хука-Джівса (Hooke-Jeeves pattern search).

    Параметри
    ----------
    func     : цільова функція F(X) -> float
    X0       : початкове наближення (список/вектор)
    delta    : початковий крок (список ΔX = [Δx1, Δx2, ...])
    q        : коефіцієнт зменшення кроку (0 < q < 1)
    p        : коефіцієнт збільшення кроку при успішному ході за зразком
    eps1     : критерій зупинки за кроком
    eps2     : критерій зупинки за значенням функції
    max_iter : максимальна кількість ітерацій

    Повертає
    --------
    X_best   : знайдений мінімум
    trajectory : список точок траєкторії
    steps    : кількість кроків
    """
    n = len(X0)
    X_base = list(X0)
    step = list(delta)
    trajectory = [list(X_base)]
    total_steps = 0

    for iteration in range(max_iter):
        # ── Дослідницький пошук навколо базової точки ──
        X_new = list(X_base)
        for i in range(n):
            # Крок у позитивному напрямку
            X_try = list(X_new)
            X_try[i] += step[i]
            if func(X_try) < func(X_new):
                X_new = X_try
            else:
                # Крок у негативному напрямку
                X_try[i] = X_new[i] - step[i]
                if func(X_try) < func(X_new):
                    X_new = X_try

        total_steps += 1
        trajectory.append(list(X_new))

        # ── Перевірка успіху дослідницького пошуку ──
        if func(X_new) < func(X_base):
            # Хід за зразком (pattern move)
            X_pattern = [X_new[i] + p * (X_new[i] - X_base[i]) for i in range(n)]
            X_base = list(X_new)

            # Повторний дослідницький пошук від точки зразка
            X_explore = list(X_pattern)
            for i in range(n):
                X_try = list(X_explore)
                X_try[i] += step[i]
                if func(X_try) < func(X_explore):
                    X_explore = X_try
                else:
                    X_try[i] = X_explore[i] - step[i]
                    if func(X_try) < func(X_explore):
                        X_explore = X_try

            total_steps += 1
            trajectory.append(list(X_explore))

            if func(X_explore) < func(X_base):
                X_base = list(X_explore)
        else:
            # Зменшення кроку
            step = [s * q for s in step]

        # ── Критерії зупинки ──
        step_norm = math.sqrt(sum(s**2 for s in step))
        if step_norm < eps1 and func(X_base) < eps2:
            print(f"  Зупинка: крок = {step_norm:.2e} < eps1={eps1}, Φ = {func(X_base):.2e} < eps2={eps2}")
            break
        if step_norm < eps1 * 0.01:
            print(f"  Зупинка: крок дуже малий ({step_norm:.2e})")
            break

    return X_base, trajectory, total_steps


# ─────────────────────────────────────────────────
# 3. Побудова графіків системи
# ─────────────────────────────────────────────────

def plot_system():
    x1 = np.linspace(-3, 3, 400)
    x2 = np.linspace(-3, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)

    F1 = X1**2 + X2**2 - 4   # кола радіуса 2
    F2 = X1 * X2 - 1          # гіпербола

    plt.figure(figsize=(7, 6))
    c1 = plt.contour(X1, X2, F1, levels=[0], colors='blue')
    c2 = plt.contour(X1, X2, F2, levels=[0], colors='red')
    plt.clabel(c1, fmt='f1=0')
    plt.clabel(c2, fmt='f2=0')

    # Точні розв'язки: x1^2+x2^2=4, x1*x2=1
    # => x1^4 - 4x1^2 + 1 = 0 => x1^2 = (4 ± sqrt(12))/2
    for sign in [1, -1]:
        x1s_sq = (4 + sign * math.sqrt(12)) / 2
        if x1s_sq > 0:
            for s2 in [1, -1]:
                x1s = s2 * math.sqrt(x1s_sq)
                x2s = 1 / x1s
                plt.plot(x1s, x2s, 'k*', markersize=12, label=f'({x1s:.3f}, {x2s:.3f})')

    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title('Система нелінійних рівнянь\n(синя: x₁²+x₂²=4,  червона: x₁x₂=1)')
    plt.legend(title='Розв\'язки')
    plt.tight_layout()
    plt.savefig(out('system_plot.png'), dpi=120)
    plt.close()
    print(f"  Графік системи збережено: {out('system_plot.png')}")


# ─────────────────────────────────────────────────
# 4. Запуск та виведення результатів
# ─────────────────────────────────────────────────

def run_test():
    """Пункт 3: тест методу на простій функції Розенброка."""
    print("\n" + "="*60)
    print("ТЕСТ методу Хука-Джівса на функції Розенброка")
    print("  f(x1,x2) = (1-x1)^2 + 100*(x2-x1^2)^2")
    print("  Мінімум: (1, 1),  f_min = 0")
    print("="*60)

    def rosenbrock(X):
        x1, x2 = X
        return (1 - x1)**2 + 100 * (x2 - x1**2)**2

    X0 = [-1.0, 1.0]
    delta = [0.5, 0.5]
    X_min, traj, steps = hooke_jeeves(rosenbrock, X0, delta,
                                      q=0.5, p=2.0, eps1=1e-6, eps2=1e-10)
    print(f"  X0          = {X0}")
    print(f"  Результат   = [{X_min[0]:.8f}, {X_min[1]:.8f}]")
    print(f"  f(X_min)    = {rosenbrock(X_min):.2e}")
    print(f"  Кроки       = {steps}")


def run_main():
    """Пункти 1, 2, 4, 5: основне завдання."""

    # ── Пункт 1: Графіки системи ──────────────────
    print("\n" + "="*60)
    print("ПУНКТ 1: Побудова графіків системи рівнянь")
    print("="*60)
    plot_system()

    # ── Пункт 2 & 4: Метод Хука-Джівса ───────────
    print("\n" + "="*60)
    print("ПУНКТИ 2 & 4: Метод Хука-Джівса для системи НР")
    print("  f1(x1,x2) = x1² + x2² - 4 = 0")
    print("  f2(x1,x2) = x1·x2 - 1     = 0")
    print("  Φ(X) = f1² + f2²  (цільова функція)")
    print("="*60)

    # Параметри
    X0    = [1.5, 0.5]          # початкове наближення (базисна точка)
    delta = [0.5, 0.5]          # початкова величина кроку ΔX
    q     = 0.5                 # коефіцієнт зменшення кроку
    p     = 2.0                 # коефіцієнт збільшення при ході за зразком
    eps1  = 1e-6                # критерій за кроком
    eps2  = 1e-10               # критерій за значенням функції

    print(f"\n  X⁽⁰⁾  = {X0}")
    print(f"  ΔX    = {delta}")
    print(f"  q={q}, p={p}, ε₁={eps1}, ε₂={eps2}")

    X_sol, trajectory, steps = hooke_jeeves(Phi, X0, delta,
                                             q=q, p=p, eps1=eps1, eps2=eps2)

    print(f"\n  ── Результат ──────────────────────────")
    print(f"  X*     = [{X_sol[0]:.8f}, {X_sol[1]:.8f}]")
    print(f"  Φ(X*)  = {Phi(X_sol):.4e}")
    print(f"  f1(X*) = {f1(*X_sol):.4e}")
    print(f"  f2(X*) = {f2(*X_sol):.4e}")
    print(f"  Кроки  = {steps}")

    # ── Пункт 5: Збереження траєкторії у файл ─────
    print("\n" + "="*60)
    print("ПУНКТ 5: Збереження координат траєкторії у файл")
    print("="*60)

    traj_file = out('trajectory.txt')
    with open(traj_file, 'w', encoding='utf-8') as f:
        f.write("Крок\tx1\t\t\tx2\t\t\tΦ(X)\n")
        f.write("-" * 65 + "\n")
        for i, point in enumerate(trajectory):
            phi_val = Phi(point)
            f.write(f"{i}\t{point[0]: .10f}\t{point[1]: .10f}\t{phi_val:.6e}\n")
        f.write("-" * 65 + "\n")
        f.write(f"Всього кроків: {steps}\n")
        f.write(f"Розв'язок: x1={X_sol[0]:.8f}, x2={X_sol[1]:.8f}\n")
        f.write(f"Φ(X*) = {Phi(X_sol):.4e}\n")

    print(f"  Файл збережено: trajectory.txt  ({len(trajectory)} точок)")

    # ── Побудова траєкторії на контурному графіку ──
    x1 = np.linspace(-0.5, 2.5, 400)
    x2 = np.linspace(-0.5, 2.5, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X1**2 + X2**2 - 4)**2 + (X1 * X2 - 1)**2

    traj_arr = np.array(trajectory)

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X1, X2, np.log1p(Z), levels=40, cmap='viridis')
    plt.colorbar(cp, label='log(1 + Φ)')
    plt.contour(X1, X2, Z, levels=[0], colors='white', linewidths=2)
    plt.plot(traj_arr[:, 0], traj_arr[:, 1], 'w.-', linewidth=0.8,
             markersize=3, alpha=0.7, label='Траєкторія')
    plt.plot(X0[0], X0[1], 'go', markersize=10, label=f'Старт {X0}')
    plt.plot(X_sol[0], X_sol[1], 'r*', markersize=14,
             label=f'Мінімум ({X_sol[0]:.4f}, {X_sol[1]:.4f})')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(f'Траєкторія методу Хука-Джівса\n(кроків: {steps})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out('trajectory_plot.png'), dpi=120)
    plt.close()
    print(f"  Графік траєкторії збережено: {out('trajectory_plot.png')}")

    return X_sol, steps


# ─────────────────────────────────────────────────
# ГОЛОВНА ПРОГРАМА
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  МЕТОД ХУКА-ДЖІВСА: СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ")
    print("=" * 60)

    # Пункт 3: тест на функції Розенброка
    run_test()

    # Пункти 1, 2, 4, 5: основна задача
    X_sol, steps = run_main()

    print("\n" + "="*60)
    print("  ПІДСУМОК")
    print("="*60)
    print(f"  Розв'язок системи: x1 ≈ {X_sol[0]:.6f}, x2 ≈ {X_sol[1]:.6f}")
    print(f"  Перевірка: f1 = {f1(*X_sol):.2e}, f2 = {f2(*X_sol):.2e}")
    print(f"  Кількість кроків: {steps}")
    print("\n  Збережені файли:")
    print("    system_plot.png    — графіки системи рівнянь")
    print("    trajectory_plot.png — траєкторія спуску")
    print("    trajectory.txt     — координати всіх точок траєкторії")
    print("="*60)