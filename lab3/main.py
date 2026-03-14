"""
Лабораторна робота: Знаходження алгебраїчних многочленів
найкращого квадратичного наближення методом найменших квадратів (МНК)

Варіант: середньомісячні температури (24 місяці)
"""

import math
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================
# 1. ВХІДНІ ДАНІ (табуляція / зчитування з CSV)
# ============================================================

def tabulate_from_csv(filename: str):
    """Зчитує вузли {x_i, f_i} з CSV-файлу (колонки Month, Temp)."""
    x_nodes, y_nodes = [], []
    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_nodes.append(float(row['Month']))
            y_nodes.append(float(row['Temp']))
    return x_nodes, y_nodes


def default_data():
    """Повертає вбудовані дані (якщо CSV не знайдено)."""
    months = list(range(1, 25))
    temps  = [-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0,
              -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3]
    return months, temps


# ============================================================
# 2. ФУНКЦІЇ МНК
# ============================================================

def form_matrix_B(x: list, m: int) -> list:
    """
    Формує матрицю B розміром (m+1) × (m+1).
    B[i][j] = Σ_k  x[k]^(i+j),  i,j = 0..m
    """
    n = len(x)
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum(x[k] ** (i + j) for k in range(n))
    return A


def form_vector_C(x: list, y: list, m: int) -> list:
    """
    Формує вектор C розміром (m+1).
    C[i] = Σ_k  y[k] * x[k]^i,  i = 0..m
    """
    n = len(x)
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * x[k] ** i for k in range(n))
    return b


def gauss_solve(A_in: list, b_in: list) -> list:
    """
    Розв'язує СЛАР методом Гаусса з вибором головного елемента по стовпцю.
    Повертає вектор коефіцієнтів a.
    """
    n = len(b_in)
    # Копії щоб не змінювати оригінал
    A = [row[:] for row in A_in]
    b = b_in[:]

    # Прямий хід
    for k in range(n - 1):
        # Вибір головного елемента
        max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            if A[k][k] == 0:
                continue
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    # Зворотний хід
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - sum(A[i][j] * x_sol[j] for j in range(i + 1, n))) / A[i][i]
    return x_sol


def polynomial(x_vals: list, coef: list) -> list:
    """
    Обчислює значення многочлена φ(x) = Σ coef[i] * x^i
    для кожного x у списку x_vals.
    """
    result = []
    for x in x_vals:
        val = sum(coef[i] * (x ** i) for i in range(len(coef)))
        result.append(val)
    return result


def compute_error(y_true: list, y_approx: list) -> list:
    """
    Функція похибки ε(x_i) = |f(x_i) - φ(x_i)|
    """
    return [abs(y_true[i] - y_approx[i]) for i in range(len(y_true))]


def variance(y_true: list, y_approx: list) -> float:
    """
    Дисперсія: D = (1/n) * Σ (f_i - φ_i)^2
    """
    n = len(y_true)
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n


# ============================================================
# 3. ОСНОВНИЙ АЛГОРИТМ
# ============================================================

def run_lab(x: list, y: list, max_degree: int = 10):
    """
    Для m = 1..max_degree:
      - будує матрицю B та вектор C
      - розв'язує СЛАР → коефіцієнти многочлена
      - обчислює дисперсію
    Повертає словник з результатами.
    """
    all_coefs    = []
    all_vars     = []
    all_approx   = []

    print("=" * 60)
    print("  ТАБЛИЦЯ ДИСПЕРСІЙ ПО СТУПЕНЯХ МНОГОЧЛЕНА")
    print("=" * 60)
    print(f"  {'m':>3}  |  {'Дисперсія':>14}")
    print("-" * 30)

    for m in range(1, max_degree + 1):
        B    = form_matrix_B(x, m)
        C    = form_vector_C(x, y, m)
        coef = gauss_solve(B, C)
        ya   = polynomial(x, coef)
        var  = variance(y, ya)

        all_coefs.append(coef)
        all_vars.append(var)
        all_approx.append(ya)

        print(f"  m={m:>2}  |  {var:>14.6f}")

    print("=" * 60)

    # Оптимальний ступінь — мінімальна дисперсія
    opt_idx = all_vars.index(min(all_vars))
    opt_m   = opt_idx + 1

    print(f"\n  ► Оптимальний ступінь: m = {opt_m}")
    print(f"  ► Мінімальна дисперсія: {all_vars[opt_idx]:.6f}")

    # Коефіцієнти оптимального многочлена
    print(f"\n  Коефіцієнти оптимального многочлена (a_i):")
    for i, c in enumerate(all_coefs[opt_idx]):
        print(f"    a[{i}] = {c:.8f}")

    return {
        "all_coefs":  all_coefs,
        "all_vars":   all_vars,
        "all_approx": all_approx,
        "opt_m":      opt_m,
        "opt_idx":    opt_idx,
        "opt_coef":   all_coefs[opt_idx],
        "opt_approx": all_approx[opt_idx],
    }


# ============================================================
# 4. ПРОГНОЗ НА НАСТУПНІ 3 МІСЯЦІ
# ============================================================

def forecast(coef: list, x_future: list):
    y_future = polynomial(x_future, coef)
    print("\n  ПРОГНОЗ НА НАСТУПНІ 3 МІСЯЦІ:")
    for x, y in zip(x_future, y_future):
        print(f"    Місяць {x:>2}: {y:.2f} °C")
    return y_future


# ============================================================
# 5. ТАБУЛЯЦІЯ ПОХИБКИ
# ============================================================

def tabulate_error(x: list, y: list, y_approx: list, m: int, h: float):
    """
    Табулює похибку з кроком h_tab = (x_n - x_0) / (20n).
    Тут для наочності використовуємо вже наявні вузли.
    """
    err = compute_error(y, y_approx)
    print(f"\n  ТАБЛИЦЯ ПОХИБКИ (m = {m}, крок h = {h:.4f}):")
    print(f"  {'i':>3} | {'x_i':>6} | {'f(x_i)':>8} | {'φ(x_i)':>10} | {'ε(x_i)':>10}")
    print("  " + "-" * 46)
    for i, (xi, fi, phi, ei) in enumerate(zip(x, y, y_approx, err)):
        print(f"  {i:>3} | {xi:>6.1f} | {fi:>8.2f} | {phi:>10.4f} | {ei:>10.6f}")


# ============================================================
# 6. ПОБУДОВА ГРАФІКІВ
# ============================================================

def plot_results(x: list, y: list, results: dict, x_future: list, y_future: list):
    opt_m      = results["opt_m"]
    opt_approx = results["opt_approx"]
    all_vars   = results["all_vars"]
    all_approx = results["all_approx"]
    all_coefs  = results["all_coefs"]

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("МНК — Апроксимація температурних даних", fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    degrees = list(range(1, len(all_vars) + 1))
    colors  = plt.cm.tab10.colors

    # --- Графік 1: Фактичні дані + апроксимація для всіх m ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, y, 'ko', markersize=5, label='Фактичні дані', zorder=5)
    for m_idx, ya in enumerate(all_approx):
        m = m_idx + 1
        lw   = 2.5 if m == opt_m else 0.8
        ls   = '-'  if m == opt_m else '--'
        alpha = 1.0 if m == opt_m else 0.45
        label = f'm={m} (опт.)' if m == opt_m else f'm={m}'
        ax1.plot(x, ya, color=colors[m_idx % 10], lw=lw, ls=ls,
                 alpha=alpha, label=label)
    # Прогноз
    ax1.plot(x_future, y_future, 'r^', markersize=8, label='Прогноз 25-27', zorder=6)
    ax1.set_title(f'Апроксимуючі многочлени m=1..10 (оптимальний m={opt_m})')
    ax1.set_xlabel('Місяць')
    ax1.set_ylabel('Температура, °C')
    ax1.legend(fontsize=7, ncol=6, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- Графік 2: Дисперсія vs ступінь ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(degrees, all_vars, 'b.-', lw=1.5, markersize=7)
    ax2.plot(opt_m, all_vars[results["opt_idx"]], 'ro', markersize=10,
             label=f'min D при m={opt_m}')
    ax2.set_title('Залежність дисперсії від ступеня m')
    ax2.set_xlabel('Ступінь m')
    ax2.set_ylabel('Дисперсія D')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(degrees)

    # --- Графік 3: Похибка оптимального многочлена ---
    err_opt = compute_error(y, opt_approx)
    ax3 = fig.add_subplot(gs[1, 1])
    colors_bar = ['#2ecc71' if e < 1.0 else '#e74c3c' for e in err_opt]
    ax3.bar(x, err_opt, color=colors_bar, edgecolor='none', alpha=0.8)
    ax3.set_title(f'Похибка ε(x_i) = |f - φ|  (m={opt_m})')
    ax3.set_xlabel('Місяць')
    ax3.set_ylabel('|Похибка|')
    ax3.grid(True, alpha=0.3, axis='y')

    # --- Графік 4: Фактичні дані + оптимальний + прогноз ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(x, y, 'ko-', markersize=5, lw=1, label='Фактичні дані')
    ax4.plot(x, opt_approx, 'b-', lw=2, label=f'Оптимальний m={opt_m}')
    ax4.plot(x_future, y_future, 'r^--', markersize=8, lw=1.5, label='Прогноз')
    for xf, yf in zip(x_future, y_future):
        ax4.annotate(f'{yf:.1f}°C', (xf, yf), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=9, color='red')
    ax4.set_title('Оптимальна апроксимація та прогноз')
    ax4.set_xlabel('Місяць')
    ax4.set_ylabel('Температура, °C')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # --- Графік 5: Похибка зі знаком (f - φ) ---
    signed_err = [y[i] - opt_approx[i] for i in range(len(y))]
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(x, signed_err,
            color=['#27ae60' if e >= 0 else '#e74c3c' for e in signed_err],
            edgecolor='none', alpha=0.8)
    ax5.axhline(0, color='black', lw=0.8)
    ax5.set_title(f'Похибка зі знаком f(x) − φ(x)  (m={opt_m})')
    ax5.set_xlabel('Місяць')
    ax5.set_ylabel('f − φ')
    ax5.grid(True, alpha=0.3, axis='y')

    plt.savefig('lab_mnk_results.png', dpi=150, bbox_inches='tight')
    print("\n  Графіки збережено у файл: lab_mnk_results.png")
    plt.show()


# ============================================================
# 7. ГОЛОВНА ПРОГРАМА
# ============================================================

def main():
    # --- Крок 1: Зчитування даних ---
    csv_file = 'temperatures.csv'
    if os.path.exists(csv_file):
        print(f"  Зчитуємо дані з файлу: {csv_file}")
        x, y = tabulate_from_csv(csv_file)
    else:
        print("  CSV не знайдено — використовуємо вбудовані дані.")
        x, y = default_data()

    n  = len(x)
    x0 = x[0]
    xn = x[-1]
    h  = (xn - x0) / n        # крок табуляції з умови завдання (п.1)
    h_tab = (xn - x0) / (20 * n)  # крок для побудови графіка похибки (п.4)

    print(f"\n  Кількість вузлів n = {n}")
    print(f"  Відрізок [{x0}, {xn}],  крок h = {h:.4f}")

    # --- Крок 2-3: МНК для m=1..10 ---
    results = run_lab(x, y, max_degree=10)

    # --- Крок 4: Табуляція похибки (оптимальний m) ---
    tabulate_error(x, y, results["opt_approx"], results["opt_m"], h_tab)

    # --- Крок 5: Прогноз на 3 місяці ---
    x_future = [xn + 1, xn + 2, xn + 3]
    y_future = forecast(results["opt_coef"], x_future)

    # --- Крок 6: Графіки ---
    plot_results(x, y, results, x_future, y_future)


if __name__ == '__main__':
    main()