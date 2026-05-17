import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────
#  Параметри задачі
# ─────────────────────────────────────────────────────────
a_glob, b_glob, h_step = -3.0, 3.0, 0.1
EPS   = 1e-10
MAX_ITER = 100_000

def F(x):
    """F(x) = x^3 - 3x + 1"""
    return x**3 - 3.0*x + 1.0

def dF(x):
    """F'(x) = 3x^2 - 3"""
    return 3.0*x**2 - 3.0

def d2F(x):
    """F''(x) = 6x"""
    return 6.0*x

# ═══════════════════════════════════════════════════════════
#  ПУНКТ 1 — ТАБУЛЯЦІЯ ФУНКЦІЇ
# ═══════════════════════════════════════════════════════════
print("=" * 65)
print("ПУНКТ 1: Табуляція F(x) на [{}, {}], h={}".format(
    a_glob, b_glob, h_step))
print("=" * 65)

tab_x, tab_y = [], []
x = a_glob
while x <= b_glob + 1e-12:
    tab_x.append(round(x, 10))
    tab_y.append(F(x))
    x += h_step

# --- запис у файл ---
TAB_FILE = "tabulation.txt"
with open(TAB_FILE, "w", encoding="utf-8") as f:
    f.write("Табуляція F(x) = x^3 - 3x + 1\n")
    f.write(f"Відрізок [{a_glob}, {b_glob}], крок h = {h_step}\n")
    f.write("-" * 32 + "\n")
    f.write(f"{'x':>10}  {'F(x)':>16}\n")
    f.write("-" * 32 + "\n")
    for xi, yi in zip(tab_x, tab_y):
        f.write(f"{xi:>10.4f}  {yi:>16.8f}\n")
print(f"Таблицю збережено у файл: {TAB_FILE}")

# --- виведення на екран ---
print(f"\n{'x':>10}  {'F(x)':>16}")
print("-" * 30)
for xi, yi in zip(tab_x, tab_y):
    print(f"{xi:>10.4f}  {yi:>16.8f}")

#знаходження відрізків зміни знаку
# Перша точка перетину — де F зростає
# Друга точка перетину — де F спадає
sign_changes = []
for i in range(len(tab_y) - 1):
    if tab_y[i] * tab_y[i+1] < 0:
        behavior = "зростаюча" if tab_y[i+1] > tab_y[i] else "спадна"
        # лінійна інтерполяція для початкового наближення
        x0_approx = tab_x[i] - tab_y[i] * (tab_x[i+1] - tab_x[i]) / \
                    (tab_y[i+1] - tab_y[i])
        sign_changes.append({
            "a": tab_x[i], "b": tab_x[i+1],
            "fa": tab_y[i], "fb": tab_y[i+1],
            "behavior": behavior, "x0": x0_approx
        })

print("\nТочки перетину з віссю x (зміна знаку):")
for sc in sign_changes:
    print(f"  [{sc['a']:.2f}, {sc['b']:.2f}]  "
          f"({sc['behavior']}),  x0 ≈ {sc['x0']:.6f}")
print(f"\nЗнайдено {len(sign_changes)} початкових наближень.")

# ═══════════════════════════════════════════════════════════
#  ПУНКТ 3 — критерій зупинки
# ═══════════════════════════════════════════════════════════
def converged(xn, xprev, fn, eps=EPS):
    """
    Критерій (пункт 3): одночасно виконуються дві умови:
      |F(x_{n+1})| < eps   ТА   |x_{n+1} - x_n| < eps
    """
    return abs(fn) < eps and abs(xn - xprev) < eps

# ═══════════════════════════════════════════════════════════
#  ПУНКТ 2 — ШІСТЬ МЕТОДІВ РОЗВ'ЯЗАННЯ F(x)=0
# ═══════════════════════════════════════════════════════════

def simple_iteration(x0, a0, b0):
    # M береться лише по відрізку локалізації кореня
    pts = [a0 + k * 0.001 * (b0 - a0) for k in range(1001)]
    M = max(abs(dF(xi)) for xi in pts)
    if M < 1e-15:
        return None, 0, []

    def g(x):
        return x - F(x) / M

    # Перевіряємо умову збіжності q = max|1 - F'(x)/M| < 1
    q = max(abs(1.0 - dF(xi) / M) for xi in pts)
    if q >= 1.0:
        # Якщо умова не виконана — збільшуємо M до безпечного значення
        M2 = max(abs(dF(xi)) for xi in pts) * 1.5
        def g(x):
            return x - F(x) / M2

    xprev = x0
    xn = g(xprev)
    iters = 1
    path = [xprev, xn]
    while not converged(xn, xprev, F(xn)) and iters < MAX_ITER:
        xprev = xn
        xn = g(xprev)
        iters += 1
        path.append(xn)
        # Захист від розбіжності: якщо вийшли за відрізок більш ніж на 2
        if abs(xn - x0) > abs(b0 - a0) * 20:
            break
    return xn, iters, path

# ──────────────────────────────────────────────────────────
#  Метод Ньютона (дотичних)
# ──────────────────────────────────────────────────────────
def newton(x0):
    xprev = x0
    if abs(dF(xprev)) < 1e-14:
        xprev += 1e-6
    xn = xprev - F(xprev) / dF(xprev)
    iters = 1
    path = [xprev, xn]
    while not converged(xn, xprev, F(xn)) and iters < MAX_ITER:
        xprev = xn
        df = dF(xprev)
        if abs(df) < 1e-14:
            break
        xn = xprev - F(xprev) / df
        iters += 1
        path.append(xn)
    return xn, iters, path

# ──────────────────────────────────────────────────────────
#  Метод Чебишева (кубічна збіжність)
#  x_{n+1} = x_n - F/F' - F^2*F'' / (2*(F')^3)
# ──────────────────────────────────────────────────────────
def chebyshev(x0):
    xprev = x0
    path = [xprev]
    iters = 0
    xn = xprev  # ініціалізація
    while iters < MAX_ITER:
        fx  = F(xprev)
        dfx = dF(xprev)
        d2fx = d2F(xprev)
        if abs(dfx) < 1e-14:
            break
        xn = xprev - fx / dfx - (fx**2 * d2fx) / (2.0 * dfx**3)
        iters += 1
        path.append(xn)
        if converged(xn, xprev, F(xn)):
            break
        xprev = xn
    return xn, iters, path

# ──────────────────────────────────────────────────────────
#  Метод хорд
#  Фіксується кінець з тим самим знаком, що й F''(x)
# ──────────────────────────────────────────────────────────
def chord(a0, b0):
    # Фіксуємо точку, де F(x)*F''(x) > 0
    if F(a0) * d2F(a0) > 0:
        fixed, moving = a0, b0
    else:
        fixed, moving = b0, a0

    ff    = F(fixed)
    xprev = moving
    fx    = F(xprev)
    denom = fx - ff
    if abs(denom) < 1e-15:
        return xprev, 0, [xprev]
    xn = xprev - fx * (xprev - fixed) / denom
    iters = 1
    path = [xprev, xn]
    while not converged(xn, xprev, F(xn)) and iters < MAX_ITER:
        xprev = xn
        fx    = F(xprev)
        ff    = F(fixed)
        denom = fx - ff
        if abs(denom) < 1e-15:
            break
        xn = xprev - fx * (xprev - fixed) / denom
        iters += 1
        path.append(xn)
    return xn, iters, path

# ──────────────────────────────────────────────────────────
#  Метод парабол (Мюллера)
#  Апроксимуємо F(x) параболою через 3 точки
# ──────────────────────────────────────────────────────────
def parabola(a0, b0):
    x0p = a0
    x1p = (a0 + b0) / 2.0
    x2p = b0
    xprev = x2p
    xn    = x2p
    path  = [x0p, x1p, x2p]
    iters = 0
    for _ in range(MAX_ITER):
        f0, f1, f2 = F(x0p), F(x1p), F(x2p)
        h1 = x1p - x0p
        h2 = x2p - x1p
        if abs(h1) < 1e-15 or abs(h2) < 1e-15:
            break
        d1   = (f1 - f0) / h1
        d2   = (f2 - f1) / h2
        a_c  = (d2 - d1) / (h2 + h1)
        b_c  = a_c * h2 + d2
        c_c  = f2
        disc = b_c**2 - 4.0 * a_c * c_c
        if disc < 0:
            disc = 0.0
        # Вибираємо знак, щоб знаменник був більшим
        if b_c >= 0:
            xn = x2p - 2.0 * c_c / (b_c + math.sqrt(disc))
        else:
            xn = x2p - 2.0 * c_c / (b_c - math.sqrt(disc))
        iters += 1
        path.append(xn)
        if converged(xn, xprev, F(xn)):
            break
        x0p, x1p, x2p = x1p, x2p, xn
        xprev = xn
    return xn, iters, path

# ──────────────────────────────────────────────────────────
#  Метод зворотної інтерполяції (формула Лагранжа)
#  Будуємо таблицю (y_i, x_i) і шукаємо x при y=0
# ──────────────────────────────────────────────────────────
def inverse_interpolation(a0, b0, n_pts=6):
    nodes_x = [a0 + i * (b0 - a0) / (n_pts - 1) for i in range(n_pts)]
    nodes_y = [F(xi) for xi in nodes_x]

    def lagrange_at_zero():
        """Лагранжева інтерполяція: аргументи = F(x_i), значення = x_i"""
        result = 0.0
        for i in range(n_pts):
            term = nodes_x[i]
            for j in range(n_pts):
                if j != i:
                    denom = nodes_y[i] - nodes_y[j]
                    if abs(denom) < 1e-15:
                        term = 0.0
                        break
                    term *= (0.0 - nodes_y[j]) / denom
            result += term
        return result

    x_approx = lagrange_at_zero()
    # Уточнення методом Ньютона
    xprev = x_approx
    iters = 1
    path  = [xprev]
    for _ in range(MAX_ITER):
        df = dF(xprev)
        if abs(df) < 1e-14:
            break
        xn = xprev - F(xprev) / df
        iters += 1
        path.append(xn)
        if converged(xn, xprev, F(xn)):
            break
        xprev = xn
    return xn, iters, path

# ═══════════════════════════════════════════════════════════
#  ПУНКТ 4 — ЗАПУСК ВСІХ МЕТОДІВ, ПІДРАХУНОК ІТЕРАЦІЙ
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ПУНКТИ 2–4: Знаходження коренів усіма методами")
print(f"Точність: eps = {EPS}")
print("=" * 65)

all_roots_info = []
for idx, sc in enumerate(sign_changes):
    a0, b0, x0 = sc["a"], sc["b"], sc["x0"]
    print(f"\n--- Корінь #{idx+1} на [{a0:.2f}, {b0:.2f}], "
          f"x0 ≈ {x0:.6f} ({sc['behavior']}) ---")

    rd = {"sc": sc, "methods": {}}

    # 1. Проста ітерація
    r, it, path = simple_iteration(x0, a0, b0)
    rd["methods"]["Проста ітерація"]  = (r, it, path)
    print(f"  Проста ітерація    : x = {r:.15f},  ітерацій = {it}")

    # 2. Ньютон
    r, it, path = newton(x0)
    rd["methods"]["Ньютон"]           = (r, it, path)
    print(f"  Метод Ньютона      : x = {r:.15f},  ітерацій = {it}")

    # 3. Чебишев
    r, it, path = chebyshev(x0)
    rd["methods"]["Чебишев"]          = (r, it, path)
    print(f"  Метод Чебишева     : x = {r:.15f},  ітерацій = {it}")

    # 4. Хорди
    r, it, path = chord(a0, b0)
    rd["methods"]["Хорди"]            = (r, it, path)
    print(f"  Метод хорд         : x = {r:.15f},  ітерацій = {it}")

    # 5. Параболи
    r, it, path = parabola(a0, b0)
    rd["methods"]["Параболи"]         = (r, it, path)
    print(f"  Метод парабол      : x = {r:.15f},  ітерацій = {it}")

    # 6. Зворотна інтерполяція
    r, it, path = inverse_interpolation(a0, b0)
    rd["methods"]["Зворот. інтерп."]  = (r, it, path)
    print(f"  Зворотна інтерп.   : x = {r:.15f},  ітерацій = {it}")

    all_roots_info.append(rd)

# ═══════════════════════════════════════════════════════════
#  ПУНКТ 5 — АЛГЕБРАЇЧНЕ РІВНЯННЯ 3-ГО ПОРЯДКУ + ГРАФІК
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ПУНКТ 5: Алгебраїчне рівняння 3-го порядку")
print("Умова: один дійсний корінь + два комплексно-спряжені")
print("=" * 65)

# p(x) = (x-2)(x^2 - 2x + 5)  =>  x=2,  x=1±2i
# Розкладаємо: x^3 - 4x^2 + 9x - 10
P3_COEFFS = [1, -4, 9, -10]   # [a3, a2, a1, a0]

def P3(x):
    return x**3 - 4*x**2 + 9*x - 10

print("\nРівняння: p(x) = x^3 - 4x^2 + 9x - 10 = 0")
print("Теоретичні корені: x=2 (дійсний),  x=1+2i,  x=1-2i (комплексні)")

# графік рівняння
xx_p = np.linspace(-1.0, 5.0, 600)
yy_p = [P3(xi) for xi in xx_p]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ліворуч: крива p(x)
ax1 = axes[0]
ax1.plot(xx_p, yy_p, 'b-', linewidth=2, label=r'$p(x)=x^3-4x^2+9x-10$')
ax1.axhline(0, color='k', linewidth=0.8)
ax1.axvline(0, color='k', linewidth=0.8)
ax1.scatter([2], [0], color='red', zorder=5, s=90,
            label='Дійсний корінь x=2')
# Позначимо реальні частини комплексних коренів
ax1.axvline(1.0, color='purple', linewidth=1.2, linestyle='--',
            label='Re(x)=1  (комплексні корені 1±2i)')
mask = np.array(yy_p) < 0

ax1.fill_between(
    xx_p,
    yy_p,
    0,
    where=mask,
    alpha=0.07,
    color='blue',
    label='p(x)<0'
)
ax1.set_xlim(-1, 5); ax1.set_ylim(-25, 65)
ax1.set_title("Графік p(x): 1 дійсний корінь", fontsize=11)
ax1.set_xlabel("x"); ax1.set_ylabel("p(x)")
ax1.legend(fontsize=8); ax1.grid(True)

#праворуч: комплексна площина
ax2 = axes[1]
# Зображуємо всі три корені у комплексній площині
real_parts  = [2.0,  1.0,  1.0]
imag_parts  = [0.0,  2.0, -2.0]
colors_c    = ['red', 'purple', 'purple']
labels_c    = ['x=2 (дійсний)', 'x=1+2i', 'x=1-2i']
for rp, ip, col, lb in zip(real_parts, imag_parts, colors_c, labels_c):
    ax2.scatter([rp], [ip], color=col, s=100, zorder=5, label=lb)
# вісь дійсна / уявна
ax2.axhline(0, color='k', linewidth=0.8)
ax2.axvline(0, color='k', linewidth=0.8)
# пунктирні лінії до коренів
for rp, ip in zip(real_parts, imag_parts):
    ax2.plot([0, rp], [0, ip], 'k--', linewidth=0.6, alpha=0.4)
ax2.set_xlim(-0.5, 3.5); ax2.set_ylim(-3.5, 3.5)
ax2.set_title("Корені у комплексній площині", fontsize=11)
ax2.set_xlabel("Re(x)"); ax2.set_ylabel("Im(x)")
ax2.legend(fontsize=8); ax2.grid(True)

plt.suptitle("Пункт 5: p(x)=x³-4x²+9x-10 — один дійсний, два комплексних корені",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("poly3_plot.png", dpi=130, bbox_inches='tight')
plt.close()
print("Графік збережено: poly3_plot.png")

# ═══════════════════════════════════════════════════════════
#  ПУНКТ 6 — ЗАПИС КОЕФІЦІЄНТІВ У ТЕКСТОВИЙ ФАЙЛ
# ═══════════════════════════════════════════════════════════
COEFF_FILE = "polynomial_coeffs.txt"
with open(COEFF_FILE, "w", encoding="utf-8") as f:
    f.write("Коефіцієнти алгебраїчного рівняння 3-го порядку\n")
    f.write("p(x) = a3*x^3 + a2*x^2 + a1*x + a0\n")
    f.write("-" * 34 + "\n")
    f.write(f"a3 = {P3_COEFFS[0]}\n")
    f.write(f"a2 = {P3_COEFFS[1]}\n")
    f.write(f"a1 = {P3_COEFFS[2]}\n")
    f.write(f"a0 = {P3_COEFFS[3]}\n")
print(f"\nКоефіцієнти збережено у файл: {COEFF_FILE}")

# ═══════════════════════════════════════════════════════════
#  ПУНКТ 7 — ЗЧИТУВАННЯ КОЕФІЦІЄНТІВ + СХЕМА ГОРНЕРА
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ПУНКТ 7: Зчитування коефіцієнтів з файлу і обчислення P(x)")
print("=" * 65)

def read_poly_coeffs(filename):

    raw = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and line.startswith("a"):
                key, val = line.split("=", 1)
                raw[key.strip()] = float(val.strip())
    degree = max(int(k[1:]) for k in raw)
    return [raw.get(f"a{i}", 0.0) for i in range(degree, -1, -1)]

def evaluate_poly(coeffs, x):

    result = 0.0
    for c in coeffs:
        result = result * x + c
    return result

loaded_coeffs = read_poly_coeffs(COEFF_FILE)
print(f"Зчитані коефіцієнти (від старшого): {loaded_coeffs}")
print("\nОбчислення P(x) за схемою Горнера для перевірки:")
for test_x in [2.0, 0.0, 1.0, -1.0]:
    val = evaluate_poly(loaded_coeffs, test_x)
    print(f"  P({test_x:>5.1f}) = {val:>14.8f}")

# ═══════════════════════════════════════════════════════════
#  ПУНКТ 8 — ДІЙСНІ КОРЕНІ: МЕТОД НЬЮТОНА ЗІ СХЕМОЮ ГОРНЕРА
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ПУНКТ 8: Дійсні корені методом Ньютона зі схемою Горнера")
print("=" * 65)

def horner_val_deriv(coeffs, x):

    n = len(coeffs) - 1
    b = coeffs[0]   # b_n
    c = 0.0
    for i in range(1, n + 1):
        c = c * x + b       # c накопичує похідну
        b = b * x + coeffs[i]   # b = P(x) на кроці i
    # b = P(x), c = P'(x)
    return b, c

def deflate(coeffs, root):

    n = len(coeffs) - 1
    new_c = [0.0] * n
    new_c[0] = coeffs[0]
    for i in range(1, n):
        new_c[i] = new_c[i-1] * root + coeffs[i]
    remainder = new_c[-1] * root + coeffs[-1]
    return new_c, remainder

def newton_horner_single(coeffs, x0, eps_loc=EPS):

    xk = x0
    iters = 0
    for _ in range(MAX_ITER):
        px, dpx = horner_val_deriv(coeffs, xk)
        if abs(dpx) < 1e-15:
            xk += 0.01
            continue
        xk_new = xk - px / dpx
        iters += 1
        if abs(xk_new - xk) < eps_loc and abs(px) < eps_loc:
            xk = xk_new
            break
        xk = xk_new
    return xk, iters

def find_real_roots_newton_horner(coeffs, starts=None, eps_loc=EPS):

    if starts is None:
        starts = [-5, -3, -1, 0, 0.5, 1, 2, 3, 4, 5]
    remaining = list(coeffs)
    roots, iter_counts = [], []

    while len(remaining) - 1 >= 1:
        found = False
        for x0 in starts:
            root, it = newton_horner_single(remaining, x0, eps_loc)
            px, _ = horner_val_deriv(remaining, root)
            if abs(px) < 1e-7:
                roots.append(root)
                iter_counts.append(it)
                remaining, _ = deflate(remaining, root)
                # нормалізуємо
                if abs(remaining[0]) > 1e-15:
                    remaining = [c / remaining[0] for c in remaining]
                found = True
                break
        if not found:
            break

    return roots, iter_counts

real_roots_8, iters_8 = find_real_roots_newton_horner(loaded_coeffs)
print("Знайдені дійсні корені p(x)=0 (Ньютон + схема Горнера):")
total_iters_8 = 0
for i, (r, it) in enumerate(zip(real_roots_8, iters_8)):
    pval = evaluate_poly(loaded_coeffs, r)
    print(f"  Корінь {i+1}: x = {r:.15f},  P(x) = {pval:.3e},  ітерацій = {it}")
    total_iters_8 += it
print(f"  Всього ітерацій (пункт 8): {total_iters_8}")

# ═══════════════════════════════════════════════════════════
#  ПУНКТ 9 — КОМПЛЕКСНІ КОРЕНІ: МЕТОД ЛІНА (БЕРСТОУ)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ПУНКТ 9: Комплексні корені методом Ліна (Берстоу)")
print("=" * 65)

def bairstow(coeffs, eps_loc=EPS):

    def bairstow_step(a, p_init, q_init):
        n = len(a) - 1
        p, q = p_init, q_init
        total_iters = 0

        for _ in range(MAX_ITER):
            b = [0.0] * (n + 1)
            c = [0.0] * (n + 1)

            # Прямий хід (ділення)
            b[0] = a[0]
            if n >= 1: b[1] = a[1] - p * b[0]
            for i in range(2, n + 1):
                b[i] = a[i] - p * b[i - 1] - q * b[i - 2]

            # Другий хід (для похідних)
            c[0] = b[0]
            if n >= 1: c[1] = b[1] - p * c[0]
            for i in range(2, n):  # Розраховуємо до c[n-1]
                c[i] = b[i] - p * c[i - 1] - q * c[i - 2]

            # Залишки
            R, S = b[n - 1], b[n]


            cc = c[n - 2] if n >= 2 else 0.0
            cc_minus_1 = c[n - 3] if n >= 3 else 0.0

            c_n_1 = -p * c[n - 2] - q * c[n - 3] if n >= 3 else -p * c[n - 2] if n == 2 else 0.0

            det = c[n - 2] ** 2 - c[n - 1] * c[n - 3] if n >= 3 else c[n - 2] ** 2

            v = c[n - 2]
            u = c[n - 1]  # це b[n-1] - p*c[n-2] - q*c[n-3]
            g = c[n - 3] if n >= 3 else 0.0
            h = c[n - 2]

            det = v * v - g * (b[n - 1] - v)  # Спрощений Якобіан

            # АБО найпростіший робочий варіант для невеликих n:
            det = c[n - 2] * c[n - 2] - c[n - 3] * c[n - 1] if n >= 3 else c[n - 2] * c[n - 2]

            if abs(det) < 1e-16:
                p += 0.1;
                q += 0.1
                continue

            dp = (b[n - 1] * c[n - 2] - b[n] * c[n - 3]) / det if n >= 3 else b[n - 1] / c[n - 2]
            dq = (b[n] * c[n - 2] - b[n - 1] * c[n - 1]) / det if n >= 3 else b[n] / c[n - 2]

            p += dp
            q += dq
            total_iters += 1

            if abs(dp) < eps_loc and abs(dq) < eps_loc:
                break

        # Розв'язуємо квадратне рівняння x^2 + p*x + q = 0
        disc = p * p - 4.0 * q
        if disc >= 0:
            r1 = (-p + math.sqrt(disc)) / 2.0
            r2 = (-p - math.sqrt(disc)) / 2.0
            roots_pair = [complex(r1, 0.0), complex(r2, 0.0)]
        else:
            re = -p / 2.0
            im = math.sqrt(-disc) / 2.0
            roots_pair = [complex(re, im), complex(re, -im)]

        # Коефіцієнти частки після дефляції на x^2+px+q
        m = n - 2
        if m < 0:
            quotient = []
        else:
            quotient = [0.0] * (m + 1)
            quotient[0] = a[0]
            if m >= 1:
                quotient[1] = a[1] - p * quotient[0]
            for i in range(2, m + 1):
                quotient[i] = a[i] - p * quotient[i-1] - q * quotient[i-2]

        return roots_pair, quotient, p, q, total_iters

    # Нормалізуємо
    a_norm = [c / coeffs[0] for c in coeffs]
    all_roots = []
    total_iters_all = 0
    remaining = list(a_norm)

    while len(remaining) - 1 >= 2:
        n_rem = len(remaining) - 1
        # Початкове наближення: береться з коефіцієнтів
        p0 = -remaining[1] if len(remaining) > 1 else 0.5
        q0 =  remaining[2] if len(remaining) > 2 else 1.0

        pairs, quotient, pf, qf, iters_step = bairstow_step(
            remaining, p0, q0)
        all_roots.extend(pairs)
        total_iters_all += iters_step

        if len(quotient) == 0:
            break
        if abs(quotient[0]) > 1e-15:
            quotient = [c / quotient[0] for c in quotient]
        remaining = quotient

    # Якщо залишився лінійний множник (x - r)
    if len(remaining) - 1 == 1:
        r = -remaining[1] / remaining[0]
        all_roots.append(complex(r, 0.0))

    return all_roots, total_iters_all

all_complex_roots, lin_iters = bairstow(loaded_coeffs)

print(f"Метод Ліна (Берстоу), всього ітерацій = {lin_iters}:")
for i, r in enumerate(all_complex_roots):
    if abs(r.imag) < 1e-8:
        print(f"  Корінь {i+1}: x = {r.real:.12f}  (дійсний)")
    else:
        sign = "+" if r.imag >= 0 else "-"
        print(f"  Корінь {i+1}: x = {r.real:.8f} {sign} {abs(r.imag):.8f}i  (комплексний)")

# Перевірка через numpy
np_roots = np.roots(loaded_coeffs)
print("\nПеревірка через numpy.roots:")
for i, r in enumerate(sorted(np_roots, key=lambda z: (-abs(z.imag), z.real))):
    if abs(r.imag) < 1e-8:
        print(f"  x{i+1} = {r.real:.12f}")
    else:
        print(f"  x{i+1} = {r.real:.8f} ± {abs(r.imag):.8f}i")

# ═══════════════════════════════════════════════════════════
#  ГРАФІК F(x) з позначенням трьох коренів
# ═══════════════════════════════════════════════════════════
xx_m = np.linspace(a_glob, b_glob, 600)
yy_m = [F(xi) for xi in xx_m]

plt.figure(figsize=(11, 5))
plt.plot(xx_m, yy_m, 'b-', linewidth=2,
         label=r'$F(x) = x^3 - 3x + 1$')
plt.axhline(0, color='k', linewidth=0.8)
plt.axvline(0, color='k', linewidth=0.8)

colors3 = ['red', 'green', 'purple']
for idx_r, ri in enumerate(all_roots_info):
    root_val = ri["methods"]["Ньютон"][0]
    plt.scatter([root_val], [0], color=colors3[idx_r],
                zorder=5, s=90,
                label=f'Корінь #{idx_r+1} ≈ {root_val:.5f}')

plt.title(r"$F(x) = x^3 - 3x + 1$ та корені рівняння", fontsize=13)
plt.xlabel("x"); plt.ylabel("F(x)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("Fx_roots_plot.png", dpi=130)
plt.close()
print("\nГрафік F(x) збережено: Fx_roots_plot.png")

# ═══════════════════════════════════════════════════════════
#  ПІДСУМКОВИЙ ЗВІТ
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("ПІДСУМОК")
print("=" * 65)
print(f"Функція: F(x) = x^3 - 3x + 1")
print(f"Відрізок: [{a_glob}, {b_glob}],  h={h_step},  eps={EPS}\n")

print("Корені F(x)=0 (всі методи):")
for idx_r, ri in enumerate(all_roots_info):
    print(f"\n  Корінь #{idx_r+1} на [{ri['sc']['a']:.2f},{ri['sc']['b']:.2f}]:")
    for mname, (root_v, iters_v, _) in ri["methods"].items():
        print(f"    {mname:<22}: x={root_v:.12f},  ітерацій={iters_v}")

print(f"\nКорені p(x)=0 (пункт 8, Ньютон+Горнер):")
for r8 in real_roots_8:
    print(f"  x = {r8:.12f}  (дійсний)")

print(f"\nКорені p(x)=0 (пункт 9, метод Ліна/Берстоу), ітерацій={lin_iters}:")
for r9 in all_complex_roots:
    if abs(r9.imag) < 1e-8:
        print(f"  x = {r9.real:.8f}  (дійсний)")
    else:
        sign = "+" if r9.imag >= 0 else "-"
        print(f"  x = {r9.real:.6f} {sign} {abs(r9.imag):.6f}i  (комплексний)")

print("\nСтворені файли:")
print(f"  {TAB_FILE:<28} — таблиця значень F(x)")
print(f"  {COEFF_FILE:<28} — коефіцієнти p(x)")
print(f"  poly3_plot.png               — графік p(x) + комплексна площина")
print(f"  Fx_roots_plot.png            — графік F(x) з коренями")
print("\nГотово!")