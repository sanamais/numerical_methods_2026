import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ============================================================
# ЗАДАНА ФУНКЦІЯ та інтервал
# ============================================================
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

# ============================================================
# ПУНКТ 1: Графік функції f(x)
# ============================================================
x_plot = np.linspace(a, b, 1000)
y_plot = f(x_plot)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(x_plot, y_plot, color='steelblue', linewidth=2,
         label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$')
ax1.set_title('Графік функції навантаження на сервер')
ax1.set_xlabel('Час, x (год)')
ax1.set_ylabel('Навантаження, f(x)')
ax1.grid(True)
ax1.legend()
plt.tight_layout()
plt.savefig('plot_function.png', dpi=150)
plt.close()

# ============================================================
# ПУНКТ 2: Точне значення інтегралу I0
# ============================================================
I0, _ = quad(f, a, b)
print(f"I0 = {I0:.15f}")

# ============================================================
# ПУНКТ 3: Складова формула Сімпсона I(N)
# ============================================================
def simpson(f, a, b, N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    fx = f(x)
    S = fx[0] + fx[-1]
    S += 4 * np.sum(fx[1:-1:2])
    S += 2 * np.sum(fx[2:-2:2])
    return S * h / 3

for N_test in [10, 100, 1000]:
    val = simpson(f, a, b, N_test)
    print(f"I({N_test:4d}) = {val:.15f}   похибка = {abs(val - I0):.2e}")

# ============================================================
# ПУНКТ 4: Залежність e(N) = |I(N) - I0|, N = 10..1000
# ============================================================
N_values = np.arange(10, 1002, 2)
eps_values = np.array([abs(simpson(f, a, b, int(N)) - I0) for N in N_values])

target_eps = 1e-12
N_opt = None
eps_opt = None
for N, eps in zip(N_values, eps_values):
    if eps <= target_eps:
        N_opt = int(N)
        eps_opt = eps
        break

if N_opt is None:
    idx = np.argmin(eps_values)
    N_opt = int(N_values[idx])
    eps_opt = eps_values[idx]
    print(f"Точність 1e-12 не досягнута в діапазоні N=10..1000")

print(f"N_opt  = {N_opt}")
print(f"epsopt = {eps_opt:.2e}")

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.semilogy(N_values, eps_values, color='darkorange', linewidth=1.5)
ax2.axvline(N_opt, color='red', linestyle='--', label=f'N_opt = {N_opt}')
ax2.axhline(target_eps, color='green', linestyle=':', label='e = 1e-12')
ax2.set_xlabel('N (кількість вузлів)')
ax2.set_ylabel('e(N) = |I(N) - I0|')
ax2.set_title('Залежність похибки Сімпсона від N')
ax2.legend()
ax2.grid(True, which='both', alpha=0.4)
plt.tight_layout()
plt.savefig('plot_eps_N.png', dpi=150)
plt.close()

# ============================================================
# ПУНКТ 5: Похибка eps0 при N0 (N0 ~ N_opt/10, кратне 8)
# ============================================================
N0 = int(round((N_opt / 10) / 8) * 8)
if N0 < 8:
    N0 = 8

I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)
print(f"N0    = {N0}")
print(f"I(N0) = {I_N0:.15f}")
print(f"eps0  = {eps0:.6e}")

# ============================================================
# ПУНКТ 6: Метод Рунге-Ромберга
# ============================================================
N0_half = N0 // 2
I_N0_half = simpson(f, a, b, N0_half)
I_R = I_N0 + (I_N0 - I_N0_half) / 15
epsR = abs(I_R - I0)

print(f"I(N0/2) = {I_N0_half:.15f}")
print(f"I_R     = {I_R:.15f}")
print(f"epsR    = {epsR:.6e}")

# ============================================================
# ПУНКТ 7: Метод Ейткена
# ============================================================
N0_q2 = N0 // 2
N0_q4 = N0 // 4

I_N0_q2 = simpson(f, a, b, N0_q2)
I_N0_q4 = simpson(f, a, b, N0_q4)

numerator   = (I_N0_q2)**2 - I_N0 * I_N0_q4
denominator = 2 * I_N0_q2 - (I_N0 + I_N0_q4)

if abs(denominator) < 1e-30:
    print("Знаменник = 0, метод Ейткена не застосовний при цих N")
    I_E = I_N0
    epsE = eps0
    p = float('nan')
else:
    I_E = numerator / denominator
    epsE = abs(I_E - I0)

    ratio_num = abs(I_N0_q4 - I_N0_q2)
    ratio_den = abs(I_N0_q2 - I_N0)
    if ratio_den > 1e-30 and ratio_num > 1e-30:
        p = (1 / np.log(2)) * np.log(ratio_num / ratio_den)
    else:
        p = float('nan')

print(f"I_E  = {I_E:.15f}")
print(f"epsE = {epsE:.6e}")
print(f"p    = {p:.4f}")

# ============================================================
# ПУНКТ 8: Аналіз похибок різних методів
# ============================================================
methods = ['Сімпсон (N0)', 'Рунге-Ромберг', 'Ейткен']
errors  = [eps0, epsR, epsE]
colors  = ['steelblue', 'darkorange', 'seagreen']

for m, e in zip(methods, errors):
    print(f"  {m:<20} похибка = {e:.6e}")

fig3, ax3 = plt.subplots(figsize=(8, 5))
bars = ax3.bar(methods, errors, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
ax3.set_yscale('log')
ax3.set_ylabel('Похибка (log-scale)')
ax3.set_title('Порівняння похибок чисельного інтегрування')
for bar, err in zip(bars, errors):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5,
             f'{err:.2e}', ha='center', va='bottom', fontsize=9)
ax3.grid(True, axis='y', alpha=0.4)
plt.subplots_adjust(wspace=0.35)
plt.savefig('plot_methods_comparison.png', dpi=150)
plt.close()

# ============================================================
# ПУНКТ 9: Адаптивний алгоритм
# ============================================================
def adaptive_simpson(f, a, b, tol=1e-6, max_depth=50):
    call_count = [0]

    def simpson_basic(a, b):
        mid = (a + b) / 2
        call_count[0] += 3
        return (b - a) / 6 * (f(a) + 4 * f(mid) + f(b))

    def recursive(a, b, tol, whole, depth):
        mid = (a + b) / 2
        left  = simpson_basic(a, mid)
        right = simpson_basic(mid, b)
        delta = left + right - whole
        if depth >= max_depth or abs(delta) <= 15 * tol:
            return left + right + delta / 15
        return (recursive(a, mid, tol / 2, left, depth + 1) +
                recursive(mid, b, tol / 2, right, depth + 1))

    whole = simpson_basic(a, b)
    result = recursive(a, b, tol, whole, 0)
    return result, call_count[0]

tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
adapt_errors = []
adapt_calls  = []

print(f"\n{'Допуск d':>12}  {'I_adap':>20}  {'Похибка':>12}  {'Обч. f':>8}")
print('-' * 60)
for tol in tolerances:
    I_a, calls = adaptive_simpson(f, a, b, tol=tol)
    err = abs(I_a - I0)
    adapt_errors.append(err)
    adapt_calls.append(calls)
    print(f"{tol:12.0e}  {I_a:20.12f}  {err:12.4e}  {calls:8d}")

fig4, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].loglog(adapt_calls, adapt_errors, 'o-', color='purple', linewidth=1.5)
axes[0].set_xlabel('Кількість обчислень f')
axes[0].set_ylabel('Похибка')
axes[0].set_title('Адаптивний алгоритм: похибка vs обчислення')
axes[0].grid(True, which='both', alpha=0.4)

axes[1].loglog(tolerances, adapt_errors, 's-', color='crimson', linewidth=1.5)
axes[1].set_xlabel('Допуск d')
axes[1].set_ylabel('Фактична похибка')
axes[1].set_title('Адаптивний алгоритм: похибка vs допуск')
axes[1].grid(True, which='both', alpha=0.4)

plt.tight_layout()
plt.savefig('plot_adaptive.png', dpi=150)
plt.close()