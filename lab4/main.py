import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Функція та її аналітична похідна
# ============================================================

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_analytical(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# ============================================================
# Пункт 1: Аналітичне розв'язання при t0 = x0 = 1
# ============================================================

x0 = 1.0
y_exact = dM_analytical(x0)
print("=" * 60)
print("1. АНАЛІТИЧНЕ РОЗ'ЯЗАННЯ")
print("=" * 60)
print(f"M(t)  = 50*e^(-0.1t) + 5*sin(t)")
print(f"M'(t) = -5*e^(-0.1t) + 5*cos(t)")
print(f"M'({x0}) = {y_exact:.6f}")

# ============================================================
# Пункт 2: Залежність похибки від кроку h (h = 10^-20 .. 10^3)
# ============================================================

print("\n" + "=" * 60)
print("2. ЗАЛЕЖНІСТЬ ПОХИБКИ ВІД КРОКУ h")
print("=" * 60)

def central_diff(f, t, h):
    return (f(t + h) - f(t - h)) / (2 * h)

h_values = [10**k for k in range(-20, 4)]
errors = []
best_h = None
best_R = float('inf')
best_D = None

print(f"{'h':>12}  {'D(h)':>14}  {'R = |D-M|':>14}")
for h in h_values:
    D = central_diff(M, x0, h)
    R = abs(D - y_exact)
    errors.append((h, D, R))
    if R < best_R:
        best_R = R
        best_h = h
        best_D = D
    print(f"{h:>12.2e}  {D:>14.8f}  {R:>14.8e}")

print(f"\nОптимальний крок h0 = {best_h:.2e}")
print(f"D(h0) = {best_D:.8f},  R0 = {best_R:.2e}")

# ============================================================
# Пункт 3: Прийнятий крок h = 10^-3
# ============================================================

print("\n" + "=" * 60)
print("3. ПРИЙНЯТИЙ КРОК h = 10^-3")
print("=" * 60)

h = 1e-3
print(f"h = {h}")

# ============================================================
# Пункт 4: Значення похідної при кроках h та 2h
# ============================================================

print("\n" + "=" * 60)
print("4. ЗНАЧЕННЯ ПОХІДНОЇ ПРИ КРОКАХ h ТА 2h")
print("=" * 60)

Dh   = central_diff(M, x0, h)
Dh2  = central_diff(M, x0, 2 * h)
print(f"y'(h)  = (f(x0+h)  - f(x0-h))  / (2h)  = {Dh:.8f}")
print(f"y'(2h) = (f(x0+2h) - f(x0-2h)) / (4h)  = {Dh2:.8f}")

# ============================================================
# Пункт 5: Похибка при кроці h
# ============================================================

print("\n" + "=" * 60)
print("5. ПОХИБКА ПРИ КРОЦІ h")
print("=" * 60)

R1 = abs(Dh - y_exact)
print(f"R1 = |y'(h) - y'(x0)| = {R1:.8e}")

# ============================================================
# Пункт 6: Метод Рунге–Ромберга
# ============================================================

print("\n" + "=" * 60)
print("6. МЕТОД РУНГЕ–РОМБЕРГА")
print("=" * 60)

# y'_R = y'(h) + (y'(h) - y'(2h)) / (2^2 - 1)
D_RR = Dh + (Dh - Dh2) / (2**2 - 1)
R2 = abs(D_RR - y_exact)
print(f"y'_R = y'(h) + (y'(h) - y'(2h)) / 3")
print(f"y'_R = {Dh:.8f} + ({Dh:.8f} - {Dh2:.8f}) / 3")
print(f"y'_R = {D_RR:.8f}")
print(f"R2 = |y'_R - y'(x0)| = {R2:.8e}")
print(f"Зміна похибки: R1/R2 = {R1/R2:.2f}  (похибка зменшилась у ~{R1/R2:.1f} разів)")

# ============================================================
# Пункт 7: Метод Ейткена
# ============================================================

print("\n" + "=" * 60)
print("7. МЕТОД ЕЙТКЕНА")
print("=" * 60)

Dh4 = central_diff(M, x0, 4 * h)
print(f"y'(h)  = {Dh:.8f}")
print(f"y'(2h) = {Dh2:.8f}")
print(f"y'(4h) = {Dh4:.8f}")

# Формула Ейткена з методички:
# y'_E = (y'(2h)^2 - y'(4h)*y'(h)) / (2*y'(2h) - (y'(4h) + y'(h)))
numerator_E   = Dh2**2 - Dh4 * Dh
denominator_E = 2 * Dh2 - (Dh4 + Dh)

if abs(denominator_E) > 1e-30:
    D_Aitken = numerator_E / denominator_E
else:
    D_Aitken = Dh
    print("Знаменник ~0, використовуємо D_h")

R3 = abs(D_Aitken - y_exact)
print(f"\ny'_E = (y'(2h)^2 - y'(4h)*y'(h)) / (2*y'(2h) - (y'(4h) + y'(h)))")
print(f"Чисельник  = {numerator_E:.10f}")
print(f"Знаменник  = {denominator_E:.10f}")
print(f"y'_E = {D_Aitken:.8f}")
print(f"R3 = |y'_E - y'(x0)| = {R3:.8e}")

# Порядок точності з методички:
# p = (1/ln2) * ln(|y'(4h) - y'(2h)| / |y'(2h) - y'(h)|)
if abs(Dh4 - Dh2) > 1e-30 and abs(Dh2 - Dh) > 1e-30:
    p = (1 / np.log(2)) * np.log(abs(Dh4 - Dh2) / abs(Dh2 - Dh))
    print(f"\nПорядок точності p = (1/ln2)*ln(|y'(4h)-y'(2h)| / |y'(2h)-y'(h)|) = {p:.4f}")

# ============================================================
# Пункт 8: Підсумкова таблиця та висновок
# ============================================================

print("\n" + "=" * 60)
print("8. ПІДСУМКОВА ТАБЛИЦЯ")
print("=" * 60)

print(f"\n{'Метод':<30} {'Значення':>14}  {'Похибка':>14}")
print("-" * 62)
print(f"{'Аналітичне M(x0)  ':<30} {y_exact:>14.8f}  {'-':>14}")
print(f"{'Центральна різниця D(h)':<30} {Dh:>14.8f}  {R1:>14.8e}")
print(f"{'Рунге–Ромберг D_RR':<30} {D_RR:>14.8f}  {R2:>14.8e}")
print(f"{'Ейткен D_E':<30} {D_Aitken:>14.8f}  {R3:>14.8e}")

print("\n" + "=" * 60)
print("ВИСНОВОК")
print("=" * 60)
print(f"Оптимальний крок для мінімальної похибки: h0 = {best_h:.2e}")
print(f"Метод Рунге–Ромберга зменшив похибку у ~{R1/R2:.1f} разів.")
if R3 < R1:
    print(f"Метод Ейткена зменшив похибку у ~{R1/R3:.1f} разів.")
print("Найточніший результат: метод Ейткена (якщо знаменник ≠ 0).")

# ============================================================
# ГРАФІКИ
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("M(t) = 50e^(-0.1t) + 5sin(t)", fontsize=13, fontweight='bold')

t_arr = np.linspace(0, 20, 500)

# --- Графік 1: M(t) — як у методичці ---
ax = axes[0]
ax.plot(t_arr, M(t_arr), color="steelblue", linewidth=2)
ax.set_title("Soil Moisture Model M(t)")
ax.set_xlabel("t")
ax.set_ylabel("M(t)")
ax.grid(True, alpha=0.3)

# --- Графік 2: Залежність похибки від кроку h (таблиця пункт 2) ---
ax = axes[1]
h_plot = [e[0] for e in errors if e[2] > 0 and 1e-15 <= e[0] <= 1]
R_plot = [e[2] for e in errors if e[2] > 0 and 1e-15 <= e[0] <= 1]
ax.loglog(h_plot, R_plot, 'o-', color="mediumseagreen", linewidth=2, markersize=6)
ax.axvline(best_h, color="red", linestyle="--", linewidth=1.5, label=f"h_opt = {best_h:.0e}")
ax.set_title("Залежність похибки R від кроку h")
ax.set_xlabel("h")
ax.set_ylabel("R = |D(h) - M'(x0)|")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("lab_plots.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nГрафіки збережено: lab_plots.png")