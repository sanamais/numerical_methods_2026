import requests
import numpy as np
import matplotlib.pyplot as plt
# URL запиту
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results)

with open("elevation_data.txt", "w") as f:
    f.write("№ | Latitude | Longitude | Elevation (m)\n")
    for i, p in enumerate(results):
        line = f"{i:2d} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}\n"
        print(line.strip())
        f.write(line)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # радіус Землі
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = np.array([p["elevation"] for p in results])

distances = [0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

distances = np.array(distances)
plt.figure()
plt.plot(distances, elevations, 'o-')
plt.xlabel("кумулятивна відстань (м)")
plt.ylabel("висота (м)")
plt.title("маршрут Заросляк – Говерла")
plt.grid()
plt.show()

def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    x = np.zeros(n)
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n):
       denom = b[i] - a[i] * c_[i-1]
    c_[i] = c[i] / denom if i < n-1 else 0
    d_[i] = (d[i] - a[i]*d_[i-1]) / denom
    x[-1] = d_[-1]
    for i in reversed(range(n-1)):
       x[i] = d_[i] - c_[i]*x[i+1]
    return x
def cubic_spline_coefficients(x, y):
    n = len(x)
    h = np.diff(x)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    for i in range(1, n-1):
        a[i] = h[i-1]
        b[i] = 2*(h[i-1]+h[i])
        c[i] = h[i]
        d[i] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    b[0] = b[-1] = 1
    d[0] = d[-1] = 0
    C = thomas_algorithm(a, b, c, d)
    return C
C = cubic_spline_coefficients(distances, elevations)
print("коефіцієнти кубічних сплайнів c_i:")
for i, val in enumerate(C):
    print(f"c[{i}] = {val}")

# п.9: обчислити коефіцієнти a_i, b_i, c_i, d_i кубічних сплайнів
def get_abcd_coefficients(x, y, C):
    n = len(x)
    h = np.diff(x)
    print("\nкоефіцієнти a_i, b_i, c_i, d_i кубічних сплайнів:")
    print(f"{'i':>3} | {'a_i':>12} | {'b_i':>12} | {'c_i':>12} | {'d_i':>12}")
    print("-" * 60)
    ai_list, bi_list, ci_list, di_list = [], [], [], []
    for i in range(n - 1):
        ai = y[i]
        bi = (y[i+1] - y[i]) / h[i] - h[i] * (2*C[i] + C[i+1]) / 6
        ci = C[i] / 2
        di = (C[i+1] - C[i]) / (6 * h[i])
        ai_list.append(ai); bi_list.append(bi)
        ci_list.append(ci); di_list.append(di)
        print(f"{i:>3} | {ai:>12.4f} | {bi:>12.4f} | {ci:>12.4f} | {di:>12.6f}")
    return np.array(ai_list), np.array(bi_list), np.array(ci_list), np.array(di_list)
ai_arr, bi_arr, ci_arr, di_arr = get_abcd_coefficients(distances, elevations, C)
def plot_spline_subset(k):
    idx = np.linspace(0, n-1, k, dtype=int)
    x = distances[idx]
    y = elevations[idx]
    C = cubic_spline_coefficients(x, y)
    plt.figure()
    plt.plot(distances, elevations, 'k.', label="оригінал")
    plt.plot(x, y, 'ro', label=f"{k} вузлів")
    plt.xlabel("відстань (м)")
    plt.ylabel("висота (м)")
    plt.legend()
    plt.title(f"кількість вузлів: {k}")
    plt.grid()
    plt.show()

for k in [10, 15, 20]:
    plot_spline_subset(k)

# Інтерполяція для всієї дистанції
from scipy.interpolate import CubicSpline

cs = CubicSpline(distances, elevations)
x_new = np.linspace(distances.min(), distances.max(), 300)
y_spline = cs(x_new)
y_real = np.interp(x_new, distances, elevations)
error = y_real - y_spline

plt.figure()
plt.plot(x_new, y_real, label="реальна функція")
plt.plot(x_new, y_spline, label="кубічний сплайн")
plt.legend(); plt.grid(); plt.show()

plt.figure()
plt.plot(x_new, error)
plt.title("похибка")
plt.grid(); plt.show()

# додатково
print("загальна довжина маршруту (м):", distances[-1])
total_ascent = sum(max(elevations[i]-elevations[i-1], 0) for i in range(1, n))
print("сумарний набір висоти (м):", total_ascent)

total_descent = sum(max(elevations[i-1]-elevations[i], 0) for i in range(1, n))
print("сумарний спуск (м):", total_descent)

cs = CubicSpline(distances, elevations)
xx = np.linspace(distances.min(), distances.max(), 300)
yy_full = cs(xx)

# градієнт (%)
grad_full = np.gradient(yy_full, xx) * 100

print("максимальний підйом (%):", np.max(grad_full))
print("максимальний спуск (%):", np.min(grad_full))
print("середній градієнт (%):", np.mean(np.abs(grad_full)))

# ділянки крутіші за 15%
steep_sections = xx[np.abs(grad_full) > 15]
print("кількість точок зі схилом > 15%:", len(steep_sections))

mass = 80  # кг
g = 9.81   # м/с^2
energy = mass * g * total_ascent
print("механічна робота (Дж):", energy)
print("механічна робота (кДж):", energy/1000)
print("енергія (ккал):", energy / 4184)

# п.12: графік заданої функції y = f(x), її наближеного значення y_наближ
# та похибки ε = |y - y_наближ| на відрізку [x_0, x_n]
def eval_spline_manual(x_query, x_nodes, a, b, c, d):
    """Обчислює значення кубічного сплайну у довільній точці x_query."""
    y_out = np.zeros_like(x_query, dtype=float)
    for j, xq in enumerate(x_query):
        # знаходимо відповідний інтервал
        idx = np.searchsorted(x_nodes, xq, side='right') - 1
        idx = np.clip(idx, 0, len(a) - 1)
        dx = xq - x_nodes[idx]
        y_out[j] = a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
    return y_out

x_plot = np.linspace(distances[0], distances[-1], 300)
y_approx = eval_spline_manual(x_plot, distances, ai_arr, bi_arr, ci_arr, di_arr)
y_true = np.interp(x_plot, distances, elevations)
epsilon = np.abs(y_true - y_approx)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(x_plot, y_true, 'b-', label="f(x) – реальні дані (лінійна інтерп.)")
axes[0].plot(x_plot, y_approx, 'r--', label=r"$y_{\mathrm{набл}}$ – кубічний сплайн")
axes[0].plot(distances, elevations, 'ko', markersize=4, label="вузли")
axes[0].set_xlabel("відстань (м)")
axes[0].set_ylabel("висота (м)")
axes[0].set_title(r"$y = f(x)$ та її наближення $y_{\mathrm{набл}}(x)$")
axes[0].legend()
axes[0].grid()

axes[1].plot(x_plot, epsilon, 'g-', label=r"$\varepsilon = |y - y_{\mathrm{набл}}|$")
axes[1].set_xlabel("відстань (м)")
axes[1].set_ylabel("похибка (м)")
axes[1].set_title(r"Похибка $\varepsilon$ на відрізку $[x_0,\, x_n]$")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

print(f"\nмаксимальна похибка ε: {np.max(epsilon):.4f} м")
print(f"середня похибка ε:     {np.mean(epsilon):.4f} м")