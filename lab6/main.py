import numpy as np

# ============================================================
# ПУНКТ 1: Генерація матриці A та вектора B, запис у файли
# ============================================================

n = 100
x_true = np.full(n, 2.5)  # розв'язок xi = 2.5 для всіх i

# Генерація випадкової діагонально домінантної матриці A
np.random.seed(42)
A = np.random.uniform(-10, 10, (n, n))
for i in range(n):
    A[i, i] = np.sum(np.abs(A[i])) + 10

# Обчислення вектора вільних членів b = A * x
b = A @ x_true

# Запис матриці A у текстовий файл
with open('matrix_A.txt', 'w') as f:
    for row in A:
        f.write(' '.join(f'{val:.6f}' for val in row) + '\n')

# Запис вектора B у текстовий файл
with open('vector_B.txt', 'w') as f:
    for val in b:
        f.write(f'{val:.6f}\n')

print("Пункт 1: Матрицю A i вектор B збережено у matrix_A.txt та vector_B.txt")

# ============================================================
# ПУНКТ 2: Функції для роботи з LU-розкладом
# ============================================================

def read_matrix(filename, n):
    """Зчитування матриці A з текстового файлу."""
    A = []
    with open(filename, 'r') as f:
        for line in f:
            row = list(map(float, line.strip().split()))
            A.append(row)
    return np.array(A)

def read_vector(filename):
    """Зчитування вектора B з текстового файлу."""
    b = []
    with open(filename, 'r') as f:
        for line in f:
            b.append(float(line.strip()))
    return np.array(b)

def lu_decomposition(A):
    """LU-розклад матриці A з частковим вибором головного елемента."""
    n = len(A)
    L = np.zeros((n, n))
    U = A.copy().astype(float)
    P = np.eye(n)

    for k in range(n):
        max_idx = np.argmax(np.abs(U[k:, k])) + k
        if max_idx != k:
            U[[k, max_idx]] = U[[max_idx, k]]
            P[[k, max_idx]] = P[[max_idx, k]]
            if k > 0:
                L[[k, max_idx], :k] = L[[max_idx, k], :k]

        L[k, k] = 1.0
        for i in range(k + 1, n):
            if U[k, k] == 0:
                continue
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    return L, U, P

def save_lu(L, U, P, filename):
    """Запис LU-розкладу матриці у текстовий файл."""
    with open(filename, 'w') as f:
        f.write('=== matrix L ===\n')
        for row in L:
            f.write(' '.join(f'{val:.6f}' for val in row) + '\n')
        f.write('=== matrix U ===\n')
        for row in U:
            f.write(' '.join(f'{val:.6f}' for val in row) + '\n')
        f.write('=== matrix P  ===\n')
        for row in P:
            f.write(' '.join(f'{val:.0f}' for val in row) + '\n')

def solve_lu(L, U, P, b):
    """Розв'язок системи AX = B за допомогою LU-розкладу."""
    Pb = P @ b
    n = len(b)

    # Пряма підстановка: Ly = Pb
    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    # Зворотна підстановка: Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x

def mat_vec_product(A, x):
    """Добуток матриці на вектор."""
    return A @ x

def vector_norm(v):
    """Норма вектора (максимум абсолютних значень)."""
    return np.max(np.abs(v))

print("Пункт 2: Всi функцiї визначено")

# ============================================================
# ПУНКТ 3: Розв'язання системи AX = B за допомогою LU-розкладу
# ============================================================

A_loaded = read_matrix('matrix_A.txt', n)
b_loaded = read_vector('vector_B.txt')

L, U, P = lu_decomposition(A_loaded)
save_lu(L, U, P, 'matrix_LU.txt')

x_sol = solve_lu(L, U, P, b_loaded)

print("\nПункт 3: Систему розв'язано методом LU-розкладу")
print(f"  Першi 5 елементiв розв'язку: {x_sol[:5]}")
print(f"  (Очiкується: всi = 2.5)")

# ============================================================
# ПУНКТ 4: Оцінка точності розв'язку
# ============================================================

residual = mat_vec_product(A_loaded, x_sol) - b_loaded
eps = vector_norm(residual)

print(f"\nПункт 4: Точнiсть розв'язку")
print(f"  eps = max|A*x - b| = {eps:.6e}")

# ============================================================
# ПУНКТ 5: Уточнення розв'язку ітераційним методом
# ============================================================

eps0 = 1e-14
x_iter = x_sol.copy()
num_iter = 0

print(f"\nПункт 5: Iтерацiйне уточнення розв'язку (eps0 = {eps0:.0e})")

for _ in range(1000):
    r = b_loaded - mat_vec_product(A_loaded, x_iter)
    norm_r = vector_norm(r)

    if norm_r <= eps0:
        break

    delta = solve_lu(L, U, P, r)
    x_iter = x_iter + delta
    num_iter += 1

residual_iter = mat_vec_product(A_loaded, x_iter) - b_loaded
eps_iter = vector_norm(residual_iter)

print(f"  Кiлькiсть iтерацiй: {num_iter}")
print(f"  Точнiсть пiсля уточнення: eps = {eps_iter:.6e}")
print(f"  Першi 5 елементiв уточненого розв'язку: {x_iter[:5]}")
