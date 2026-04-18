import numpy as np

# ============================================================
# ПУНКТ 1: Генерація матриці A з діагональним переважанням,
#           обчислення вектора B, запис у файли
# ============================================================

n = 100
x_true = np.full(n, 2.5)

np.random.seed(42)
A = np.random.uniform(-10, 10, (n, n))
for i in range(n):
    A[i, i] = np.sum(np.abs(A[i])) + 10  # діагональне переважання

b = A @ x_true

with open('matrix_A.txt', 'w') as f:
    for row in A:
        f.write(' '.join(f'{val:.6f}' for val in row) + '\n')

with open('vector_B.txt', 'w') as f:
    for val in b:
        f.write(f'{val:.6f}\n')

print("Пункт 1: Матрицю A і вектор B збережено у matrix_A.txt та vector_B.txt")

# ============================================================
# ПУНКТ 2: Функції
# ============================================================

def read_matrix(filename):
    A = []
    with open(filename, 'r') as f:
        for line in f:
            row = list(map(float, line.strip().split()))
            A.append(row)
    return np.array(A)

def read_vector(filename):
    b = []
    with open(filename, 'r') as f:
        for line in f:
            b.append(float(line.strip()))
    return np.array(b)

def mat_vec_product(A, x):
    """Добуток матриці на вектор."""
    return A @ x

def vector_norm(v):
    """Норма вектора (max абсолютних значень)."""
    return np.max(np.abs(v))

def matrix_norm(A):
    """Норма матриці (максимальна сума рядка)."""
    return np.max(np.sum(np.abs(A), axis=1))

def solve_simple_iteration(A, b, x0, eps0=1e-14, max_iter=10000):
    """
    Метод простої ітерації.
    Перетворення: x = B*x + c, де B = I - D^{-1}*A, c = D^{-1}*b
    D - діагональна матриця з елементів A.
    """
    n = len(b)
    d = np.diag(A)
    B = np.eye(n) - A / d[:, np.newaxis]
    c = b / d

    x = x0.copy()
    for k in range(1, max_iter + 1):
        x_new = mat_vec_product(B, x) + c
        if vector_norm(x_new - x) <= eps0:
            return x_new, k
        x = x_new
    return x, max_iter

def solve_jacobi(A, b, x0, eps0=1e-14, max_iter=10000):
    """Метод Якобі."""
    n = len(b)
    d = np.diag(A)
    x = x0.copy()
    for k in range(1, max_iter + 1):
        x_new = np.zeros(n)
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - s) / A[i, i]
        if vector_norm(x_new - x) <= eps0:
            return x_new, k
        x = x_new
    return x, max_iter

def solve_seidel(A, b, x0, eps0=1e-14, max_iter=10000):
    """Метод Зейделя."""
    n = len(b)
    x = x0.copy()
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            x[i] = (b[i] - s) / A[i, i]
        if vector_norm(x - x_old) <= eps0:
            return x, k
    return x, max_iter

print("Пункт 2: Всі функції визначено")

# ============================================================
# ПУНКТ 3: Початкове наближення x0 = 1.0
# ============================================================

A_loaded = read_matrix('matrix_A.txt')
b_loaded = read_vector('vector_B.txt')

x0 = np.ones(n)  # x_i^(0) = 1.0

print(f"\nПункт 3: Початкове наближення x0 = 1.0 для всіх i")
print(f"  Норма матриці A: {matrix_norm(A_loaded):.6f}")

# ============================================================
# ПУНКТ 4: Розв'язок трьома методами, кількість ітерацій
# ============================================================

eps0 = 1e-14
print(f"\nПункт 4: Розв'язок системи (eps0 = {eps0:.0e})\n")

# Метод простої ітерації
x_si, iter_si = solve_simple_iteration(A_loaded, b_loaded, x0, eps0)
eps_si = vector_norm(mat_vec_product(A_loaded, x_si) - b_loaded)
print(f"Метод простої ітерації:")
print(f"  Кількість ітерацій: {iter_si}")
print(f"  Точність eps = max|A*x - b| = {eps_si:.6e}")
print(f"  Перші 5 елементів: {x_si[:5]}\n")

# Метод Якобі
x_jac, iter_jac = solve_jacobi(A_loaded, b_loaded, x0, eps0)
eps_jac = vector_norm(mat_vec_product(A_loaded, x_jac) - b_loaded)
print(f"Метод Якобі:")
print(f"  Кількість ітерацій: {iter_jac}")
print(f"  Точність eps = max|A*x - b| = {eps_jac:.6e}")
print(f"  Перші 5 елементів: {x_jac[:5]}\n")

# Метод Зейделя
x_sei, iter_sei = solve_seidel(A_loaded, b_loaded, x0, eps0)
eps_sei = vector_norm(mat_vec_product(A_loaded, x_sei) - b_loaded)
print(f"Метод Зейделя:")
print(f"  Кількість ітерацій: {iter_sei}")
print(f"  Точність eps = max|A*x - b| = {eps_sei:.6e}")
print(f"  Перші 5 елементів: {x_sei[:5]}\n")

# Порівняння
print(f"{'Метод':<25} {'Ітерацій':>10} {'Точність':>15}")
print('-' * 52)
print(f"{'Проста ітерація':<25} {iter_si:>10} {eps_si:>15.6e}")
print(f"{'Якобі':<25} {iter_jac:>10} {eps_jac:>15.6e}")
print(f"{'Зейдель':<25} {iter_sei:>10} {eps_sei:>15.6e}")