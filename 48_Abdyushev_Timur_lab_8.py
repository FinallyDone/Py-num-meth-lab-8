import numpy as np

Var = 1
ALPHA = 2.0 + 0.5 * Var
T = 1.0
h = 0.1
N = 1 + int(T / h)


def func_1(t, y1, y2):
    return np.sin(ALPHA * y1 ** 2) + t + y2


def func_2(t, y1, y2):
    return t + y1 - ALPHA * y2 ** 2 + 1


# метод Хойна (Хьюна)
def solve_Hoyna(tau, t_n, y1, y2):
    for i in range(N - 1):
        p1 = y1[i] + tau * func_1(t_n[i], y1[i], y2[i])
        p2 = y2[i] + tau * func_2(t_n[i], y1[i], y2[i])

        y1[i + 1] = y1[i] + (tau / 2) * (func_1(t_n[i], y1[i], y2[i]) + func_1(t_n[i + 1], p1, p2))
        y2[i + 1] = y2[i] + (tau / 2) * (func_2(t_n[i], y1[i], y2[i]) + func_2(t_n[i + 1], y1[i + 1], p2))


# метод Рунге-Кутты 3-го порядка
def solve_Runge_Kutta(tau, t_n, y1, y2):
    for i in range(N - 1):
        k1_1 = func_1(t_n[i], y1[i], y2[i])
        k1_2 = func_2(t_n[i], y1[i], y2[i])

        k2_1 = func_1(t_n[i] + tau / 2.0, y1[i] + tau / 2.0 * k1_1, y2[i] + tau / 2.0 * k1_2)
        k2_2 = func_2(t_n[i] + tau / 2.0, y1[i] + tau / 2.0 * k1_1, y2[i] + tau / 2.0 * k1_2)

        k3_1 = func_1(t_n[i] + tau, y1[i] - tau * k1_1 + 2.0 * tau * k2_1, y2[i] - tau * k1_2 + 2.0 * tau * k2_2)
        k3_2 = func_2(t_n[i] + tau, y1[i] - tau * k1_1 + 2.0 * tau * k2_1, y2[i] - tau * k1_2 + 2.0 * tau * k2_2)

        y1[i + 1] = y1[i] + tau * (k1_1 + 4.0 * k2_1 + k3_1) / 6.0
        y2[i + 1] = y2[i] + tau * (k1_2 + 4.0 * k2_2 + k3_2) / 6.0


if __name__ == '__main__':
    ts = np.array([h * i for i in range(N)])
    y1 = np.zeros(N)
    y2 = np.zeros(N)
    u1 = np.zeros(N)
    u2 = np.zeros(N)

    # начальные условия
    y1[0] = u1[0] = 1.0
    y2[0] = u2[0] = 0.5

    # метод Хойна
    solve_Hoyna(h, ts, y1, y2)
    # метод Рунге-Кутты 3-го порядка
    solve_Runge_Kutta(h, ts, u1, u2)

    print('      Метод Хойна (Хьюна)   Метод Рунге-Кутты (3-го порядка)')
    print('t   |   y1        y2      |   y1        y2')
    for i in range(N):
        print(f'{ts[i]:.1f} | {y1[i]:8.4f}  {y2[i]:8.4f}  | {u1[i]:8.4f}  {u2[i]:8.4f}')