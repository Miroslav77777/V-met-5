import numpy as np
import matplotlib.pyplot as plt

def read_input(filename='input.txt'):
    with open(filename, 'r') as file:
        data = {}
        for line in file:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                data[key.strip()] = val.strip()
        return data

def build_function(f_str):
    # f_str должен быть выражением от x и y, например: 'y - x**2 + 1'
    def f(x, y):
        return eval(f_str, {"x": x, "y": y, "np": np})
    return f

def runge_kutta_4(f, x0, y0, h, steps):
    xs = [x0]
    ys = [y0]
    for _ in range(steps):
        k1 = f(x0, y0)
        k2 = f(x0 + h/2, y0 + h*k1/2)
        k3 = f(x0 + h/2, y0 + h*k2/2)
        k4 = f(x0 + h, y0 + h*k3)
        y0 = y0 + h*(k1 + 2*k2 + 2*k3 + k4)/6
        x0 = x0 + h
        xs.append(x0)
        ys.append(y0)
    return xs, ys

def adams_moulton_4(f, x0, y0, h, Xmax):
    steps = int((Xmax - x0) / h)
    xs, ys = runge_kutta_4(f, x0, y0, h, 3)

    for n in range(3, steps):
        x_next = xs[-1] + h

        def G(y_guess):
            f0 = f(xs[-1], ys[-1])
            f1 = f(xs[-2], ys[-2])
            f2 = f(xs[-3], ys[-3])
            f_new = f(x_next, y_guess)
            return ys[-1] + h/24 * (9*f_new + 19*f0 - 5*f1 + f2)

        y_guess = ys[-1]
        for _ in range(5):
            y_guess = G(y_guess)

        xs.append(x_next)
        ys.append(y_guess)

    return xs, ys

# ======= Главный блок =======

input_data = read_input('input.txt')

# Парсинг входных данных
f_str = input_data['f']
x0 = float(input_data['x0'])
y0 = float(input_data['y0'])
h = float(input_data['h'])
Xmax = float(input_data['Xmax'])

# Создание функции f(x, y)
f = build_function(f_str)

# Решение методом Адамса-Мултона
x_vals, y_vals = adams_moulton_4(f, x0, y0, h, Xmax)

# Визуализация
plt.plot(x_vals, y_vals, 'bo-', label='Adams-Moulton (4-step)')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Неявный метод Адамса 4-го порядка")
plt.grid()
plt.legend()
plt.show()
