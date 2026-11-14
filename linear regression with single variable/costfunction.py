import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1,2])
y_train = np.array([2,4])


print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

def cost_function(x,y,w):
    cost = 0
    m = x.shape[0]
    f_w = np.zeros(m)
    for i in range(m):
        f_w = w * x[i]
        cost_sum = (f_w - y[i]) ** 2
        cost += cost_sum
    total_sum = (1 / 2 * m) * cost_sum
    return total_sum

print(cost_function(x_train,y_train,2))