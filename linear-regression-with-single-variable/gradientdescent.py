import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1,2])
y_train = np.array([300,500])

def compute_gradient_descent(x,y,w,b):
    m = x.shape[0]
    dj_wb = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_wb += dj_dw_i
        dj_db += dj_db_i

    dj_db = dj_db / m
    dj_wb = dj_wb / m

    return dj_wb,dj_db

compute_gradient_descent(x_train,y_train,)




