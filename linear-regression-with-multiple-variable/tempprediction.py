import math
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

X_train = np.array([[2104,5,1,45],
                   [1416,3,2,40],
                   [852,2,1,35]])

y_train = np.array([460,232,178])

print(X_train.shape)
print(X_train)

print(y_train)

w_init = np.array([0.39133535,18.75376741,-53.36032453,-26.42131618])
b_init = 785.1811367994083

def predict_single_loop(x,w,b):
    m = x.shape[0]
    p = 0
    for i in range(m):
        p_i = x[i]*w[i]
        p = p + p_i
    p = p + b
    return p

def predict(x,w,b):
    p = np.dot(x,w) + b
    return p

x = X_train[0,:]
print(x.shape)
print(x)

f_wb = predict_single_loop(X_train[0,:],w_init,b_init)
f_wbk = f_wb.shape
print(f_wbk)
print(f_wb)
print(predict(x,w_init,b_init))