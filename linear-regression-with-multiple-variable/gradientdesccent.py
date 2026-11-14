import numpy as np

X_train = np.array([[2104,5,1,45],
                   [1416,3,2,40],
                   [852,2,1,35]])

y_train = np.array([460,232,178])

w_init = np.array([0.39133535,18.75376741,-53.36032453,-26.42131618])
b_init = 785.1811367994083

def compute_gradient(X,y,w,b):

    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i],w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw,dj_db

temp_dj_dw,temp_dj_db = compute_gradient(X_train,y_train,w_init,b_init)
print(f"dj_dw at initial w,b : {temp_dj_dw}")
print(f"dj_db at initial w,b : {temp_dj_db}")