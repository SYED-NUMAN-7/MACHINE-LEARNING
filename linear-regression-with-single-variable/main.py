import math
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1,2,3])
y_train = np.array([300,500,700])

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = (1 / (2 * m)) * cost
    return total_cost

def compute_gradient_descent(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = (w * x[i]) + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_db = dj_db / m
    dj_dw = dj_dw / m

    return dj_dw , dj_db

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,cost_function,compute_gradient_descent):

    J_history = []
    p_history = []

    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw,dj_db = compute_gradient_descent(x,y,w,b)

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < 100000:
            J_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration : {i:4} , Cost : {J_history[-1]}, ",
                  f"dj-dw : {dj_dw:0.2e} , dj_db : {dj_db:0.3e}, ",
                  f"w : {w:0.3e} , b : {b:0.5e}")

    return w,b,J_history,p_history

w_in = 0
b_in = 0
iterations = 10000
tmp_alpha = 1.0e-1

w_final,b_final,j_history,p_history = gradient_descent(x_train,y_train,w_in,b_in,tmp_alpha,iterations,compute_cost,compute_gradient_descent)

print(f" (w,b) found by gradient descent : ({w_final:8.4f},{b_final:8.4f})")

def compute_model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

output = compute_model_output(x_train,w_final,b_final)

plt.scatter(x_train,y_train,c='r',marker='x',label='Actual Values')
plt.plot(x_train,output,c='b',label='Model Prediction')
plt.xlabel('Area in Sqft ( in 1000s of meters)')
plt.ylabel('Price ( in 1000s of dollars )')
plt.title('Housing Prediction')
plt.show()

x_input = float(input("Enter the size of the house : "))

print(x_input)
house_predict = x_input * w_final + b_final
print(f"The Price of the house for {x_input*1000} = {house_predict*1000:0.0f}")