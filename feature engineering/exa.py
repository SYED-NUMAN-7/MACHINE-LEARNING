import numpy as np
import matplotlib.pyplot as plt
import math

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = (1 / (2 * m)) * cost
    return total_cost

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw,dj_db

def gradient_descent(x,y,w_in,b_in,iterations,alpha,cost_function,gradient_function):

    J_history = []
    p_history = []
    w = w_in
    b = b_in

    for i in range(iterations):
        dj_dw,dj_db = gradient_function(x,y,w,b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])
        
        if i % math.ceil(iterations/10) == 0:
            print(f"Iteration : {i:4} , Cost : {J_history[-1]}, ",
                  f"dj-dw : {dj_dw:0.2e} , dj_db : {dj_db:0.3e}, ",
                  f"w : {w:0.3e} , b : {b:0.5e}")
        
            return w,b,J_history,p_history

iterations = 1000
alpha = 1e-2

w_in = 0
b_in = 0
x = np.arange(0,20,1)
y = 1 + x**2
X = x**2



print(x)
print(y)
print(X)

model_w,model_b,J_history,p_history = gradient_descent(X,y,w_in,b_in,iterations,alpha,compute_cost,compute_gradient)

def compute_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

output = compute_output(X,model_w,model_b)

# plt.scatter(x,y,marker='x',c='r',label='Actual Value')
# plt.plot(x,output,label="Predicted Value")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("No Feature Engineering")
# plt.legend()
# plt.show()

plt.scatter(x,y,c='r',m='x',label='Actual Values')
plt.plot(x,output,label='Predicted Values')
plt.title("Added x**2 feature")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()