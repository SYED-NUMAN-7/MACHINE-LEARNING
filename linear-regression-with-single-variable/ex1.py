""" 
THIS PROGRAM IS FAILING TO OUTPUT THE ACCURATE PREDICTED OUTPUT DATA BECUASE THERE ARE PRICE CHANGES IN
THE DATA WITH EVERY INPUT CHANGE AND IT IS NOT VARYING IN A SEQUENCE 

LET'S TRY THIS WITH MULIPLE VARIABLE REGRESSION
""" 

import numpy as np
import matplotlib.pyplot as plt
import os

x_train = np.array([1., 1.2, 1.25, 1.5, 1.6, 1.6, 1.8, 2.])
y_train = np.array([65, 51, 60, 130, 121, 120, 95, 170])

print(x_train)
print(y_train)

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = cost / (2 * m)
    return total_cost

def compute_gradient_descent(x,y,w,b):
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

def gradient_descent(x,y,w_in,b_in,alpha,num_iters,compute_cost,compute_gradient_descent):

    J_history = []
    P_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):

        dj_dw,dj_db = compute_gradient_descent(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(compute_cost(x,y,w,b))
            P_history.append([w,b])

        assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        os.makedirs(assets_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(J_history)), J_history, color="#1f77b4", linewidth=2)
    plt.title("Gradient Descent Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("Cost J(w, b)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(os.path.join(assets_dir, "cost_convergence.png"), dpi=300)
    plt.show()
        
    return w,b,J_history,P_history

# Change these values in your code
w_in = 0
b_in = 0
num_iters = 15000  # Give the gradient enough steps to settle
alpha = 0.1        # Smaller step size prevents the model from overshooting


w_final,b_final,J_history,P_history = gradient_descent(x_train,y_train,w_in,b_in,alpha,num_iters,compute_cost,compute_gradient_descent)

print(f"w = {w_final:.2f},b = {b_final:.2f}")

def compute_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

output = compute_output(x_train,w_final,b_final)
print(output)

assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(assets_dir, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.scatter(x_train,y_train,marker='s',c='r',label="Actual Values")
plt.plot(x_train,output,c='b',label='Model Prediction (Random Parameters)')
plt.xlabel("Area in sqft")
plt.ylabel("Price (in lakhs)")
plt.title("HOUSING PRICE (IN MALAKPET)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, "regression_fit.png"), dpi=300)
plt.show()