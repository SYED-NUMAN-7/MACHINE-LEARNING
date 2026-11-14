"""
WE ARE GOING TO TAKE THE SAME DATA FROM SINGLE VARIABLE LINEAR REGRESSION AND THEN CHECK IT BY APPLYING 
MULTI - VARIABLE REGRESSION

WE WILL BE TAKING BEDROOMS AND AGE OF THE HOUSE AS MULTIPLE INPUTS ALONGSIDE AREA OF THE HOUSE 
"""
import math,copy
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[1100,2,8,1],
                   [1218,2,14,4],
                   [1260,2,5,2],
                   [1500,3,2,1],
                   [1625,3,6,3],
                   [1643,3,7,3],
                   [1800,3,15,5],
                   [2000,3,1,2]])

y_train = np.array([65,51,60,130,121,120,95,170])

w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def compute_cost(X,y,w,b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = np.dot(X[i],w) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

def compute_gradient(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i],w) + b ) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw,dj_db

def gradient_descent(X,y,w_in,b_in,cost_function,gradient_function,alpha,num_iters):

    J_history = []

    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw,dj_db = gradient_function(X,y,w,b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(X,y,w,b))

        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4d} : Cost {J_history[-1]:8.2f}")

    return w,b,J_history

def zscore_normalization_feature(X):
    
    #Find the Mean of each comlumn 
    mu = np.mean(X, axis=0)

    # Find the Standard deviation of each column/feature
    sigma = np.std(X,axis=0)

    # Element wise , Subtract mu for each column from each example , div by std for that column
    X_Norm = (X - mu) / sigma

    return (X_Norm, mu, sigma)

X_norm, X_mu, X_sigma = zscore_normalization_feature(x_train)

initial_w = np.zeros(x_train.shape[1])
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 0.1
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_norm, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_norm[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

x = X_norm[1,:]
print(x)

def predict(x,w,b):
    p = np.dot(x,w) + b
    return p 

output = predict(x,w_final,b_final)
print(f"\nBasic Sample Prediction :  {output:.0f} (in Lakhs)")

# --- VISUALIZING THE MULTI-VARIABLE MODEL ACCURACY ---

# 1. Generate predictions for all houses in your training set
m = x_train.shape[0]
predicted_prices = np.zeros(m)
for i in range(m):
    predicted_prices[i] = np.dot(X_norm[i], w_final) + b_final

# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Actual vs Predicted Scatter Plot
ax1.scatter(y_train, predicted_prices, color='blue', marker='o', s=100, label='Model Predictions')
# Draw a perfect diagonal reference line (where Actual = Predicted)
ideal_line = np.linspace(min(y_train), max(y_train), 100)
ax1.plot(ideal_line, ideal_line, color='red', linestyle='--', label='Perfect Accuracy Line')

ax1.set_xlabel("Actual Price (₹ Lakhs)")
ax1.set_ylabel("Predicted Price (₹ Lakhs)")
ax1.set_title("Actual vs. Predicted Housing Prices")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# Plot 2: Size vs Price (Isolating the primary feature)
# This shows how your multi-variable model smoothly handles the non-sequential noise
ax2.scatter(x_train[:, 0], y_train, color='darkorange', marker='s', s=100, label='Actual Data')
# Sort by size to plot a smooth prediction profile line
sort_idx = np.argsort(x_train[:, 0])
ax2.plot(x_train[sort_idx, 0], predicted_prices[sort_idx], color='teal', lw=2, label='Multi-Variable Prediction Trend')

ax2.set_xlabel("Area (sq. ft.)")
ax2.set_ylabel("Price (₹ Lakhs)")
ax2.set_title("How Model Handles Size vs Price Noise")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.suptitle("Malakpet Multi-Variable Regression Performance Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
