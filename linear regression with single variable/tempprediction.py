# THIS IS JUST AN UNTRAINED MODEL TYPE
# IN THIS PLOTTING , THE PLOT IS JUST DONE BY DETERMINED W AND B

import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0,2.0,3.0])
y_train = np.array([300.0,500.0,700.0])
m = x_train.shape[0]
w = 199.9929
b = 100.0116

print(f"x_train : {x_train}")
print(f"y_train : {y_train}")
print(f"m : {m}")
print(f"w : {w}")
print(f"b : {b}")

def compute_model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

temp_f_wb = compute_model_output(x_train,w,b)

plt.plot(x_train,temp_f_wb,c='b',label = "Our prediction")
plt.scatter(x_train,y_train,marker='x',c='r',label='Actual Values')
plt.title('Housing Prices')
plt.xlabel('Size (1000 sqft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.show()