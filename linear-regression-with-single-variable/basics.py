import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

print(f"x_train : {x_train}")
print(f"y_train : {y_train}")

m = x_train.shape

print(f"Number of training examples is : {m}")

m = len(x_train)

print(f"Number of training examples is : {m}")

i = 0
x_i = x_train[i]
y_i = y_train[i]

print(f"(x^({i}),y^({i})) = ({x_i} , {y_i})")

i = 1
x_i = x_train[i]
y_i = y_train[i]

print(f"(x^({i}),y^({i})) = ({x_i} , {y_i})")

w = 100
b = 100

def compute_model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

temp_wb = compute_model_output(x_train,w,b)

plt.scatter(x_train,y_train,marker='x',c='r',label="Actual Values")
plt.plot(x_train,temp_wb,c='b',label='Model Prediction')
plt.title("Housing Prices")
plt.ylabel('Prices (in 1000s of Dollars)')
plt.xlabel('Size (1000 sqft)')
plt.show()