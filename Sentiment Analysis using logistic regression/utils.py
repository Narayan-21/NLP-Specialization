import numpy as np

def Sigmoid(z):
    h = 1/(1+np.exp(-z))
    return z


def gradientDescent(x, y, theta, alpha, num_iter):
    m = x.shape[0]
    for i in range(0, num_iter):
        z = np.dot(x, theta)
        h = Sigmoid(z)
        J = (-1/m)*((np.dot(y.T, np.log(h))) + (np.dot((1-y).T, np.log(1-h))))
        theta = theta - (alpha/m) * (np.dot(x.T, (h-y)))
    J = float(J)
    return J, theta
