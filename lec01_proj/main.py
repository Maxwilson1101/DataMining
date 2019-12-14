import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = './data/diabetes.csv'


def load_data(data_file):
    try:
        data_arr = np.loadtxt(data_file, dilimiter=',', skiprows=1)
    except Exception as e:
        print(e)
    return data_arr


def get_gradient(theta, X, Y):
    m = X.shape[0]
    Y_estimate = X.dot(theta)
    error = Y_estimate - Y
    grad = 1.0/m * X.T.dot(error)
    cost = 1.0/(2*m) * np.sum(error ** 2)
    return grad, cost


def gradient_descend(theta, X, Y, max_iter=500, learning_rate=0.0001):
    """
        initialize theta
    """
    theta = np.random.randn(2)
    """
        set tolerance
    """
    tolerance = 1e-3
    """
        Perform Gradient Descent
    """
    iterations = 1
    is_converged = False
    while not is_converged:
        grad, cost = get_gradient(theta, X, Y)
        new_theta = theta - learning_rate * grad
        """
            Stop iteration condition
        """
        if np.sum(abs(theta - new_theta)) < tolerance:
            is_converged = True
            print('parameter is converged')

        if iterations > max_iter:
            is_converged = True
            print('The maximum iteration has been reached')

    return theta


def main():
    pass


if __name__ == '__main__':
    pass
