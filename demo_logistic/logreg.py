import numpy as np
import copy
import matplotlib.pyplot as plt


def logcostfun(theta, data):
    data0 = copy.deepcopy(data)
    data_len = len(data0)
    par_len = len(data0[0])
    cost = 0
    for i in range(data_len):
        y = data0[i][par_len - 1]
        data0[i].pop()
        data0[i].append(1)
        vec = np.mat(data0[i]).T
        a = theta * vec
        cost = cost + (np.log(1 + np.exp(a)) - y * a)
    cost = cost / data_len
    return float(cost)


def costfun_gra(theta, data):
    data0 = copy.deepcopy(data)
    gradient = 0
    par_len = len(data0[0])
    data_len = len(data0)
    for i in range(data_len):
        y = data0[i][par_len - 1]
        data0[i].pop()
        data0[i].append(1)
        vec = np.mat(data0[i]).T
        a = theta * vec
        gradient = gradient + float(np.exp(a)/(1 + np.exp(a)) - y) * vec
    gradient = gradient / data_len
    return gradient


def logreg(data, alpha, maxtimes=50000):
    threshold = 0.00001
    par_len = len(data[0])
    cost1, cost0, times = 1, 0, 0
    theta = np.mat(np.random.random(par_len))
    x_time, y_cost = [], []
    while (abs(cost1 - cost0) > threshold) & (times < maxtimes):
        times = times + 1
        cost0 = logcostfun(theta, data)
        a = alpha * costfun_gra(theta, data)
        theta = theta - a.T
        cost1 = logcostfun(theta, data)
        x_time.append(times)
        y_cost.append(cost1)
    plt.close()
    plt.scatter(x_time, y_cost, c='r')
    plt.show()
    return theta


def predict(theta, data):
    data0 = copy.deepcopy(data)
    data_len = len(data0)
    for i in range(data_len):
        data0[i].pop()
        data0[i].append(1)
    vec = np.mat(data0).T
    a = theta * vec
    a = a.tolist()
    b = []
    for i in range(len(a[0])):
        if a[0][i] > 0:
            b.append(1)
        else:
            b.append(0)
    return b


def logplot(theta, data):
    data_len = len(data)
    x1_1, x1_2, x0_1, x0_2 = [], [], [], []
    for i in range(data_len):
        if data[i][2] == 1:
            x1_1.append(data[i][0])
            x1_2.append(data[i][1])
        else:
            x0_1.append(data[i][0])
            x0_2.append(data[i][1])
    plt.close()
    plt.scatter(x1_1, x1_2, c='r')
    plt.scatter(x0_1, x0_2, c='g')
    theta = theta.tolist()
    xmax, xmin = max([i[0] for i in data]), min([i[0] for i in data])
    ymax, ymin = max([i[1] for i in data]), min([i[1] for i in data])
    x = np.arange(xmin-0.1, xmax+0.1, 0.01)
    y = np.arange(ymin-0.1, ymax+0.1, 0.01)
    x, y = np.meshgrid(x, y)
    z = theta[0][0]*x + theta[0][1]*y + theta[0][2]
    plt.contour(x, y, z, 0)
    plt.show()