import load_data
import normal_equation
import math
import types
import numpy as np

def prepare_data(polynomial):
    # test_data = load_data.loaddata('test')
    theta_list = []
    # data = data.reshape((2, num))
    x_test = []
    y_test = []
    for da in load_data.loaddata('test'):
        y_test.append(da[0])
        x_line = []
        for i in range(polynomial):
            x_line.append(math.pow(float(da[1]), i))
        x_test.append(x_line)
    return x_test, y_test

def crmse(theta_list):
    polynomial = len(theta_list)
    x_test, y_test = prepare_data(polynomial)
    loss = 0
    hh = 0
    for x,y in zip(x_test, y_test):
        total = 0
        for i in range(len(theta_list)):
            total += x[i] * theta_list[i]
        if type(total) != types.FloatType:
            total = np.sum(total)
        loss_single = float(y) - total
        loss += math.pow(loss_single, 2)
        
        if hh < 10:
            print loss
            hh = hh + 1
            print hh
    loss = math.sqrt(float(loss) / len(x_test))
    # print 'loss',loss
    return loss

if __name__ == '__main__':
    theta_list = normal_equation.get_thetalist(2)
    print theta_list
    loss = crmse(theta_list)
    print loss

