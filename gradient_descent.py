import load_data
import random
import numpy as np
import math
import types
# data = 
# num = len(data)
# data = np.asarray(data, dtype='float32')

# def gradient(theta_index):

def prepare_data(parameter_num):
    theta_list = []
    # data = data.reshape((2, num))
    y_train = []
    x_train = []
    for da in load_data.loaddata():
        y_train.append(da[0])
        x_line = []
        for i in range(parameter_num):
            x_line.append(math.pow(float(da[1]), i))
        x_train.append(x_line)
    return x_train, y_train


def rantheta(parameter_num):
    theta_list = []
    for i in range(parameter_num):
        the = random.randint(290,310)
        theta_list.append(the)
    return theta_list

def gradient(theta_list, index, x, y):
    gradient = 0
    total = 0
    parameter_num = len(theta_list)
    for i in range(parameter_num):
        ee = x[i] * theta_list[i]
        total = total + ee
        gradient = (total - float(y)) * x[index]
    return gradient

def gradient_batch(theta_list, x_batch, y_batch):
    # x_batch = np.asarray(x_batch, dtype='float32')
    theta_list = np.asarray(theta_list)
    x_batch = np.asarray(x_batch)
    y_batch = np.asarray(y_batch)

    # tx = theta_list.dot(x_batch) - y_batch
    tx = x_batch.dot(theta_list)
    return tx


def newton_method(alpha, theta_list, x, y):

    theta_array = np.asarray(theta_list)
    
    polynomial = len(theta_list)
    
    # for x,y in zip(x_train, y_train):
    
    delta_thetaz_array = gradient_batch(theta_list, x, y)
    x_len = len(x)
    hessian = []
    # for i in range(x_len):
    #     # for j in range(x_len):
    #     #     hessian.append(x[i] * x[j])

    hession_matrix = x.T.dot(x)

    tt = hession_matrix.I
    print 'he',hession_matrix
    de = delta_thetaz_array.dot(hession_matrix)
    de = np.squeeze(np.asarray(de))
    theta_array = theta_array - alpha * de
    print theta_array
    return theta_array



def batch_train_gd(alpha, theta_list):
    # alpha = 0.00000015
    # theta_list = list(theta_list)
    parameter_num = len(theta_list)
    x_train, y_train = prepare_data(parameter_num)
    

    for i in range(parameter_num):
        gradient_batch = 0
        for x,y in zip(x_train, y_train):
            single_gradient = gradient(theta_list, i, x, y)
            gradient_batch += single_gradient
        theta_list[i] = theta_list[i] - alpha * gradient_batch
    # print theta_list
    return theta_list

def stochastic_train_gd(alpha, theta_list):
    parameter_num = len(theta_list)
    x_train, y_train = prepare_data(parameter_num)
    for x, y in zip(x_train, y_train):
        for i in range(parameter_num):
            gradient_single = gradient(theta_list, i, x, y)
            theta_list[i] = theta_list[i] - alpha * gradient_single
    return theta_list


def loss(theta_list):
    polynomial = len(theta_list)
    x_train, y_train = prepare_data(polynomial)
    loss = 0
    for x,y in zip(x_train, y_train):
        total = 0
        for i in range(len(theta_list)):
            total += x[i] * theta_list[i]
        if type(total) != types.FloatType:
            total = np.sum(total)
        loss_single = float(y) - total
        loss += 0.5 * math.pow(loss_single, 2)
    # print 'loss',loss
    return loss

if __name__ == '__main__':
    import argparse
    stop_flag = False
    parser = argparse.ArgumentParser(description='Select batch or stochastic.')
    parser.add_argument('--learning_rate', help = 'define the learning_rate default is 0.00000001555555')
    parser.add_argument('--method', help = 'if batch is batch or stochastic' )
    args = parser.parse_args()
    if not args.learning_rate:
        alpha = 0.00000001555555
    else:
        alpha = float(args.learning_rate)
    is_batch = args.method

    theta_list = rantheta(2)
    if is_batch == 'batch':
        filew = 'batch.out'
    elif is_batch == 'newton':
        filew = 'newton.out'
    else:
        filew = 'stochastic.out'

    filewriter = open('filew', 'w')
    loss_b = 0
    if is_batch == 'batch':
        print 'start batch'
        while stop_flag == False:
            theta_list = batch_train_gd(alpha, theta_list)
            filewriter.write(str(theta_list))
            filewriter.write('\n')
            loss_ = loss(theta_list)
            print loss_
            if abs(loss_ - loss_b) < 0.00001:
                stop_flag = True
            loss_b = loss_
            filewriter.write(str(loss_))
            filewriter.write('\n')
    elif is_batch == 'newton':
        print 'newton'
        x_train, y_train = prepare_data(2)
        while stop_flag == False:
            # for x,y in zip(x_train, y_train):
            theta_list = newton_method(alpha, theta_list ,x_train ,y_train)
            filewriter.write(str(theta_list))
            filewriter.write('\n')
            loss_ = loss(theta_list)
            print loss_
            if loss_ - loss_b < 5:
                stop_flag = True
            loss_b = loss_
            filewriter.write(str(loss_))
            filewriter.write('\n')
    else:
        print 'start scho'
        while stop_flag == False:
            theta_list = stochastic_train_gd(alpha, theta_list)
            filewriter.write(str(theta_list))
            filewriter.write('\n')
            loss_ = loss(theta_list)
            print loss_
            if loss_ - loss_b < 50000:
                stop_flag = True
            loss_b = loss_
            filewriter.write(str(loss_))
            filewriter.write('\n')
    print theta_list
    print 'ok'
    filewriter.write(str(theta_list))
    filewriter.write('\n')
    filewriter.write('ok')
    filewriter.close()
    # print loss([-355010.8, 749.25, -0.15212923])