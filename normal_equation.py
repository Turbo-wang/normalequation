import numpy as np
import load_data
import math

def prepare_data(parameter_num):
    theta_list = []
    # data = data.reshape((2, num))
    y_train = []
    x_train = []
    for da in load_data.loaddata('train'):
        y_train.append(da[0])
        x_line = []
        for i in range(parameter_num):
            x_line.append(math.pow(float(da[1]), i))
        x_train.append(x_line)
    return x_train, y_train

def get_thetalist(parameter_num):
    X, Y = prepare_data(parameter_num)
    X = np.asmatrix(X, dtype='float32')
    Y = np.asarray(Y, dtype='float32')
    XT = X.transpose()
    XTX = XT.dot(X)
    X_1 = XTX.I
    XTX_1XT = X_1.dot(XT)
    theta_list = XTX_1XT.dot(Y)
    return theta_list
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Select batch or stochastic.')
    parser.add_argument('--polynomial', help = 'Polynomial')
    # parser.add_argument('--method', help = 'if batch is batch or stochastic' )
    args = parser.parse_args()

    polynomial = args.polynomial
    filewriter = open('normal_equation.out', 'w+')
    theta_list = get_thetalist(int(polynomial))
    for i in  theta_list.tolist()[0]:
        print '%.2f' % float(i)
    print 'ok'
    filewriter.write(str(theta_list))
    filewriter.write('\n')
    filewriter.write('ok')
    filewriter.write('\n')
    filewriter.close()


