#! -*- coding:utf-8 -*-
# Author: gpxlcj
# This module comes from the homework of NCTU CS machine learning course.
# You can use it for your experiment but not homework.


from math import sin, cos, log, e, sqrt, pi, floor
from matrix_tool import *
from generate_model import univariate_gaussian_data_generate


class LogisticRegression:

    d1_list = list()
    d2_list = list()
    train_data = list()

    theta = [0.01, 0.01, 0.01]
    alpha = 0.01

    accuracy = 0
    sensitivity = 0
    specificity = 0
    confusion_matrix = [[0, 0], [0, 0]]
    hessian_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    xm1 = 0
    xv1 = 1
    ym1 = 0
    yv1 = 1
    xm2 = 0
    xv2 = 1
    ym2 = 0
    yv2 = 1
    n = 500
    iter = 100

    def __init__(self, xm1, xv1, ym1, yv1, xm2, xv2, ym2, yv2, n):
        temp_x = univariate_gaussian_data_generate(xm1, xv1, n)
        temp_y = univariate_gaussian_data_generate(ym1, yv1, n)
        self.d1_list = [[1, temp_x[i], temp_y[i], 1] for i in range(n)]

        temp_x = univariate_gaussian_data_generate(xm2, xv2, n)
        temp_y = univariate_gaussian_data_generate(ym2, yv2, n)
        self.d2_list = [[1, temp_x[i], temp_y[i], 0] for i in range(n)]
        self.train_data.extend(self.d1_list)
        self.train_data.extend(self.d2_list)


    def gradient_descent(self):
        n = self.n
        j_func = 0
        temp_theta = self.theta
        for i in range(self.iter):
            theta_sum = [0, 0, 0]
            for record in self.train_data:
                for j in range(3):
                    temp = e**((temp_theta[0]*record[0])+(temp_theta[1]*record[1])+(temp_theta[2]*record[2]))
                    theta_sum[j] += ((temp/(1+temp)) - record[3]) * record[j]
            for j in range(3):
                temp_theta[j] = temp_theta[j] - self.alpha * theta_sum[j]/n
        self.theta = temp_theta
        return temp_theta


    def newton_method(self):
        temp_theta = self.theta

        temp = 0
        for i in range(self.iter):
            self.hessian_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            g_list = [[0, 0, 0]]
            for record in self.train_data:
                for j in range(3):
                    temp = e ** ((temp_theta[0]*record[0]) + (temp_theta[1]*record[1]) + (temp_theta[2]*record[2]))
                    g_list[0][j] += record[j] * (record[3] - temp/(1+temp))
                    for k in range(3):
                        self.hessian_matrix[j][k] += -record[j] * record[k] * temp/(1+temp)**2
            for j in range(3):
                g_list[0][j] = g_list[0][j] / self.n
                for k in range(3):
                    self.hessian_matrix[j][k] = self.hessian_matrix[j][k] / self.n

            self.hessian_matrix = lu_decomposition(self.hessian_matrix, 3)
            temp_matrix = matrix_time(g_list, self.hessian_matrix)[0]
            temp_matrix = [[-k for k in temp_matrix]]
            temp_theta = matrix_add([temp_theta], temp_matrix)[0]
            self.theta = temp_theta
        return temp_theta


    def evaluate(self):
        conf_m = self.confusion_matrix
        self.sensitivity = float(conf_m[1][1]) / (conf_m[1][1] + conf_m[1][0])
        self.specificity = float(conf_m[0][0]) / (conf_m[0][0] + conf_m[0][1])


    def test(self):
        self.confusion_matrix = [[0, 0], [0, 0]]
        temp_theta = self.theta
        print len(self.train_data)
        for record in self.train_data:
            temp = e ** ((temp_theta[0] * record[0]) + (temp_theta[1] * record[1]) + (temp_theta[2] * record[2]))
            if (temp/(1+temp))> 0.5:
                temp_y = 1
            else:
                temp_y = 0
            self.confusion_matrix[record[3]][temp_y] += 1


    def train(self):
        # theta_list = self.gradient_descent()
        theta_list = self.newton_method()
        print theta_list


    def test(self):
        predict_list = list()
        for i in len(self.test_img):
            max_ans = 0
            max_label = -1
            for j in range(10):
                temp = self.compare(self.test_img[i], self.center[j])
                if (temp > max_ans):
                    max_ans = temp
                    max_label = j
            predict_list.append(max_label)


    def evaluate(self):
        conf_m = self.confusion_matrix
        self.sensitivity = float(conf_m[1][1]) / (conf_m[1][1] + conf_m[1][0])
        self.specificity = float(conf_m[0][0]) / (conf_m[0][0] + conf_m[0][1])

class RidgeRegression:
    """
    Attribute:
    n: int, basic function number
    n_lambda: int, the parameter of regularization part
    data_length: int, the number of input matrix row
    rlse: int, the regularization least square error
    file_name: str, the input file name
    data_list: list, input data records, E.g [[1, 4], [3, 5]]
    """

    # Public Variable
    n = int()
    n_lambda = int()
    data_length = int()
    file_name = str()
    data_list = list()

    x = list()
    K = list()
    b = list()
    A_trans = list()
    A = list()
    m_expand = list()

    y_list = list()
    rlse = int()

    def __init__(self, n, n_lambda, file_name):
        """
        :param n: int
        :param n_lambda: int
        :param file_name: str
        """
        self.n = n
        self.n_lambda = n_lambda
        self.file_name = file_name
        for i in range(n):
            self.m_expand.append(list())
            for j in range(n):
                self.m_expand[i].append(0)
            self.m_expand[i][i] = 1

    # matrix multiplication
    def matrix_time(self, m_A, m_B):
        """
        :param m_A: list(left matrix)
        :param m_B: list(right matrix)
        :return: list(result matrix)
        """
        m_ans = list()
        for i in range(len(m_A)):
            m_ans.append(list())
            for j in range(len(m_B[0])):
                temp = 0
                for l in range(len(m_A[0])):
                    temp = temp + m_A[i][l] * m_B[l][j]
                m_ans[i].append(temp)
        return m_ans

    # elementary matrix: row-switching
    def swap(self, line_a, line_b):
        """
        :param line_a: list(row A)
        :param line_b: list(row B)
        :return: 0
        """
        temp = self.m_expand[line_a]
        self.m_expand[line_a] = self.m_expand[line_b]
        self.m_expand[line_b] = temp
        temp = self.K[line_a]
        self.K[line_a] = self.K[line_b]
        self.K[line_b] = temp
        return 0

    # elementary matrix: row-addition
    def add(self, line_a, line_b, w):
        """
        :param line_a: list(row A)
        :param line_b: list(changed row B)
        :param w: int(multiple)
        :return: 0
        """
        add_list = [i * w for i in self.m_expand[line_a]]
        self.m_expand[line_b] = [x + y for x, y in zip(self.m_expand[line_b], add_list)]
        add_list = [i * w for i in self.K[line_a]]
        self.K[line_b] = [x + y for x, y in zip(self.K[line_b], add_list)]
        return 0

    # elementary matrix: row-multiplying
    def multiply(self, line_a, w):
        """
        :param line_a: list(changed row A)
        :param w: int(multiple)
        :return:  0
        """
        self.m_expand[line_a] = [item * w for item in self.m_expand[line_a]]
        self.K[line_a] = [item * w for item in self.K[line_a]]
        return 0

    # read input data file
    def read_file(self, file_name=''):
        """
        :param file_name: string
        :return: 0
        """
        data_list = list()
        r_file = open(self.file_name, 'r')
        raw_data_list = r_file.readlines()
        for data in raw_data_list:
            temp = data.replace('\n', '').split(',')
            temp = [float(i) for i in temp]
            data_list.append(temp)

        r_file.close()
        self.data_list = data_list
        self.data_length = len(data_list)
        return 0

    # get the matrix that we will use in the next step
    def get_matrix_list(self):
        """
        :return: tuple(K, A, A_trans, b)
        """
        K = list()
        b = list()
        A = list()
        n = self.n

        A_trans = list()
        for i in range(n):
            A_trans.append(list())
        for pair in self.data_list:
            b.append([pair[1]])
            temp = 1
            temp_list = list()
            for i in range(n):
                temp *= pair[0]
                temp_list.append(temp)
                A_trans[i].append(temp)
            A.append(temp_list)
        for i in range(n):
            K.append(list())
            for j in range(n):
                temp = 0
                for l in range(len(A)):
                    temp += A_trans[i][l] * A[l][j]
                K[i].append(temp)
        for i in range(n):
            K[i][i] += self.n_lambda

        self.K = K
        self.A = A
        self.A_trans = A_trans
        self.b = b
        return K, A, A_trans, b

    # LU Decomposition based on Gaussian elimination
    def lu_decomposition(self):
        """
        :return: 0
        """
        x = list()
        self.x = x
        for i in range(self.n):
            self.multiply(i, 1.0 / self.K[i][i])
            for j in range(i + 1, self.n):
                self.add(i, j, -self.K[j][i])
            else:
                if ((self.n == i + 1) or (self.K[i + 1][i + 1] != 0)):
                    continue
                else:
                    for l in range(i + 1, self.n):
                        if self.K[l] != 0:
                            self.swap(i, l)
        for i in range(self.n)[::-1]:
            for j in range(i)[::-1]:
                self.add(i, j, -self.K[j][i])

        return 0

    # calculate the weight vector
    def calculate_weight(self):
        """
        :return: 0
        """
        self.x = self.matrix_time(self.matrix_time(self.m_expand, self.A_trans), self.b)
        return 0

    def train(self):
        """
        :return: 0
        """
        self.get_matrix_list()
        self.lu_decomposition()
        self.calculate_weight()
        return 0

    def predict(self):
        """
        :return: 0
        """
        self.y_list = self.matrix_time(self.A, self.x)
        self.__get_rlse__()
        return 0

    def __get_rlse__(self):

        lse = sum([(x[0] - y[0]) ** 2 for x, y in zip(self.y_list, self.b)])
        self.rlse = lse + self.n_lambda * sum([i[0] ** 2 for i in self.x])

        return 0

