#! -*- coding:utf-8 -*-
# Author: gpxlcj
# This module comes from the homework of NCTU CS machine learning course.
# You can use it for your experiment but not homework.


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
            K[i][i] += n_lambda

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
                if ((n == i + 1) or (self.K[i + 1][i + 1] != 0)):
                    continue
                else:
                    for l in range(i + 1, n):
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

