#! -*- coding:utf-8 -*-
# Author: gpxlcj
# This module comes from the homework of NCTU CS machine learning course.
# You can use it for your experiment but not homework.

import argparse
from Regression import RidgeRegression

def main(file_name, n, n_lambda):
    """
    :param file_name: str
    :param n: int
    :param n_lambda: int
    :return: null
    """
    regression = RidgeRegression(n, n_lambda, file_name)
    regression.read_file()
    regression.train()
    regression.predict()
    print('best fitting line: ' + str(regression.x))
    print('RLSE: ' + str(regression.rlse))
    print ('predict label: ' + str(regression.y_list))

if __name__ == '__main__':
    n = 3
    n_lambda = 1
    file_name = 'test.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help='please input data file name, like test.txt')
    parser.add_argument("n", type=int, help='the number of polynomial bases')
    parser.add_argument("n_lambda", type=float, help='regularized parameter, lambda')
    param = parser.parse_args()
    file_name = param.file_name
    n = param.n
    n_lambda = param.n_lambda
    if n<1:
        print('n should larger than 0!')
        exit(0)
    main(file_name, n, n_lambda)
