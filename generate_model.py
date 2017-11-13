#! -*- encoding:utf-8 -*-
#author: gpxlcj


from random import randint
from sys import maxint
from math import sin, cos, log, e, sqrt, pi, floor
from matrix_tool import *
from mnist import MNIST


'''
univariate gaussian data generate model
'''
def univariate_gaussian_data_generate(m=0, s=1, n=20):
    uniform_data_u = [randint(0, maxint)/float(maxint) for i in range(0, n)]
    uniform_data_v = [randint(0, maxint)/float(maxint) for i in range(0, n)]

    data = list()
    for i in range(0, n):
        u = uniform_data_u[i]
        v = uniform_data_v[i]
        x = sqrt(-2*log(u, e)) * cos(2*pi*v)
        x = m+s*x
        data.append(x)
    return data
