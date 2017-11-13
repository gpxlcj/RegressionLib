#! -*- coding:utf-8 -*-


def matrix_time(m_A, m_B):
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


def matrix_add(m_A, m_B):
    """
    :param m_A: list(left matrix)
    :param m_B: list(right matrix)
    :return: list(result matrix)
    """
    m_ans = list()
    for i in range(len(m_A)):
        m_ans.append(list())
        for j in range(len(m_B[0])):
            temp = m_A[i][j] + m_B[i][j]
            m_ans[i].append(temp)
    return m_ans

def ele_swap(m_A, line_a, line_b):
    """
    :param line_a: list(row A)
    :param line_b: list(row B)
    :return: 0
    """
    temp = m_A[line_a]
    m_A[line_a] = m_A[line_b]
    m_A[line_b] = temp
    return m_A


def ele_add(m_A, line_a, line_b, w):
    """
    :param line_a: list(row A)
    :param line_b: list(changed row B)
    :param w: int(multiple)
    :return: 0
    """
    add_list = [i * w for i in m_A[line_a]]
    m_A[line_b] = [x + y for x, y in zip(m_A[line_b], add_list)]
    return m_A


def ele_multiply(m_A, line_a, w):
    """
    :param line_a: list(changed row A)
    :param w: int(multiple)
    :return:  0
    """
    m_A[line_a] = [item * w for item in m_A[line_a]]
    return m_A


def lu_decomposition(m_A, n):
    m_B = list()
    for i in range(n):
        m_B.append(list())
        for j in range(n):
            m_B[i].append(0)
        m_B[i][i] = 1
    for i in range(n):
        if (m_A[i][i] == 0):
            for l in range(i + 1, n):
                if m_A[l] != 0:
                    m_B = ele_swap(m_B, i, l)
                    m_A = ele_swap(m_A, i, l)
        m_B = ele_multiply(m_B, i, 1.0 / m_A[i][i])
        m_A = ele_multiply(m_A, i, 1.0 / m_A[i][i])
        for j in range(i + 1, n):
            m_B = ele_add(m_B, i, j, -m_A[j][i])
            m_A = ele_add(m_A, i, j, -m_A[j][i])
        else:
            if ((n == i + 1) or (m_A[i + 1][i + 1] != 0)):
                continue
            else:
                for l in range(i + 1, n):
                    if m_A[l] != 0:
                        m_B = ele_swap(m_B, i, l)
                        m_A = ele_swap(m_A, i, l)
    for i in range(n)[::-1]:
        for j in range(i)[::-1]:
            m_B = ele_add(m_B, i, j, -m_A[j][i])
            m_A = ele_add(m_A, i, j, -m_A[j][i])
    return m_B