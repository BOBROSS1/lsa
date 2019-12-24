# python implementation of least square appproximation (LSA)
# program will fit data optimal to linear model (minimal sum squares)
# of form (y = a + bx)
import numpy as np


def build_matrix(data):
    ''' function to build up a matrix from raw datapoints (x, y) '''
    return [(1, point[0]) for point in data]


def build_b(data):
    ''' funtion to build up b vector of lsa formula '''
    return [point[1] for point in data]


def matrix_transpose(A):
    ''' function to transpase matrix A '''
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def lsa(data):
    ''' function that calculates the lsa using formula: A'Ax = A'b returns 
        1. optimal linear function  2.vector x (with unknows a & b) '''
    A = np.array(build_matrix(data))
    b = np.array(build_b(data))
    At = np.array(matrix_transpose(A))

    AtAinv = np.linalg.inv(np.matmul(At, A))
    Atb = np.matmul(At, b)
    x = np.matmul(AtAinv, Atb)
    formula = 'y = %s + %sx' % (round(x[0], 3), round(x[1], 3))
    return formula, x


if __name__ == "__main__":
    user_input = input("Enter your data x1,y1; x2,y2; etc..: ")
    if user_input:
        data = []
        for item in user_input.split(";"):
            data.append(tuple(map(int, item.split(","))))
    else:
        data = [(1, 2), (2, 2), (3, 4)]

    print("x:", lsa(data)[1])
    print("Linear formula:", lsa(data)[0])
