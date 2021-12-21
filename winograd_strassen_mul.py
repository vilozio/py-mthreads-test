import numpy as np
from typing import Union


def is_power_of_two(n: Union[int, float]):
    while n % 2 == 0 and n > 1:
        n /= 2
    return n == 1


def get_right_power_of_two(n):
    exp = 1
    right_n = 1
    while right_n < n:
        right_n = 2 ** exp
        exp += 1
    return right_n


class MatrixSlices:
    @staticmethod
    def top_left(x, n):
        return x[:n, :n]

    @staticmethod
    def top_right(x, n):
        return x[:n, n:]

    @staticmethod
    def bottom_left(x, n):
        return x[n:, :n]

    @staticmethod
    def bottom_right(x, n):
        return x[n:, n:]

counter = 0


def recursive_mul(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    div_length = n // 2


    if div_length == 1:
        global counter
        counter += 1
        return np.matmul(a, b)
    a11 = MatrixSlices.top_left(a, div_length)
    a12 = MatrixSlices.top_right(a, div_length)
    a21 = MatrixSlices.bottom_left(a, div_length)
    a22 = MatrixSlices.bottom_right(a, div_length)

    b11 = MatrixSlices.top_left(b, div_length)
    b12 = MatrixSlices.top_right(b, div_length)
    b21 = MatrixSlices.bottom_left(b, div_length)
    b22 = MatrixSlices.bottom_right(b, div_length)

    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    s5 = b12 - b11
    s6 = b22 - s5
    s7 = b22 - b12
    s8 = s6 - b21

    p1 = recursive_mul(div_length, s2, s6)
    p2 = recursive_mul(div_length, a11, b11)
    p3 = recursive_mul(div_length, a12, b21)
    p4 = recursive_mul(div_length, s3, s7)
    p5 = recursive_mul(div_length, s1, s5)
    p6 = recursive_mul(div_length, s4, b22)
    p7 = recursive_mul(div_length, a22, s8)

    t1 = p1 + p2
    t2 = t1 + p4

    c11 = p2 + p3
    c12 = t1 + p5 + p6
    c21 = t2 - p7
    c22 = t2 + p5
    c = np.block([[c11, c12], [c21, c22]])
    return c


def mul_square_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape[0] == b.shape[0]
    assert a.shape[0] == a.shape[1] and b.shape[0] == b.shape[1]
    n = a.shape[0]
    if not is_power_of_two(n):
        good_n = get_right_power_of_two(n)
        a = np.pad(a, (0, good_n - n), constant_values=0)
        b = np.pad(b, (0, good_n - n), constant_values=0)
        c = recursive_mul(good_n, a, b)
        c = c[:n, :n]
    else:
        c = recursive_mul(n, a, b)
    return c


def main():
    a = np.array([
        [4, 33, 53, 79, 13, 44, 51, 69, 87],
        [32, 90, 8, 37, 34, 66, 70, 28, 92],
        [74, 57, 30, 31, 77, 26, 63, 77, 9],
        [13, 18, 62, 26, 11, 85, 32, 11, 80],
        [91, 85, 26, 20, 34, 51, 89, 59, 71],
        [4, 78, 44, 71, 29, 21, 97, 2, 56],
        [88, 81, 0, 7, 64, 97, 65, 3, 39],
        [28, 73, 90, 55, 23, 59, 28, 91, 93],
        [26, 24, 70, 5, 86, 28, 8, 94, 36]
    ])
    b = a
    c = mul_square_matrices(a, b)
    # print(c)
    print(counter)


if __name__ == '__main__':
    main()
