from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import numpy as np


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


def recursive_mul(n: int, a: np.ndarray, b: np.ndarray, pool: ThreadPoolExecutor) -> np.ndarray:
    div_length = n // 2
    if div_length == 1:
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

    p1 = pool.submit(recursive_mul, div_length, s2, s6, pool)
    p2 = pool.submit(recursive_mul, div_length, a11, b11, pool)
    p3 = pool.submit(recursive_mul, div_length, a12, b21, pool)
    p4 = pool.submit(recursive_mul, div_length, s3, s7, pool)
    p5 = pool.submit(recursive_mul, div_length, s1, s5, pool)
    p6 = pool.submit(recursive_mul, div_length, s4, b22, pool)
    p7 = pool.submit(recursive_mul, div_length, a22, s8, pool)
    futures = [p1, p2, p3, p4, p5, p6, p7]
    futures = {future: i for i, future in enumerate(futures)}
    results = {}
    for future in as_completed(futures):
        p_idx = futures[future]
        results[p_idx] = future.result()

    t1 = results[0] + results[1]
    t2 = t1 + results[3]

    c11 = results[1] + results[2]
    c12 = t1 + results[4] + results[5]
    c21 = t2 - results[6]
    c22 = t2 + results[4]
    c = np.block([[c11, c12], [c21, c22]])
    return c


def mul_square_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape[0] == b.shape[0]
    assert a.shape[0] == a.shape[1] and b.shape[0] == b.shape[1]
    n = a.shape[0]
    with ThreadPoolExecutor() as pool:
        if not is_power_of_two(n):
            good_n = get_right_power_of_two(n)
            a = np.pad(a, (0, good_n - n), constant_values=0)
            b = np.pad(b, (0, good_n - n), constant_values=0)
            c = recursive_mul(good_n, a, b, pool)
            c = c[:n, :n]
        else:
            c = recursive_mul(n, a, b, pool)
    return c


def main():
    a = np.array([
        [72, 88, 19, 75, 54, 28, 75],
        [94, 58, 39, 39, 36, 58, 96],
        [30, 63, 90, 58, 44, 23, 0],
        [83, 29, 28, 52, 26, 92, 40],
        [91, 95, 70, 15, 30, 26, 24],
        [84, 83, 80, 34, 14, 95, 69],
        [84, 78, 39, 20, 33, 81, 57]
    ])
    b = np.array([
        [19, 28, 7, 53, 30, 74, 99],
        [38, 16, 65, 74, 87, 84, 26],
        [10, 86, 38, 49, 58, 75, 59],
        [92, 40, 37, 71, 32, 12, 81],
        [82, 95, 88, 30, 74, 1, 83],
        [92, 55, 74, 52, 71, 95, 86],
        [95, 59, 42, 80, 77, 0, 2]
    ])
    c = mul_square_matrices(a, b)
    # print(c)


if __name__ == '__main__':
    main()
