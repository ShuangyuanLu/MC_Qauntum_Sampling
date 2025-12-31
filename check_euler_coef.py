import numpy as np
import math


def polynomial_coef(n, m):
    coef = 0
    for k in range(m + 1):
        coef += (-1) ** m * math.comb(2 * n + 1, 2 * k + 1) * math.comb(n - k, m - k)
    return coef


def reduced_polynomial_coef(n, m):
    return polynomial_coef(n, m) / (2 * n + 1)


def print_polynomial_coef(n):
    for m in range(n + 1):
        print(reduced_polynomial_coef(n, m))


def check_polynomial_coef(n):
    theta = 0.1
    print("value:", np.sin((2 * n + 1) * theta))

    poly_value = 0
    for m in range(n + 1):
        poly_value += polynomial_coef(n, m) * np.sin(theta) ** (2 * m + 1)
    print("poly_value:", poly_value)


def check_reduced_polynomial_coef_m(m):
    coef_list = []
    coef_list_formula = []
    for n in range(1, 20):
        if n >= m:
            coef_list.append(reduced_polynomial_coef(n, m))
            if m == 0:
                coef_list_formula.append(1)
            if m == 1:
                coef_list_formula.append(-2 / 3 * n * (n + 1))
            if m == 2:
                coef_list_formula.append(2 / 3 / 5 * n * (n-1) * (n + 1) * (n + 2))
            if m == 3:
                coef_list_formula.append(- 4 / 5 / 7 / 9 * (n-2) * (n-1) * n * (n + 1) * (n + 2) * (n+3))
            if m == 4:
                coef_list_formula.append(2 / 5 / 7 / 9 / 9 * (n-3) * (n-2) * (n-1) * n * (n + 1) * (n + 2) * (n+3) * (n+4))
            if m == 5:
                coef_list_formula.append(-4 / 5 / 5 / 7 / 9 / 9 / 11 * (n-4) * (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n+5))
            if m == 6:
                coef_list_formula.append(-4 / 3 / 5 / 5 / 7 / 9 / 9 / 11 / 13 * (n-5) * (n-4) * (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n+5) * (n+6))

    print(coef_list)
    print(coef_list_formula)


def check_expansion_coef(n):
    theta = 0.001
    exact_poly_value = np.sin(theta * (2 * n + 1)) / np.sin(theta)/ (2* n+1)
    print("exact_poly_value", exact_poly_value, 1-exact_poly_value)

    p_n = np.sin(theta * (2 * n + 1)) ** 2
    print("p_n", p_n)
    p = np.sin(theta) ** 2

    alpha_1 = 2/3 * n * (n+1) /((2*n+1) ** 2)
    r_1 = 1 - alpha_1 * p_n
    print("r_1", r_1, 1-r_1)

    alpha_2 = 2/45 * n * (n + 1) * (17 * n ** 2 + 17 * n + 6) / ((2 * n + 1) ** 4)
    r_2 = 1- alpha_2 * p_n ** 2

    alpha_3 = 8/105 * n * (n+1) * (27 *n**4 + 54*n**3 +42*n**2 + 15*n +2) / (2*n+1) **6
    r_3 = 1 - alpha_3 * p_n ** 3

    #x = -14 * n * (n+1) * (7*n**2 + 7*n + 6) + 70 * n**2 * (n+1)**2 + 21 * (n-1) * n * (n+1) * (n+2) -3 * (n-2) * (n-1) * (n+2 ) * (n+ 3) - 28 * n * (n+1) * ( 17 *n**2 +17*n +6)
    #alpha_3 = -4 * n * (n+1)  / 3/5/7/9 * x #/(2*n+1) ** 6
    # print("alpha_3", alpha_3 * p**3)
    # r_3 = 1 - alpha_3 * p ** 3
    #a_3 = -8/135 * n **2* (n+1)**2 * (7*n**2 + 7*n + 6) + 8/27 * n**3 * (n+1)**3 + 4/45 * (n-1) * n**2 * (n+1)**2 * (n+2) -4/315 * (n-2) * (n-1) *n*(n+1)* (n+2 ) * (n+ 3) - 16/135 *n**2 *(n+1)**2 * ( 17 *n**2 +17*n +6)
    #a_3 = 4 * n * (n+1) /3/5/7/9 *18 * ((n-1)*(n+2)*(n**2 +n+1) - 14 * n * (n+1) * (2 * n**2 + 2 * n + 1))
    # a_3 = 8*n*(n+1)/105*(27 *n**4 +54*n**3 +42*n**2 +15*n + 2)
    # a_3 = a_3 * p** 3
    # print("a_3:", a_3)

    difference= 1 - exact_poly_value / r_1
    print("difference_1", difference)
    difference = 1 - exact_poly_value / r_1 / r_2
    print("difference_2", difference)
    difference = 1 - exact_poly_value / r_1 / r_2 / r_3
    print("difference_3", difference)

    # print(exact_poly_value - 1)
    # print(exact_poly_value - (1 - 2 / 3 * n * (n + 1) * p))
    # print(exact_poly_value - (1 - 2/3 *n*(n+1)*p +2/15*(n-1)*n*(n+1)*(n+2)*p**2))
    # print(exact_poly_value - (1 - 2 / 3 * n * (n + 1) * p + 2 / 15 * (n - 1) * n * (n + 1) * (n + 2) * p ** 2 -4/5/7/9 * (n-2)*(n-1)*n*(n+1)*(n+2)*(n+3)* p**3))

    # print(1/r_1 - (1 + 2/3 * n * (n+1) * p))
    # print(1/r_1 - (1 + 2/3 * n * (n+1) * p - 4/9 * n**2 * (n+1) ** 2 * p ** 2))
    #print(1/r_1 - (1 + 2/3 * n * (n+1) * p - 4/9 * n**2 * (n+1) ** 2 * p ** 2 - 8/135 * n**2 * (n+1)**2 * (7 * n ** 2 + 7 * n + 6) * p ** 3  ))

    # print(p_n/(2 * n + 1) ** 2 /p - (1))
    # print(p_n /(2 * n + 1) ** 2 / p - (1 - 2/3 * n * (n+1) * p) ** 2)
    # print(p_n /(2 * n + 1) ** 2 / p - (1 - 2/3 * n * (n+1) * p + 2/15 * (n-1) * n * (n+1) *(n+2) * p**2 ) ** 2)
    #
    # x = 1 - 2/3 * n * (n+1) * p + 2/15 * (n-1) * n * (n+1) *(n+2) * p**2
    # print(1/r_1 - (1 + 2/3 * n * (n+1) * p * x ** 2))
    # print(1 / r_1 - (1 + 2 / 3 * n * (n + 1) * p * x ** 2 + 4/9 * n**2 * (n+1)**2 * p ** 2 * x ** 4))
    # print(1 / r_1 - (1 + 2 / 3 * n * (n + 1) * p * x ** 2 + 4 / 9 * n ** 2 * (n + 1) ** 2 * p ** 2 * x ** 4 + 8/27 * n** 3 *(n+1)** 3 * p** 3 * x ** 6))
    #
    # print(1/r_2 -1)
    # print(1/r_2 - (1 + 2/45 * n * (n+1) * (17 * n ** 2 + 17 * n + 6) * p ** 2 - 16/ 135 * n **2 * (n+1) ** 2 * (17 * n ** 2 + 17 * n + 6) * p ** 3))

    # print(1- (1 - 2 / 3 * n * (n + 1) * p + 2 / 15 * (n - 1) * n * (n + 1) * (n + 2) * p ** 2 -4/5/7/9 * (n-2)*(n-1)*n*(n+1)*(n+2)*(n+3)* p**3) * \
    #       (1 + 2/3 * n * (n+1) * p - 4/9 * n**2 * (n+1) ** 2 * p ** 2 - 8/135 * n**2 * (n+1)**2 * (7 * n ** 2 + 7 * n + 6) * p ** 3  ) * \
    #       (1 + 2/45 * n * (n+1) * (17 * n ** 2 + 17 * n + 6) * p ** 2 - 16/ 135 * n **2 * (n+1) ** 2 * (17 * n ** 2 + 17 * n + 6) * p ** 3))
    #
    #
    # print(- 8/135 * n**2 * (n+1)**2 * (7 * n ** 2 + 7 * n + 6) * p ** 3)
    # print(- 2 / 3 * n * (n + 1) * p *(- 4/9 * n**2 * (n+1) ** 2 * p ** 2))
    # print(2 / 15 * (n - 1) * n * (n + 1) * (n + 2) * p ** 2 * 2/3 * n * (n+1) * p)
    # print(-4 / 5 / 7 / 9 * (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * p ** 3)
    # print(-16/ 135 * n **2 * (n+1) ** 2 * (17 * n ** 2 + 17 * n + 6) * p ** 3)




check_expansion_coef(50)






