"""
This file is not relevant to the overall project.
This is just a place for me (Aidan) to practice dynamic programming.
"""



def Test(cases, func):
    for case, answer in cases:
        evaluated = func(case)
        print(f"Evaluated: {evaluated}, Expected: {answer},", "PASS" if evaluated == answer else "FAIL")



######################################
# Trailing Integer
######################################

def TrailingInt(string, i=-1, out=-1, d=1):
    """
    Returns -1 if there is not trailing integer.
    :param string:
    :param i:
    :param out:
    :param d: What digit is to be added to out next, ie 1, 10, 100, 1000, etc.
    :return:
    """
    #base case
    if abs(i) > len(string):
        return out

    #base case
    if not string[i].isdigit():
        return out

    #general case
    else:
        if out == -1: #boundary case for the first digit
            out = 0
        inc = int(string[i]) * d
        return TrailingInt(string, i-1, out+inc, d*10)


def TestTrailingInt():
    tests = [
        ("string1045", 1045),
        ("test", -1),
        ("words3", 3),
        ("694", 694)
    ]

    for string, answer in tests:
        test = TrailingInt(string)
        print(f"Calculated: {test}, Expected: {answer},", "PASS" if test == answer else "FAIL")





######################################
# Decimal To Binary and vice versa
######################################
def BinToDec(b, i=-1, a=0):
    """
    Assumes b is not empty.
    :param b: Input binary string
    :param i: current index of b being analyzed
    :param a: accumulator of the result
    :return:
    """
    #base case, end of string (b)
    if abs(i) > len(b):
        return a

    #general case
    else:
        if b[i] == "1":
            return BinToDec(b, i-1, a + 2**(abs(i)-1))
        else: #b[i] == "0"
            return BinToDec(b, i-1, a)

def DecToBin(d):
    """

    :param d:
    :return:
    """
    return 0

def TestBinDec():
    tests = [
        ("10101", 21),
        ("11", 3),
        ("1000", 8),
        ("011011", 27)
    ]
    Test(tests, BinToDec)

    tests = [
        (21, "10101"),
        (3, "11"),
        (8, "1000"),
        (27, "11011")
    ]
    Test(tests, DecToBin)






######################################
# Fast and Space efficient Fibonacci
######################################
def fibonacci(n):
    # base case
    if n == 0:
        return 0
    if n == 1:
        return 1

    #F[i%3] is the i-th fibonacci number. Only need space for 3 to calculate.
    F = [0, 1, -1]

    # Count
    i = 2
    while i <= n:
        F[i % 3] = F[(i-1) % 3] + F[(i-2) % 3]
        i += 1
    return F[n % 3]












def main():
    for i in range(10):
        print(i, fibonacci(i))


if __name__ == "__main__":
    main()


