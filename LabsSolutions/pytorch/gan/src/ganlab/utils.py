# coding: utf-8

# Standard imports
import inspect

class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def tab(n):
    return " " * 4 * n


def fail(msg):
    print(
        colors.FAIL
        + tab(1)
        + f"[FAILED] From {inspect.stack()[1][3]}"
        + msg
        + colors.ENDC
    )


def succeed(msg=""):
    print(colors.OKGREEN + tab(1) + "[PASSED]" + msg + colors.ENDC)


def head(msg):
    print(colors.HEADER + msg + colors.ENDC)


def info(msg):
    print(colors.OKBLUE + tab(1) + msg + colors.ENDC)


def test_equal(l1, l2, eps):
    return all([abs(l1i - l2i) <= eps for l1i, l2i in zip(l1, l2)])

