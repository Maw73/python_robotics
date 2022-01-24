from sympy.abc import x, y, z
from sympy import LT, poly, div
from sympy.polys.orderings import monomial_key


def poly_div(f, divs, mo):
    # Sorting the monomials by the chosen ordering (lex, grlex, grevlex)
    ordered = sorted(divs, reverse=True, key=monomial_key(mo, [x, y, z]))
    # Creating a zero polynomial for the remainder
    r = poly(0, x, y, z)
    p = f
    # Finding the number of polynomials of divs
    s = ordered.__len__()
    # Creating an array with s polynomials of zero value
    q = [poly(0, x, y, z)] * s


    # Polynomial division algorithm
    while p != 0:
        i = 0
        div_occurred = False
        while i < s and div_occurred == False:
            # If the remainder of the division between LT(p) and LT(fi) is zero, then LT(fi) is divisible by LT(p)
            if div(LT(p, order=mo), LT(ordered[i], order=mo))[1] == 0:
                q[i] = q[i] + LT(p, order=mo) / LT(ordered[i], order=mo)
                p = p - (LT(p, order=mo) / LT(ordered[i], order=mo)) * ordered[i]
                div_occurred = True
            else:
                i = i + 1
        if div_occurred == False:
            r = r + LT(p, order=mo)
            p = p - LT(p, order=mo)
    dict = {'q': q, 'r': r}
    return dict

