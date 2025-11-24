import sympy as sp


def taylor_coeff(expr: sp.Expr, var: sp.Symbol, m: int, point: sp.Expr = 0) -> sp.Expr:
    """
    Compute the coefficient of (var - point)^m in the Taylor expansion of expr about `point`.
    """
    if m < 0:
        raise ValueError("m must be non-negative")

    # Use series to avoid intermediate 0/0 evaluations that can yield NaN.
    order = m + 4  # a few extra terms for safety
    ser = sp.series(expr, var, point, order).removeO().expand()
    coeff = ser.coeff(var, m)
    # Keep structured factors like (2*n+1)**k by combining and factoring.
    return sp.factor(sp.together(coeff))


def euler_expr(x: sp.Symbol, n: sp.Symbol) -> sp.Expr:
    """
    Return f(x) = x^(1/2) / (2n+1) / sin(arcsin(x^(1/2))/(2n+1)).
    This form is analytic at x=0; lim_{x->0} f(x) = 1.
    """
    return x ** sp.Rational(1, 2) / (2 * n + 1) / sp.sin(
        sp.asin(x ** sp.Rational(1, 2)) / (2 * n + 1)
    )


def series_coeff(expr: sp.Expr, var: sp.Symbol, power: sp.Expr, around: sp.Expr, order: int = 10) -> sp.Expr:
    """
    Extract the coefficient of var**power from the series of expr around `around`.
    Uses Sympy's series; works for power being int or Rational.
    """
    ser = sp.series(expr, var, around, order).removeO().expand()
    return sp.simplify(ser.coeff(var, power))


def euler_product_coeffs(expr: sp.Expr, var: sp.Symbol, order: int = 5) -> dict[int, sp.Expr]:
    """
    For a series f(x) with f(0)=1, compute c_k such that
    f(x) = product_{k=1..order} (1 + c_k x^k) + O(x^{order+1}).
    """
    # Normalize so the constant term is 1, using the limit to avoid 0/0.
    f0 = sp.simplify(sp.limit(expr, var, 0))
    if f0 == 0:
        raise ValueError("f(0) is zero; cannot form product with constant term 1.")
    expr_norm = sp.simplify(expr / f0)

    # Plain Maclaurin series to required order
    ser = sp.series(expr_norm, var, 0, order + 1).removeO().expand()
    if sp.simplify(ser.subs(var, 0) - 1) != 0:
        raise ValueError("Series normalization failed to produce constant term 1.")

    # Convert series to coefficient dict
    residual = {m: sp.simplify(ser.coeff(var, m)) for m in range(order + 1)}

    def mul_series(a: dict[int, sp.Expr], b: dict[int, sp.Expr]) -> dict[int, sp.Expr]:
        out: dict[int, sp.Expr] = {k: 0 for k in range(order + 1)}
        for ea, va in a.items():
            if va == 0:
                continue
            for eb, vb in b.items():
                m = ea + eb
                if m > order:
                    continue
                out[m] += va * vb
        return out

    coeffs: dict[int, sp.Expr] = {}
    for k in range(1, order + 1):
        ck = residual.get(k, 0)
        coeffs[k] = sp.factor(sp.together(ck))

        # Build inverse of (1 + c_k x^k) truncated: 1 - c_k x^k + c_k^2 x^{2k} - ...
        inv_factor = {j * k: (-ck) ** j for j in range(0, order // k + 1)}
        residual = mul_series(residual, inv_factor)
        residual = {m: sp.simplify(v) for m, v in residual.items()}
        # Re-normalize small drift in constant term
        residual[0] = sp.Integer(1)
    return coeffs


if __name__ == "__main__":
    x, n = sp.symbols("x n", positive=True)
    a = sp.symbols("a")
    alpha_1 = sp.symbols("alpha_1")

    # Work in terms of a = 1/(2n+1) to keep series small, then substitute back.
    expr_a = sp.sqrt(x) * a / sp.sin(a * sp.asin(sp.sqrt(x)))

    c_prod_a = euler_product_coeffs(expr_a, x, order=10)
    c_prod = {k: sp.factor(sp.together(v.subs(a, 1 / (2 * n + 1)))) for k, v in c_prod_a.items()}

    print("\nEuler product coefficients c_k (up to order 10):")
    for k, ck in c_prod.items():
        print(f"c_{k} = {ck}")

    print("\nLaTeX for coefficients (names alpha_k, n unchanged):")
    for k, ck in c_prod.items():
        print(f"alpha_{{{k}}} = {sp.latex(ck)}")
