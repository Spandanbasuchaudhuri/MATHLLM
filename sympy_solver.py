import sympy
from sympy.parsing.sympy_parser import parse_expr

def sympy_check(step_text: str):
    """
    Improved check: tries to parse and validate simple math equalities.
    Returns (is_valid: bool, message: str)
    """
    # Look for simple equalities like "expr1 = expr2"
    if "=" in step_text:
        parts = step_text.split("=")
        if len(parts) == 2:
            lhs, rhs = parts[0].strip(), parts[1].strip()
            try:
                lhs_expr = parse_expr(lhs)
                rhs_expr = parse_expr(rhs)
                if sympy.simplify(lhs_expr - rhs_expr) == 0:
                    return True, "Step is mathematically correct."
                else:
                    return False, "Step is mathematically incorrect."
            except Exception as e:
                return False, f"Could not parse step: {e}"
    # Fallback: pass if not an equality
    return True, "No equality to check."

def sympy_solve_expression(expr: str):
    """
    Placeholder function to parse math from LLM and solve with SymPy.
    """
    x = sympy.Symbol('x')
    try:
        parsed_expr = parse_expr(expr)
        solution = sympy.solve(parsed_expr, x)
        return solution
    except Exception:
        return None

def parse_math_expression(expr_str: str):
    """
    Parse a mathematical expression string into a SymPy expression.
    """
    try:
        parsed_expr = parse_expr(expr_str)
        return parsed_expr
    except Exception as e:
        return f"Error parsing expression: {e}"