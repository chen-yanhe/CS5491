"""Deterministic online evaluator and reproducible Weibull instances."""

import ast
import math
import random


CAPACITY = 100
FIRST_FIT = "0"
BEST_FIT = "-remaining"
NAMES = {"item", "remaining", "gap"}
FUNCTIONS = {"abs": abs, "min": min, "max": max}
OPERATORS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.UAdd, ast.USub,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.And, ast.Or, ast.Not,
)


def weibull_instances(seed, count, items_per_instance):
    """Sample Weibull(scale=45, shape=3) and round into [1, 100]."""
    rng = random.Random(seed)
    return [
        [max(1, min(CAPACITY, round(rng.weibullvariate(45, 3))))
         for _ in range(items_per_instance)]
        for _ in range(count)
    ]


def check_expression(source):
    """Accept a bounded numeric expression without attribute access or statements."""
    if not isinstance(source, str) or len(source) > 500 or "\n" in source:
        raise ValueError("Provide one Python expression of at most 500 characters.")
    tree = ast.parse(source.strip(), mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 100:
        raise ValueError("The expression has too many operations.")

    # Function names are allowed only in the call position, with numeric arguments.
    callees = {node.func for node in nodes if isinstance(node, ast.Call)}
    for node in nodes:
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) not in (int, float) or not -1000 <= value <= 1000:
                raise ValueError("Use real numeric literals in [-1000, 1000].")
            node.value = float(value)
        elif isinstance(node, ast.Name):
            allowed = FUNCTIONS if node in callees else NAMES
            if node.id not in allowed:
                raise ValueError(f"Unknown name: {node.id}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ValueError("Use direct calls to abs, min, or max.")
            if node.func.id == "abs" and len(node.args) != 1:
                raise ValueError("abs takes one argument.")
            if node.func.id in {"min", "max"} and not 2 <= len(node.args) <= 4:
                raise ValueError("min and max take two to four arguments.")
        elif not isinstance(node, (
            ast.Expression, ast.Load, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.BoolOp, ast.IfExp, *OPERATORS,
        )):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
    return compile(tree, "<priority>", "eval")


def priorities(source):
    """Evaluate the checked expression on every feasible (item, remaining) pair."""
    program = check_expression(source)
    scope = {"__builtins__": {}, **FUNCTIONS}
    table = [[0.0] * (CAPACITY + 1) for _ in range(CAPACITY + 1)]
    for item in range(1, CAPACITY + 1):
        for remaining in range(item, CAPACITY + 1):
            score = eval(program, scope, {
                "item": item / CAPACITY,
                "remaining": remaining / CAPACITY,
                "gap": (remaining - item) / CAPACITY,
            })
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("The score must be a finite number for every feasible pair.")
            table[item][remaining] = score
    return table


def bin_count(items, table):
    """Pack items in arrival order, taking the oldest feasible bin on a score tie."""
    bins = []
    for item in items:
        if not 1 <= item <= CAPACITY:
            raise ValueError("Item size must be in [1, 100].")
        row = table[item]
        best_index = None
        best_score = -math.inf
        for index, remaining in enumerate(bins):
            if remaining >= item and row[remaining] > best_score:
                best_index = index
                best_score = row[remaining]
        if best_index is None:
            bins.append(CAPACITY - item)
        else:
            bins[best_index] -= item
    return len(bins)


def evaluate(source, instances):
    """Return one bin count per instance using one shared priority table."""
    table = priorities(source)
    return [bin_count(items, table) for items in instances]
