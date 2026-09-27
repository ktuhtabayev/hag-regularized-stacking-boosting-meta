from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from hag_regularized_stacking_boosting_meta.domain.params import MajorizingConfig


MajorizingFn = Callable[[np.ndarray], np.ndarray]


def _to_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        out: List[float] = []
        for v in x:
            out.append(_to_float(v, 0.0))
        return out
    return []


# -----------------------
# Built-in majorizers
# -----------------------

def identity(x: np.ndarray) -> np.ndarray:
    return x


def sigmoid_factory(k: float, x0: float) -> MajorizingFn:
    def f(x: np.ndarray) -> np.ndarray:
        z = k * (x - x0)
        return 1.0 / (1.0 + np.exp(-z))
    return f


def exponential_factory(k: float, x0: float, scale: float) -> MajorizingFn:
    """
    exponential: scale * exp(k*(x-x0))
    """
    def f(x: np.ndarray) -> np.ndarray:
        return scale * np.exp(k * (x - x0))
    return f


def quadratic_factory(a: float, b: float, c: float) -> MajorizingFn:
    """
    quadratic: a*x^2 + b*x + c
    """
    def f(x: np.ndarray) -> np.ndarray:
        return a * (x ** 2) + b * x + c
    return f


def piecewise_linear_factory(x_points: List[float], y_points: List[float]) -> MajorizingFn:
    """
    piecewise_linear via interpolation.
    - x_points: strictly increasing recommended
    - y outside range clamps to endpoints.
    """
    if len(x_points) < 2 or len(x_points) != len(y_points):
        raise ValueError("piecewise_linear requires params.x and params.y lists of same length >= 2")

    xp = np.asarray(x_points, dtype=float)
    yp = np.asarray(y_points, dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # np.interp clamps outside range automatically
        return np.interp(x, xp, yp)
    return f


# -----------------------
# Registry / factory
# -----------------------

def get_majorizing_function(cfg: MajorizingConfig) -> MajorizingFn:
    """
    Returns f: ndarray -> ndarray.

    Supported names:
      identity
      sigmoid, logistic
      exponential
      quadratic
      piecewise_linear
    """
    name = (cfg.name or "identity").lower().strip()
    params: Dict[str, Any] = dict(cfg.params or {})

    if name in ("identity", "id", "none"):
        return identity

    if name in ("sigmoid", "logistic"):
        k = _to_float(params.get("k", 1.0), 1.0)
        x0 = _to_float(params.get("x0", 0.0), 0.0)
        return sigmoid_factory(k=k, x0=x0)

    if name in ("exponential", "exp"):
        k = _to_float(params.get("k", 1.0), 1.0)
        x0 = _to_float(params.get("x0", 0.0), 0.0)
        scale = _to_float(params.get("scale", 1.0), 1.0)
        return exponential_factory(k=k, x0=x0, scale=scale)

    if name in ("quadratic", "poly2"):
        a = _to_float(params.get("a", 1.0), 1.0)
        b = _to_float(params.get("b", 0.0), 0.0)
        c = _to_float(params.get("c", 0.0), 0.0)
        return quadratic_factory(a=a, b=b, c=c)

    if name in ("piecewise_linear", "pwl"):
        xs = _to_float_list(params.get("x"))
        ys = _to_float_list(params.get("y"))
        return piecewise_linear_factory(xs, ys)

    raise ValueError(
        f"Unknown majorizing function '{cfg.name}'. "
        "Supported: identity, sigmoid/logistic, exponential, quadratic, piecewise_linear."
    )
