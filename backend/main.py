"""
Interactive Curve Fitting Lab - Backend API
FastAPI server for curve fitting and function analysis
"""
from __future__ import annotations

import numpy as np
from typing import Literal, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Model fitting imports
from scipy import optimize, integrate
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(
    title="Curve Fitting Lab API",
    description="Backend for interactive curve fitting and function analysis",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class Point(BaseModel):
    x: float
    y: float


class FitRequest(BaseModel):
    points: list[Point]
    objective: Literal['accuracy', 'interpretability', 'balanced'] = 'accuracy'


class FitStatistics(BaseModel):
    r2: float
    rmse: float
    mae: float
    aic: Optional[float] = None
    bic: Optional[float] = None


class FitResult(BaseModel):
    expression: str
    expressionLatex: str
    statistics: FitStatistics
    quality: Literal['bad', 'regular', 'good']
    curvePoints: list[Point]
    modelType: str
    heuristics: list[str]


class AnalyzeRequest(BaseModel):
    expression: str


class Extremum(BaseModel):
    type: Literal['maximum', 'minimum']
    x: float
    y: float


class Asymptote(BaseModel):
    type: Literal['vertical', 'horizontal', 'oblique']
    value: Union[float, str]


class AnalyticalProperties(BaseModel):
    firstDerivative: str
    secondDerivative: str
    extrema: list[Extremum]
    asymptotes: list[Asymptote]


class IntegralRequest(BaseModel):
    expression: str
    a: float
    b: float


class IntegralResult(BaseModel):
    value: float
    expression: str


# Model fitting functions
def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a linear model: y = mx + b"""
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    m, b = model.coef_[0], model.intercept_
    y_pred = model.predict(x.reshape(-1, 1))
    expr = f"y = {m:.6g}x + {b:.6g}" if b >= 0 else f"y = {m:.6g}x - {abs(b):.6g}"
    return expr, y_pred, {'type': 'Linear', 'complexity': 1}


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int = 2) -> tuple[str, np.ndarray, dict]:
    """Fit a polynomial model"""
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.01))
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    # Get coefficients for expression
    coeffs = model.named_steps['ridge'].coef_
    intercept = model.named_steps['ridge'].intercept_

    # Build expression
    terms = []
    if abs(intercept) > 1e-10:
        terms.append(f"{intercept:.6g}")
    for i in range(1, degree + 1):
        if i < len(coeffs) and abs(coeffs[i]) > 1e-10:
            if i == 1:
                terms.append(f"{coeffs[i]:.6g}x")
            else:
                terms.append(f"{coeffs[i]:.6g}x^{i}")

    expr = "y = " + " + ".join(terms) if terms else "y = 0"
    expr = expr.replace("+ -", "- ")

    return expr, y_pred, {'type': f'Polynomial (degree {degree})', 'complexity': degree}


def fit_exponential(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit an exponential model: y = a * exp(b*x) + c

    This is the improved 3-parameter exponential with:
    - Vertical shift parameter c
    - Robust initial guesses using multiple c seeds
    - Overflow protection via safe_exp
    - NaN for overflow values
    """
    MAX_EXP_ARG = 700  # Avoid overflow (exp(700) ~ 1e304)

    def safe_exp(arg):
        """Compute exp with overflow protection, returns NaN for overflow"""
        with np.errstate(over='ignore'):
            result = np.where(
                np.abs(arg) > MAX_EXP_ARG,
                np.nan,
                np.exp(np.clip(arg, -MAX_EXP_ARG, MAX_EXP_ARG))
            )
        return result

    def exponential_func(x, a, b, c):
        """Exponential function with overflow protection"""
        return a * safe_exp(b * x) + c

    try:
        # Try multiple c candidates (vertical shift seeds)
        y_min = np.min(y)
        y_max = np.max(y)
        y_mean = np.mean(y)
        y_p10 = np.percentile(y, 10)
        y_p90 = np.percentile(y, 90)

        # Candidate c values: below min(y), at percentiles, and 0
        c_candidates = [
            y_min - abs(y_max - y_min) * 0.1,  # Below minimum
            y_min - 1,
            y_p10 - 1,
            0,
            y_mean,
        ]

        best_result = None
        best_residual = float('inf')

        for c_init in c_candidates:
            try:
                # Estimate a and b using log transform on (y - c)
                y_shifted = y - c_init
                valid_mask = y_shifted > 0
                if np.sum(valid_mask) < 3:
                    continue

                x_valid = x[valid_mask]
                y_valid = y_shifted[valid_mask]

                # Linear regression on log(y - c) vs x
                log_y = np.log(y_valid)
                model = LinearRegression()
                model.fit(x_valid.reshape(-1, 1), log_y)

                b_init = model.coef_[0]
                a_init = np.exp(model.intercept_)

                # Refine with nonlinear least squares
                popt, _ = optimize.curve_fit(
                    exponential_func,
                    x, y,
                    p0=[a_init, b_init, c_init],
                    maxfev=5000,
                    bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                )

                # Calculate residual
                y_pred_test = exponential_func(x, *popt)
                valid_pred = np.isfinite(y_pred_test)
                if np.sum(valid_pred) > len(x) * 0.5:  # At least 50% valid
                    residual = np.sum((y[valid_pred] - y_pred_test[valid_pred]) ** 2)
                    if residual < best_residual:
                        best_residual = residual
                        best_result = popt
            except Exception:
                continue

        if best_result is None:
            raise ValueError("Exponential fit failed for all c candidates")

        a, b, c = best_result

        # Generate predictions with overflow protection
        y_pred = exponential_func(x, a, b, c)

        # Build expression string
        c_sign = "+" if c >= 0 else "-"
        c_val = abs(c)

        expr = f"y = {a:.4g} * exp({b:.4g}x) {c_sign} {c_val:.4g}"

        return expr, y_pred, {
            'type': 'Exponential',
            'complexity': 3,
            'params': {'a': a, 'b': b, 'c': c}
        }
    except Exception:
        raise ValueError("Exponential fit failed")


def fit_logarithmic(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a logarithmic model: y = a * ln(x) + b"""
    try:
        # Only fit if all x values are positive
        if np.any(x <= 0):
            return fit_linear(x, y)

        log_x = np.log(x)
        model = LinearRegression()
        model.fit(log_x.reshape(-1, 1), y)

        a, b = model.coef_[0], model.intercept_
        y_pred = a * np.log(x) + b

        expr = f"y = {a:.6g} * ln(x) + {b:.6g}" if b >= 0 else f"y = {a:.6g} * ln(x) - {abs(b):.6g}"
        return expr, y_pred, {'type': 'Logarithmic', 'complexity': 2}
    except Exception:
        return fit_linear(x, y)


def fit_power(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a power model: y = a * x^b"""
    try:
        if np.any(x <= 0) or np.any(y <= 0):
            return fit_linear(x, y)

        log_x = np.log(x)
        log_y = np.log(y)
        model = LinearRegression()
        model.fit(log_x.reshape(-1, 1), log_y)

        b = model.coef_[0]
        a = np.exp(model.intercept_)

        y_pred = a * np.power(x, b)

        expr = f"y = {a:.6g} * x^{b:.6g}"
        return expr, y_pred, {'type': 'Power', 'complexity': 2}
    except Exception:
        return fit_linear(x, y)


def fit_rational(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a simple rational model: y = (ax + b) / (cx + d)"""
    try:
        def rational_func(x, a, b, c, d):
            return (a * x + b) / (c * x + d)

        # Initial guess
        p0 = [1, 1, 1, 1]

        popt, _ = optimize.curve_fit(rational_func, x, y, p0=p0, maxfev=5000)
        a, b, c, d = popt

        y_pred = rational_func(x, *popt)

        expr = f"y = ({a:.4g}x + {b:.4g}) / ({c:.4g}x + {d:.4g})"
        return expr, y_pred, {'type': 'Rational', 'complexity': 3}
    except Exception:
        return fit_linear(x, y)


def fit_spline(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a smoothing spline"""
    try:
        # Sort by x
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Remove duplicates
        _, unique_idx = np.unique(x_sorted, return_index=True)
        x_unique = x_sorted[unique_idx]
        y_unique = y_sorted[unique_idx]

        if len(x_unique) < 4:
            return fit_linear(x, y)

        spline = UnivariateSpline(x_unique, y_unique, s=len(x_unique) * 0.1)
        y_pred = spline(x)

        expr = "y = Spline(x)"
        return expr, y_pred, {'type': 'Smoothing Spline', 'complexity': 5}
    except Exception:
        return fit_linear(x, y)


def fit_sinusoidal(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a sinusoidal model: y = A * sin(Bx + C) + D"""
    try:
        def sin_func(x, A, B, C, D):
            return A * np.sin(B * x + C) + D

        # Estimate initial parameters
        A0 = (y.max() - y.min()) / 2
        D0 = y.mean()
        # Estimate frequency from zero crossings
        B0 = 2 * np.pi / (x.max() - x.min())
        C0 = 0

        popt, _ = optimize.curve_fit(
            sin_func, x, y,
            p0=[A0, B0, C0, D0],
            maxfev=5000
        )
        A, B, C, D = popt

        y_pred = sin_func(x, *popt)

        expr = f"y = {A:.4g} * sin({B:.4g}x + {C:.4g}) + {D:.4g}"
        return expr, y_pred, {'type': 'Sinusoidal', 'complexity': 4}
    except Exception:
        return fit_linear(x, y)


def fit_reciprocal_shifted(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a shifted reciprocal model: y = a/(x - c) + d

    This model captures 1/x-like relationships with horizontal and vertical shifts.
    Returns NaN for x values where |x - c| < epsilon to avoid Infinity.
    """
    EPSILON = 1e-10

    try:
        def reciprocal_func(x, a, c, d):
            """Reciprocal function with safe evaluation"""
            denom = x - c
            # Return large value instead of inf for fitting (scipy handles this)
            with np.errstate(divide='ignore', invalid='ignore'):
                result = a / denom + d
            return result

        def reciprocal_safe(x, a, c, d):
            """Safe evaluation returning NaN near asymptote"""
            denom = x - c
            result = np.where(
                np.abs(denom) < EPSILON,
                np.nan,
                a / denom + d
            )
            return result

        # Estimate initial parameters
        # c: estimate from where y changes sign or has largest gradient
        x_sorted_idx = np.argsort(x)
        x_sorted = x[x_sorted_idx]
        y_sorted = y[x_sorted_idx]

        # Estimate c from the x value where |dy/dx| is largest
        if len(x_sorted) > 2:
            dy = np.diff(y_sorted)
            dx = np.diff(x_sorted)
            with np.errstate(divide='ignore', invalid='ignore'):
                gradients = np.abs(dy / dx)
            gradients = np.nan_to_num(gradients, nan=0)
            max_grad_idx = np.argmax(gradients)
            c0 = (x_sorted[max_grad_idx] + x_sorted[max_grad_idx + 1]) / 2
        else:
            c0 = x.mean()

        # d: estimate from the mean of y values far from suspected asymptote
        mask_far = np.abs(x - c0) > (x.max() - x.min()) / 4
        if np.sum(mask_far) > 0:
            d0 = np.median(y[mask_far])
        else:
            d0 = np.median(y)

        # a: estimate from y - d at a point away from asymptote
        x_far = x[mask_far] if np.sum(mask_far) > 0 else x
        y_far = y[mask_far] if np.sum(mask_far) > 0 else y
        if len(x_far) > 0:
            idx = len(x_far) // 2
            a0 = (y_far[idx] - d0) * (x_far[idx] - c0)
        else:
            a0 = 1.0

        # Try multiple initial guesses for c
        c_candidates = [c0, x.min() - 0.1, x.max() + 0.1, 0.0]
        best_result = None
        best_residual = float('inf')

        for c_init in c_candidates:
            try:
                # Exclude points too close to the candidate asymptote
                mask_valid = np.abs(x - c_init) > EPSILON * 1000
                if np.sum(mask_valid) < 3:
                    continue

                x_fit = x[mask_valid]
                y_fit = y[mask_valid]

                popt, _ = optimize.curve_fit(
                    reciprocal_func,
                    x_fit, y_fit,
                    p0=[a0, c_init, d0],
                    maxfev=5000,
                    bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
                )

                # Calculate residual on valid points
                y_pred_test = reciprocal_safe(x_fit, *popt)
                valid_mask = ~np.isnan(y_pred_test)
                if np.sum(valid_mask) > 0:
                    residual = np.sum((y_fit[valid_mask] - y_pred_test[valid_mask]) ** 2)
                    if residual < best_residual:
                        best_residual = residual
                        best_result = popt
            except Exception:
                continue

        if best_result is None:
            raise ValueError("Could not fit reciprocal model")

        a, c, d = best_result

        # Generate predictions with NaN near asymptote
        y_pred = reciprocal_safe(x, a, c, d)

        # Build expression string
        c_sign = "-" if c >= 0 else "+"
        c_val = abs(c)
        d_sign = "+" if d >= 0 else "-"
        d_val = abs(d)

        expr = f"y = {a:.4g}/(x {c_sign} {c_val:.4g}) {d_sign} {d_val:.4g}"

        return expr, y_pred, {
            'type': 'Reciprocal',
            'complexity': 3,
            'asymptote_x': c,
            'params': {'a': a, 'c': c, 'd': d}
        }
    except Exception:
        raise ValueError("Reciprocal fit failed")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> FitStatistics:
    """Calculate fit statistics, handling NaN/Inf values safely"""
    # Filter out NaN and Inf values from predictions
    valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    n = len(y_true_valid)

    if n < 2:
        # Not enough valid points
        return FitStatistics(r2=-999.0, rmse=float('inf'), mae=float('inf'), aic=None, bic=None)

    r2 = r2_score(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    mae = mean_absolute_error(y_true_valid, y_pred_valid)

    # AIC and BIC
    residuals = y_true_valid - y_pred_valid
    ss_res = np.sum(residuals ** 2)

    if ss_res > 0 and n > n_params:
        log_likelihood = -n/2 * np.log(2 * np.pi) - n/2 * np.log(ss_res/n) - n/2
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood
    else:
        aic = None
        bic = None

    return FitStatistics(r2=r2, rmse=rmse, mae=mae, aic=aic, bic=bic)


def determine_quality(r2: float, assumptions_met: bool = True) -> Literal['bad', 'regular', 'good']:
    """Determine fit quality based on R² and assumptions"""
    if r2 < 0.6 or not assumptions_met:
        return 'bad'
    elif r2 < 0.85:
        return 'regular'
    else:
        return 'good'


@app.get("/")
def root():
    return {"status": "ok", "message": "Curve Fitting Lab API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/fit", response_model=FitResult)
def fit_curve(request: FitRequest):
    """Fit a curve to the provided points"""
    if len(request.points) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 points")

    # Extract data
    x = np.array([p.x for p in request.points])
    y = np.array([p.y for p in request.points])

    # Define models to try based on objective
    models = [
        ('linear', fit_linear, 1),
        ('poly2', lambda x, y: fit_polynomial(x, y, 2), 2),
        ('poly3', lambda x, y: fit_polynomial(x, y, 3), 3),
        ('exponential', fit_exponential, 2),
        ('logarithmic', fit_logarithmic, 2),
        ('power', fit_power, 2),
        ('sinusoidal', fit_sinusoidal, 4),
        ('reciprocal', fit_reciprocal_shifted, 3),
    ]

    if request.objective == 'accuracy':
        models.append(('poly4', lambda x, y: fit_polynomial(x, y, 4), 4))
        models.append(('spline', fit_spline, 5))
        models.append(('rational', fit_rational, 3))

    # Fit all models and collect results
    results = []
    heuristics = []

    for name, fit_func, n_params in models:
        try:
            expr, y_pred, info = fit_func(x, y)
            metrics = calculate_metrics(y, y_pred, n_params)

            # Score based on objective
            if request.objective == 'accuracy':
                score = metrics.r2
            elif request.objective == 'interpretability':
                # Prefer simpler models
                score = metrics.r2 - 0.1 * info['complexity']
            else:  # balanced
                score = metrics.r2 - 0.05 * info['complexity']

            results.append({
                'name': name,
                'expr': expr,
                'y_pred': y_pred,
                'metrics': metrics,
                'info': info,
                'score': score
            })

            heuristics.append(f"{info['type']}: R²={metrics.r2:.4f}, RMSE={metrics.rmse:.4f}")
        except Exception as e:
            heuristics.append(f"{name}: failed ({str(e)[:50]})")

    if not results:
        raise HTTPException(status_code=500, detail="All models failed to fit")

    # Select best model
    best = max(results, key=lambda r: r['score'])

    # Generate curve points for visualization
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    x_curve = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)

    # Refit best model on full range
    try:
        _, y_curve, _ = [m for m in models if m[0] == best['name']][0][1](x, y)
        # Interpolate for curve points
        from scipy.interpolate import interp1d
        interp_func = interp1d(x, best['y_pred'], kind='linear', fill_value='extrapolate')
        y_curve = interp_func(x_curve)
    except Exception:
        y_curve = np.interp(x_curve, x, best['y_pred'])

    curve_points = [Point(x=float(xi), y=float(yi)) for xi, yi in zip(x_curve, y_curve)]

    return FitResult(
        expression=best['expr'],
        expressionLatex=best['expr'],  # Could be enhanced with LaTeX formatting
        statistics=best['metrics'],
        quality=determine_quality(best['metrics'].r2),
        curvePoints=curve_points,
        modelType=best['info']['type'],
        heuristics=heuristics
    )


@app.post("/analyze", response_model=AnalyticalProperties)
def analyze_function(request: AnalyzeRequest):
    """Compute analytical properties of a function"""
    try:
        x = sp.Symbol('x')

        # Parse the expression (handle common formats)
        expr_str = request.expression
        expr_str = expr_str.replace('y = ', '').replace('y=', '')
        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace('ln', 'log')

        # Handle special cases
        if 'Spline' in expr_str or 'spline' in expr_str:
            return AnalyticalProperties(
                firstDerivative="Spline derivative (numerical)",
                secondDerivative="Spline second derivative (numerical)",
                extrema=[],
                asymptotes=[]
            )

        expr = parse_expr(expr_str)

        # Compute derivatives
        first_deriv = sp.diff(expr, x)
        second_deriv = sp.diff(first_deriv, x)

        # Find extrema (where first derivative = 0)
        extrema = []
        try:
            critical_points = sp.solve(first_deriv, x)
            for cp in critical_points:
                if cp.is_real:
                    cp_float = float(cp)
                    y_val = float(expr.subs(x, cp))
                    second_val = float(second_deriv.subs(x, cp))

                    if second_val < 0:
                        extrema.append(Extremum(type='maximum', x=cp_float, y=y_val))
                    elif second_val > 0:
                        extrema.append(Extremum(type='minimum', x=cp_float, y=y_val))
        except Exception:
            pass

        # Find asymptotes
        asymptotes = []
        try:
            # Horizontal asymptotes
            limit_pos = sp.limit(expr, x, sp.oo)
            limit_neg = sp.limit(expr, x, -sp.oo)

            if limit_pos.is_finite:
                asymptotes.append(Asymptote(type='horizontal', value=float(limit_pos)))
            if limit_neg.is_finite and limit_neg != limit_pos:
                asymptotes.append(Asymptote(type='horizontal', value=float(limit_neg)))

            # Vertical asymptotes (where denominator = 0)
            # This is simplified; would need more robust handling
        except Exception:
            pass

        return AnalyticalProperties(
            firstDerivative=str(first_deriv),
            secondDerivative=str(second_deriv),
            extrema=extrema,
            asymptotes=asymptotes
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not analyze expression: {str(e)}")


@app.post("/integrate", response_model=IntegralResult)
def compute_integral(request: IntegralRequest):
    """Compute definite integral of a function"""
    try:
        x = sp.Symbol('x')

        expr_str = request.expression
        expr_str = expr_str.replace('y = ', '').replace('y=', '')
        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace('ln', 'log')

        # For splines, use numerical integration
        if 'Spline' in expr_str:
            raise HTTPException(status_code=400, detail="Cannot integrate spline symbolically")

        expr = parse_expr(expr_str)

        # Try symbolic integration first
        try:
            integral = sp.integrate(expr, (x, request.a, request.b))
            value = float(integral)
            integral_expr = str(sp.integrate(expr, x))
        except Exception:
            # Fall back to numerical integration
            f = sp.lambdify(x, expr, 'numpy')
            value, _ = integrate.quad(f, request.a, request.b)
            integral_expr = "∫f(x)dx (numerical)"

        return IntegralResult(
            value=value,
            expression=f"∫[{request.a}, {request.b}] {expr_str} dx = {integral_expr}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not compute integral: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
