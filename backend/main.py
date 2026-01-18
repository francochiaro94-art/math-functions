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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import warnings
import time
from dataclasses import dataclass, field
from typing import Callable, Any

warnings.filterwarnings('ignore')


# ============================================================================
# CANDIDATE SEARCH ENGINE - Multi-Start Model Selection
# ============================================================================

@dataclass
class CandidateResult:
    """Result from fitting a single candidate seed"""
    family: str
    seed_id: int
    params: dict
    expression: str
    cv_rmse: float
    cv_mae: float
    cv_r2: float
    train_rmse: float
    train_r2: float
    n_params: int
    coverage: float
    eval_func: Callable | None = None
    y_pred: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    status: str = 'valid'  # 'valid', 'rejected', 'failed'
    rejection_reason: str = ''


@dataclass
class FamilyResult:
    """Best result for a model family"""
    family: str
    best_candidate: CandidateResult | None
    seeds_tried: int
    seeds_valid: int
    final_score: float
    status: str  # 'selected', 'rejected', 'skipped', 'failed'
    rejection_reason: str = ''
    top_candidates: list[CandidateResult] = field(default_factory=list)


@dataclass
class SearchTrace:
    """Trace of the entire search process"""
    families: list[FamilyResult]
    selected_family: str
    total_seeds: int
    total_time_ms: float
    early_exit: bool = False
    early_exit_reason: str = ''


class CandidateSearchEngine:
    """
    Multi-start candidate search engine for fair, reliable model selection.

    Tries multiple parameter seeds per model family and selects using
    generalization-based scoring (CV/holdout) with complexity regularization.
    """

    # Default configuration
    DEFAULT_CONFIG = {
        'max_poly_degree': 4,  # Cap at 4 to prevent overfitting
        'min_valid_coverage': 0.9,
        'cv_folds': 3,  # 3-fold CV for speed (5 for large datasets handled adaptively)
        'holdout_repeats': 3,
        'lambda_complexity': 0.25,
        'max_total_seeds': 60,
        'max_total_ms': 3000,  # 3 seconds max for good UX
        'max_iters_per_seed': 50,
        'early_exit_threshold': 0.001,  # If RMSE < this, consider early exit
        'seed_counts': {
            'sinusoidal': 12,  # Reduced for speed while maintaining coverage
            'exponential': 6,
            'logarithmic': 6,
            'log_shifted': 6,
            'reciprocal': 6,
            'sqrt': 4,
            'power': 4,
            'rational': 4,
        }
    }

    def __init__(self, config: dict | None = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.trace: SearchTrace | None = None

    def search(self, x: np.ndarray, y: np.ndarray, objective: str = 'accuracy', forced_family: str | None = None) -> tuple[CandidateResult, SearchTrace]:
        """
        Run multi-start candidate search across all model families, or fit a single forced family.

        Args:
            x: Input values
            y: Output values
            objective: 'accuracy', 'interpretability', or 'balanced'
            forced_family: If provided, only fit this model family (e.g., 'poly2', 'sinusoidal')

        Returns:
            Tuple of (best_candidate, search_trace)
        """
        start_time = time.time()
        n = len(x)

        # Determine scoring strategy based on N
        use_cv = n >= 25

        family_results: list[FamilyResult] = []
        total_seeds = 0

        # Define all model families with their fitters and seed generators
        families = self._get_model_families(x, y, objective)

        # If forced_family is specified, only keep that family
        if forced_family:
            if forced_family not in families:
                raise ValueError(f"Model family '{forced_family}' is not available for this data. Available families: {list(families.keys())}")
            families = {forced_family: families[forced_family]}

        # Track best score for early exit
        best_global_score = float('inf')
        early_exit = False
        early_exit_reason = ''

        for family_name, family_config in families.items():
            # Check time budget
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.config['max_total_ms']:
                family_results.append(FamilyResult(
                    family=family_name,
                    best_candidate=None,
                    seeds_tried=0,
                    seeds_valid=0,
                    final_score=float('inf'),
                    status='skipped',
                    rejection_reason='time budget exceeded'
                ))
                continue

            # Check seed budget
            if total_seeds >= self.config['max_total_seeds']:
                family_results.append(FamilyResult(
                    family=family_name,
                    best_candidate=None,
                    seeds_tried=0,
                    seeds_valid=0,
                    final_score=float('inf'),
                    status='skipped',
                    rejection_reason='seed budget exceeded'
                ))
                continue

            # Run multi-start for this family
            result = self._fit_family(
                family_name, family_config, x, y,
                use_cv=use_cv, objective=objective
            )

            total_seeds += result.seeds_tried
            family_results.append(result)

            # Check for early exit (very good fit found)
            if result.best_candidate and result.final_score < best_global_score:
                best_global_score = result.final_score

                # Early exit if we found an excellent fit
                if best_global_score < self.config['early_exit_threshold']:
                    early_exit = True
                    early_exit_reason = f"Excellent fit found (score={best_global_score:.6f})"
                    break

        # Select the winning family
        valid_families = [f for f in family_results if f.best_candidate is not None]

        if not valid_families:
            raise ValueError("All model families failed to produce valid candidates")

        # Score and select best family
        best_family = min(valid_families, key=lambda f: f.final_score)
        best_family.status = 'selected'

        # Update status for non-selected families
        for f in family_results:
            if f.best_candidate is not None and f != best_family:
                f.status = 'rejected'

        elapsed_ms = (time.time() - start_time) * 1000

        trace = SearchTrace(
            families=family_results,
            selected_family=best_family.family,
            total_seeds=total_seeds,
            total_time_ms=elapsed_ms,
            early_exit=early_exit,
            early_exit_reason=early_exit_reason
        )

        self.trace = trace
        return best_family.best_candidate, trace

    def _get_model_families(self, x: np.ndarray, y: np.ndarray, objective: str) -> dict:
        """Define model families with their configurations"""
        families = {}

        # Polynomial families (stable, no multi-start needed for fitting itself)
        max_degree = min(self.config['max_poly_degree'], 6 if objective == 'accuracy' else 4)
        for deg in range(1, max_degree + 1):
            families[f'poly{deg}'] = {
                'n_params': deg,
                'seed_count': 1,  # Polynomials are convex, single fit is sufficient
                'seeds': [{}],  # No seed parameters needed
                'complexity': deg,
            }

        # Exponential family
        families['exponential'] = {
            'n_params': 3,
            'seed_count': self.config['seed_counts']['exponential'],
            'seeds': self._generate_exponential_seeds(x, y),
            'complexity': 3,
        }

        # Logarithmic families
        if np.all(x > 0):
            families['logarithmic'] = {
                'n_params': 2,
                'seed_count': 1,
                'seeds': [{}],
                'complexity': 2,
            }

        families['log_shifted'] = {
            'n_params': 3,
            'seed_count': self.config['seed_counts']['log_shifted'],
            'seeds': self._generate_log_shifted_seeds(x, y),
            'complexity': 3,
        }

        # Square root family
        families['sqrt_shifted'] = {
            'n_params': 3,
            'seed_count': self.config['seed_counts']['sqrt'],
            'seeds': self._generate_sqrt_seeds(x, y),
            'complexity': 3,
        }

        # Power family
        if np.all(x > 0) and np.all(y > 0):
            families['power'] = {
                'n_params': 2,
                'seed_count': 1,
                'seeds': [{}],
                'complexity': 2,
            }

        # Sinusoidal family (needs multi-start for frequency/phase)
        families['sinusoidal'] = {
            'n_params': 4,
            'seed_count': self.config['seed_counts']['sinusoidal'],
            'seeds': self._generate_sinusoidal_seeds(x, y),
            'complexity': 4,
        }

        # Reciprocal family
        families['reciprocal'] = {
            'n_params': 3,
            'seed_count': self.config['seed_counts']['reciprocal'],
            'seeds': self._generate_reciprocal_seeds(x, y),
            'complexity': 3,
        }

        # Rational family (only for accuracy objective)
        if objective == 'accuracy':
            families['rational'] = {
                'n_params': 4,
                'seed_count': self.config['seed_counts']['rational'],
                'seeds': self._generate_rational_seeds(x, y),
                'complexity': 4,
            }

        return families

    def _generate_exponential_seeds(self, x: np.ndarray, y: np.ndarray) -> list[dict]:
        """Generate seed candidates for exponential: y = a * exp(b*x) + c"""
        y_min, y_max = np.min(y), np.max(y)
        y_mean = np.mean(y)
        y_p10 = np.percentile(y, 10)
        y_p25 = np.percentile(y, 25)
        y_range = abs(y_max - y_min)

        # Various c (vertical shift) candidates
        c_candidates = [
            y_min - y_range * 0.2,
            y_min - y_range * 0.1,
            y_min - 1,
            y_p10 - 1,
            y_p25 - 1,
            0,
            y_mean - y_range,
            -abs(y_max),
        ]

        # b (growth rate) candidates - both growth and decay
        b_candidates = [0.1, 0.5, 1.0, -0.1, -0.5, -1.0]

        seeds = []
        for c in c_candidates:
            for b in b_candidates[:2]:  # Limit combinations
                seeds.append({'c_init': c, 'b_hint': b})
                if len(seeds) >= self.config['seed_counts']['exponential']:
                    break
            if len(seeds) >= self.config['seed_counts']['exponential']:
                break

        return seeds[:self.config['seed_counts']['exponential']]

    def _generate_log_shifted_seeds(self, x: np.ndarray, y: np.ndarray) -> list[dict]:
        """Generate seed candidates for log shifted: y = a * ln(x - c) + d"""
        x_min = np.min(x)
        x_range = np.max(x) - x_min

        # c (horizontal shift) must be < x_min for valid domain
        c_candidates = [
            x_min - x_range * 0.5,
            x_min - x_range * 0.2,
            x_min - x_range * 0.1,
            x_min - 2.0,
            x_min - 1.0,
            x_min - 0.5,
            x_min - 0.1,
            0 if x_min > 1 else x_min - 3.0,
            np.percentile(x, 5) - 1.0,
            np.percentile(x, 10) - 0.5,
        ]

        return [{'c_init': c} for c in c_candidates[:self.config['seed_counts']['log_shifted']]]

    def _generate_sqrt_seeds(self, x: np.ndarray, y: np.ndarray) -> list[dict]:
        """Generate seed candidates for sqrt: y = a * sqrt(x - c) + d"""
        x_min = np.min(x)
        x_range = np.max(x) - x_min

        # c (horizontal shift) must be <= x_min for valid domain
        c_candidates = [
            x_min - x_range * 0.3,
            x_min - x_range * 0.1,
            x_min - 1.0,
            x_min - 0.5,
            x_min - 0.1,
            x_min,
            np.percentile(x, 5) - 0.5,
            0 if x_min > 0.5 else x_min - 2.0,
        ]

        return [{'c_init': c} for c in c_candidates[:self.config['seed_counts']['sqrt']]]

    def _generate_sinusoidal_seeds(self, x: np.ndarray, y: np.ndarray) -> list[dict]:
        """Generate seed candidates for sinusoidal: y = A * sin(B*x + C) + D"""
        x_range = np.max(x) - np.min(x)
        y_min, y_max = np.min(y), np.max(y)

        # Amplitude and offset estimates
        A_est = (y_max - y_min) / 2
        D_est = np.mean(y)

        # Frequency candidates based on plausible periods
        # Period = 2*pi/B, so B = 2*pi/period
        period_fractions = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        B_candidates = [2 * np.pi / (x_range * frac) for frac in period_fractions]

        # Add FFT-based frequency estimate if enough points
        if len(x) >= 8:
            try:
                # Simple periodicity detection via autocorrelation
                y_centered = y - np.mean(y)
                autocorr = np.correlate(y_centered, y_centered, mode='full')
                autocorr = autocorr[len(autocorr)//2:]

                # Find first significant peak after zero
                peaks = []
                for i in range(1, len(autocorr) - 1):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        if autocorr[i] > 0.1 * autocorr[0]:
                            peaks.append(i)

                if peaks:
                    # Estimate period from first peak
                    period_idx = peaks[0]
                    estimated_period = (x_range / len(x)) * period_idx
                    if estimated_period > 0:
                        B_fft = 2 * np.pi / estimated_period
                        B_candidates.insert(0, B_fft)
            except Exception:
                pass

        # Phase candidates
        C_candidates = [0, np.pi/4, np.pi/2, np.pi, -np.pi/4, -np.pi/2]

        seeds = []
        for B in B_candidates:
            for C in C_candidates[:3]:  # Limit combinations
                seeds.append({
                    'A_init': A_est,
                    'B_init': B,
                    'C_init': C,
                    'D_init': D_est
                })
                if len(seeds) >= self.config['seed_counts']['sinusoidal']:
                    break
            if len(seeds) >= self.config['seed_counts']['sinusoidal']:
                break

        return seeds[:self.config['seed_counts']['sinusoidal']]

    def _generate_reciprocal_seeds(self, x: np.ndarray, y: np.ndarray) -> list[dict]:
        """Generate seed candidates for reciprocal: y = a/(x - c) + d"""
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min

        # Estimate c from gradient analysis
        x_sorted_idx = np.argsort(x)
        x_sorted = x[x_sorted_idx]
        y_sorted = y[x_sorted_idx]

        c_estimates = []

        # From maximum gradient
        if len(x_sorted) > 2:
            dy = np.diff(y_sorted)
            dx = np.diff(x_sorted)
            with np.errstate(divide='ignore', invalid='ignore'):
                gradients = np.abs(dy / dx)
            gradients = np.nan_to_num(gradients, nan=0)
            max_grad_idx = np.argmax(gradients)
            c_estimates.append((x_sorted[max_grad_idx] + x_sorted[max_grad_idx + 1]) / 2)

        # Other candidates
        c_candidates = [
            *c_estimates,
            x_min - x_range * 0.1,
            x_max + x_range * 0.1,
            x_min - 0.5,
            x_max + 0.5,
            0.0,
            np.mean(x),
            np.median(x),
        ]

        # Remove duplicates and limit
        seen = set()
        unique_seeds = []
        for c in c_candidates:
            c_rounded = round(c, 4)
            if c_rounded not in seen:
                seen.add(c_rounded)
                unique_seeds.append({'c_init': c})

        return unique_seeds[:self.config['seed_counts']['reciprocal']]

    def _generate_rational_seeds(self, x: np.ndarray, y: np.ndarray) -> list[dict]:
        """Generate seed candidates for rational: y = (ax + b) / (cx + d)"""
        # Simple initial guesses
        seeds = [
            {'a': 1, 'b': 1, 'c': 1, 'd': 1},
            {'a': 1, 'b': 0, 'c': 0, 'd': 1},
            {'a': 0, 'b': 1, 'c': 1, 'd': 0},
            {'a': 1, 'b': -1, 'c': 1, 'd': 1},
            {'a': -1, 'b': 1, 'c': 1, 'd': -1},
            {'a': 2, 'b': 1, 'c': 1, 'd': 2},
        ]
        return seeds[:self.config['seed_counts']['rational']]

    def _fit_family(
        self, family_name: str, config: dict,
        x: np.ndarray, y: np.ndarray,
        use_cv: bool, objective: str
    ) -> FamilyResult:
        """Fit all seed candidates for a model family and select the best"""
        seeds = config['seeds']
        n_params = config['n_params']
        complexity = config['complexity']

        candidates: list[CandidateResult] = []
        seeds_tried = 0
        seeds_valid = 0

        for seed_id, seed in enumerate(seeds):
            seeds_tried += 1

            try:
                # Fit this seed
                result = self._fit_single_seed(
                    family_name, seed_id, seed, n_params, complexity,
                    x, y, use_cv
                )

                if result.status == 'valid':
                    seeds_valid += 1
                    candidates.append(result)

            except Exception as e:
                # Record failed seed
                candidates.append(CandidateResult(
                    family=family_name,
                    seed_id=seed_id,
                    params=seed,
                    expression='',
                    cv_rmse=float('inf'),
                    cv_mae=float('inf'),
                    cv_r2=-999,
                    train_rmse=float('inf'),
                    train_r2=-999,
                    n_params=n_params,
                    coverage=0,
                    status='failed',
                    rejection_reason=str(e)[:100]
                ))

        # Select best candidate for this family
        valid_candidates = [c for c in candidates if c.status == 'valid']

        if not valid_candidates:
            return FamilyResult(
                family=family_name,
                best_candidate=None,
                seeds_tried=seeds_tried,
                seeds_valid=0,
                final_score=float('inf'),
                status='failed',
                rejection_reason='No valid candidates'
            )

        # Compute final score for each candidate
        n = len(x)
        lambda_c = self.config['lambda_complexity']

        for c in valid_candidates:
            # FinalScore = CV_RMSE + lambda*(k/N)
            c.info['final_score'] = c.cv_rmse + lambda_c * (c.n_params / n)

        # Sort by final score
        valid_candidates.sort(key=lambda c: c.info['final_score'])
        best = valid_candidates[0]

        return FamilyResult(
            family=family_name,
            best_candidate=best,
            seeds_tried=seeds_tried,
            seeds_valid=seeds_valid,
            final_score=best.info['final_score'],
            status='candidate',
            top_candidates=valid_candidates[:3]  # Keep top 3 for debugging
        )

    def _fit_single_seed(
        self, family: str, seed_id: int, seed: dict,
        n_params: int, complexity: int,
        x: np.ndarray, y: np.ndarray, use_cv: bool
    ) -> CandidateResult:
        """Fit a single seed and compute metrics"""

        # Get the appropriate fitting function based on family
        fit_result = self._call_fitter(family, x, y, seed)

        if fit_result is None:
            raise ValueError(f"Fitter returned None for {family}")

        expr, y_pred, info = fit_result

        # Check coverage
        valid_mask = np.isfinite(y_pred)
        coverage = np.sum(valid_mask) / len(y)

        if coverage < self.config['min_valid_coverage']:
            return CandidateResult(
                family=family,
                seed_id=seed_id,
                params=seed,
                expression=expr,
                cv_rmse=float('inf'),
                cv_mae=float('inf'),
                cv_r2=-999,
                train_rmse=float('inf'),
                train_r2=-999,
                n_params=n_params,
                coverage=coverage,
                status='rejected',
                rejection_reason=f'Insufficient coverage: {coverage*100:.1f}%'
            )

        # Compute training metrics
        y_valid = y[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        train_rmse = np.sqrt(np.mean((y_valid - y_pred_valid) ** 2))
        train_r2 = 1 - np.sum((y_valid - y_pred_valid)**2) / np.sum((y_valid - np.mean(y_valid))**2)

        # Compute CV/holdout metrics
        cv_rmse, cv_mae, cv_r2 = self._compute_cv_metrics(
            family, x, y, seed, use_cv
        )

        return CandidateResult(
            family=family,
            seed_id=seed_id,
            params=seed,
            expression=expr,
            cv_rmse=cv_rmse,
            cv_mae=cv_mae,
            cv_r2=cv_r2,
            train_rmse=train_rmse,
            train_r2=train_r2,
            n_params=n_params,
            coverage=coverage,
            eval_func=info.get('eval_func'),
            y_pred=y_pred,
            info=info,
            status='valid'
        )

    def _call_fitter(self, family: str, x: np.ndarray, y: np.ndarray, seed: dict):
        """Call the appropriate fitter for a model family with seed parameters"""

        # Import fitters (they're defined later in the file, but we access them via globals)
        # This is a workaround since the fitters are defined at module level

        if family.startswith('poly'):
            degree = int(family[4:])
            return fit_polynomial_with_seed(x, y, degree)
        elif family == 'exponential':
            return fit_exponential_with_seed(x, y, seed)
        elif family == 'logarithmic':
            return fit_logarithmic_basic(x, y)
        elif family == 'log_shifted':
            return fit_log_shifted_with_seed(x, y, seed)
        elif family == 'sqrt_shifted':
            return fit_sqrt_with_seed(x, y, seed)
        elif family == 'power':
            return fit_power_basic(x, y)
        elif family == 'sinusoidal':
            return fit_sinusoidal_with_seed(x, y, seed)
        elif family == 'reciprocal':
            return fit_reciprocal_with_seed(x, y, seed)
        elif family == 'rational':
            return fit_rational_with_seed(x, y, seed)
        else:
            raise ValueError(f"Unknown family: {family}")

    def _compute_cv_metrics(
        self, family: str, x: np.ndarray, y: np.ndarray,
        seed: dict, use_cv: bool
    ) -> tuple[float, float, float]:
        """Compute cross-validation or holdout metrics"""
        n = len(x)

        if use_cv:
            # K-fold cross-validation
            n_folds = self.config['cv_folds']
            return self._kfold_cv(family, x, y, seed, n_folds)
        else:
            # Repeated holdout for small datasets
            n_repeats = self.config['holdout_repeats']
            return self._repeated_holdout(family, x, y, seed, n_repeats)

    def _kfold_cv(
        self, family: str, x: np.ndarray, y: np.ndarray,
        seed: dict, n_folds: int
    ) -> tuple[float, float, float]:
        """K-fold cross-validation"""
        n = len(x)
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)

        fold_size = n // n_folds
        rmse_scores, mae_scores, r2_scores = [], [], []

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n

            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            if len(train_idx) < 2 or len(val_idx) < 1:
                continue

            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            try:
                result = self._call_fitter(family, x_train, y_train, seed)
                if result is None:
                    continue

                _, _, info = result
                eval_func = info.get('eval_func')

                if eval_func:
                    y_pred_val = eval_func(x_val)
                else:
                    continue

                valid_mask = np.isfinite(y_pred_val) & np.isfinite(y_val)
                if np.sum(valid_mask) < 1:
                    continue

                y_v = y_val[valid_mask]
                y_p = y_pred_val[valid_mask]

                rmse_scores.append(np.sqrt(np.mean((y_v - y_p) ** 2)))
                mae_scores.append(np.mean(np.abs(y_v - y_p)))

                ss_res = np.sum((y_v - y_p) ** 2)
                ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                r2_scores.append(r2)

            except Exception:
                continue

        if not rmse_scores:
            return float('inf'), float('inf'), -999

        return np.mean(rmse_scores), np.mean(mae_scores), np.mean(r2_scores)

    def _repeated_holdout(
        self, family: str, x: np.ndarray, y: np.ndarray,
        seed: dict, n_repeats: int
    ) -> tuple[float, float, float]:
        """Repeated holdout validation (80/20 split)"""
        n = len(x)
        rmse_scores, mae_scores, r2_scores = [], [], []

        for rep in range(n_repeats):
            np.random.seed(42 + rep)
            indices = np.arange(n)
            np.random.shuffle(indices)

            split_point = int(0.8 * n)
            train_idx = indices[:split_point]
            val_idx = indices[split_point:]

            if len(train_idx) < 2 or len(val_idx) < 1:
                continue

            x_train, y_train = x[train_idx], y[train_idx]
            x_val, y_val = x[val_idx], y[val_idx]

            try:
                result = self._call_fitter(family, x_train, y_train, seed)
                if result is None:
                    continue

                _, _, info = result
                eval_func = info.get('eval_func')

                if eval_func:
                    y_pred_val = eval_func(x_val)
                else:
                    continue

                valid_mask = np.isfinite(y_pred_val) & np.isfinite(y_val)
                if np.sum(valid_mask) < 1:
                    continue

                y_v = y_val[valid_mask]
                y_p = y_pred_val[valid_mask]

                rmse_scores.append(np.sqrt(np.mean((y_v - y_p) ** 2)))
                mae_scores.append(np.mean(np.abs(y_v - y_p)))

                ss_res = np.sum((y_v - y_p) ** 2)
                ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                r2_scores.append(r2)

            except Exception:
                continue

        if not rmse_scores:
            return float('inf'), float('inf'), -999

        return np.mean(rmse_scores), np.mean(mae_scores), np.mean(r2_scores)


# ============================================================================
# SEED-AWARE FITTING FUNCTIONS
# These wrap the original fitters to accept seed parameters
# ============================================================================

def fit_polynomial_with_seed(x: np.ndarray, y: np.ndarray, degree: int):
    """Polynomial fitting (wrapper for consistency)"""
    if degree == 1:
        return fit_linear(x, y)
    return fit_polynomial(x, y, degree)


def fit_exponential_with_seed(x: np.ndarray, y: np.ndarray, seed: dict):
    """Exponential fitting with seed hint for c"""
    MAX_EXP_ARG = 700

    def safe_exp(arg):
        with np.errstate(over='ignore'):
            result = np.where(
                np.abs(arg) > MAX_EXP_ARG,
                np.nan,
                np.exp(np.clip(arg, -MAX_EXP_ARG, MAX_EXP_ARG))
            )
        return result

    def exponential_func(x, a, b, c):
        return a * safe_exp(b * x) + c

    c_init = seed.get('c_init', np.min(y) - 1)

    try:
        y_shifted = y - c_init
        valid_mask = y_shifted > 0
        if np.sum(valid_mask) < 3:
            raise ValueError("Not enough positive values")

        x_valid = x[valid_mask]
        y_valid = y_shifted[valid_mask]

        log_y = np.log(y_valid)
        model = LinearRegression()
        model.fit(x_valid.reshape(-1, 1), log_y)

        b_init = seed.get('b_hint', model.coef_[0])
        a_init = np.exp(model.intercept_)

        popt, _ = optimize.curve_fit(
            exponential_func, x, y,
            p0=[a_init, b_init, c_init],
            maxfev=2000,
            bounds=([-np.inf, -10, -np.inf], [np.inf, 10, np.inf])
        )

        a, b, c = popt
        y_pred = exponential_func(x, a, b, c)

        c_sign = "+" if c >= 0 else "-"
        c_val = abs(c)
        expr = f"y = {a:.4g} * exp({b:.4g}x) {c_sign} {c_val:.4g}"

        def eval_func(x_new):
            return exponential_func(x_new, a, b, c)

        return expr, y_pred, {
            'type': 'Exponential',
            'complexity': 3,
            'params': {'a': a, 'b': b, 'c': c},
            'eval_func': eval_func
        }
    except Exception:
        raise ValueError("Exponential fit failed")


def fit_logarithmic_basic(x: np.ndarray, y: np.ndarray):
    """Basic logarithmic fitting: y = a * ln(x) + b"""
    if np.any(x <= 0):
        raise ValueError("Log requires positive x")

    log_x = np.log(x)
    model = LinearRegression()
    model.fit(log_x.reshape(-1, 1), y)

    a, b = model.coef_[0], model.intercept_
    y_pred = a * np.log(x) + b

    def eval_func(x_new):
        x_arr = np.atleast_1d(x_new)
        return np.where(x_arr > 0, a * np.log(x_arr) + b, np.nan)

    expr = f"y = {a:.6g} * ln(x) + {b:.6g}" if b >= 0 else f"y = {a:.6g} * ln(x) - {abs(b):.6g}"
    return expr, y_pred, {'type': 'Logarithmic', 'complexity': 2, 'eval_func': eval_func}


def fit_log_shifted_with_seed(x: np.ndarray, y: np.ndarray, seed: dict):
    """Shifted log fitting with seed hint for c"""
    EPSILON = 1e-9
    c_init = seed.get('c_init', np.min(x) - 1)

    def log_func(x, a, c, d):
        arg = x - c
        with np.errstate(invalid='ignore', divide='ignore'):
            return a * np.log(np.maximum(arg, EPSILON)) + d

    def log_safe(x, a, c, d):
        arg = x - c
        return np.where(arg > EPSILON, a * np.log(arg) + d, np.nan)

    try:
        mask_valid = (x - c_init) > EPSILON
        if np.sum(mask_valid) < 3:
            raise ValueError("Not enough valid points")

        x_fit = x[mask_valid]
        y_fit = y[mask_valid]

        log_x = np.log(x_fit - c_init)
        if np.any(~np.isfinite(log_x)):
            raise ValueError("Invalid log transform")

        model = LinearRegression()
        model.fit(log_x.reshape(-1, 1), y_fit)
        a0, d0 = model.coef_[0], model.intercept_

        popt, _ = optimize.curve_fit(
            log_func, x_fit, y_fit,
            p0=[a0, c_init, d0],
            maxfev=2000,
            bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.min(x) - EPSILON, np.inf])
        )

        a, c, d = popt
        y_pred = log_safe(x, a, c, d)

        c_sign = "-" if c >= 0 else "+"
        c_val = abs(c)
        d_sign = "+" if d >= 0 else "-"
        d_val = abs(d)

        expr = f"y = {a:.4g} * ln(x {c_sign} {c_val:.4g}) {d_sign} {d_val:.4g}"

        def eval_func(x_new):
            return log_safe(x_new, a, c, d)

        return expr, y_pred, {
            'type': 'Logarithmic (Shifted)',
            'complexity': 3,
            'domain_boundary': c,
            'params': {'a': a, 'c': c, 'd': d},
            'eval_func': eval_func
        }
    except Exception:
        raise ValueError("Log shifted fit failed")


def fit_sqrt_with_seed(x: np.ndarray, y: np.ndarray, seed: dict):
    """Sqrt fitting with seed hint for c"""
    EPSILON = 1e-9
    c_init = seed.get('c_init', np.min(x) - 0.1)

    def sqrt_func(x, a, c, d):
        arg = x - c
        with np.errstate(invalid='ignore'):
            return a * np.sqrt(np.maximum(arg, 0)) + d

    def sqrt_safe(x, a, c, d):
        arg = x - c
        return np.where(arg >= -EPSILON, a * np.sqrt(np.maximum(arg, 0)) + d, np.nan)

    try:
        mask_valid = (x - c_init) >= -EPSILON
        if np.sum(mask_valid) < 3:
            raise ValueError("Not enough valid points")

        x_fit = x[mask_valid]
        y_fit = y[mask_valid]

        sqrt_x = np.sqrt(np.maximum(x_fit - c_init, EPSILON))
        if np.any(~np.isfinite(sqrt_x)):
            raise ValueError("Invalid sqrt transform")

        a0 = (y_fit.max() - y_fit.min()) / (sqrt_x.max() - sqrt_x.min() + EPSILON)
        d0 = y_fit.min()

        popt, _ = optimize.curve_fit(
            sqrt_func, x_fit, y_fit,
            p0=[a0, c_init, d0],
            maxfev=2000,
            bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.min(x) + EPSILON, np.inf])
        )

        a, c, d = popt
        y_pred = sqrt_safe(x, a, c, d)

        c_sign = "-" if c >= 0 else "+"
        c_val = abs(c)
        d_sign = "+" if d >= 0 else "-"
        d_val = abs(d)

        expr = f"y = {a:.4g} * sqrt(x {c_sign} {c_val:.4g}) {d_sign} {d_val:.4g}"

        def eval_func(x_new):
            return sqrt_safe(x_new, a, c, d)

        return expr, y_pred, {
            'type': 'Square Root',
            'complexity': 3,
            'domain_boundary': c,
            'params': {'a': a, 'c': c, 'd': d},
            'eval_func': eval_func
        }
    except Exception:
        raise ValueError("Sqrt fit failed")


def fit_power_basic(x: np.ndarray, y: np.ndarray):
    """Power fitting: y = a * x^b"""
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("Power requires positive x and y")

    log_x = np.log(x)
    log_y = np.log(y)
    model = LinearRegression()
    model.fit(log_x.reshape(-1, 1), log_y)

    b_exp = model.coef_[0]
    a_coef = np.exp(model.intercept_)

    y_pred = a_coef * np.power(x, b_exp)

    def eval_func(x_new):
        x_arr = np.atleast_1d(x_new)
        return np.where(x_arr > 0, a_coef * np.power(x_arr, b_exp), np.nan)

    # Generate expression with proper formatting for LaTeX
    # Use braces around exponent for proper rendering: x^{b}
    expr = f"y = {a_coef:.6g} * x^{{{b_exp:.6g}}}"
    return expr, y_pred, {
        'type': 'Power',
        'complexity': 2,
        'eval_func': eval_func,
        'params': {'a': a_coef, 'b': b_exp}
    }


def fit_sinusoidal_with_seed(x: np.ndarray, y: np.ndarray, seed: dict):
    """Sinusoidal fitting with multi-start seeds"""
    def sin_func(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    A_init = seed.get('A_init', (np.max(y) - np.min(y)) / 2)
    B_init = seed.get('B_init', 2 * np.pi / (np.max(x) - np.min(x)))
    C_init = seed.get('C_init', 0)
    D_init = seed.get('D_init', np.mean(y))

    try:
        popt, _ = optimize.curve_fit(
            sin_func, x, y,
            p0=[A_init, B_init, C_init, D_init],
            maxfev=2000,
            bounds=(
                [-np.inf, 0.001, -2*np.pi, -np.inf],
                [np.inf, 100, 2*np.pi, np.inf]
            )
        )

        A, B, C, D = popt
        y_pred = sin_func(x, *popt)

        def eval_func(x_new):
            return A * np.sin(B * x_new + C) + D

        expr = f"y = {A:.4g} * sin({B:.4g}x + {C:.4g}) + {D:.4g}"
        return expr, y_pred, {
            'type': 'Sinusoidal',
            'complexity': 4,
            'params': {'A': A, 'B': B, 'C': C, 'D': D},
            'eval_func': eval_func
        }
    except Exception:
        raise ValueError("Sinusoidal fit failed")


def fit_reciprocal_with_seed(x: np.ndarray, y: np.ndarray, seed: dict):
    """Reciprocal fitting with seed hint for c"""
    EPSILON = 1e-10
    c_init = seed.get('c_init', 0)

    def reciprocal_func(x, a, c, d):
        denom = x - c
        with np.errstate(divide='ignore', invalid='ignore'):
            return a / denom + d

    def reciprocal_safe(x, a, c, d):
        denom = x - c
        return np.where(np.abs(denom) < EPSILON, np.nan, a / denom + d)

    try:
        mask_valid = np.abs(x - c_init) > EPSILON * 1000
        if np.sum(mask_valid) < 3:
            raise ValueError("Not enough valid points")

        x_fit = x[mask_valid]
        y_fit = y[mask_valid]

        # Estimate d from far-from-asymptote values
        mask_far = np.abs(x - c_init) > (np.max(x) - np.min(x)) / 4
        d0 = np.median(y[mask_far]) if np.sum(mask_far) > 0 else np.median(y)

        # Estimate a
        x_far = x[mask_far] if np.sum(mask_far) > 0 else x
        y_far = y[mask_far] if np.sum(mask_far) > 0 else y
        if len(x_far) > 0:
            idx = len(x_far) // 2
            a0 = (y_far[idx] - d0) * (x_far[idx] - c_init)
        else:
            a0 = 1.0

        popt, _ = optimize.curve_fit(
            reciprocal_func, x_fit, y_fit,
            p0=[a0, c_init, d0],
            maxfev=2000
        )

        a, c, d = popt
        y_pred = reciprocal_safe(x, a, c, d)

        c_sign = "-" if c >= 0 else "+"
        c_val = abs(c)
        d_sign = "+" if d >= 0 else "-"
        d_val = abs(d)

        expr = f"y = {a:.4g}/(x {c_sign} {c_val:.4g}) {d_sign} {d_val:.4g}"

        def eval_func(x_new):
            return reciprocal_safe(x_new, a, c, d)

        return expr, y_pred, {
            'type': 'Reciprocal',
            'complexity': 3,
            'asymptote_x': c,
            'params': {'a': a, 'c': c, 'd': d},
            'eval_func': eval_func
        }
    except Exception:
        raise ValueError("Reciprocal fit failed")


def fit_rational_with_seed(x: np.ndarray, y: np.ndarray, seed: dict):
    """Rational fitting with seed parameters"""
    def rational_func(x, a, b, c, d):
        return (a * x + b) / (c * x + d)

    a0 = seed.get('a', 1)
    b0 = seed.get('b', 1)
    c0 = seed.get('c', 1)
    d0 = seed.get('d', 1)

    try:
        popt, _ = optimize.curve_fit(
            rational_func, x, y,
            p0=[a0, b0, c0, d0],
            maxfev=2000
        )

        a, b, c, d = popt
        y_pred = rational_func(x, *popt)

        def eval_func(x_new):
            x_arr = np.atleast_1d(x_new)
            denom = c * x_arr + d
            return np.where(np.abs(denom) > 1e-10, (a * x_arr + b) / denom, np.nan)

        expr = f"y = ({a:.4g}x + {b:.4g}) / ({c:.4g}x + {d:.4g})"
        return expr, y_pred, {
            'type': 'Rational',
            'complexity': 4,
            'params': {'a': a, 'b': b, 'c': c, 'd': d},
            'eval_func': eval_func
        }
    except Exception:
        raise ValueError("Rational fit failed")

app = FastAPI(
    title="Curve Fitting Lab API",
    description="Backend for interactive curve fitting and function analysis",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
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
    selectedModel: Optional[str] = 'auto'  # 'auto' or a model family ID


class FitStatistics(BaseModel):
    r2: float
    rmse: float
    mae: float
    aic: Optional[float] = None
    bic: Optional[float] = None


class ModelParameter(BaseModel):
    name: str
    value: float
    label: str
    hint: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None


class ModelParameterSchema(BaseModel):
    modelFamily: str
    expressionTemplate: str
    parameters: list[ModelParameter]


class FitResult(BaseModel):
    expression: str
    expressionLatex: str
    statistics: FitStatistics
    quality: Literal['bad', 'regular', 'good']
    curvePoints: list[Point]
    modelType: str
    heuristics: list[str]
    parameterSchema: Optional[ModelParameterSchema] = None
    mode: Literal['auto', 'forced'] = 'auto'  # Whether the model was auto-selected or forced


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


class EvaluateRequest(BaseModel):
    modelFamily: str
    parameters: dict[str, float]
    points: list[Point]
    xRange: Optional[tuple[float, float]] = None


class EvaluateResult(BaseModel):
    expression: str
    expressionLatex: str
    statistics: FitStatistics
    quality: Literal['bad', 'regular', 'good']
    curvePoints: list[Point]
    valid: bool
    error: Optional[str] = None


class ModelInfo(BaseModel):
    modelId: str
    displayName: str
    parameterCount: int
    supportsEditing: bool = True
    domain: Optional[str] = None  # e.g., "x > 0" for log/sqrt


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


# ============================================================================
# PARAMETER SCHEMA GENERATOR
# ============================================================================

def get_parameter_schema(model_type: str, params: dict) -> ModelParameterSchema:
    """Generate parameter schema for a model type with current parameter values"""

    SCHEMA_REGISTRY = {
        'Linear': {
            'expressionTemplate': 'y = {m}x + {b}',
            'parameters': [
                {'name': 'm', 'label': 'Slope', 'hint': 'Rate of change', 'step': 0.1},
                {'name': 'b', 'label': 'Intercept', 'hint': 'Y-intercept', 'step': 0.1},
            ]
        },
        'Polynomial (degree 2)': {
            'expressionTemplate': 'y = {a2}x² + {a1}x + {a0}',
            'parameters': [
                {'name': 'a2', 'label': 'a₂', 'hint': 'Quadratic coefficient', 'step': 0.01},
                {'name': 'a1', 'label': 'a₁', 'hint': 'Linear coefficient', 'step': 0.1},
                {'name': 'a0', 'label': 'a₀', 'hint': 'Constant term', 'step': 0.1},
            ]
        },
        'Polynomial (degree 3)': {
            'expressionTemplate': 'y = {a3}x³ + {a2}x² + {a1}x + {a0}',
            'parameters': [
                {'name': 'a3', 'label': 'a₃', 'hint': 'Cubic coefficient', 'step': 0.001},
                {'name': 'a2', 'label': 'a₂', 'hint': 'Quadratic coefficient', 'step': 0.01},
                {'name': 'a1', 'label': 'a₁', 'hint': 'Linear coefficient', 'step': 0.1},
                {'name': 'a0', 'label': 'a₀', 'hint': 'Constant term', 'step': 0.1},
            ]
        },
        'Polynomial (degree 4)': {
            'expressionTemplate': 'y = {a4}x⁴ + {a3}x³ + {a2}x² + {a1}x + {a0}',
            'parameters': [
                {'name': 'a4', 'label': 'a₄', 'hint': 'Quartic coefficient', 'step': 0.0001},
                {'name': 'a3', 'label': 'a₃', 'hint': 'Cubic coefficient', 'step': 0.001},
                {'name': 'a2', 'label': 'a₂', 'hint': 'Quadratic coefficient', 'step': 0.01},
                {'name': 'a1', 'label': 'a₁', 'hint': 'Linear coefficient', 'step': 0.1},
                {'name': 'a0', 'label': 'a₀', 'hint': 'Constant term', 'step': 0.1},
            ]
        },
        'Exponential': {
            'expressionTemplate': 'y = {a} × exp({b}x) + {c}',
            'parameters': [
                {'name': 'a', 'label': 'Amplitude', 'hint': 'Scaling factor', 'step': 0.1},
                {'name': 'b', 'label': 'Growth Rate', 'hint': 'Exponential rate (+ growth, - decay)', 'step': 0.01, 'min': -10, 'max': 10},
                {'name': 'c', 'label': 'Vertical Shift', 'hint': 'Asymptotic value', 'step': 0.1},
            ]
        },
        'Logarithmic': {
            'expressionTemplate': 'y = {a} × ln(x) + {b}',
            'parameters': [
                {'name': 'a', 'label': 'Scale', 'hint': 'Logarithm coefficient', 'step': 0.1},
                {'name': 'b', 'label': 'Vertical Shift', 'hint': 'Y-offset', 'step': 0.1},
            ]
        },
        'Logarithmic (Shifted)': {
            'expressionTemplate': 'y = {a} × ln(x - {c}) + {d}',
            'parameters': [
                {'name': 'a', 'label': 'Scale', 'hint': 'Logarithm coefficient', 'step': 0.1},
                {'name': 'c', 'label': 'Domain Shift', 'hint': 'Horizontal shift (domain starts at x > c)', 'step': 0.1},
                {'name': 'd', 'label': 'Vertical Shift', 'hint': 'Y-offset', 'step': 0.1},
            ]
        },
        'Square Root': {
            'expressionTemplate': 'y = {a} × √(x - {c}) + {d}',
            'parameters': [
                {'name': 'a', 'label': 'Scale', 'hint': 'Coefficient', 'step': 0.1},
                {'name': 'c', 'label': 'Domain Shift', 'hint': 'Horizontal shift (domain starts at x ≥ c)', 'step': 0.1},
                {'name': 'd', 'label': 'Vertical Shift', 'hint': 'Y-offset', 'step': 0.1},
            ]
        },
        'Power': {
            'expressionTemplate': 'y = {a} × x^{b}',
            'parameters': [
                {'name': 'a', 'label': 'Coefficient', 'hint': 'Scaling factor', 'step': 0.1},
                {'name': 'b', 'label': 'Exponent', 'hint': 'Power', 'step': 0.1},
            ]
        },
        'Sinusoidal': {
            'expressionTemplate': 'y = {A} × sin({B}x + {C}) + {D}',
            'parameters': [
                {'name': 'A', 'label': 'Amplitude', 'hint': 'Peak deviation from center', 'step': 0.1},
                {'name': 'B', 'label': 'Frequency', 'hint': 'Angular frequency (period = 2π/B)', 'step': 0.01, 'min': 0.001},
                {'name': 'C', 'label': 'Phase', 'hint': 'Horizontal shift', 'step': 0.1},
                {'name': 'D', 'label': 'Vertical Shift', 'hint': 'Center line', 'step': 0.1},
            ]
        },
        'Reciprocal': {
            'expressionTemplate': 'y = {a}/(x - {c}) + {d}',
            'parameters': [
                {'name': 'a', 'label': 'Numerator', 'hint': 'Scaling factor', 'step': 0.1},
                {'name': 'c', 'label': 'Asymptote', 'hint': 'Vertical asymptote at x = c', 'step': 0.1},
                {'name': 'd', 'label': 'Horizontal Asymptote', 'hint': 'Value as x → ±∞', 'step': 0.1},
            ]
        },
        'Rational': {
            'expressionTemplate': 'y = ({a}x + {b})/({c}x + {d})',
            'parameters': [
                {'name': 'a', 'label': 'a', 'hint': 'Numerator x coefficient', 'step': 0.1},
                {'name': 'b', 'label': 'b', 'hint': 'Numerator constant', 'step': 0.1},
                {'name': 'c', 'label': 'c', 'hint': 'Denominator x coefficient', 'step': 0.1},
                {'name': 'd', 'label': 'd', 'hint': 'Denominator constant', 'step': 0.1},
            ]
        },
    }

    schema_def = SCHEMA_REGISTRY.get(model_type)
    if not schema_def:
        return None

    # Build parameters with current values
    parameters = []
    for param_def in schema_def['parameters']:
        param_name = param_def['name']
        param_value = params.get(param_name, 0.0)
        parameters.append(ModelParameter(
            name=param_name,
            value=param_value,
            label=param_def['label'],
            hint=param_def.get('hint'),
            min=param_def.get('min'),
            max=param_def.get('max'),
            step=param_def.get('step', 0.1),
        ))

    return ModelParameterSchema(
        modelFamily=model_type,
        expressionTemplate=schema_def['expressionTemplate'],
        parameters=parameters
    )


def evaluate_with_params(model_family: str, params: dict, x: np.ndarray) -> tuple[np.ndarray, str, bool, str]:
    """Evaluate a model with given parameters and return predictions, expression, validity, and error message"""
    MAX_EXP_ARG = 700
    EPSILON = 1e-9

    try:
        if model_family == 'Linear':
            m, b = params.get('m', 1), params.get('b', 0)
            y = m * x + b
            b_sign = '+' if b >= 0 else '-'
            expr = f"y = {m:.4g}x {b_sign} {abs(b):.4g}"
            return y, expr, True, ''

        elif model_family.startswith('Polynomial'):
            if 'degree 2' in model_family:
                a2, a1, a0 = params.get('a2', 0), params.get('a1', 0), params.get('a0', 0)
                y = a2 * x**2 + a1 * x + a0
                expr = f"y = {a2:.4g}x² + {a1:.4g}x + {a0:.4g}"
            elif 'degree 3' in model_family:
                a3, a2, a1, a0 = params.get('a3', 0), params.get('a2', 0), params.get('a1', 0), params.get('a0', 0)
                y = a3 * x**3 + a2 * x**2 + a1 * x + a0
                expr = f"y = {a3:.4g}x³ + {a2:.4g}x² + {a1:.4g}x + {a0:.4g}"
            elif 'degree 4' in model_family:
                a4, a3, a2, a1, a0 = params.get('a4', 0), params.get('a3', 0), params.get('a2', 0), params.get('a1', 0), params.get('a0', 0)
                y = a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0
                expr = f"y = {a4:.4g}x⁴ + {a3:.4g}x³ + {a2:.4g}x² + {a1:.4g}x + {a0:.4g}"
            else:
                return np.zeros_like(x), '', False, 'Unsupported polynomial degree'
            return y, expr, True, ''

        elif model_family == 'Exponential':
            a, b, c = params.get('a', 1), params.get('b', 1), params.get('c', 0)
            with np.errstate(over='ignore'):
                exp_arg = b * x
                y = np.where(
                    np.abs(exp_arg) > MAX_EXP_ARG,
                    np.nan,
                    a * np.exp(np.clip(exp_arg, -MAX_EXP_ARG, MAX_EXP_ARG)) + c
                )
            c_sign = '+' if c >= 0 else '-'
            expr = f"y = {a:.4g} * exp({b:.4g}x) {c_sign} {abs(c):.4g}"
            return y, expr, True, ''

        elif model_family == 'Logarithmic':
            a, b = params.get('a', 1), params.get('b', 0)
            y = np.where(x > EPSILON, a * np.log(x) + b, np.nan)
            b_sign = '+' if b >= 0 else '-'
            expr = f"y = {a:.4g} * ln(x) {b_sign} {abs(b):.4g}"
            return y, expr, True, ''

        elif model_family == 'Logarithmic (Shifted)':
            a, c, d = params.get('a', 1), params.get('c', 0), params.get('d', 0)
            arg = x - c
            y = np.where(arg > EPSILON, a * np.log(arg) + d, np.nan)
            c_sign = '-' if c >= 0 else '+'
            d_sign = '+' if d >= 0 else '-'
            expr = f"y = {a:.4g} * ln(x {c_sign} {abs(c):.4g}) {d_sign} {abs(d):.4g}"
            return y, expr, True, ''

        elif model_family == 'Square Root':
            a, c, d = params.get('a', 1), params.get('c', 0), params.get('d', 0)
            arg = x - c
            y = np.where(arg >= -EPSILON, a * np.sqrt(np.maximum(arg, 0)) + d, np.nan)
            c_sign = '-' if c >= 0 else '+'
            d_sign = '+' if d >= 0 else '-'
            expr = f"y = {a:.4g} * sqrt(x {c_sign} {abs(c):.4g}) {d_sign} {abs(d):.4g}"
            return y, expr, True, ''

        elif model_family == 'Power':
            a, b = params.get('a', 1), params.get('b', 1)
            y = np.where(x > 0, a * np.power(x, b), np.nan)
            # Use braces around exponent for proper LaTeX rendering
            expr = f"y = {a:.4g} * x^{{{b:.4g}}}"
            return y, expr, True, ''

        elif model_family == 'Sinusoidal':
            A, B, C, D = params.get('A', 1), params.get('B', 1), params.get('C', 0), params.get('D', 0)
            y = A * np.sin(B * x + C) + D
            expr = f"y = {A:.4g} * sin({B:.4g}x + {C:.4g}) + {D:.4g}"
            return y, expr, True, ''

        elif model_family == 'Reciprocal':
            a, c, d = params.get('a', 1), params.get('c', 0), params.get('d', 0)
            denom = x - c
            y = np.where(np.abs(denom) > EPSILON, a / denom + d, np.nan)
            c_sign = '-' if c >= 0 else '+'
            d_sign = '+' if d >= 0 else '-'
            expr = f"y = {a:.4g}/(x {c_sign} {abs(c):.4g}) {d_sign} {abs(d):.4g}"
            return y, expr, True, ''

        elif model_family == 'Rational':
            a, b, c, d = params.get('a', 1), params.get('b', 0), params.get('c', 0), params.get('d', 1)
            denom = c * x + d
            y = np.where(np.abs(denom) > EPSILON, (a * x + b) / denom, np.nan)
            expr = f"y = ({a:.4g}x + {b:.4g}) / ({c:.4g}x + {d:.4g})"
            return y, expr, True, ''

        else:
            return np.zeros_like(x), '', False, f'Unknown model family: {model_family}'

    except Exception as e:
        return np.zeros_like(x), '', False, str(e)


# Model fitting functions
def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a linear model: y = mx + b"""
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    m, b = model.coef_[0], model.intercept_
    y_pred = model.predict(x.reshape(-1, 1))
    expr = f"y = {m:.6g}x + {b:.6g}" if b >= 0 else f"y = {m:.6g}x - {abs(b):.6g}"

    def eval_func(x_new):
        return m * x_new + b

    return expr, y_pred, {'type': 'Linear', 'complexity': 1, 'eval_func': eval_func}


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

    def eval_func(x_new):
        x_arr = np.atleast_1d(x_new)
        return model.predict(x_arr.reshape(-1, 1))

    return expr, y_pred, {'type': f'Polynomial (degree {degree})', 'complexity': degree, 'eval_func': eval_func}


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

        # Create evaluation function for CV
        def eval_func(x_new):
            return exponential_func(x_new, a, b, c)

        return expr, y_pred, {
            'type': 'Exponential',
            'complexity': 3,
            'params': {'a': a, 'b': b, 'c': c},
            'eval_func': eval_func
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

        def eval_func(x_new):
            x_arr = np.atleast_1d(x_new)
            result = np.where(x_arr > 0, a * np.log(x_arr) + b, np.nan)
            return result

        expr = f"y = {a:.6g} * ln(x) + {b:.6g}" if b >= 0 else f"y = {a:.6g} * ln(x) - {abs(b):.6g}"
        return expr, y_pred, {'type': 'Logarithmic', 'complexity': 2, 'eval_func': eval_func}
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

        b_exp = model.coef_[0]
        a_coef = np.exp(model.intercept_)

        y_pred = a_coef * np.power(x, b_exp)

        def eval_func(x_new):
            x_arr = np.atleast_1d(x_new)
            result = np.where(x_arr > 0, a_coef * np.power(x_arr, b_exp), np.nan)
            return result

        # Generate expression with proper formatting for LaTeX
        expr = f"y = {a_coef:.6g} * x^{{{b_exp:.6g}}}"
        return expr, y_pred, {
            'type': 'Power',
            'complexity': 2,
            'eval_func': eval_func,
            'params': {'a': a_coef, 'b': b_exp}
        }
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

        def eval_func(x_new):
            x_arr = np.atleast_1d(x_new)
            denom = c * x_arr + d
            result = np.where(np.abs(denom) > 1e-10, (a * x_arr + b) / denom, np.nan)
            return result

        expr = f"y = ({a:.4g}x + {b:.4g}) / ({c:.4g}x + {d:.4g})"
        return expr, y_pred, {'type': 'Rational', 'complexity': 3, 'eval_func': eval_func}
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

        def eval_func(x_new):
            return A * np.sin(B * x_new + C) + D

        expr = f"y = {A:.4g} * sin({B:.4g}x + {C:.4g}) + {D:.4g}"
        return expr, y_pred, {'type': 'Sinusoidal', 'complexity': 4, 'eval_func': eval_func}
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

        # Create evaluation function for CV
        def eval_func(x_new):
            return reciprocal_safe(x_new, a, c, d)

        return expr, y_pred, {
            'type': 'Reciprocal',
            'complexity': 3,
            'asymptote_x': c,
            'params': {'a': a, 'c': c, 'd': d},
            'eval_func': eval_func
        }
    except Exception:
        raise ValueError("Reciprocal fit failed")


def fit_sqrt_shifted(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a shifted square root model: y = a * sqrt(x - c) + d

    This model captures sqrt-like relationships with horizontal and vertical shifts.
    Returns NaN for x values where x - c < 0 (domain restriction).
    """
    EPSILON = 1e-9

    try:
        def sqrt_func(x, a, c, d):
            """Sqrt function for fitting"""
            arg = x - c
            with np.errstate(invalid='ignore'):
                result = a * np.sqrt(np.maximum(arg, 0)) + d
            return result

        def sqrt_safe(x, a, c, d):
            """Safe evaluation returning NaN outside domain"""
            arg = x - c
            result = np.where(
                arg >= -EPSILON,
                a * np.sqrt(np.maximum(arg, 0)) + d,
                np.nan
            )
            return result

        # Estimate initial parameters
        # c: should be near or below min(x) for sqrt to be valid
        x_min = x.min()
        x_max = x.max()

        # Try multiple c candidates
        c_candidates = [
            x_min - 0.1 * (x_max - x_min),  # Slightly below min
            x_min,
            x_min - 1.0,
            np.percentile(x, 5) - 0.5,
        ]

        best_result = None
        best_residual = float('inf')

        for c_init in c_candidates:
            try:
                # Only use points where x - c >= 0
                mask_valid = (x - c_init) >= -EPSILON
                if np.sum(mask_valid) < 3:
                    continue

                x_fit = x[mask_valid]
                y_fit = y[mask_valid]

                # Estimate a and d using linear regression on sqrt-transformed x
                sqrt_x = np.sqrt(np.maximum(x_fit - c_init, EPSILON))
                if np.any(~np.isfinite(sqrt_x)):
                    continue

                # Initial estimates
                a0 = (y_fit.max() - y_fit.min()) / (sqrt_x.max() - sqrt_x.min() + EPSILON)
                d0 = y_fit.min()

                popt, _ = optimize.curve_fit(
                    sqrt_func,
                    x_fit, y_fit,
                    p0=[a0, c_init, d0],
                    maxfev=5000,
                    bounds=([-np.inf, -np.inf, -np.inf], [np.inf, x_min + EPSILON, np.inf])
                )

                # Calculate residual on valid points
                y_pred_test = sqrt_safe(x_fit, *popt)
                valid_mask = np.isfinite(y_pred_test)
                if np.sum(valid_mask) > 0:
                    residual = np.sum((y_fit[valid_mask] - y_pred_test[valid_mask]) ** 2)
                    if residual < best_residual:
                        best_residual = residual
                        best_result = popt
            except Exception:
                continue

        if best_result is None:
            raise ValueError("Could not fit sqrt model")

        a, c, d = best_result

        # Generate predictions with NaN outside domain
        y_pred = sqrt_safe(x, a, c, d)

        # Check valid coverage
        valid_count = np.sum(np.isfinite(y_pred))
        if valid_count < len(x) * 0.8:
            raise ValueError("Sqrt model has insufficient domain coverage")

        # Build expression string
        c_sign = "-" if c >= 0 else "+"
        c_val = abs(c)
        d_sign = "+" if d >= 0 else "-"
        d_val = abs(d)

        expr = f"y = {a:.4g} * sqrt(x {c_sign} {c_val:.4g}) {d_sign} {d_val:.4g}"

        def eval_func(x_new):
            return sqrt_safe(x_new, a, c, d)

        return expr, y_pred, {
            'type': 'Square Root',
            'complexity': 3,
            'domain_boundary': c,
            'params': {'a': a, 'c': c, 'd': d},
            'eval_func': eval_func
        }
    except Exception:
        raise ValueError("Sqrt fit failed")


def fit_log_shifted(x: np.ndarray, y: np.ndarray) -> tuple[str, np.ndarray, dict]:
    """Fit a shifted logarithmic model: y = a * ln(x - c) + d

    This model captures log-like relationships with horizontal shift.
    Returns NaN for x values where x - c <= 0 (domain restriction).
    """
    EPSILON = 1e-9

    try:
        def log_func(x, a, c, d):
            """Log function for fitting"""
            arg = x - c
            with np.errstate(invalid='ignore', divide='ignore'):
                result = a * np.log(np.maximum(arg, EPSILON)) + d
            return result

        def log_safe(x, a, c, d):
            """Safe evaluation returning NaN outside domain"""
            arg = x - c
            result = np.where(
                arg > EPSILON,
                a * np.log(arg) + d,
                np.nan
            )
            return result

        # Estimate initial parameters
        # c: should be below min(x) for log to be valid
        x_min = x.min()
        x_max = x.max()

        # Try multiple c candidates
        c_candidates = [
            x_min - 0.1 * (x_max - x_min),  # Below min
            x_min - 1.0,
            x_min - 0.5,
            np.percentile(x, 5) - 1.0,
            0.0 if x_min > 1 else x_min - 2.0,
        ]

        best_result = None
        best_residual = float('inf')

        for c_init in c_candidates:
            try:
                # Only use points where x - c > 0
                mask_valid = (x - c_init) > EPSILON
                if np.sum(mask_valid) < 3:
                    continue

                x_fit = x[mask_valid]
                y_fit = y[mask_valid]

                # Estimate a and d using linear regression on log-transformed x
                log_x = np.log(x_fit - c_init)
                if np.any(~np.isfinite(log_x)):
                    continue

                # Linear regression: y = a * log_x + d
                model = LinearRegression()
                model.fit(log_x.reshape(-1, 1), y_fit)
                a0, d0 = model.coef_[0], model.intercept_

                popt, _ = optimize.curve_fit(
                    log_func,
                    x_fit, y_fit,
                    p0=[a0, c_init, d0],
                    maxfev=5000,
                    bounds=([-np.inf, -np.inf, -np.inf], [np.inf, x_min - EPSILON, np.inf])
                )

                # Calculate residual on valid points
                y_pred_test = log_safe(x_fit, *popt)
                valid_mask = np.isfinite(y_pred_test)
                if np.sum(valid_mask) > 0:
                    residual = np.sum((y_fit[valid_mask] - y_pred_test[valid_mask]) ** 2)
                    if residual < best_residual:
                        best_residual = residual
                        best_result = popt
            except Exception:
                continue

        if best_result is None:
            raise ValueError("Could not fit log model")

        a, c, d = best_result

        # Generate predictions with NaN outside domain
        y_pred = log_safe(x, a, c, d)

        # Check valid coverage
        valid_count = np.sum(np.isfinite(y_pred))
        if valid_count < len(x) * 0.8:
            raise ValueError("Log model has insufficient domain coverage")

        # Build expression string
        c_sign = "-" if c >= 0 else "+"
        c_val = abs(c)
        d_sign = "+" if d >= 0 else "-"
        d_val = abs(d)

        expr = f"y = {a:.4g} * ln(x {c_sign} {c_val:.4g}) {d_sign} {d_val:.4g}"

        def eval_func(x_new):
            return log_safe(x_new, a, c, d)

        return expr, y_pred, {
            'type': 'Logarithmic (Shifted)',
            'complexity': 3,
            'domain_boundary': c,
            'params': {'a': a, 'c': c, 'd': d},
            'eval_func': eval_func
        }
    except Exception:
        raise ValueError("Log shifted fit failed")


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


@app.get("/models", response_model=ModelsResponse)
def get_models():
    """Return list of available model families for fitting"""
    models = [
        ModelInfo(modelId="linear", displayName="Linear", parameterCount=2),
        ModelInfo(modelId="poly2", displayName="Polynomial (degree 2)", parameterCount=3),
        ModelInfo(modelId="poly3", displayName="Polynomial (degree 3)", parameterCount=4),
        ModelInfo(modelId="poly4", displayName="Polynomial (degree 4)", parameterCount=5),
        ModelInfo(modelId="exponential", displayName="Exponential", parameterCount=3),
        ModelInfo(modelId="logarithmic", displayName="Logarithmic", parameterCount=2, domain="x > 0"),
        ModelInfo(modelId="log_shifted", displayName="Logarithmic (Shifted)", parameterCount=3, domain="x > c"),
        ModelInfo(modelId="sqrt_shifted", displayName="Square Root", parameterCount=3, domain="x ≥ c"),
        ModelInfo(modelId="power", displayName="Power", parameterCount=2, domain="x > 0"),
        ModelInfo(modelId="sinusoidal", displayName="Sinusoidal", parameterCount=4),
        ModelInfo(modelId="reciprocal", displayName="Reciprocal", parameterCount=3),
        ModelInfo(modelId="rational", displayName="Rational", parameterCount=4),
    ]
    return ModelsResponse(models=models)


# Mapping from modelId to internal family name and model type
MODEL_ID_TO_FAMILY = {
    'linear': ('poly1', 'Linear'),
    'poly2': ('poly2', 'Polynomial (degree 2)'),
    'poly3': ('poly3', 'Polynomial (degree 3)'),
    'poly4': ('poly4', 'Polynomial (degree 4)'),
    'exponential': ('exponential', 'Exponential'),
    'logarithmic': ('logarithmic', 'Logarithmic'),
    'log_shifted': ('log_shifted', 'Logarithmic (Shifted)'),
    'sqrt_shifted': ('sqrt_shifted', 'Square Root'),
    'power': ('power', 'Power'),
    'sinusoidal': ('sinusoidal', 'Sinusoidal'),
    'reciprocal': ('reciprocal', 'Reciprocal'),
    'rational': ('rational', 'Rational'),
}


def compute_validation_score(x: np.ndarray, y: np.ndarray, fit_func, n_folds: int = 5) -> float:
    """Compute k-fold cross-validation score for generalization assessment"""
    n = len(x)
    if n < n_folds * 2:
        # Not enough data for k-fold, use leave-one-out style
        n_folds = max(2, n // 2)

    # Shuffle indices
    indices = np.arange(n)
    np.random.seed(42)  # Reproducible
    np.random.shuffle(indices)

    fold_size = n // n_folds
    val_errors = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n

        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        if len(train_indices) < 2 or len(val_indices) < 1:
            continue

        x_train, y_train = x[train_indices], y[train_indices]
        x_val, y_val = x[val_indices], y[val_indices]

        try:
            _, y_pred_train, info = fit_func(x_train, y_train)

            # Use evaluation function if available (for proper nonlinear model evaluation)
            if 'eval_func' in info and info['eval_func'] is not None:
                y_pred_val = info['eval_func'](x_val)
            else:
                # Fallback to cubic interpolation for smoother curves
                from scipy.interpolate import interp1d
                # Sort by x for proper interpolation
                sort_idx = np.argsort(x_train)
                x_sorted = x_train[sort_idx]
                y_sorted = y_pred_train[sort_idx]
                # Use cubic interpolation when possible, linear as fallback
                kind = 'cubic' if len(x_train) >= 4 else 'linear'
                interp_func = interp1d(x_sorted, y_sorted, kind=kind,
                                       fill_value='extrapolate', bounds_error=False)
                y_pred_val = interp_func(x_val)

            # Filter valid predictions
            valid_mask = np.isfinite(y_pred_val)
            if np.sum(valid_mask) > 0:
                mse = np.mean((y_val[valid_mask] - y_pred_val[valid_mask]) ** 2)
                val_errors.append(mse)
        except Exception:
            continue

    if len(val_errors) == 0:
        return float('inf')

    return np.mean(val_errors)


@app.post("/fit", response_model=FitResult)
def fit_curve(request: FitRequest):
    """Fit a curve to the provided points using multi-start candidate search"""
    if len(request.points) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 points")

    # Extract data
    x = np.array([p.x for p in request.points])
    y = np.array([p.y for p in request.points])

    # Determine if we're using auto or forced mode
    selected_model = request.selectedModel or 'auto'
    is_forced = selected_model.lower() != 'auto'
    forced_family = None
    forced_model_type = None

    if is_forced:
        # Map the modelId to the internal family name
        if selected_model not in MODEL_ID_TO_FAMILY:
            raise HTTPException(status_code=400, detail=f"Unknown model: {selected_model}")
        forced_family, forced_model_type = MODEL_ID_TO_FAMILY[selected_model]

    # Use the CandidateSearchEngine for multi-start model selection
    engine = CandidateSearchEngine()

    try:
        best_candidate, trace = engine.search(x, y, request.objective, forced_family=forced_family)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build heuristics from trace
    heuristics = []
    heuristics.append(f"Search: {trace.total_seeds} seeds in {trace.total_time_ms:.0f}ms")

    for family_result in trace.families:
        if family_result.status == 'selected':
            heuristics.append(
                f"✓ {family_result.family}: CV_RMSE={family_result.final_score:.4f} "
                f"({family_result.seeds_valid}/{family_result.seeds_tried} valid)"
            )
        elif family_result.status == 'rejected' and family_result.best_candidate:
            heuristics.append(
                f"  {family_result.family}: CV_RMSE={family_result.final_score:.4f} "
                f"({family_result.seeds_valid}/{family_result.seeds_tried} valid)"
            )
        elif family_result.status == 'skipped':
            heuristics.append(f"  {family_result.family}: skipped ({family_result.rejection_reason})")
        elif family_result.status == 'failed':
            heuristics.append(f"  {family_result.family}: failed ({family_result.rejection_reason})")

    if trace.early_exit:
        heuristics.append(f"Early exit: {trace.early_exit_reason}")

    # Calculate training metrics for the best candidate
    if best_candidate.y_pred is not None:
        metrics = calculate_metrics(y, best_candidate.y_pred, best_candidate.n_params)
    else:
        # Fallback: use CV metrics
        metrics = FitStatistics(
            r2=best_candidate.cv_r2,
            rmse=best_candidate.cv_rmse,
            mae=best_candidate.cv_mae,
            aic=None,
            bic=None
        )

    # Generate curve points for visualization
    x_data_min, x_data_max = x.min(), x.max()
    x_data_range = x_data_max - x_data_min

    x_extend = max(x_data_range * 2, 15)
    x_curve_min = min(x_data_min - x_extend, -20)
    x_curve_max = max(x_data_max + x_extend, 20)

    N_SAMPLES = 1000
    x_curve = np.linspace(x_curve_min, x_curve_max, N_SAMPLES)

    # Use eval_func for curve generation
    if best_candidate.eval_func is not None:
        y_curve = best_candidate.eval_func(x_curve)
    elif best_candidate.y_pred is not None:
        # Fallback: interpolation
        try:
            from scipy.interpolate import interp1d
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = best_candidate.y_pred[sort_idx]
            kind = 'cubic' if len(x) >= 4 else 'linear'
            interp_func = interp1d(x_sorted, y_sorted, kind=kind,
                                   fill_value='extrapolate', bounds_error=False)
            y_curve = interp_func(x_curve)
        except Exception:
            y_curve = np.interp(x_curve, x, best_candidate.y_pred)
    else:
        y_curve = np.zeros_like(x_curve)

    # Filter out NaN/Infinity values
    curve_points = [
        Point(x=float(xi), y=float(yi))
        for xi, yi in zip(x_curve, y_curve)
        if np.isfinite(xi) and np.isfinite(yi)
    ]

    # Get model type from info or family name
    # Use forced_model_type if we're in forced mode
    model_type = forced_model_type if is_forced and forced_model_type else best_candidate.info.get('type', best_candidate.family.title())

    # Get parameters from info and generate schema
    params = best_candidate.info.get('params', {})

    # For Linear/Polynomial, we need to extract params differently
    # Regex pattern for numbers (including scientific notation)
    import re
    num_pattern = r'-?\d+\.?\d*(?:e[+-]?\d+)?'

    if model_type == 'Linear':
        # Parse from expression: y = mx + b
        match = re.match(rf'y\s*=\s*({num_pattern})x\s*([+-])\s*({num_pattern})', best_candidate.expression)
        if match:
            m_val = float(match.group(1))
            sign = match.group(2)
            b_val = float(match.group(3))
            if sign == '-':
                b_val = -b_val
            params = {'m': m_val, 'b': b_val}
    elif 'Polynomial' in model_type:
        # Extract all numbers from expression
        terms = re.findall(num_pattern, best_candidate.expression)
        # Filter out empty strings and convert to float
        terms = [float(t) for t in terms if t and t not in ['+', '-']]

        if 'degree 2' in model_type and len(terms) >= 3:
            params = {'a2': terms[0], 'a1': terms[1], 'a0': terms[2]}
        elif 'degree 3' in model_type and len(terms) >= 4:
            params = {'a3': terms[0], 'a2': terms[1], 'a1': terms[2], 'a0': terms[3]}
        elif 'degree 4' in model_type and len(terms) >= 5:
            params = {'a4': terms[0], 'a3': terms[1], 'a2': terms[2], 'a1': terms[3], 'a0': terms[4]}

    parameter_schema = get_parameter_schema(model_type, params)

    return FitResult(
        expression=best_candidate.expression,
        expressionLatex=best_candidate.expression,
        statistics=metrics,
        quality=determine_quality(metrics.r2),
        curvePoints=curve_points,
        modelType=model_type,
        heuristics=heuristics,
        parameterSchema=parameter_schema,
        mode='forced' if is_forced else 'auto'
    )


def preprocess_expression(expr_str: str) -> str:
    """Preprocess expression string for sympy parsing"""
    import re

    # Remove "y = " prefix
    expr_str = expr_str.replace('y = ', '').replace('y=', '')

    # Replace ^ with ** (but do this AFTER handling other patterns)
    # We'll do this last to avoid conflicts

    # Replace ln with log (sympy uses log for natural log)
    expr_str = expr_str.replace('ln', 'log')

    # Fix coefficient before x outside of functions (e.g., "0.5x" -> "0.5*x", "2x" -> "2*x")
    # Be careful not to add * inside function calls
    expr_str = re.sub(r'(\d+\.?\d*)x(?!\w)', r'\1*x', expr_str)

    # Fix patterns like exp(0.5x) -> exp(0.5*x)
    # This handles the case inside exp()
    expr_str = re.sub(r'exp\((-?\d+\.?\d*)x\)', r'exp(\1*x)', expr_str)

    # Now replace ^ with **
    expr_str = expr_str.replace('^', '**')

    return expr_str


@app.post("/analyze", response_model=AnalyticalProperties)
def analyze_function(request: AnalyzeRequest):
    """Compute analytical properties of a function"""
    try:
        x = sp.Symbol('x')

        # Parse the expression (handle common formats)
        expr_str = preprocess_expression(request.expression)

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
            # Convert to single fraction and find zeros of denominator
            together = sp.together(expr)
            _, denom = sp.fraction(together)
            if denom != 1:
                vert_asymp = sp.solve(denom, x)
                for va in vert_asymp:
                    if va.is_real:
                        asymptotes.append(Asymptote(type='vertical', value=float(va)))
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

        expr_str = preprocess_expression(request.expression)

        expr = parse_expr(expr_str)

        # Check for vertical asymptotes within integration bounds
        a, b = min(request.a, request.b), max(request.a, request.b)
        try:
            together = sp.together(expr)
            _, denom = sp.fraction(together)
            if denom != 1:
                vert_asymp = sp.solve(denom, x)
                for va in vert_asymp:
                    if va.is_real:
                        va_float = float(va)
                        if a < va_float < b:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Integral undefined (bounds cross vertical asymptote at x = {va_float:.4g})"
                            )
        except HTTPException:
            raise
        except Exception:
            pass  # Continue even if asymptote check fails

        # Check for sqrt/log domain boundaries
        try:
            # Find sqrt arguments and check domain
            for sqrt_arg in expr.atoms(sp.sqrt):
                # Get the argument inside sqrt
                inner = sqrt_arg.args[0]
                # Solve for when inner = 0 (domain boundary)
                boundary_solutions = sp.solve(inner, x)
                for bs in boundary_solutions:
                    if bs.is_real:
                        bs_float = float(bs)
                        if a < bs_float:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Integral undefined (bounds include invalid domain for sqrt at x = {bs_float:.4g})"
                            )

            # Find log arguments and check domain
            for log_func in expr.atoms(sp.log):
                inner = log_func.args[0]
                # Solve for when inner = 0 (domain boundary for log)
                boundary_solutions = sp.solve(inner, x)
                for bs in boundary_solutions:
                    if bs.is_real:
                        bs_float = float(bs)
                        if a <= bs_float:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Integral undefined (bounds include invalid domain for ln at x = {bs_float:.4g})"
                            )
        except HTTPException:
            raise
        except Exception:
            pass  # Continue even if domain check fails

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
            expression=f"∫[{request.a}, {request.b}] {request.expression.replace('y = ', '')} dx = {integral_expr}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not compute integral: {str(e)}")


@app.post("/evaluate", response_model=EvaluateResult)
def evaluate_function(request: EvaluateRequest):
    """Evaluate a model with custom parameters"""
    if len(request.points) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 points")

    # Extract data
    x_data = np.array([p.x for p in request.points])
    y_data = np.array([p.y for p in request.points])

    # Evaluate with parameters
    y_pred, expression, valid, error = evaluate_with_params(
        request.modelFamily,
        request.parameters,
        x_data
    )

    if not valid:
        return EvaluateResult(
            expression='',
            expressionLatex='',
            statistics=FitStatistics(r2=-999, rmse=float('inf'), mae=float('inf')),
            quality='bad',
            curvePoints=[],
            valid=False,
            error=error
        )

    # Check coverage (how many valid predictions)
    valid_mask = np.isfinite(y_pred)
    coverage = np.sum(valid_mask) / len(y_pred)

    if coverage < 0.5:
        return EvaluateResult(
            expression=expression,
            expressionLatex=expression,
            statistics=FitStatistics(r2=-999, rmse=float('inf'), mae=float('inf')),
            quality='bad',
            curvePoints=[],
            valid=False,
            error=f'Domain invalid for most data points ({coverage*100:.1f}% valid)'
        )

    # Calculate metrics
    metrics = calculate_metrics(y_data, y_pred, len(request.parameters))

    # Generate curve points for visualization
    x_data_min, x_data_max = x_data.min(), x_data.max()
    x_data_range = x_data_max - x_data_min

    if request.xRange:
        x_curve_min, x_curve_max = request.xRange
    else:
        x_extend = max(x_data_range * 2, 15)
        x_curve_min = min(x_data_min - x_extend, -20)
        x_curve_max = max(x_data_max + x_extend, 20)

    N_SAMPLES = 1000
    x_curve = np.linspace(x_curve_min, x_curve_max, N_SAMPLES)

    # Evaluate for curve
    y_curve, _, _, _ = evaluate_with_params(request.modelFamily, request.parameters, x_curve)

    # Filter out NaN/Infinity values
    curve_points = [
        Point(x=float(xi), y=float(yi))
        for xi, yi in zip(x_curve, y_curve)
        if np.isfinite(xi) and np.isfinite(yi)
    ]

    return EvaluateResult(
        expression=expression,
        expressionLatex=expression,
        statistics=metrics,
        quality=determine_quality(metrics.r2),
        curvePoints=curve_points,
        valid=True,
        error=None
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
