"""
metric_engine.py
----------------
Dedicated Track C Engine: Closed-Form Moment-Space Dimension Reduction for
Stable Inverse Modeling and Numerical Regulation of Repulsive Gravitational
Metric Perturbations (Linearized Weak-Field Einstein / Poisson-Metric System).

Mathematical Foundations (Generalizing Dr. Hyo Eun Ahn's 1994 LANL Formulations):
  1. Theorem 1 (Signed Prior Regularization, sum(mu_i) = -1, mu_i < 0):
       J(a, lambda) = 1/2 sum_{i=1}^N (a_i - mu_i)^2 + lambda^T (s - Omega a)
       a = mu + Lambda (s - Omega mu) = Lambda s + c
     Suppresses unphysical positive boundary spikes (a_i > 0) and Gibbs/Runge
     oscillations on underdetermined 3D spacetime grids (N >> M = 35).
  2. Theorem 2 (Modulus-Weighted Potential Collapse — The Fall-Out of N):
       chi^2(v) = sum_{i=1}^N |mu_i| [Phi_exact(r_i) - Phi_approx(r_i)]^2
       v = Lambda'^T D Lambda s + Lambda'^T D c = K s + v_0   (35 x 35)
     Using w_i = |mu_i| > 0 guarantees a strictly convex quadratic form and
     symmetric positive-definite (SPD) Gram matrix Omega {Omega |mu|}^T.
  3. Theorem 3 (Direct 3D Repulsive Acceleration Field Fit & Zero Self-Acceleration):
       S(v_g) = sum_{i=1}^N |mu_i| sum_{j=1}^3 [g_{x_j}^exact(r_i) - g_{x_j}^approx(r_i)]^2
       v_g = K_g s + v_{g,0}   (34 x 35)
       <g_approx>_{|mu|} = <g_exact>_{|mu|} = 0  (Zero artificial self-acceleration)

Author: Hyo Eun Ahn (Independent Researcher, hyo.eun.ahn@proton.me)
Architecture: DDD / Clean Architecture / Spec-Driven Design (Zero Hard-Coding)
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import time
import numpy as np


@dataclass(frozen=True)
class RepulsiveMetricSpec:
    """Configuration specification for the 3D Repulsive Metric Moment Engine."""
    num_grid_points: int = 1000
    max_polynomial_order: int = 4
    spatial_dimensions: int = 3
    domain_half_width: float = 1.0
    source_core_anisotropy: Tuple[float, float, float] = (0.55, 0.48, 0.62)
    prior_total_signature: float = -1.0
    regularization_eps: float = 1e-12
    random_seed: int = 1994


class RepulsiveMetricMomentEngine:
    """
    Closed-form M x M (35 x 35 potential and 34 x 35 3D acceleration field)
    inverse modeling and numerical regulation engine for linearized repulsive
    gravitational metric perturbations.
    """

    def __init__(
        self,
        spec: RepulsiveMetricSpec,
        custom_grid: Optional[np.ndarray] = None,
        custom_signed_prior: Optional[np.ndarray] = None,
    ) -> None:
        self.spec = spec
        self.N = spec.num_grid_points
        self.order = spec.max_polynomial_order

        self.exponents: List[Tuple[int, int, int]] = self._generate_monomial_exponents(
            self.order
        )
        self.M = len(self.exponents)
        self.M_field = self.M - 1  # 34 non-constant monomials for 3D field fit

        if custom_grid is not None:
            if custom_grid.shape != (self.N, self.spec.spatial_dimensions):
                raise ValueError(
                    f"Expected grid shape {(self.N, self.spec.spatial_dimensions)}, got {custom_grid.shape}"
                )
            self.grid = custom_grid.copy()
        else:
            self.grid = self._initialize_3d_manifold_grid()

        if custom_signed_prior is not None:
            if custom_signed_prior.shape != (self.N,):
                raise ValueError(
                    f"Expected prior shape {(self.N,)}, got {custom_signed_prior.shape}"
                )
            self.mu = self._normalize_signed_prior(custom_signed_prior)
        else:
            self.mu = self._initialize_signed_prior()

        # Strictly positive modulus weights w_i = |mu_i| > 0 for SPD Gram matrices
        self.weights = np.abs(self.mu)
        self.weights = self.weights / float(np.sum(self.weights))

        self._precompute_metric_operators()

    @staticmethod
    def _generate_monomial_exponents(max_order: int) -> List[Tuple[int, int, int]]:
        exps: List[Tuple[int, int, int]] = []
        for total in range(max_order + 1):
            for p in range(total, -1, -1):
                for q in range(total - p, -1, -1):
                    r = total - p - q
                    exps.append((p, q, r))
        return exps

    def _initialize_3d_manifold_grid(self) -> np.ndarray:
        rng = np.random.default_rng(self.spec.random_seed)
        hw = self.spec.domain_half_width
        return rng.uniform(-hw, hw, size=(self.N, self.spec.spatial_dimensions))

    def _normalize_signed_prior(self, raw_mu: np.ndarray) -> np.ndarray:
        abs_sum = float(np.sum(np.abs(raw_mu)))
        if abs_sum <= 0.0:
            raise ValueError("Prior vector must have non-zero norm.")
        # Enforce negative-definite reference profile summing to prior_total_signature (-1.0)
        return self.spec.prior_total_signature * (np.abs(raw_mu) / abs_sum)

    def _initialize_signed_prior(self) -> np.ndarray:
        sx, sy, sz = self.spec.source_core_anisotropy
        x, y, z = self.grid[:, 0], self.grid[:, 1], self.grid[:, 2]
        r2_scaled = (x / sx) ** 2 + (y / sy) ** 2 + (z / sz) ** 2
        raw_profile = np.exp(-0.5 * r2_scaled)
        return self._normalize_signed_prior(raw_profile)

    def _evaluate_monomial_and_gradients(
        self, coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        n_pts = coords.shape[0]

        omega = np.empty((self.M, n_pts), dtype=np.float64)
        omega_x = np.zeros((self.M, n_pts), dtype=np.float64)
        omega_y = np.zeros((self.M, n_pts), dtype=np.float64)
        omega_z = np.zeros((self.M, n_pts), dtype=np.float64)

        for k, (p, q, r) in enumerate(self.exponents):
            xp = np.ones_like(x) if p == 0 else x**p
            yq = np.ones_like(y) if q == 0 else y**q
            zr = np.ones_like(z) if r == 0 else z**r
            omega[k, :] = xp * yq * zr

            if p > 0:
                xp_m1 = np.ones_like(x) if p == 1 else x ** (p - 1)
                omega_x[k, :] = p * xp_m1 * yq * zr
            if q > 0:
                yq_m1 = np.ones_like(y) if q == 1 else y ** (q - 1)
                omega_y[k, :] = q * xp * yq_m1 * zr
            if r > 0:
                zr_m1 = np.ones_like(z) if r == 1 else z ** (r - 1)
                omega_z[k, :] = r * xp * yq * zr_m1

        return omega, omega_x, omega_y, omega_z

    def _precompute_metric_operators(self) -> None:
        self.Omega, self.Omega_x, self.Omega_y, self.Omega_z = (
            self._evaluate_monomial_and_gradients(self.grid)
        )

        # 1. Theorem 1: Underdetermined signed source inversion (a = Lambda s + c)
        gram_unweighted = (
            self.Omega @ self.Omega.T
            + self.spec.regularization_eps * np.eye(self.M)
        )
        self.Lambda = self.Omega.T @ np.linalg.inv(gram_unweighted)  # (N, 35)
        self.c = self.mu - self.Lambda @ (self.Omega @ self.mu)  # (N,)

        # 2. Pairwise Green's function kernels for linearized metric perturbation
        diff = self.grid[:, np.newaxis, :] - self.grid[np.newaxis, :, :]  # (N, N, 3)
        dist_sq = np.sum(diff**2, axis=-1)  # (N, N)
        np.fill_diagonal(dist_sq, 1.0)  # Temporary 1.0 on diagonal to prevent 0/0 or 0*inf NaN
        dist = np.sqrt(dist_sq)
        dist_cubed = dist_sq * dist

        # Linearized gravitational scalar potential kernel: Phi(r_i) = sum_j D_ij a_j
        # For negative source a_j < 0, D_ij = -1 / |r_i - r_j| produces Phi > 0 hill
        # and outward repulsive acceleration g = -grad(Phi) > 0.
        # Exclude self-interaction explicitly (D_ii = 0, G_ii = 0):
        self.D = -1.0 / dist  # (N, N)
        self.G_x = -diff[:, :, 0] / dist_cubed  # (N, N)
        self.G_y = -diff[:, :, 1] / dist_cubed  # (N, N)
        self.G_z = -diff[:, :, 2] / dist_cubed  # (N, N)

        np.fill_diagonal(self.D, 0.0)
        np.fill_diagonal(self.G_x, 0.0)
        np.fill_diagonal(self.G_y, 0.0)
        np.fill_diagonal(self.G_z, 0.0)

        # 3. Theorem 2: Modulus-weighted 35x35 Potential Operator (w_i = |mu_i| > 0)
        omega_w = self.Omega * self.weights[np.newaxis, :]  # (35, N)
        gram_w = (
            omega_w @ self.Omega.T
            + self.spec.regularization_eps * np.eye(self.M)
        )
        self.Lambda_prime_T = np.linalg.inv(gram_w) @ omega_w  # (35, N)
        self.K = self.Lambda_prime_T @ (self.D @ self.Lambda)  # (35, 35)
        self.v_0 = self.Lambda_prime_T @ (self.D @ self.c)  # (35,)

        # 4. Theorem 3: Direct 3D Repulsive Acceleration Field Operator (34 x 35)
        self.Omega_tilde_x = self.Omega_x[1:, :]  # (34, N)
        self.Omega_tilde_y = self.Omega_y[1:, :]  # (34, N)
        self.Omega_tilde_z = self.Omega_z[1:, :]  # (34, N)

        ox_w = self.Omega_tilde_x * self.weights[np.newaxis, :]
        oy_w = self.Omega_tilde_y * self.weights[np.newaxis, :]
        oz_w = self.Omega_tilde_z * self.weights[np.newaxis, :]

        gram_field_w = (
            ox_w @ self.Omega_tilde_x.T
            + oy_w @ self.Omega_tilde_y.T
            + oz_w @ self.Omega_tilde_z.T
            + self.spec.regularization_eps * np.eye(self.M_field)
        )
        gram_field_w_inv = np.linalg.inv(gram_field_w)  # (34, 34)

        field_proj_n = ox_w @ self.G_x + oy_w @ self.G_y + oz_w @ self.G_z  # (34, N)
        self.Lambda_g_T = -gram_field_w_inv @ field_proj_n  # (34, N)
        self.K_g = self.Lambda_g_T @ self.Lambda  # (34, 35)
        self.v_g0 = self.Lambda_g_T @ self.c  # (34,)

    def compute_moments(self, signed_sources: np.ndarray) -> np.ndarray:
        return self.Omega @ signed_sources

    def reconstruct_source_distribution(
        self, moments: np.ndarray, custom_prior: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if custom_prior is None:
            return self.Lambda @ moments + self.c
        return custom_prior + self.Lambda @ (moments - self.Omega @ custom_prior)

    def runtime_potential_coefficients(self, moments: np.ndarray) -> np.ndarray:
        """O(M^2) closed-form 35x35 metric potential coefficient evaluation."""
        return self.K @ moments + self.v_0

    def runtime_acceleration_field_coefficients(
        self, moments: np.ndarray
    ) -> np.ndarray:
        """O(M_E * M) closed-form 34x35 3D repulsive acceleration field evaluation."""
        return self.K_g @ moments + self.v_g0

    def verify_potential_moment_conservation(
        self, sources: np.ndarray
    ) -> Dict[str, np.ndarray]:
        w = self.weights
        phi_exact = self.D @ sources
        omega_w = self.Omega * w[np.newaxis, :]
        gram = omega_w @ self.Omega.T + self.spec.regularization_eps * np.eye(self.M)
        v_full = np.linalg.solve(gram, omega_w @ phi_exact)
        phi_approx = self.Omega.T @ v_full

        exact_avgs = omega_w @ phi_exact
        approx_avgs = omega_w @ phi_approx
        return {
            "v_full": v_full,
            "exact_potential_averages_35": exact_avgs,
            "approx_potential_averages_35": approx_avgs,
        }

    def verify_acceleration_and_zero_self_force_theorem(
        self, sources: np.ndarray
    ) -> Dict[str, np.ndarray]:
        w = self.weights

        gx_exact = self.G_x @ sources
        gy_exact = self.G_y @ sources
        gz_exact = self.G_z @ sources

        ox_w = self.Omega_tilde_x * w[np.newaxis, :]
        oy_w = self.Omega_tilde_y * w[np.newaxis, :]
        oz_w = self.Omega_tilde_z * w[np.newaxis, :]

        gram_g = (
            ox_w @ self.Omega_tilde_x.T
            + oy_w @ self.Omega_tilde_y.T
            + oz_w @ self.Omega_tilde_z.T
            + self.spec.regularization_eps * np.eye(self.M_field)
        )
        rhs_exact_34 = ox_w @ gx_exact + oy_w @ gy_exact + oz_w @ gz_exact
        v_g_full = -np.linalg.solve(gram_g, rhs_exact_34)

        gx_approx = -(self.Omega_tilde_x.T @ v_g_full)
        gy_approx = -(self.Omega_tilde_y.T @ v_g_full)
        gz_approx = -(self.Omega_tilde_z.T @ v_g_full)

        lhs_approx_34 = ox_w @ gx_approx + oy_w @ gy_approx + oz_w @ gz_approx

        avg_g_exact = np.array(
            [np.sum(w * gx_exact), np.sum(w * gy_exact), np.sum(w * gz_exact)]
        )
        avg_g_approx = np.array(
            [np.sum(w * gx_approx), np.sum(w * gy_approx), np.sum(w * gz_approx)]
        )

        return {
            "v_g_full": v_g_full,
            "exact_field_averages_34": rhs_exact_34,
            "approx_field_averages_34": lhs_approx_34,
            "avg_g_exact": avg_g_exact,
            "avg_g_approx": avg_g_approx,
        }

    def benchmark_latency(
        self, moments: np.ndarray, num_iterations: int = 10000
    ) -> Tuple[float, float]:
        for _ in range(100):
            _ = self.runtime_potential_coefficients(moments)
            _ = self.runtime_acceleration_field_coefficients(moments)

        t0 = time.perf_counter()
        for _ in range(num_iterations):
            _ = self.runtime_potential_coefficients(moments)
        pot_us = ((time.perf_counter() - t0) / num_iterations) * 1e6

        t1 = time.perf_counter()
        for _ in range(num_iterations):
            _ = self.runtime_acceleration_field_coefficients(moments)
        field_us = ((time.perf_counter() - t1) / num_iterations) * 1e6

        return pot_us, field_us

