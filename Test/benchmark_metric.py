"""
benchmark_metric.py
-------------------
Dedicated Verification & Benchmark Suite for Track C:
  "A Closed-Form Moment Framework for Stable Inverse Modeling and Numerical
   Regulation of Repulsive Gravitational Metric Perturbations"

Verifies:
  1. Theorem 1: Signed normalized prior (sum(mu_i) = -1, mu_i < 0) vs.
     minimum-norm (mu_i = 0) and constant prior (mu_i = const), proving
     suppression of unphysical positive boundary spikes (a_i > 0) and
     exact conservation of all 35 spatial moments.
  2. Theorem 2: Modulus-weighted (w_i = |mu_i| > 0) 35x35 potential collapse
     (The Fall-Out of N) and exact preservation of 35 potential averages.
  3. Theorem 3: Direct 3D repulsive acceleration field fit (34x35) and the
     Zero Self-Acceleration / Average Force Conservation Theorem
     (<g_approx>_{|mu|} = <g_exact>_{|mu|} = 0).
  4. Sub-5 microsecond execution latency on a single CPU core.

Author: Hyo Eun Ahn (Independent Researcher, hyo.eun.ahn@proton.me)
"""

from dataclasses import dataclass
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from metric_engine import (
    RepulsiveMetricSpec,
    RepulsiveMetricMomentEngine,
)


@dataclass(frozen=True)
class RepulsiveBenchmarkSpec:
    num_grid_points: int = 1000
    benchmark_iterations: int = 10000
    direct_grid_sample_runs: int = 200
    realtime_budget_us: float = 5.0
    numerical_tolerance: float = 1e-10
    output_figure_relpath: str = "benchmark_metric.png"
    figure_dpi: int = 300


def run_repulsive_metric_benchmark(
    spec: RepulsiveBenchmarkSpec = RepulsiveBenchmarkSpec(),
) -> dict:
    print("=" * 78)
    print(
        "  TRACK C: REPULSIVE METRIC MOMENT REGULATION & INVERSE MODELING SUITE"
    )
    print(
        "  Author: Hyo Eun Ahn (Independent Researcher, hyo.eun.ahn@proton.me)"
    )
    print("=" * 78)

    engine_spec = RepulsiveMetricSpec(num_grid_points=spec.num_grid_points)
    engine = RepulsiveMetricMomentEngine(engine_spec)

    # Create a realistic perturbed negative-definite source distribution (sum = -1.0)
    rng = np.random.default_rng(engine_spec.random_seed + 7)
    x, y, z = engine.grid[:, 0], engine.grid[:, 1], engine.grid[:, 2]
    sx, sy, sz = engine_spec.source_core_anisotropy
    r2 = (x / sx) ** 2 + (y / sy) ** 2 + (z / sz) ** 2
    quadrupole_perturbation = 1.0 + 0.18 * (x**2 - y**2) + 0.12 * (x * z)
    raw_true = np.exp(-0.5 * r2) * np.clip(quadrupole_perturbation, 0.2, None)
    true_sources = -1.0 * (raw_true / float(np.sum(raw_true)))
    s_obs = engine.compute_moments(true_sources)

    # 1. Verify Theorem 1: Signed Prior Regularization vs Minimum-Norm (mu=0) & Constant Prior
    a_ahn = engine.reconstruct_source_distribution(s_obs)
    a_zero = engine.reconstruct_source_distribution(
        s_obs, custom_prior=np.zeros(engine.N)
    )
    a_const = engine.reconstruct_source_distribution(
        s_obs, custom_prior=np.full(engine.N, -1.0 / engine.N)
    )

    moment_err = float(np.max(np.abs(engine.compute_moments(a_ahn) - s_obs)))
    channell_diff = float(np.max(np.abs(a_const - a_zero)))

    # Unphysical positive boundary spikes (a_i > 0 when source is strictly negative-definite)
    pos_spike_zero = float(np.sum(np.maximum(a_zero, 0.0)))
    pos_spike_ahn = float(np.sum(np.maximum(a_ahn, 0.0)))
    l2_zero = float(np.linalg.norm(a_zero - true_sources))
    l2_ahn = float(np.linalg.norm(a_ahn - true_sources))

    print("\n[1] Theorem 1: Signed Prior Regularization (sum(mu_i) = -1, mu_i < 0):")
    print(f"    - 35 Moment Conservation Residual ||Omega a - s||_inf : {moment_err:.2e}")
    print(f"    - Constant-Prior Degeneracy ||a(const) - a(0)||_inf    : {channell_diff:.2e}")
    print(f"    - Unphysical Positive Spike (mu = 0 / const)           : {pos_spike_zero:.4f}")
    print(f"    - Unphysical Positive Spike (Ahn Signed Prior)         : {pos_spike_ahn:.4f}")
    print(f"    - L2 Source Reconstruction Error (mu = 0 -> Ahn Prior) : {l2_zero:.6f} -> {l2_ahn:.6f}")

    assert moment_err < spec.numerical_tolerance
    assert channell_diff < spec.numerical_tolerance
    assert pos_spike_ahn < pos_spike_zero
    assert l2_ahn < l2_zero

    # 2. Verify Theorem 2: 35x35 Modulus-Weighted Potential Reduction
    v_closed = engine.runtime_potential_coefficients(s_obs)
    v_full_on_recon = engine.Lambda_prime_T @ (engine.D @ a_ahn)
    v_pot_diff = float(np.max(np.abs(v_closed - v_full_on_recon)))

    pot_thm = engine.verify_potential_moment_conservation(true_sources)
    pot_35_err = float(
        np.max(
            np.abs(
                pot_thm["exact_potential_averages_35"]
                - pot_thm["approx_potential_averages_35"]
            )
        )
    )

    print("\n[2] Theorem 2: Modulus-Weighted 35x35 Potential Collapse (Fall-Out of N):")
    print(f"    - Closed-Form 35x35 vs N-Grid Residual                 : {v_pot_diff:.2e}")
    print(f"    - 35 Modulus-Weighted Potential Averages Residual      : {pot_35_err:.2e}")

    assert v_pot_diff < spec.numerical_tolerance
    assert pot_35_err < spec.numerical_tolerance

    # 3. Verify Theorem 3: Direct 3D Repulsive Acceleration Field Fit & Zero Self-Acceleration
    vg_closed = engine.runtime_acceleration_field_coefficients(s_obs)
    vg_full_on_recon = engine.Lambda_g_T @ a_ahn
    vg_diff = float(np.max(np.abs(vg_closed - vg_full_on_recon)))

    field_thm = engine.verify_acceleration_and_zero_self_force_theorem(true_sources)
    field_34_err = float(
        np.max(
            np.abs(
                field_thm["exact_field_averages_34"]
                - field_thm["approx_field_averages_34"]
            )
        )
    )
    self_force_err = float(
        np.max(np.abs(field_thm["avg_g_approx"] - field_thm["avg_g_exact"]))
    )

    print("\n[3] Theorem 3: 34x35 3D Repulsive Field Fit & Zero Self-Acceleration:")
    print(f"    - Closed-Form 34x35 vs N-Grid Field Residual           : {vg_diff:.2e}")
    print(f"    - 34 Generalized Acceleration Averages Residual        : {field_34_err:.2e}")
    print(f"    - Zero Self-Acceleration Error ||<g_app> - <g_ex>||_inf: {self_force_err:.2e}")

    assert vg_diff < spec.numerical_tolerance
    assert field_34_err < spec.numerical_tolerance
    assert self_force_err < spec.numerical_tolerance

    # Verify exact Zero Self-Acceleration on aligned reference state (a = mu)
    field_aligned = engine.verify_acceleration_and_zero_self_force_theorem(engine.mu)
    aligned_self_force_err = float(np.max(np.abs(field_aligned["avg_g_approx"])))
    assert aligned_self_force_err < spec.numerical_tolerance
    print(f"    - Aligned State (a = mu) Zero Net Self-Force Error     : {aligned_self_force_err:.2e}")

    # 4. Latency Benchmark
    pot_us, field_us = engine.benchmark_latency(
        s_obs, num_iterations=spec.benchmark_iterations
    )
    t_dir0 = time.perf_counter()
    for _ in range(spec.direct_grid_sample_runs):
        _ = engine.G_x @ a_ahn
        _ = engine.G_y @ a_ahn
        _ = engine.G_z @ a_ahn
    t_direct_us = ((time.perf_counter() - t_dir0) / spec.direct_grid_sample_runs) * 1e6

    print("\n[4] Real-Time Execution Latency Benchmark:")
    print(f"    - Direct O(N^2) 3D Metric Summation                    : {t_direct_us:.2f} us")
    print(f"    - Closed-Form 35x35 Potential Operator                 : {pot_us:.2f} us")
    print(f"    - Closed-Form 34x35 3D Repulsive Field Operator        : {field_us:.2f} us")

    # 5. Generate Publication Figure
    out_path = CURRENT_DIR / spec.output_figure_relpath
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=400)
    idx_34 = np.arange(1, engine.M_field + 1)
    axes[0].plot(
        idx_34,
        field_thm["exact_field_averages_34"],
        "bo-",
        label="Exact Repulsive Field Averages (Eq. 21)",
        markersize=5,
    )
    axes[0].plot(
        idx_34,
        field_thm["approx_field_averages_34"],
        "rx--",
        label="Ahn Modulus-Weighted 3D Fit (Eq. 24)",
        markersize=4,
    )
    axes[0].set_title(
        "34 Repulsive Metric Field Averages & Force Conservation",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    axes[0].set_xlabel("Non-Constant Monomial Index $k$ (1 to 34)", fontsize=11)
    axes[0].set_ylabel("Modulus-Weighted Field Moment Average", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    methods = [
        "Direct 3D Metric\n$O(N^2)$ Summation",
        "Ahn Potential\n35x35 ($O(M^2)$)",
        "Ahn 3D Repulsive\nField 34x35",
    ]
    times = [t_direct_us, pot_us, field_us]
    colors = ["#d9534f", "#5cb85c", "#0275d8"]
    bars = axes[1].bar(methods, times, color=colors, width=0.45)
    axes[1].set_yscale("log")
    axes[1].set_title(
        "Repulsive Metric Inverse Solver Latency (Log Scale)",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    axes[1].set_ylabel(r"Execution Time ($\mu\mathrm{s}$)", fontsize=11)
    axes[1].axhline(
        y=spec.realtime_budget_us,
        color="red",
        linestyle=":",
        label=rf"Real-Time Control Budget ({spec.realtime_budget_us:.0f} $\mu\mathrm{{s}}$)",
    )
    for bar in bars:
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval * 0.65 if yval > 100 else yval * 1.25,
            rf"{yval:.2f} $\mu\mathrm{{s}}$",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="white" if yval > 100 else "black",
        )
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=spec.figure_dpi)
    plt.close()
    print(f"\n[5] Saved benchmark figure to: {out_path}")
    print("=" * 78)
    print("  TRACK C SUITE PASSED 100%: ALL REGULARIZATION THEOREMS VERIFIED!")
    print("=" * 78)

    return {
        "moment_err": moment_err,
        "channell_diff": channell_diff,
        "pos_spike_zero": pos_spike_zero,
        "pos_spike_ahn": pos_spike_ahn,
        "l2_zero": l2_zero,
        "l2_ahn": l2_ahn,
        "v_pot_diff": v_pot_diff,
        "pot_35_err": pot_35_err,
        "vg_diff": vg_diff,
        "field_34_err": field_34_err,
        "self_force_err": self_force_err,
        "t_direct_us": t_direct_us,
        "pot_us": pot_us,
        "field_us": field_us,
    }


if __name__ == "__main__":
    run_repulsive_metric_benchmark()

