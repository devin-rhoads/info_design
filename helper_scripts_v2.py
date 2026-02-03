# helper_scripts.py
# Unified utilities for K-path subnetwork construction, UE/SO benchmarks, and behavioral equilibrium
# (with tidy flow exports and a robust plotting helper)
#
# Design goal (per user request):
#   - "Global" algorithm settings (tolerances, iteration limits, etc.) live in ONE place:
#       SolverConfig
#   - No separate benchmark_tol vs tol; everything uses the same config unless you override it.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List, Callable, Union

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import beta as beta_dist, truncnorm


# =============================================================================
# Global algorithm settings (ONE place)
# =============================================================================

@dataclass(frozen=True)
class SolverConfig:
    """Central place for algorithm knobs so they aren't duplicated across functions.

    Notes:
      - max_iter is used for all iterative loops (UE, SO, BE, and optional reference-point fixed-points).
      - tol is used as the convergence threshold everywhere.
      - sensitivity is used for multinomial logit (softmax temperature).
      - grid_points controls discretization resolution for belief integration.
    """
    max_iter: int = 500
    tol: float = 1e-6
    sensitivity: float = 1.0
    grid_points: int = 100


DEFAULT_CONFIG = SolverConfig()


# =============================================================================
# Network Preparation
# =============================================================================

def build_network_data(
    G: nx.DiGraph,
    s_node: Any,
    t_node: Any,
    K: int = 3,
) -> Optional[Dict[str, Any]]:
    """Build a K-path subnetwork (K shortest simple paths by 'free_flow_time').

    Returns a dictionary:
      - paths: list of node lists (each path)
      - links: sorted list of unique directed edges used by those paths
      - path_link_matrix: DataFrame A (rows=links, cols=paths); A[link, k]=1 if link on path k
      - bpr_params: dict[(u,v)] -> {'free_flow_time','capacity','alpha','beta'}
      - bpr_arrays: NumPy arrays aligned with `links` for fast computations
      - s_node, t_node, K (actual K_found)
      - _graph_base: a copy of the graph (for reproducibility)

    Returns None if no path exists.
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph (directed).")
    if K <= 0:
        raise ValueError("K must be a positive integer.")

    # Generate up to K shortest simple paths by free-flow time
    try:
        gen = nx.shortest_simple_paths(G, s_node, t_node, weight="free_flow_time")
        paths: List[List[Any]] = []
        for i, pth in enumerate(gen):
            if i >= K:
                break
            paths.append(list(pth))
    except nx.NetworkXNoPath:
        warnings.warn(f"No path exists from {s_node} to {t_node}. Returning None.")
        return None
    except nx.NetworkXError as e:
        raise ValueError(f"NetworkX error while enumerating paths: {e}")

    if len(paths) == 0:
        warnings.warn(f"No paths found from {s_node} to {t_node}. Returning None.")
        return None

    K_found = len(paths)
    if K_found < K:
        warnings.warn(f"Requested K={K} paths but only found {K_found}. Proceeding with {K_found}.")

    # Collect all unique links used by these paths
    all_links_set = set()
    for pth in paths:
        for u, v in zip(pth[:-1], pth[1:]):
            all_links_set.add((u, v))
    links = sorted(list(all_links_set))

    # Validate links exist in the graph
    for e in links:
        if e not in G.edges:
            raise KeyError(f"Link {e} found in path list but not present in graph edges.")

    # Build link-path incidence matrix A (links x paths)
    # Build link-path incidence matrix A (links x paths)
    link_to_row = {e: i for i, e in enumerate(links)}
    A_mat = np.zeros((len(links), K_found), dtype=np.float64)
    for k, pth in enumerate(paths):
        for u, v in zip(pth[:-1], pth[1:]):
            A_mat[link_to_row[(u, v)], k] = 1.0

    A = pd.DataFrame(
        A_mat,
        index=pd.MultiIndex.from_tuples(links, names=["u", "v"]),
        columns=list(range(K_found)),
        dtype=np.float64
    )



    # Extract BPR params and build aligned arrays
    required = ["free_flow_time", "capacity", "alpha", "beta"]
    bpr_params: Dict[Tuple[Any, Any], Dict[str, float]] = {}

    n_links = len(links)
    fft_arr = np.zeros(n_links, dtype=np.float64)
    cap_arr = np.zeros(n_links, dtype=np.float64)
    alpha_arr = np.zeros(n_links, dtype=np.float64)
    beta_arr = np.zeros(n_links, dtype=np.float64)

    for i, (u, v) in enumerate(links):
        attrs = G.edges[(u, v)]
        missing = [k for k in required if k not in attrs]
        if missing:
            raise KeyError(f"Edge {(u, v)} is missing required BPR attributes: {missing}")

        fft = float(attrs["free_flow_time"])
        cap = float(attrs["capacity"])
        a = float(attrs["alpha"])
        b = float(attrs["beta"])

        bpr_params[(u, v)] = {"free_flow_time": fft, "capacity": cap, "alpha": a, "beta": b}
        fft_arr[i] = fft
        cap_arr[i] = cap
        alpha_arr[i] = a
        beta_arr[i] = b

    return {
        "paths": paths,
        "links": links,
        "path_link_matrix": A,
        "bpr_params": bpr_params,
        "bpr_arrays": {"fft": fft_arr, "cap": cap_arr, "alpha": alpha_arr, "beta": beta_arr},
        "s_node": s_node,
        "t_node": t_node,
        "K": K_found,
        "_graph_base": G.copy(),
    }


# =============================================================================
# Core calculation helpers
# =============================================================================

def _align_link_series(link_flows: Union[pd.Series, np.ndarray, List[float]], network_data: Dict[str, Any]) -> pd.Series:
    """Return link flows as a Series aligned to network_data['links'] order.

    This protects vectorized routines from accidental index/order mismatches when users pass custom Series.
    Internal calls already align correctly, so this is primarily a safety net.
    """
    links = network_data["links"]
    if isinstance(link_flows, pd.Series):
        s = link_flows.reindex(links).fillna(0.0)
    else:
        arr = np.asarray(link_flows, dtype=np.float64).reshape(-1)
        if arr.shape[0] != len(links):
            raise ValueError(f"link_flows length must be {len(links)}, got {arr.shape[0]}")
        s = pd.Series(arr, index=links, dtype=np.float64)
    return s.astype(np.float64).clip(lower=0.0)


def vectorized_bpr(link_flows: pd.Series, network_data: Dict[str, Any]) -> pd.Series:
    """Vectorized BPR travel time: t = t0 * (1 + alpha * (x/c)^beta).

    Notes:
      - Internally, link flows are always aligned to network_data['links'].
      - If you pass a custom Series, it is reindexed to that same order.
    """
    link_flows = _align_link_series(link_flows, network_data)

    fft = network_data["bpr_arrays"]["fft"]
    cap = network_data["bpr_arrays"]["cap"]
    alpha = network_data["bpr_arrays"]["alpha"]
    beta = network_data["bpr_arrays"]["beta"]

    x = np.asarray(link_flows.values, dtype=np.float64)
    x = np.maximum(x, 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.divide(x, cap, out=np.zeros_like(x), where=(cap != 0))
        pow_term = ratio ** beta
        t = fft * (1.0 + alpha * pow_term)

        # Penalize non-positive capacity links
        bad = cap <= 0
        if np.any(bad):
            t[bad] = fft[bad] * 1000.0

    t = np.nan_to_num(t, nan=0.0, posinf=np.max(fft) * 1000.0, neginf=0.0)
    t = np.maximum(t, fft)  # ensure >= free-flow time

    return pd.Series(t, index=network_data["links"], dtype=np.float64)



def vectorized_m_cost(link_flows: pd.Series, network_data: Dict[str, Any]) -> pd.Series:
    """Marginal cost for BPR.

    For t(x) = t0 * (1 + alpha * (x/c)^beta), the link marginal cost is:
        MC(x) = t(x) + x * t'(x) = t(x) + t0 * alpha * beta * (x/c)^beta.

    The returned Series is aligned to network_data['links'] order.
    """
    link_flows = _align_link_series(link_flows, network_data)

    links = network_data["links"]
    fft = network_data["bpr_arrays"]["fft"]
    cap = network_data["bpr_arrays"]["cap"]
    alpha = network_data["bpr_arrays"]["alpha"]
    beta = network_data["bpr_arrays"]["beta"]

    x = np.asarray(link_flows.values, dtype=np.float64)
    t = vectorized_bpr(link_flows, network_data).values

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = np.divide(x, cap, out=np.zeros_like(x), where=(cap != 0))
        x_dt = fft * alpha * beta * (ratio ** beta)

    mc = np.maximum(t + x_dt, t)
    mc = np.nan_to_num(mc, nan=np.max(fft) * 1000.0, posinf=np.max(fft) * 1000.0, neginf=0.0)
    return pd.Series(mc.astype(np.float64), index=links, dtype=np.float64)



def get_network_state(
    path_flows: Union[np.ndarray, List[float], pd.Series],
    network_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute link flows/times, path times, and TSTT from path flows."""
    pf = np.asarray(path_flows, dtype=np.float64).reshape(-1)
    K = int(network_data["K"])
    if pf.shape[0] != K:
        raise ValueError(f"path_flows length must be {K}, got {pf.shape[0]}")
    pf = np.maximum(pf, 0.0)

    A = network_data["path_link_matrix"].values  # (n_links, K)
    links = network_data["links"]

    link_flows = pd.Series(A @ pf, index=links, dtype=np.float64)
    link_times = vectorized_bpr(link_flows, network_data)

    path_times = pd.Series(A.T @ link_times.values, index=list(range(K)), dtype=np.float64)
    TSTT = float(np.dot(link_flows.values, link_times.values))

    return {
        "path_flows": pd.Series(pf, index=list(range(K)), dtype=np.float64),
        "link_flows": link_flows,
        "link_times": link_times,
        "path_times": path_times,
        "TSTT": TSTT,
    }


def multinomial_logit(
    utilities: Union[np.ndarray, List[float]],
    sensitivity: Optional[float] = None,
    config: SolverConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Stable softmax; if sensitivity is None, uses config.sensitivity."""
    u = np.asarray(utilities, dtype=np.float64)
    u = np.nan_to_num(u, nan=-1e10, posinf=1e10, neginf=-1e10)

    if sensitivity is None:
        sensitivity = float(config.sensitivity)
    if (not np.isfinite(sensitivity)) or sensitivity <= 0:
        sensitivity = 1.0

    u = u - np.max(u)
    z = np.clip(sensitivity * u, -700.0, 700.0)

    expz = np.exp(z)
    s = np.sum(expz)
    if (not np.isfinite(s)) or s <= 0:
        return np.ones_like(u) / float(len(u))

    p = expz / s
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p)
    return p.astype(np.float64)


# =============================================================================
# UE / SO (MSA on K-path subnetwork)
# =============================================================================

def find_network_ue(
    network_data: Dict[str, Any],
    total_demand: float,
    config: SolverConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """K-path UE using MSA (settings from SolverConfig)."""
    if total_demand < 0:
        raise ValueError("total_demand must be nonnegative.")

    A = network_data["path_link_matrix"].values
    links = network_data["links"]
    K = int(network_data["K"])

    path_flows = np.zeros(K, dtype=np.float64)
    link_flows = pd.Series(np.zeros(len(links), dtype=np.float64), index=links, dtype=np.float64)

    converged = False
    it = 0
    for it in range(1, int(config.max_iter) + 1):
        link_times = vectorized_bpr(link_flows, network_data)
        path_times = A.T @ link_times.values

        sp = int(np.argmin(path_times))
        target = np.zeros(K, dtype=np.float64)
        target[sp] = float(total_demand)

        step = 1.0 / it
        new_pf = path_flows + step * (target - path_flows)
        change = np.linalg.norm(new_pf - path_flows)

        path_flows = new_pf
        link_flows = pd.Series(A @ path_flows, index=links, dtype=np.float64)

        if it >= 10 and change < float(config.tol):
            converged = True
            break

    state = get_network_state(path_flows, network_data)
    state["iterations"] = it
    state["converged"] = converged
    return state


def find_network_so(
    network_data: Dict[str, Any],
    total_demand: float,
    config: SolverConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """K-path SO using MSA on marginal costs (settings from SolverConfig)."""
    if total_demand < 0:
        raise ValueError("total_demand must be nonnegative.")

    A = network_data["path_link_matrix"].values
    links = network_data["links"]
    K = int(network_data["K"])

    path_flows = np.zeros(K, dtype=np.float64)
    link_flows = pd.Series(np.zeros(len(links), dtype=np.float64), index=links, dtype=np.float64)

    converged = False
    it = 0
    for it in range(1, int(config.max_iter) + 1):
        mc = vectorized_m_cost(link_flows, network_data)
        path_mc = A.T @ mc.values

        sp = int(np.argmin(path_mc))
        target = np.zeros(K, dtype=np.float64)
        target[sp] = float(total_demand)

        step = 1.0 / it
        new_pf = path_flows + step * (target - path_flows)
        change = np.linalg.norm(new_pf - path_flows)

        path_flows = new_pf
        link_flows = pd.Series(A @ path_flows, index=links, dtype=np.float64)

        if it >= 10 and change < float(config.tol):
            converged = True
            break

    state = get_network_state(path_flows, network_data)
    state["iterations"] = it
    state["converged"] = converged
    return state


# =============================================================================
# Beliefs
# =============================================================================

def generate_belief_grid(
    belief: Dict[str, Any],
    grid_points: Optional[int] = None,
    config: SolverConfig = DEFAULT_CONFIG
) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete grid + weights for belief integration (grid_points defaults to config.grid_points)."""
    if not isinstance(belief, dict):
        raise TypeError("belief must be a dict.")

    if grid_points is None:
        grid_points = int(config.grid_points)

    dist = str(belief.get("dist", "uniform")).lower()
    low = float(belief.get("low", 0.0))
    high = float(belief.get("high", 1.0))

    if high < low:
        low, high = high, low

    if np.isclose(low, high):
        return np.array([low], dtype=np.float64), np.array([1.0], dtype=np.float64)

    grid_points = max(int(grid_points), 2)
    grid = np.linspace(low, high, grid_points, dtype=np.float64)

    if dist == "uniform":
        w = np.ones_like(grid, dtype=np.float64)

    elif dist == "beta":
        a = max(float(belief.get("a", 2.0)), 1e-6)
        b = max(float(belief.get("b", 2.0)), 1e-6)

        x = (grid - low) / (high - low)
        x = np.clip(x, 1e-12, 1 - 1e-12)
        w = beta_dist.pdf(x, a, b).astype(np.float64)
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            w = np.ones_like(grid, dtype=np.float64)

    elif dist == "normal":
        mean = float(belief.get("mean", 0.5 * (low + high)))
        std = float(belief.get("std", 0.25 * (high - low)))
        if std <= 1e-10:
            idx = int(np.argmin(np.abs(grid - mean)))
            w = np.zeros_like(grid, dtype=np.float64)
            w[idx] = 1.0
        else:
            a, b = (low - mean) / std, (high - mean) / std
            w = truncnorm.pdf(grid, a, b, loc=mean, scale=std).astype(np.float64)
            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                w = np.ones_like(grid, dtype=np.float64)

    else:
        w = np.ones_like(grid, dtype=np.float64)

    w = np.clip(w, 0.0, np.inf)
    s = w.sum()
    if s <= 0 or (not np.isfinite(s)):
        w = np.ones_like(grid, dtype=np.float64) / float(len(grid))
    else:
        w = w / s

    return grid, w


def update_belief(prior_belief: Dict[str, Any], signal_belief: Dict[str, Any], credibility: float) -> Dict[str, Any]:
    """Blend prior and signal beliefs with credibility in [0,1]."""
    c = float(np.clip(credibility, 0.0, 1.0))
    prior_dist = str(prior_belief.get("dist", "uniform")).lower()
    signal_dist = str(signal_belief.get("dist", "uniform")).lower()
    if prior_dist != signal_dist:
        raise ValueError(f"Belief dist mismatch: prior={prior_dist}, signal={signal_dist}")

    low = (1 - c) * float(prior_belief.get("low", 0.0)) + c * float(signal_belief.get("low", 0.0))
    high = (1 - c) * float(prior_belief.get("high", 1.0)) + c * float(signal_belief.get("high", 1.0))
    if high < low:
        low, high = high, low

    out = {"dist": prior_dist, "low": low, "high": high}

    if prior_dist == "beta":
        a = (1 - c) * float(prior_belief.get("a", 2.0)) + c * float(signal_belief.get("a", 2.0))
        b = (1 - c) * float(prior_belief.get("b", 2.0)) + c * float(signal_belief.get("b", 2.0))
        out["a"] = max(a, 1e-6)
        out["b"] = max(b, 1e-6)

    elif prior_dist == "normal":
        mean = (1 - c) * float(prior_belief.get("mean", 0.5*(low+high))) + c * float(signal_belief.get("mean", 0.5*(low+high)))
        std = (1 - c) * float(prior_belief.get("std", 0.25*(high-low))) + c * float(signal_belief.get("std", 0.25*(high-low)))
        out["mean"] = mean
        out["std"] = max(std, 1e-6)

    return out


# =============================================================================
# Reference points and strategic utility
# =============================================================================

def compute_ue_reference_point(
    network_data: Dict[str, Any],
    ref_demand: float,
    config: SolverConfig = DEFAULT_CONFIG
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Tau reference vector = UE path times at ref_demand."""
    ue = find_network_ue(network_data, total_demand=ref_demand, config=config)
    tau = ue["path_times"].values.astype(np.float64)
    return tau, ue


def compute_t_ff_scaled_reference_point(
    network_data: Dict[str, Any],
    scalar: float = 1.0
) -> Tuple[np.ndarray, str]:
    """Tau reference vector = min(scalar * free-flow path time), repeated K times."""
    A = network_data["path_link_matrix"].values
    fft = network_data["bpr_arrays"]["fft"]
    link_times = scalar * fft
    path_times = A.T @ link_times
    min_time = float(np.min(path_times))
    K = int(network_data["K"])
    tau = np.full(K, min_time, dtype=np.float64)
    return tau, f"t_ff_scaled_min (scalar={scalar})"


def calculate_strategic_utility(
    total_demand: float,
    network_data: Dict[str, Any],
    belief: Dict[str, Any],
    market_penetration: float,
    qU: np.ndarray,
    qI: np.ndarray,
    behavioral_params: Dict[str, Any],
    tau_refs: Optional[np.ndarray] = None,
    risk_model: str = "variance",
    utility_base: str = "outcome",
    config: SolverConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Compute per-path utility for a group, integrating over its belief grid."""
    p = float(np.clip(market_penetration, 0.0, 1.0))
    K = int(network_data["K"])
    qU = np.asarray(qU, dtype=np.float64).reshape(-1)
    qI = np.asarray(qI, dtype=np.float64).reshape(-1)
    if qU.size != K or qI.size != K:
        raise ValueError("qU and qI must be length K.")

    demand_grid, w = generate_belief_grid(belief, config=config)

    imagined_path_times = np.zeros((len(demand_grid), K), dtype=np.float64)
    for i, D in enumerate(demand_grid):
        pf = (1 - p) * float(D) * qU + p * float(D) * qI
        state = get_network_state(pf, network_data)
        imagined_path_times[i, :] = state["path_times"].values

    risk_model = str(risk_model).lower()
    utility_base = str(utility_base).lower()
    if tau_refs is None:
        tau_refs = np.zeros(K, dtype=np.float64)
    tau_refs = np.asarray(tau_refs, dtype=np.float64).reshape(-1)
    if tau_refs.size != K:
        raise ValueError("tau_refs must be length K.")

    if risk_model == "variance":
        alpha_gain = float(behavioral_params.get("alpha_gain", 0.0))
        beta_loss = float(behavioral_params.get("beta_loss", 0.0))
        r_risk = float(behavioral_params.get("r_risk", 0.0))

        mean_t = (w[:, None] * imagined_path_times).sum(axis=0)

        gains = np.maximum(0.0, tau_refs[None, :] - imagined_path_times)
        losses = np.maximum(0.0, imagined_path_times - tau_refs[None, :])

        Eg = (w[:, None] * gains).sum(axis=0)
        El = (w[:, None] * losses).sum(axis=0)

        var = (w[:, None] * (imagined_path_times - mean_t[None, :]) ** 2).sum(axis=0)

        u = -mean_t + alpha_gain * Eg - beta_loss * El - r_risk * var

    elif risk_model in ("ara", "rra"):
        a_risk = float(behavioral_params.get("a_risk", 0.01))
        gamma = float(behavioral_params.get("gamma_risk", 0.5))

        if utility_base == "deviation":
            x = imagined_path_times - tau_refs[None, :]
        else:
            x = imagined_path_times

        if risk_model == "ara":
            util_out = -np.exp(np.clip(a_risk * x, -700.0, 700.0))
        else:
            eps = 1e-12
            denom = (1 - gamma) if abs(1 - gamma) > eps else 1.0
            if utility_base == "deviation":
                util_out = -np.sign(x) * (np.abs(x) + eps) ** (1 - gamma) / denom
            else:
                util_out = -(np.maximum(x, eps) ** (1 - gamma)) / denom

        u = (w[:, None] * util_out).sum(axis=0)

    else:
        u = -(w[:, None] * imagined_path_times).sum(axis=0)

    u = np.nan_to_num(u, nan=-1e10, posinf=1e10, neginf=-1e10)
    return u.astype(np.float64)


def compute_behavioral_reference_point(
    total_demand: float,
    network_data: Dict[str, Any],
    belief: Dict[str, Any],
    market_penetration: float,
    behavioral_params: Dict[str, Any],
    risk_model: str = "variance",
    utility_base: str = "outcome",
    config: SolverConfig = DEFAULT_CONFIG
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Optional: fixed-point tau reference vector consistent with a single-belief equilibrium."""
    K = int(network_data["K"])
    p = float(np.clip(market_penetration, 0.0, 1.0))

    state0 = get_network_state(np.zeros(K), network_data)
    tau = state0["path_times"].values.astype(np.float64)

    q = np.ones(K, dtype=np.float64) / K
    last_state = state0

    for outer in range(1, int(config.max_iter) + 1):
        q_inner = q.copy()
        for inner in range(1, int(config.max_iter) + 1):
            u = calculate_strategic_utility(
                total_demand=total_demand,
                network_data=network_data,
                belief=belief,
                market_penetration=p,
                qU=q_inner,
                qI=q_inner,
                behavioral_params=behavioral_params,
                tau_refs=tau,
                risk_model=risk_model,
                utility_base=utility_base,
                config=config
            )
            target = multinomial_logit(u, config=config)
            step = 1.0 / inner
            q_new = q_inner + step * (target - q_inner)
            if inner >= 10 and np.linalg.norm(q_new - q_inner) < float(config.tol):
                q_inner = q_new
                break
            q_inner = q_new

        pf = float(total_demand) * q_inner
        last_state = get_network_state(pf, network_data)
        tau_new = last_state["path_times"].values.astype(np.float64)

        if outer >= 5 and np.linalg.norm(tau_new - tau) < float(config.tol):
            tau = tau_new
            q = q_inner
            break

        tau = tau_new
        q = q_inner

    return tau, last_state


# =============================================================================
# Mixed strategy equilibrium (two groups)
# =============================================================================

def find_mixed_strategy_equilibrium(
    total_demand: float,
    network_data: Dict[str, Any],
    prior_belief_demand: Dict[str, Any],
    signal_belief_demand: Dict[str, Any],
    market_penetration: float,
    credibility: float,
    behavioral_params: Dict[str, Any],
    reference_method: str = "ue",
    risk_model: str = "variance",
    utility_base: str = "outcome",
    t_ff_scalar: float = 1.0,
    run_benchmarks: bool = True,
    config: SolverConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Two-group mixed-strategy equilibrium. All iterative settings come from SolverConfig."""
    if total_demand < 0:
        raise ValueError("total_demand must be nonnegative.")

    p = float(np.clip(market_penetration, 0.0, 1.0))
    K = int(network_data["K"])

    results: Dict[str, Any] = {
        "inputs": {
            "total_demand": float(total_demand),
            "market_penetration": p,
            "credibility": float(credibility),
            "reference_method": str(reference_method),
            "risk_model": str(risk_model),
            "utility_base": str(utility_base),
            "t_ff_scalar": float(t_ff_scalar),
            "config": {
                "max_iter": int(config.max_iter),
                "tol": float(config.tol),
                "sensitivity": float(config.sensitivity),
                "grid_points": int(config.grid_points),
            },
        },
        "beliefs": {},
        "benchmarks": {"UE": None, "SO": None},
        "reference_point": {},
        "utilities": {},
        "equilibrium": {},
        "convergence": {},
        "iteration_history": [],
    }

    if run_benchmarks:
        try:
            results["benchmarks"]["UE"] = find_network_ue(network_data, total_demand, config=config)
        except Exception as e:
            warnings.warn(f"UE benchmark failed: {e}")
            results["benchmarks"]["UE"] = None

        try:
            results["benchmarks"]["SO"] = find_network_so(network_data, total_demand, config=config)
        except Exception as e:
            warnings.warn(f"SO benchmark failed: {e}")
            results["benchmarks"]["SO"] = None

    blended = update_belief(prior_belief_demand, signal_belief_demand, credibility)
    results["beliefs"] = {"prior": prior_belief_demand, "signal": signal_belief_demand, "blended": blended}

    reference_method = str(reference_method).lower()
    tau = np.zeros(K, dtype=np.float64)
    ref_meta: Dict[str, Any] = {"method": reference_method}

    if risk_model.lower() == "variance" or utility_base.lower() == "deviation":
        if reference_method == "ue":
            ref_demand = float(prior_belief_demand.get("high", total_demand))
            tau, ue_ref = compute_ue_reference_point(network_data, ref_demand=ref_demand, config=config)
            ref_meta.update({"ref_demand": ref_demand, "ue_ref": ue_ref})
        elif reference_method == "behavioral":
            tau, last_state = compute_behavioral_reference_point(
                total_demand=total_demand,
                network_data=network_data,
                belief=blended,
                market_penetration=p,
                behavioral_params=behavioral_params,
                risk_model=risk_model,
                utility_base=utility_base,
                config=config
            )
            ref_meta.update({"last_state": last_state})
        elif reference_method == "t_ff_scaled":
            tau, label = compute_t_ff_scaled_reference_point(network_data, scalar=t_ff_scalar)
            ref_meta.update({"label": label})
        else:
            raise ValueError(f"Unknown reference_method: {reference_method}")

    results["reference_point"] = {"tau": tau, "meta": ref_meta}

    qU = np.ones(K, dtype=np.float64) / K
    qI = np.ones(K, dtype=np.float64) / K

    converged = False
    change = np.inf
    it = 0

    for it in range(1, int(config.max_iter) + 1):
        uU = calculate_strategic_utility(
            total_demand=total_demand,
            network_data=network_data,
            belief=prior_belief_demand,
            market_penetration=p,
            qU=qU,
            qI=qI,
            behavioral_params=behavioral_params,
            tau_refs=tau,
            risk_model=risk_model,
            utility_base=utility_base,
            config=config
        )
        uI = calculate_strategic_utility(
            total_demand=total_demand,
            network_data=network_data,
            belief=blended,
            market_penetration=p,
            qU=qU,
            qI=qI,
            behavioral_params=behavioral_params,
            tau_refs=tau,
            risk_model=risk_model,
            utility_base=utility_base,
            config=config
        )

        targetU = multinomial_logit(uU, config=config)
        targetI = multinomial_logit(uI, config=config)

        step = 1.0 / it
        qU_new = qU + step * (targetU - qU)
        qI_new = qI + step * (targetI - qI)

        change = float(np.linalg.norm(qU_new - qU) + np.linalg.norm(qI_new - qI))
        qU, qI = qU_new, qI_new

        if it == 1 or it % 10 == 0:
            results["iteration_history"].append({
                "iteration": it,
                "change": change,
                "qU": qU.copy(),
                "qI": qI.copy(),
                "uU": uU.copy(),
                "uI": uI.copy(),
            })

        if it >= 10 and change < float(config.tol):
            converged = True
            break

    final_path_flows = (1 - p) * float(total_demand) * qU + p * float(total_demand) * qI
    final_state = get_network_state(final_path_flows, network_data)

    results["utilities"]["final_utility_U"] = uU
    results["utilities"]["final_utility_I"] = uI

    results["equilibrium"] = {
        "qU_vec": qU,
        "qI_vec": qI,
        "final_path_flows_vec": final_path_flows,
        "final_state": final_state,
        "iterations": it,
    }
    results["convergence"] = {"converged": converged, "final_change": change}

    return results


# =============================================================================
# Tidy exports (model-tagged link/path tables)
# =============================================================================

def state_to_link_df(state: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """State -> per-link tidy DataFrame."""
    lf = state["link_flows"]
    lt = state["link_times"]
    df = pd.DataFrame({"link": lf.index, "flow": lf.values, "time": lt.loc[lf.index].values})
    df["u"] = df["link"].apply(lambda e: e[0])
    df["v"] = df["link"].apply(lambda e: e[1])
    df["model"] = str(model_name)
    return df[["model", "u", "v", "link", "flow", "time"]]


def state_to_path_df(state: Dict[str, Any], model_name: str) -> pd.DataFrame:
    """State -> per-path tidy DataFrame."""
    pf = state["path_flows"]
    pt = state["path_times"]
    df = pd.DataFrame({"path_id": pf.index, "flow": pf.values, "time": pt.loc[pf.index].values})
    df["model"] = str(model_name)
    return df[["model", "path_id", "flow", "time"]]


def collect_model_tables(
    ue: Optional[Dict[str, Any]] = None,
    so: Optional[Dict[str, Any]] = None,
    be: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """Combine UE/SO/BE into concatenated link/path tables."""
    link_frames = []
    path_frames = []

    if ue is not None:
        link_frames.append(state_to_link_df(ue, "UE"))
        path_frames.append(state_to_path_df(ue, "UE"))
    if so is not None:
        link_frames.append(state_to_link_df(so, "SO"))
        path_frames.append(state_to_path_df(so, "SO"))
    if be is not None:
        be_state = be
        if isinstance(be, dict) and "equilibrium" in be and "final_state" in be["equilibrium"]:
            be_state = be["equilibrium"]["final_state"]
        link_frames.append(state_to_link_df(be_state, "BE"))
        path_frames.append(state_to_path_df(be_state, "BE"))

    return {
        "link_table": pd.concat(link_frames, ignore_index=True) if link_frames else pd.DataFrame(),
        "path_table": pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame(),
    }


# =============================================================================
# Plotting helper
# =============================================================================

def plot_network_flows(
    *,
    state: Optional[Dict[str, Any]] = None,
    G: Optional[nx.DiGraph] = None,
    graph_builder: Optional[Callable[[], nx.DiGraph]] = None,

    # DataFrame mode (fallback)
    df: Optional[pd.DataFrame] = None,
    flow_cols: Optional[List[str]] = None,
    edge_mapping: Optional[Dict[str, Tuple[Any, Any]]] = None,

    # Presentation
    model_name: str = "",
    title: Optional[str] = None,
    pos: Optional[Dict[Any, Tuple[float, float]]] = None,
    annotate_times: bool = True,
    annotate_flows: bool = True,
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 150,
) -> Tuple[Any, Any, Dict[Tuple[Any, Any], str], Optional[float]]:
    """Plot a directed graph with flow and BPR time labels per edge."""
    import matplotlib.pyplot as plt

    if G is None:
        if graph_builder is None:
            raise ValueError("Provide either G or graph_builder.")
        G = graph_builder()
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph (directed).")

    Gp = G.copy()

    flows_by_edge: Dict[Tuple[Any, Any], float] = {}
    TSTT_val: Optional[float] = None

    if state is not None:
        lf = state.get("link_flows", None)
        if lf is None:
            raise KeyError("state must contain 'link_flows'.")
        for (u, v), f in lf.items():
            flows_by_edge[(u, v)] = float(f)
        if "TSTT" in state:
            TSTT_val = float(state["TSTT"])

    elif df is not None and flow_cols is not None and edge_mapping is not None:
        if len(df) == 0:
            raise ValueError("df is empty.")
        row = df.iloc[0]
        missing_cols = [c for c in flow_cols if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing flow columns in df: {missing_cols}")
        for col in flow_cols:
            if col not in edge_mapping:
                raise KeyError(f"edge_mapping is missing a mapping for column '{col}'")
            e = edge_mapping[col]
            flows_by_edge[e] = float(row[col])

    else:
        raise ValueError("Provide either (state) or (df, flow_cols, edge_mapping).")

    required = ["free_flow_time", "capacity", "alpha", "beta"]
    for (u, v) in Gp.edges:
        attrs = Gp.edges[(u, v)]
        missing = [k for k in required if k not in attrs]
        if missing:
            raise KeyError(f"Edge {(u, v)} missing required BPR attrs: {missing}")

    for (u, v) in Gp.edges:
        Gp.edges[(u, v)]["flow"] = float(flows_by_edge.get((u, v), 0.0))

    def bpr_scalar(flow: float, t0: float, cap: float, a: float, b: float) -> float:
        flow = max(flow, 0.0)
        if cap <= 0:
            return t0 * 1000.0
        return t0 * (1.0 + a * (flow / cap) ** b)

    edge_labels: Dict[Tuple[Any, Any], str] = {}
    for (u, v) in Gp.edges:
        attrs = Gp.edges[(u, v)]
        f = float(Gp.edges[(u, v)].get("flow", 0.0))
        tt = bpr_scalar(f, float(attrs["free_flow_time"]), float(attrs["capacity"]), float(attrs["alpha"]), float(attrs["beta"]))

        parts = []
        if annotate_times:
            parts.append(f"t: {tt:.2f}")
        if annotate_flows:
            parts.append(f"f: {f:.0f}")
        edge_labels[(u, v)] = "\n".join(parts) if parts else ""

    if pos is None and set(Gp.nodes) >= {"s", "b", "c", "t"}:
        pos = {"s": (0, 2), "b": (1, 1), "c": (1, 3), "t": (2, 2)}

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(Gp, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=14,
            font_weight="bold", arrowsize=20, ax=ax)
    nx.draw_networkx_edge_labels(Gp, pos, edge_labels=edge_labels, font_color="red", font_size=10, ax=ax)

    if title is None:
        title = model_name if model_name else "Network flows"
    if TSTT_val is not None:
        ax.set_title(f"{title} - TSTT: {TSTT_val:.2f}", fontsize=16)  # ASCII-only dash
    else:
        ax.set_title(title, fontsize=16)

    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax, edge_labels, TSTT_val
