# helper_scripts.py
# Unified utilities for K-path subnetwork construction, UE/SO benchmarks, and behavioral equilibrium
# (with tidy flow exports and a robust plotting helper)
#
# Design goal (per user request):
#   - "Global" algorithm settings (tolerances, iteration limits, etc.) live in ONE place:
#       SolverConfig
#   - No separate benchmark_tol vs tol; everything uses the same config unless you override it.

from __future__ import annotations


# === Version ===
__version__ = "5.0.0-networkwide-trace"

import warnings
import os
import json
import gzip
import pickle
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List, Callable, Union, Iterable

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import beta as beta_dist, truncnorm


# =============================================================================
# Global algorithm settings (ONE place)
# =============================================================================


# =============================================================================
# Numeric safety constants
# =============================================================================
# exp(x) overflows in float64 for x > ~709; we clip a bit below that.
MAX_EXP_ARG: float = 700.0

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
    exp_clip: float = 700.0  # clip for ARA exp(a*x) to avoid overflow


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

def _decompose_max_flow_paths(
    G: nx.DiGraph,
    s_node: Any,
    t_node: Any,
    flow_func: Callable = nx.algorithms.flow.dinitz,
    flow_eps: float = 1e-9,
    max_paths: Optional[int] = None,
    capacity_attr: str = "capacity",
) -> List[Tuple[List[Any], float]]:
    val, flow = nx.maximum_flow(G, s_node, t_node, capacity=capacity_attr, flow_func=flow_func)
    if val <= float(flow_eps):
        return []
    residual: Dict[Any, Dict[Any, float]] = {}
    for u, nbrs in flow.items():
        for v, f in nbrs.items():
            if f > flow_eps:
                residual.setdefault(u, {})[v] = float(f)
    paths: List[Tuple[List[Any], float]] = []
    while True:
        stack = [(s_node, [s_node])]
        visited = set()
        found = None
        while stack:
            node, path = stack.pop()
            if node == t_node:
                found = path
                break
            if node in visited:
                continue
            visited.add(node)
            for v, f in residual.get(node, {}).items():
                if f > flow_eps:
                    stack.append((v, path + [v]))
        if found is None:
            break
        bottleneck = min(residual[found[i]][found[i+1]] for i in range(len(found)-1))
        paths.append((found, float(bottleneck)))
        for i in range(len(found)-1):
            u, v = found[i], found[i+1]
            residual[u][v] -= bottleneck
            if residual[u][v] <= flow_eps:
                residual[u].pop(v, None)
        if max_paths is not None and len(paths) >= max_paths:
            break
    return paths

def build_network_data_from_flow_decomposition(
    G: nx.DiGraph,
    s_node: Any,
    t_node: Any,
    flow_func: Callable = nx.algorithms.flow.dinitz,
    flow_eps: float = 1e-9,
    max_paths: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    paths_with_flow = _decompose_max_flow_paths(G, s_node, t_node, flow_func=flow_func, flow_eps=flow_eps, max_paths=max_paths)
    if len(paths_with_flow) == 0:
        return None
    paths = [p for p, _ in paths_with_flow]
    path_caps = np.asarray([f for _, f in paths_with_flow], dtype=np.float64)
    all_links_set = set()
    for pth in paths:
        for u, v in zip(pth[:-1], pth[1:]):
            all_links_set.add((u, v))
    links = sorted(list(all_links_set))
    link_to_row = {e: i for i, e in enumerate(links)}
    K_found = len(paths)
    A_mat = np.zeros((len(links), K_found), dtype=np.float64)
    for k, pth in enumerate(paths):
        for u, v in zip(pth[:-1], pth[1:]):
            A_mat[link_to_row[(u, v)], k] = 1.0
    A = pd.DataFrame(
        A_mat,
        index=pd.MultiIndex.from_tuples(links, names=["u", "v"]),
        columns=list(range(K_found)),
        dtype=np.float64,
    )
    required = ["free_flow_time", "capacity", "alpha", "beta"]
    fft_arr = np.zeros(len(links), dtype=np.float64)
    cap_arr = np.zeros(len(links), dtype=np.float64)
    alpha_arr = np.zeros(len(links), dtype=np.float64)
    beta_arr = np.zeros(len(links), dtype=np.float64)
    for i, (u, v) in enumerate(links):
        attrs = G.edges[(u, v)]
        missing = [k for k in required if k not in attrs]
        if missing:
            raise KeyError(f"Edge {(u, v)} missing attributes {missing}")
        fft_arr[i] = float(attrs["free_flow_time"])
        cap_arr[i] = float(attrs["capacity"])
        alpha_arr[i] = float(attrs["alpha"])
        beta_arr[i] = float(attrs["beta"])
    return {
        "paths": paths,
        "path_flow_caps": path_caps,
        "links": links,
        "path_link_matrix": A,
        "bpr_params": {(u, v): {"free_flow_time": fft_arr[i], "capacity": cap_arr[i], "alpha": alpha_arr[i], "beta": beta_arr[i]}
                       for i, (u, v) in enumerate(links)},
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
    z = np.clip(sensitivity * u, -MAX_EXP_ARG, MAX_EXP_ARG)

    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
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
            with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                util_out = -np.exp(np.clip(a_risk * x, -MAX_EXP_ARG, MAX_EXP_ARG))
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



# =============================================================================
# v3 network-wide extensions + belief stability patches (2026-02-02)
# =============================================================================
# This section adds:
#   - log-space belief discretization (beta/normal tails no longer machine-zero)
#   - optional linear pooling of prior + signal on a common grid (supports mixing dist families)
#   - network-wide (multi-OD) UE, SO, and two-group behavioral equilibrium with a global theta
#     demand multiplier belief applied to every OD demand.

def _stable_softmax_log_weights(logw: np.ndarray, eps_w: float = 1e-300) -> np.ndarray:
    """Convert log-weights to normalized weights stably, then apply a tiny positive floor."""
    logw = np.asarray(logw, dtype=np.float64)
    m = np.nanmax(logw)
    if not np.isfinite(m):
        w = np.ones_like(logw, dtype=np.float64)
        w = w / w.sum()
    else:
        w = np.exp(logw - m)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        s = w.sum()
        if s <= 0 or (not np.isfinite(s)):
            w = np.ones_like(logw, dtype=np.float64) / float(len(logw))
        else:
            w = w / s

    # Avoid machine-zeros at extreme tails
    eps_w = float(max(eps_w, 0.0))
    if eps_w > 0:
        w = np.maximum(w, eps_w)
        w = w / w.sum()
    return w


def _resolve_bounds(belief: Dict[str, Any], demand_ref: Optional[float] = None) -> Tuple[float, float]:
    """
    Resolve (low, high) support for a belief.
    Supports either:
      - explicit low/high
      - low_mult/high_mult (multiplied by demand_ref)
    """
    if not isinstance(belief, dict):
        raise TypeError("belief must be a dict.")
    if ("low_mult" in belief) or ("high_mult" in belief):
        if demand_ref is None:
            raise ValueError("belief uses low_mult/high_mult but demand_ref was not provided.")
        low = float(belief.get("low_mult", 0.0)) * float(demand_ref)
        high = float(belief.get("high_mult", 1.0)) * float(demand_ref)
    else:
        low = float(belief.get("low", 0.0))
        high = float(belief.get("high", 1.0))

    if high < low:
        low, high = high, low
    if high == low:
        high = low + 1e-9
    return low, high


def generate_belief_grid(
    belief: Dict[str, Any],
    grid_points: Optional[int] = None,
    config: SolverConfig = DEFAULT_CONFIG,
    demand_ref: Optional[float] = None,
    eps_w: float = 1e-300
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete grid + weights for belief integration.

    Changes vs v2:
      - beta/normal evaluated in log-space to prevent sharp machine-zero truncation
      - optional low_mult/high_mult bounds (scaled by demand_ref)
      - supports dist="grid" (pre-discretized weights, used by linear pooling)
      - applies an extremely small positive weight floor eps_w to keep extreme tails nonzero
    """
    if grid_points is None:
        grid_points = int(config.grid_points)

    dist = str(belief.get("dist", "uniform")).lower()

    # Pre-discretized belief
    if dist in ("grid", "discrete"):
        grid = np.asarray(belief.get("grid", []), dtype=np.float64)
        w = np.asarray(belief.get("weights", []), dtype=np.float64)
        if grid.ndim != 1 or w.ndim != 1 or len(grid) != len(w) or len(grid) == 0:
            raise ValueError("dist='grid' requires 1D arrays belief['grid'] and belief['weights'] of same length.")
        w = np.clip(w, 0.0, np.inf)
        s = w.sum()
        if s <= 0 or (not np.isfinite(s)):
            w = np.ones_like(w, dtype=np.float64) / float(len(w))
        else:
            w = w / s
        if eps_w > 0:
            w = np.maximum(w, float(eps_w))
            w = w / w.sum()
        return grid, w

    low, high = _resolve_bounds(belief, demand_ref=demand_ref)
    grid = np.linspace(low, high, int(grid_points), dtype=np.float64)

    if dist == "uniform":
        w = np.ones_like(grid, dtype=np.float64)
        w = w / w.sum()
        if eps_w > 0:
            w = np.maximum(w, float(eps_w))
            w = w / w.sum()
        return grid, w

    if dist == "beta":
        a = max(float(belief.get("a", 2.0)), 1e-6)
        b = max(float(belief.get("b", 2.0)), 1e-6)

        x = (grid - low) / (high - low)
        x = np.clip(x, 1e-15, 1 - 1e-15)
        logw = beta_dist.logpdf(x, a, b).astype(np.float64)
        w = _stable_softmax_log_weights(logw, eps_w=eps_w)
        return grid, w

    if dist == "normal":
        mean = float(belief.get("mean", 0.5 * (low + high)))
        std = float(belief.get("std", 0.25 * (high - low)))
        std = max(std, 1e-10)

        a, b = (low - mean) / std, (high - mean) / std
        tn = truncnorm(a, b, loc=mean, scale=std)
        logw = tn.logpdf(grid).astype(np.float64)
        w = _stable_softmax_log_weights(logw, eps_w=eps_w)
        return grid, w

    # Fallback: uniform
    w = np.ones_like(grid, dtype=np.float64) / float(len(grid))
    if eps_w > 0:
        w = np.maximum(w, float(eps_w))
        w = w / w.sum()
    return grid, w


def update_belief(
    prior_belief: Dict[str, Any],
    signal_belief: Dict[str, Any],
    credibility: float,
    method: str = "parametric",
    config: SolverConfig = DEFAULT_CONFIG,
    grid_points: Optional[int] = None,
    demand_ref: Optional[float] = None,
    eps_w: float = 1e-300
) -> Dict[str, Any]:
    """
    Update belief using either:
      - method="parametric": interpolate distribution parameters (requires same dist family, like v2)
      - method="linear_pool": discretize both beliefs on a common grid and mix weights directly
        (supports mixing different dist families, e.g., uniform prior + beta signal)
    """
    if not isinstance(prior_belief, dict) or not isinstance(signal_belief, dict):
        raise TypeError("prior_belief and signal_belief must be dicts.")

    c = float(np.clip(credibility, 0.0, 1.0))
    method = str(method).lower().strip()

    if method not in ("parametric", "linear_pool", "linear", "pool"):
        raise ValueError("method must be 'parametric' or 'linear_pool'.")

    if method == "parametric":
        prior_dist = str(prior_belief.get("dist", "uniform")).lower()
        signal_dist = str(signal_belief.get("dist", "uniform")).lower()
        if prior_dist != signal_dist:
            raise ValueError(f"Belief dist mismatch under parametric update: prior={prior_dist}, signal={signal_dist}")

        low_p, high_p = _resolve_bounds(prior_belief, demand_ref=demand_ref)
        low_s, high_s = _resolve_bounds(signal_belief, demand_ref=demand_ref)
        low = (1 - c) * low_p + c * low_s
        high = (1 - c) * high_p + c * high_s
        if high < low:
            low, high = high, low

        out: Dict[str, Any] = {"dist": prior_dist, "low": float(low), "high": float(high)}

        if prior_dist == "beta":
            a = (1 - c) * float(prior_belief.get("a", 2.0)) + c * float(signal_belief.get("a", 2.0))
            b = (1 - c) * float(prior_belief.get("b", 2.0)) + c * float(signal_belief.get("b", 2.0))
            out["a"] = float(max(a, 1e-6))
            out["b"] = float(max(b, 1e-6))

        elif prior_dist == "normal":
            mean = (1 - c) * float(prior_belief.get("mean", 0.5 * (low_p + high_p))) + c * float(signal_belief.get("mean", 0.5 * (low_s + high_s)))
            std = (1 - c) * float(prior_belief.get("std", 0.25 * (high_p - low_p))) + c * float(signal_belief.get("std", 0.25 * (high_s - low_s)))
            out["mean"] = float(mean)
            out["std"] = float(max(std, 1e-10))

        return out

    # --- linear pooling on a common grid ---
    if grid_points is None:
        grid_points = int(config.grid_points)

    # Determine a common support = union of supports
    low_p, high_p = _resolve_bounds(prior_belief, demand_ref=demand_ref)
    low_s, high_s = _resolve_bounds(signal_belief, demand_ref=demand_ref)
    low = float(min(low_p, low_s))
    high = float(max(high_p, high_s))
    if high <= low:
        high = low + 1e-9

    common_grid = np.linspace(low, high, int(grid_points), dtype=np.float64)

    # Evaluate each belief on this common grid by temporarily overriding bounds
    def _as_common(b: Dict[str, Any]) -> Dict[str, Any]:
        bb = dict(b)
        bb.pop("low_mult", None)
        bb.pop("high_mult", None)
        bb["low"] = low
        bb["high"] = high
        return bb

    g1, w1 = generate_belief_grid(_as_common(prior_belief), grid_points=int(grid_points), config=config, demand_ref=None, eps_w=eps_w)
    g2, w2 = generate_belief_grid(_as_common(signal_belief), grid_points=int(grid_points), config=config, demand_ref=None, eps_w=eps_w)

    # Mix weights directly
    w_post = (1 - c) * w1 + c * w2
    w_post = np.maximum(w_post, float(eps_w))
    w_post = w_post / w_post.sum()

    return {"dist": "grid", "grid": common_grid, "weights": w_post}


# =============================================================================
# Network-wide (multi-OD) data + solvers
# =============================================================================

ODPair = Tuple[Any, Any]


def build_multi_od_network_data(
    G: nx.DiGraph,
    od_pairs: Iterable[ODPair],
    K: Optional[int] = None,
    drop_ods_with_no_path: bool = True,
    path_builder: str = "k_shortest",
    max_decomp_paths: Optional[int] = None,
    flow_eps: float = 1e-9,
    flow_func: Callable = nx.algorithms.flow.dinitz,
) -> Dict[str, Any]:
    """
    Build per-OD path sets and a global link index for joint congestion.
    path_builder: "k_shortest" (uses K shortest) or "maxflow" (Dinic + decomposition).
    """
    if not isinstance(G, nx.DiGraph):
        raise TypeError("G must be a networkx.DiGraph.")
    if path_builder not in ("k_shortest", "maxflow"):
        raise ValueError("path_builder must be 'k_shortest' or 'maxflow'.")
    if path_builder == "k_shortest" and (K is None or K <= 0):
        raise ValueError("K must be positive when path_builder='k_shortest'.")

    od_list = list(od_pairs)
    if len(od_list) == 0:
        raise ValueError("od_pairs must be non-empty.")

    per_od_local: Dict[ODPair, Dict[str, Any]] = {}
    used_links = set()

    for od in od_list:
        o, d = od
        if path_builder == "k_shortest":
            net = build_network_data(G, s_node=o, t_node=d, K=int(K))
            path_caps = None
        else:
            net = build_network_data_from_flow_decomposition(
                G, s_node=o, t_node=d,
                flow_func=flow_func,
                flow_eps=flow_eps,
                max_paths=max_decomp_paths,
            )
            path_caps = None if net is None else net["path_flow_caps"]

        if net is None:
            if drop_ods_with_no_path:
                continue
            raise ValueError(f"No path exists for OD={od}.")

        per_od_local[od] = {
            "paths": net["paths"],
            "K": int(net["K"]),
            "links_local": list(net["links"]),
            "A_local": net["path_link_matrix"],  # DataFrame
            "path_flow_caps": path_caps,
        }
        used_links.update(net["links"])

    if len(per_od_local) == 0:
        raise ValueError("No OD pairs had at least one feasible path (after filtering).")

    # Global link order: preserve graph order, then add any remaining used links
    global_links = [e for e in G.edges() if e in used_links]
    for e in used_links:
        if e not in set(global_links):
            global_links.append(e)

    idx = {e: i for i, e in enumerate(global_links)}
    L = len(global_links)

    # Build global BPR arrays
    fft = np.zeros(L, dtype=np.float64)
    cap = np.zeros(L, dtype=np.float64)
    alpha = np.zeros(L, dtype=np.float64)
    beta = np.zeros(L, dtype=np.float64)
    for i, (u, v) in enumerate(global_links):
        data = G[u][v]
        for key in ("free_flow_time", "capacity", "alpha", "beta"):
            if key not in data:
                raise KeyError(f"Edge ({u},{v}) missing attribute '{key}'.")
        fft[i] = float(data["free_flow_time"])
        cap[i] = float(data["capacity"])
        alpha[i] = float(data["alpha"])
        beta[i] = float(data["beta"])

    per_od: Dict[ODPair, Dict[str, Any]] = {}
    for od, dd in per_od_local.items():
        K_found = int(dd["K"])
        A_local: pd.DataFrame = dd["A_local"]
        links_local = dd["links_local"]
        A_glob = np.zeros((L, K_found), dtype=np.float64)
        local_vals = A_local.values.astype(np.float64)
        for r, e in enumerate(links_local):
            A_glob[idx[e], :] = local_vals[r, :]
        per_od[od] = {
            "paths": dd["paths"],
            "K": K_found,
            "A_global": A_glob,
            "links_local": links_local,
            "path_flow_caps": dd.get("path_flow_caps"),
        }

    return {
        "ods": list(per_od.keys()),
        "per_od": per_od,
        "global_links": global_links,
        "bpr_arrays": {"fft": fft, "cap": cap, "alpha": alpha, "beta": beta},
        "_graph_base": G.copy(),
        "path_builder": path_builder,
        "max_decomp_paths": max_decomp_paths,
        "flow_eps": flow_eps,
    }



def _bpr_times_from_x(x: np.ndarray, bpr: Dict[str, np.ndarray]) -> np.ndarray:
    """Vectorized BPR travel times. Accepts x as (L,) or (N,L). Returns same shape."""
    fft = bpr["fft"]
    cap = bpr["cap"]
    alpha = bpr["alpha"]
    beta = bpr["beta"]

    x = np.asarray(x, dtype=np.float64)
    # Broadcast to (..., L)
    ratio = np.maximum(x, 0.0) / np.maximum(cap, 1e-12)
    return fft * (1.0 + alpha * np.power(ratio, beta))


def _bpr_marginal_costs_from_x(x: np.ndarray, bpr: Dict[str, np.ndarray]) -> np.ndarray:
    """Marginal cost for SO: t(x) + x * dt/dx."""
    fft = bpr["fft"]
    cap = bpr["cap"]
    alpha = bpr["alpha"]
    beta = bpr["beta"]

    x = np.asarray(x, dtype=np.float64)
    ratio = np.maximum(x, 0.0) / np.maximum(cap, 1e-12)
    t = fft * (1.0 + alpha * np.power(ratio, beta))
    # dt/dx = fft * alpha * beta * (x/cap)^(beta-1) * 1/cap
    # handle beta < 1 safely with max(ratio, tiny)
    ratio_safe = np.maximum(ratio, 1e-15)
    dtdx = fft * alpha * beta * np.power(ratio_safe, beta - 1.0) / np.maximum(cap, 1e-12)
    return t + np.maximum(x, 0.0) * dtdx


def _state_from_path_flows_multi(
    net: Dict[str, Any],
    path_flows_by_od: Dict[ODPair, np.ndarray]
) -> Dict[str, Any]:
    """
    Compute global link flows/times and per-OD path times given per-OD path flows.
    """
    bpr = net["bpr_arrays"]
    global_links = net["global_links"]

    L = len(global_links)
    x = np.zeros(L, dtype=np.float64)
    for od, pf in path_flows_by_od.items():
        A = net["per_od"][od]["A_global"]
        x += A @ np.asarray(pf, dtype=np.float64)

    t = _bpr_times_from_x(x, bpr=bpr)  # (L,)

    path_times_by_od: Dict[ODPair, np.ndarray] = {}
    for od in path_flows_by_od.keys():
        A = net["per_od"][od]["A_global"]
        pt = A.T @ t  # (K,)
        path_times_by_od[od] = pt.astype(np.float64)

    TSTT = float(np.sum(x * t))

    return {
        "link_flows": pd.Series(x, index=pd.MultiIndex.from_tuples(global_links, names=["u","v"]), dtype=np.float64),
        "link_times": pd.Series(t, index=pd.MultiIndex.from_tuples(global_links, names=["u","v"]), dtype=np.float64),
        "path_flows_by_od": {od: np.asarray(pf, dtype=np.float64) for od, pf in path_flows_by_od.items()},
        "path_times_by_od": path_times_by_od,
        "TSTT": TSTT,
    }


def find_network_ue_multi_od(
    net: Dict[str, Any],
    od_demands: Dict[ODPair, float],
    config: SolverConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Multi-OD UE benchmark using a simple MSA update over the fixed K-path sets per OD.

    NOTE: This solves UE on the restricted K-path subnetwork. If K is small, UE is an approximation.
    """
    ods = [od for od in net["ods"] if od in od_demands]
    if len(ods) == 0:
        raise ValueError("No overlap between net['ods'] and provided od_demands keys.")

    # initialize uniform split per OD
    path_flows: Dict[ODPair, np.ndarray] = {}
    for od in ods:
        D = float(od_demands[od])
        K = int(net["per_od"][od]["K"])
        if K <= 0:
            continue
        path_flows[od] = np.ones(K, dtype=np.float64) * (D / float(K))

    converged = False
    change = np.inf
    it = 0

    for it in range(1, int(config.max_iter) + 1):
        state = _state_from_path_flows_multi(net, path_flows)
        change = 0.0

        # all-or-nothing to shortest path per OD under current times
        target_flows: Dict[ODPair, np.ndarray] = {}
        for od in ods:
            D = float(od_demands[od])
            pt = state["path_times_by_od"][od]
            K = len(pt)
            if K == 0:
                continue
            sp = int(np.argmin(pt))
            tf = np.zeros(K, dtype=np.float64)
            tf[sp] = D
            target_flows[od] = tf

        step = 1.0 / float(it)

        for od in ods:
            pf = path_flows[od]
            tf = target_flows[od]
            new_pf = pf + step * (tf - pf)
            change = max(change, float(np.linalg.norm(new_pf - pf)))
            path_flows[od] = new_pf

        if it >= 10 and change < float(config.tol):
            converged = True
            break

    final_state = _state_from_path_flows_multi(net, path_flows)
    final_state["iterations"] = it
    final_state["converged"] = converged
    final_state["final_change"] = float(change)
    final_state["od_demands"] = {od: float(od_demands[od]) for od in ods}
    return final_state


def find_network_so_multi_od(
    net: Dict[str, Any],
    od_demands: Dict[ODPair, float],
    config: SolverConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """
    Multi-OD SO benchmark using a simple MSA update where shortest paths are computed on marginal costs.
    """
    ods = [od for od in net["ods"] if od in od_demands]
    if len(ods) == 0:
        raise ValueError("No overlap between net['ods'] and provided od_demands keys.")

    path_flows: Dict[ODPair, np.ndarray] = {}
    for od in ods:
        D = float(od_demands[od])
        K = int(net["per_od"][od]["K"])
        if K <= 0:
            continue
        path_flows[od] = np.ones(K, dtype=np.float64) * (D / float(K))

    bpr = net["bpr_arrays"]
    global_links = net["global_links"]
    L = len(global_links)

    converged = False
    change = np.inf
    it = 0

    for it in range(1, int(config.max_iter) + 1):
        # current link flows
        x = np.zeros(L, dtype=np.float64)
        for od in ods:
            A = net["per_od"][od]["A_global"]
            x += A @ path_flows[od]

        mc = _bpr_marginal_costs_from_x(x, bpr=bpr)  # (L,)
        change = 0.0

        # all-or-nothing on marginal costs
        target_flows: Dict[ODPair, np.ndarray] = {}
        for od in ods:
            D = float(od_demands[od])
            A = net["per_od"][od]["A_global"]
            pc = A.T @ mc
            K = len(pc)
            sp = int(np.argmin(pc))
            tf = np.zeros(K, dtype=np.float64)
            tf[sp] = D
            target_flows[od] = tf

        step = 1.0 / float(it)

        for od in ods:
            pf = path_flows[od]
            tf = target_flows[od]
            new_pf = pf + step * (tf - pf)
            change = max(change, float(np.linalg.norm(new_pf - pf)))
            path_flows[od] = new_pf

        if it >= 10 and change < float(config.tol):
            converged = True
            break

    final_state = _state_from_path_flows_multi(net, path_flows)
    final_state["iterations"] = it
    final_state["converged"] = converged
    final_state["final_change"] = float(change)
    final_state["od_demands"] = {od: float(od_demands[od]) for od in ods}
    return final_state


def _belief_high(belief: Dict[str, Any], demand_ref: Optional[float] = None) -> float:
    """Best-effort 'high' endpoint for reference-point heuristics."""
    dist = str(belief.get("dist", "uniform")).lower()
    if dist in ("grid", "discrete"):
        g = np.asarray(belief.get("grid", []), dtype=np.float64)
        if g.size == 0:
            return 1.0
        return float(np.max(g))
    low, high = _resolve_bounds(belief, demand_ref=demand_ref)
    return float(high)


def calculate_strategic_utilities_multi_od(
    net: Dict[str, Any],
    od_demands_base: Dict[ODPair, float],
    belief_theta: Dict[str, Any],
    market_penetration: float,
    qU_by_od: Dict[ODPair, np.ndarray],
    qI_by_od: Dict[ODPair, np.ndarray],
    behavioral_params: Dict[str, Any],
    tau_by_od: Optional[Dict[ODPair, np.ndarray]] = None,
    risk_model: str = "variance",
    utility_base: str = "outcome",
    config: SolverConfig = DEFAULT_CONFIG,
    eps_w: float = 1e-300,
    return_debug: bool = False
) -> Union[Dict[ODPair, np.ndarray], Tuple[Dict[ODPair, np.ndarray], Dict[str, Any]]]:
    """
    Compute per-OD utility vectors under a shared theta belief (network-wide).

    Under Option A (network-wide multiplicative uncertainty):
      - Each OD has a base demand D_od.
      - Scenarios are represented by a scalar multiplier theta.
      - Scenario demand is theta * D_od for every OD.
      - With fixed route choice probabilities, link flows scale linearly in theta:
            x(theta) = theta * x_base
        where x_base is computed once from D_od and the current mixed strategy.

    If return_debug=True, also returns a diagnostics dict containing:
      - theta_grid, theta_weights (discretized belief)
      - x_base (global link flow vector at theta=1 under current strategy mix)
      - x_by_theta (N,L) and link_times_by_theta (N,L)
      - path_times_by_theta_by_od (per OD: (N,K) imagined path times)
      - summary moments per OD: expected_path_time_by_od, path_time_var_by_od
    """
    p = float(np.clip(market_penetration, 0.0, 1.0))
    bpr = net["bpr_arrays"]
    global_links = net["global_links"]
    L = len(global_links)

    # Belief over theta multipliers
    theta_grid, w = generate_belief_grid(belief_theta, config=config, eps_w=eps_w)

    # x_base from mixed strategy at theta = 1
    x_base = np.zeros(L, dtype=np.float64)
    q_mix_by_od: Dict[ODPair, np.ndarray] = {}

    for od in net["ods"]:
        if od not in od_demands_base:
            continue
        D0 = float(od_demands_base[od])
        qU = np.asarray(qU_by_od[od], dtype=np.float64)
        qI = np.asarray(qI_by_od[od], dtype=np.float64)
        q_mix = (1.0 - p) * qU + p * qI
        q_mix_by_od[od] = q_mix
        pf = D0 * q_mix  # (K,)
        A = net["per_od"][od]["A_global"]  # (L,K)
        x_base += A @ pf

    # Scenario link flows/times for each theta
    x_mat = theta_grid[:, None] * x_base[None, :]  # (N,L)
    t_mat = _bpr_times_from_x(x_mat, bpr=bpr)      # (N,L)

    risk_model_l = str(risk_model).lower().strip()
    utility_base_l = str(utility_base).lower().strip()

    # Risk/behavior parameters
    r_risk = float(behavioral_params.get("r_risk", 0.0))
    alpha_gain = float(behavioral_params.get("alpha_gain", 0.0))
    beta_loss = float(behavioral_params.get("beta_loss", 0.0))
    a_risk = float(behavioral_params.get("a_risk", 1.0))
    gamma_risk = float(behavioral_params.get("gamma_risk", 0.5))

    utilities: Dict[ODPair, np.ndarray] = {}
    debug: Dict[str, Any] = {}

    if return_debug:
        debug["theta_grid"] = theta_grid.copy()
        debug["theta_weights"] = w.copy()
        debug["x_base"] = x_base.copy()
        debug["x_by_theta"] = x_mat.copy()
        debug["link_times_by_theta"] = t_mat.copy()
        debug["q_mix_by_od"] = {od: q.copy() for od, q in q_mix_by_od.items()}
        debug["path_times_by_theta_by_od"] = {}
        debug["expected_path_time_by_od"] = {}
        debug["path_time_var_by_od"] = {}
        debug["tau_by_od"] = None if tau_by_od is None else {od: np.asarray(tau, dtype=np.float64) for od, tau in tau_by_od.items()}

    for od in net["ods"]:
        if od not in od_demands_base:
            continue

        A = net["per_od"][od]["A_global"]  # (L,K)
        imagined = t_mat @ A               # (N,K)

        # record basic moments (always meaningful, even if utility_base="deviation")
        exp_pt = (w[:, None] * imagined).sum(axis=0)
        var_pt = (w[:, None] * np.square(imagined)).sum(axis=0) - np.square(exp_pt)

        if return_debug:
            debug["path_times_by_theta_by_od"][od] = imagined.copy()
            debug["expected_path_time_by_od"][od] = exp_pt.copy()
            debug["path_time_var_by_od"][od] = var_pt.copy()

        if tau_by_od is None:
            tau = np.zeros(imagined.shape[1], dtype=np.float64)
        else:
            tau = np.asarray(tau_by_od[od], dtype=np.float64)

        if risk_model_l == "variance":
            # Mean-variance prospect-style utility (original implementation; no exponentiation):
            #   u_k = -E[t_k] + alpha_gain * E[gain_k] - beta_loss * E[loss_k] - r_risk * Var(t_k)
            # where gain_k = max(0, tau_k - t_k) and loss_k = max(0, t_k - tau_k).
            mean_t = (w[:, None] * imagined).sum(axis=0)

            gains = np.maximum(0.0, tau[None, :] - imagined)
            losses = np.maximum(0.0, imagined - tau[None, :])

            Eg = (w[:, None] * gains).sum(axis=0)
            El = (w[:, None] * losses).sum(axis=0)

            var_term = var_pt
            u = -mean_t + alpha_gain * Eg - beta_loss * El - r_risk * var_term

        elif risk_model_l in ("ara", "rra"):
            # Exponential/CARA-ish or CRRA-ish utility in the outcome/deviation domain.
            if utility_base_l == "deviation":
                x = imagined - tau[None, :]
            else:
                x = imagined

            if risk_model_l == "ara":
                z = a_risk * x
                z_clip_hi = float(min(getattr(config, 'exp_clip', MAX_EXP_ARG), MAX_EXP_ARG))
                if return_debug:
                    debug.setdefault('ara_exp_clip_by_od', {})[od] = {
                        'clip_hi': int(np.sum(z > z_clip_hi)),
                        'clip_lo': int(np.sum(z < -z_clip_hi)),
                        'z_max': float(np.max(z)),
                        'z_min': float(np.min(z)),
                        'z_clip': float(z_clip_hi),
                    }
                z = np.clip(z, -z_clip_hi, z_clip_hi)
                with np.errstate(over='ignore', under='ignore', invalid='ignore'):
                    util_out = -np.exp(z)
            else:
                eps = 1e-12
                g = float(np.clip(gamma_risk, eps, 1 - eps))
                util_out = -np.sign(x) * (np.power(np.abs(x) + eps, 1 - g) / (1 - g))

            u = (w[:, None] * util_out).sum(axis=0)

        else:
            # Risk-neutral: minimize expected travel time
            u = -exp_pt

        u = np.nan_to_num(u, nan=-1e10, posinf=1e10, neginf=-1e10)
        utilities[od] = u.astype(np.float64)

    if return_debug:
        return utilities, debug
    return utilities

def _logit_per_od(utilities_by_od: Dict[ODPair, np.ndarray], config: SolverConfig) -> Dict[ODPair, np.ndarray]:
    out: Dict[ODPair, np.ndarray] = {}
    for od, u in utilities_by_od.items():
        out[od] = multinomial_logit(np.asarray(u, dtype=np.float64), config=config)
    return out


def compute_t_ff_scaled_reference_point_multi_od(
    net: Dict[str, Any],
    scalar: float = 1.0
) -> Dict[ODPair, np.ndarray]:
    """
    Per-OD reference vector tau where all paths share the same reference = scalar * min free-flow path time.
    """
    bpr = net["bpr_arrays"]
    fft = bpr["fft"]
    tau_by_od: Dict[ODPair, np.ndarray] = {}
    for od in net["ods"]:
        A = net["per_od"][od]["A_global"]
        ff_pt = A.T @ fft  # (K,)
        tau0 = float(np.min(ff_pt)) * float(scalar)
        tau_by_od[od] = np.ones(A.shape[1], dtype=np.float64) * tau0
    return tau_by_od


def compute_ue_reference_point_multi_od(
    net: Dict[str, Any],
    od_demands_ref: Dict[ODPair, float],
    config: SolverConfig = DEFAULT_CONFIG
) -> Tuple[Dict[ODPair, np.ndarray], Dict[str, Any]]:
    """
    Reference tau as the UE path times under a reference demand matrix.
    """
    ue_state = find_network_ue_multi_od(net, od_demands_ref, config=config)
    tau_by_od = {od: ue_state["path_times_by_od"][od] for od in ue_state["path_times_by_od"].keys()}
    return tau_by_od, ue_state


def compute_behavioral_reference_point_multi_od(
    net: Dict[str, Any],
    od_demands_base: Dict[ODPair, float],
    belief_theta: Dict[str, Any],
    market_penetration: float,
    behavioral_params: Dict[str, Any],
    risk_model: str = "variance",
    utility_base: str = "outcome",
    config: SolverConfig = DEFAULT_CONFIG,
    eps_w: float = 1e-300
) -> Tuple[Dict[ODPair, np.ndarray], Dict[str, Any]]:
    """
    Compute a self-consistent (behavioral) reference point tau_by_od by solving a single-strategy
    fixed point under one belief (no informed/uninformed distinction; everyone uses q).

    This is used when reference_method="behavioral" in the two-group solver.
    """
    p = float(np.clip(market_penetration, 0.0, 1.0))

    # Initialize q per OD uniformly
    q_by_od: Dict[ODPair, np.ndarray] = {}
    for od in net["ods"]:
        if od not in od_demands_base:
            continue
        K = int(net["per_od"][od]["K"])
        q_by_od[od] = np.ones(K, dtype=np.float64) / float(K)

    # Initial tau from zero flow
    tau_by_od = compute_t_ff_scaled_reference_point_multi_od(net, scalar=1.0)
    last_state: Dict[str, Any] = {}

    for outer in range(1, 51):
        # Inner: solve for q given tau
        for inner in range(1, 201):
            # Everyone uses the same strategy => set qU=qI=q
            qU = q_by_od
            qI = q_by_od
            u_by_od = calculate_strategic_utilities_multi_od(
                net=net,
                od_demands_base=od_demands_base,
                belief_theta=belief_theta,
                market_penetration=p,
                qU_by_od=qU,
                qI_by_od=qI,
                behavioral_params=behavioral_params,
                tau_by_od=tau_by_od,
                risk_model=risk_model,
                utility_base=utility_base,
                config=config,
                eps_w=eps_w
            )
            target = _logit_per_od(u_by_od, config=config)

            step = 1.0 / float(inner)
            max_change = 0.0
            for od in target.keys():
                q_old = q_by_od[od]
                q_new = q_old + step * (target[od] - q_old)
                max_change = max(max_change, float(np.linalg.norm(q_new - q_old)))
                q_by_od[od] = q_new

            if inner >= 10 and max_change < float(config.tol):
                break

        # Update tau from resulting state at theta=1
        path_flows = {od: float(od_demands_base[od]) * q_by_od[od] for od in q_by_od.keys()}
        last_state = _state_from_path_flows_multi(net, path_flows)
        tau_new = {od: last_state["path_times_by_od"][od] for od in last_state["path_times_by_od"].keys()}

        # Convergence on tau
        diff = 0.0
        for od in tau_new.keys():
            diff = max(diff, float(np.linalg.norm(tau_new[od] - tau_by_od[od])))
        tau_by_od = tau_new

        if outer >= 5 and diff < float(config.tol):
            break

    return tau_by_od, last_state


def find_mixed_strategy_equilibrium_multi_od(
    net: Dict[str, Any],
    od_demands_base: Dict[ODPair, float],
    prior_theta_belief: Dict[str, Any],
    signal_theta_belief: Dict[str, Any],
    market_penetration: float,
    credibility: float,
    behavioral_params: Optional[Dict[str, Any]] = None,
    reference_method: str = "ue",         # ue | behavioral | t_ff_scaled
    risk_model: str = "variance",         # variance | ara | rra | (other => expected time)
    utility_base: str = "outcome",        # outcome | deviation
    pooling_method: str = "parametric",   # parametric | linear_pool
    theta_real: float = 1.0,
    run_benchmarks: bool = True,
    reference_override: Optional[Dict[str, Any]] = None,
    config: SolverConfig = DEFAULT_CONFIG,
    eps_w: float = 1e-300,
    store_full_trace: bool = True,
    store_theta_diagnostics: bool = True,
    store_states_each_iter: bool = True
) -> Dict[str, Any]:
    """
    Network-wide two-group mixed-strategy behavioral equilibrium under a *global theta multiplier*
    belief applied to every OD demand.

    - Uninformed group uses the PRIOR belief over theta.
    - Informed group uses the BLENDED belief: update_belief(prior, signal, credibility, ...)

    Returns a fully-auditable results dict. If store_full_trace=True, the output contains:
      - per-iteration strategies, utilities, logit targets, and optional network states
      - discretized prior/signal/blended beliefs (theta grid + weights)
      - per-iteration theta diagnostics ("imagined" scenario path times, link times, etc.) when enabled
    """
    if behavioral_params is None:
        behavioral_params = {}

    p = float(np.clip(market_penetration, 0.0, 1.0))
    c = float(np.clip(credibility, 0.0, 1.0))
    theta_real = float(theta_real)

    pooling_method = str(pooling_method).lower().strip()
    reference_method = str(reference_method).lower().strip()
    risk_model = str(risk_model).lower().strip()
    utility_base = str(utility_base).lower().strip()

    # Blend beliefs for informed group
    blended = update_belief(
        prior_belief=prior_theta_belief,
        signal_belief=signal_theta_belief,
        credibility=c,
        method=pooling_method,
        config=config,
        demand_ref=None,   # theta is already dimensionless
        eps_w=eps_w
    )

    # Discretize beliefs once (for transparency + reproducibility)
    prior_grid, prior_w = generate_belief_grid(prior_theta_belief, config=config, eps_w=eps_w)
    sig_grid, sig_w = generate_belief_grid(signal_theta_belief, config=config, eps_w=eps_w)
    blend_grid, blend_w = generate_belief_grid(blended, config=config, eps_w=eps_w)

    results: Dict[str, Any] = {
        "inputs": {
            "market_penetration": p,
            "credibility": c,
            "theta_real": theta_real,
            "reference_method": reference_method,
            "risk_model": risk_model,
            "utility_base": utility_base,
            "pooling_method": pooling_method,
            "grid_points": int(config.grid_points),
            "sensitivity": float(config.sensitivity),
            "tol": float(config.tol),
            "max_iter": int(config.max_iter),
        },
        "beliefs": {
            "prior_theta": prior_theta_belief,
            "signal_theta": signal_theta_belief,
            "blended_theta": blended,
        },
        "beliefs_discrete": {
            "prior": {"grid": prior_grid.copy(), "weights": prior_w.copy()},
            "signal": {"grid": sig_grid.copy(), "weights": sig_w.copy()},
            "blended": {"grid": blend_grid.copy(), "weights": blend_w.copy()},
        },
        "reference": {},
        "utilities": {},
        "equilibrium": {},
        "convergence": {},
        "benchmarks": {},
        "iteration_history": [] if store_full_trace else None,
    }
    # Reference point tau_by_od
    tau_by_od: Optional[Dict[ODPair, np.ndarray]] = None
    ref_state: Optional[Dict[str, Any]] = None

    if reference_override is not None:
        # Caller-supplied reference point (precomputed tau vectors)
        tau_by_od = reference_override.get("tau_by_od", None)
        ref_state = reference_override.get("ref_state", None)
        results["reference"] = reference_override.get("meta", {"method": "override"})
        if ref_state is not None:
            results["reference"]["ref_state"] = ref_state
    else:
        if reference_method in ("t_ff_scaled", "tff", "freeflow_scaled"):
            scalar = float(behavioral_params.get("t_ff_scalar", 1.0))
            tau_by_od = compute_t_ff_scaled_reference_point_multi_od(net, scalar=scalar)
            results["reference"] = {"method": "t_ff_scaled", "t_ff_scalar": scalar}
        else:
            # Reference demand matrix used for UE/behavioral reference points
            od_demands_ref = {od: float(od_demands_base[od]) * theta_real for od in od_demands_base.keys()}

            if reference_method in ("ue", "wardrop"):
                tau_by_od, ref_state = compute_ue_reference_point_multi_od(net, od_demands_ref, config=config)
                results["reference"] = {"method": "ue", "od_demands_ref": od_demands_ref}
            elif reference_method in ("behavioral", "be"):
                tau_by_od, ref_state = compute_behavioral_reference_point_multi_od(
                    net=net,
                    od_demands_base=od_demands_base,
                    belief_theta=blended,
                    market_penetration=p,
                    behavioral_params=behavioral_params,
                    risk_model=risk_model,
                    utility_base=utility_base,
                    config=config,
                    eps_w=eps_w
                )
                results["reference"] = {"method": "behavioral"}
            else:
                tau_by_od = None
                results["reference"] = {"method": "none"}

        if ref_state is not None:
            results["reference"]["ref_state"] = ref_state

    # Initialize strategies per OD
    qU_by_od: Dict[ODPair, np.ndarray] = {}
    qI_by_od: Dict[ODPair, np.ndarray] = {}
    for od in net["ods"]:
        if od not in od_demands_base:
            continue
        K = int(net["per_od"][od]["K"])
        qU_by_od[od] = np.ones(K, dtype=np.float64) / float(K)
        qI_by_od[od] = np.ones(K, dtype=np.float64) / float(K)

    # Iteration loop
    converged = False
    change = np.inf

    for it in range(1, int(config.max_iter) + 1):
        step = 1.0 / float(it)

        # Utilities for each group, given their belief over theta
        if store_full_trace and store_theta_diagnostics:
            uU_by_od, dbgU = calculate_strategic_utilities_multi_od(
                net=net,
                od_demands_base=od_demands_base,
                belief_theta=prior_theta_belief,
                market_penetration=p,
                qU_by_od=qU_by_od,
                qI_by_od=qI_by_od,
                behavioral_params=behavioral_params,
                tau_by_od=tau_by_od,
                risk_model=risk_model,
                utility_base=utility_base,
                config=config,
                eps_w=eps_w,
                return_debug=True
            )
            uI_by_od, dbgI = calculate_strategic_utilities_multi_od(
                net=net,
                od_demands_base=od_demands_base,
                belief_theta=blended,
                market_penetration=p,
                qU_by_od=qU_by_od,
                qI_by_od=qI_by_od,
                behavioral_params=behavioral_params,
                tau_by_od=tau_by_od,
                risk_model=risk_model,
                utility_base=utility_base,
                config=config,
                eps_w=eps_w,
                return_debug=True
            )
        else:
            dbgU = None
            dbgI = None
            uU_by_od = calculate_strategic_utilities_multi_od(
                net=net,
                od_demands_base=od_demands_base,
                belief_theta=prior_theta_belief,
                market_penetration=p,
                qU_by_od=qU_by_od,
                qI_by_od=qI_by_od,
                behavioral_params=behavioral_params,
                tau_by_od=tau_by_od,
                risk_model=risk_model,
                utility_base=utility_base,
                config=config,
                eps_w=eps_w
            )
            uI_by_od = calculate_strategic_utilities_multi_od(
                net=net,
                od_demands_base=od_demands_base,
                belief_theta=blended,
                market_penetration=p,
                qU_by_od=qU_by_od,
                qI_by_od=qI_by_od,
                behavioral_params=behavioral_params,
                tau_by_od=tau_by_od,
                risk_model=risk_model,
                utility_base=utility_base,
                config=config,
                eps_w=eps_w
            )

        # Logit best response targets (per OD)
        targetU = _logit_per_od(uU_by_od, config=config)
        targetI = _logit_per_od(uI_by_od, config=config)

        # Damped fixed-point update
        change = 0.0
        for od in targetU.keys():
            qU_old = qU_by_od[od]
            qI_old = qI_by_od[od]
            qU_new = qU_old + step * (targetU[od] - qU_old)
            qI_new = qI_old + step * (targetI[od] - qI_old)
            change = max(change, float(np.linalg.norm(qU_new - qU_old) + np.linalg.norm(qI_new - qI_old)))
            qU_by_od[od] = qU_new
            qI_by_od[od] = qI_new

        # Optional per-iteration states/flows at realized demand
        iter_state_total = None
        iter_state_U = None
        iter_state_I = None

        if store_full_trace and store_states_each_iter:
            od_demands_real = {od: float(od_demands_base[od]) * theta_real for od in od_demands_base.keys()}
            pf_total_by_od: Dict[ODPair, np.ndarray] = {}
            pf_U_by_od: Dict[ODPair, np.ndarray] = {}
            pf_I_by_od: Dict[ODPair, np.ndarray] = {}
            q_mix_by_od: Dict[ODPair, np.ndarray] = {}

            for od in net["ods"]:
                if od not in od_demands_real:
                    continue
                D = float(od_demands_real[od])
                qU = np.asarray(qU_by_od[od], dtype=np.float64)
                qI = np.asarray(qI_by_od[od], dtype=np.float64)
                q_mix = (1.0 - p) * qU + p * qI
                q_mix_by_od[od] = q_mix
                pf_total_by_od[od] = D * q_mix
                pf_U_by_od[od] = D * (1.0 - p) * qU
                pf_I_by_od[od] = D * p * qI

            iter_state_total = _state_from_path_flows_multi(net, pf_total_by_od)
            iter_state_U = _state_from_path_flows_multi(net, pf_U_by_od)
            iter_state_I = _state_from_path_flows_multi(net, pf_I_by_od)

        if store_full_trace:
            rec: Dict[str, Any] = {
                "iteration": it,
                "step": float(step),
                "change": float(change),
                "qU_by_od": {od: q.copy() for od, q in qU_by_od.items()},
                "qI_by_od": {od: q.copy() for od, q in qI_by_od.items()},
                "targetU_by_od": {od: q.copy() for od, q in targetU.items()},
                "targetI_by_od": {od: q.copy() for od, q in targetI.items()},
                "utilityU_by_od": {od: u.copy() for od, u in uU_by_od.items()},
                "utilityI_by_od": {od: u.copy() for od, u in uI_by_od.items()},
            }
            if dbgU is not None:
                rec["theta_diagnostics_uninformed"] = dbgU
            if dbgI is not None:
                rec["theta_diagnostics_informed"] = dbgI

            if iter_state_total is not None:
                rec["state_total_realized"] = iter_state_total
                rec["state_uninformed_realized"] = iter_state_U
                rec["state_informed_realized"] = iter_state_I

            results["iteration_history"].append(rec)

        if it >= 10 and change < float(config.tol):
            converged = True
            break

    # Final state at realized theta (total + per group)
    od_demands_real = {od: float(od_demands_base[od]) * theta_real for od in od_demands_base.keys()}

    final_pf_total_by_od: Dict[ODPair, np.ndarray] = {}
    final_pf_U_by_od: Dict[ODPair, np.ndarray] = {}
    final_pf_I_by_od: Dict[ODPair, np.ndarray] = {}
    final_q_mix_by_od: Dict[ODPair, np.ndarray] = {}

    for od in net["ods"]:
        if od not in od_demands_real:
            continue
        D = float(od_demands_real[od])
        qU = np.asarray(qU_by_od[od], dtype=np.float64)
        qI = np.asarray(qI_by_od[od], dtype=np.float64)
        q_mix = (1.0 - p) * qU + p * qI
        final_q_mix_by_od[od] = q_mix
        final_pf_total_by_od[od] = D * q_mix
        final_pf_U_by_od[od] = D * (1.0 - p) * qU
        final_pf_I_by_od[od] = D * p * qI

    final_state_total = _state_from_path_flows_multi(net, final_pf_total_by_od)
    final_state_U = _state_from_path_flows_multi(net, final_pf_U_by_od)
    final_state_I = _state_from_path_flows_multi(net, final_pf_I_by_od)

    # Final utilities at convergence (helpful summary)
    uU_final = calculate_strategic_utilities_multi_od(
        net=net,
        od_demands_base=od_demands_base,
        belief_theta=prior_theta_belief,
        market_penetration=p,
        qU_by_od=qU_by_od,
        qI_by_od=qI_by_od,
        behavioral_params=behavioral_params,
        tau_by_od=tau_by_od,
        risk_model=risk_model,
        utility_base=utility_base,
        config=config,
        eps_w=eps_w
    )
    uI_final = calculate_strategic_utilities_multi_od(
        net=net,
        od_demands_base=od_demands_base,
        belief_theta=blended,
        market_penetration=p,
        qU_by_od=qU_by_od,
        qI_by_od=qI_by_od,
        behavioral_params=behavioral_params,
        tau_by_od=tau_by_od,
        risk_model=risk_model,
        utility_base=utility_base,
        config=config,
        eps_w=eps_w
    )

    results["utilities"]["final_utility_U_by_od"] = {od: u.copy() for od, u in uU_final.items()}
    results["utilities"]["final_utility_I_by_od"] = {od: u.copy() for od, u in uI_final.items()}

    results["equilibrium"] = {
        "qU_by_od": {od: q.copy() for od, q in qU_by_od.items()},
        "qI_by_od": {od: q.copy() for od, q in qI_by_od.items()},
        "q_mix_by_od": {od: q.copy() for od, q in final_q_mix_by_od.items()},
        "od_demands_real": od_demands_real,
        "final_path_flows_total_by_od": final_pf_total_by_od,
        "final_path_flows_U_by_od": final_pf_U_by_od,
        "final_path_flows_I_by_od": final_pf_I_by_od,
        "final_state_total": final_state_total,
        "final_state_uninformed": final_state_U,
        "final_state_informed": final_state_I,
        "iterations": it,
    }
    results["convergence"] = {"converged": converged, "final_change": float(change)}

    # Benchmarks (UE and SO) at realized demand
    if run_benchmarks:
        ue_state = find_network_ue_multi_od(net, od_demands_real, config=config)
        so_state = find_network_so_multi_od(net, od_demands_real, config=config)
        results["benchmarks"] = {"UE": ue_state, "SO": so_state}

    return results

def collect_model_tables_multi_od(
    ue: Optional[Dict[str, Any]] = None,
    so: Optional[Dict[str, Any]] = None,
    be: Optional[Dict[str, Any]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Build tidy per-link and per-path tables (network-wide).

    - link_table: one row per global link, with model columns.
    - path_table: one row per (OD, path_id), with model columns.

    This complements collect_model_tables() (single-OD).
    """
    tables: Dict[str, pd.DataFrame] = {}

    def _link_df(state: Dict[str, Any], prefix: str) -> pd.DataFrame:
        df = pd.DataFrame({
            "link": state["link_flows"].index,
            f"{prefix}_flow": state["link_flows"].values,
            f"{prefix}_time": state["link_times"].values,
        })
        return df

    def _path_df(state: Dict[str, Any], prefix: str) -> pd.DataFrame:
        rows = []
        for (o, d), pt in state["path_times_by_od"].items():
            pf = state["path_flows_by_od"][(o, d)]
            for k in range(len(pt)):
                rows.append({
                    "origin": o,
                    "destination": d,
                    "path_id": k,
                    f"{prefix}_path_flow": float(pf[k]),
                    f"{prefix}_path_time": float(pt[k]),
                })
        return pd.DataFrame(rows)

    link_tables = []
    path_tables = []

    if ue is not None:
        link_tables.append(_link_df(ue, "UE"))
        path_tables.append(_path_df(ue, "UE"))

    if so is not None:
        link_tables.append(_link_df(so, "SO"))
        path_tables.append(_path_df(so, "SO"))

    if be is not None:
        final_state = be["equilibrium"]["final_state"]
        link_tables.append(_link_df(final_state, "BE"))
        path_tables.append(_path_df(final_state, "BE"))

    if len(link_tables) > 0:
        out = link_tables[0]
        for df in link_tables[1:]:
            out = out.merge(df, on="link", how="outer")
        tables["link_table"] = out.sort_values(by="link").reset_index(drop=True)

    if len(path_tables) > 0:
        out = path_tables[0]
        for df in path_tables[1:]:
            out = out.merge(df, on=["origin", "destination", "path_id"], how="outer")
        tables["path_table"] = out.sort_values(by=["origin", "destination", "path_id"]).reset_index(drop=True)

    return tables


# --- Override: single-OD BE with pooling_method (parametric or linear_pool) ---
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
    pooling_method: str = "parametric",
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

    blended = update_belief(prior_belief_demand, signal_belief_demand, credibility, method=pooling_method, config=config)
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
# ----------------------------
# JSON helper
# ----------------------------
def _json_fallback(o: Any):
    """Fallback serializer for json.dump."""
    try:
        import numpy as _np
        if isinstance(o, (_np.integer, _np.floating)):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _json_sanitize_keys(obj: Any):
    """Recursively convert dict keys into JSON-compatible primitives (strings).

    JSON requires object keys to be str/int/float/bool/None. Our results often use tuple keys
    such as OD pairs (o, d). This function converts such keys to stable string forms.
    """
    # dict: sanitize keys and recurse on values
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # keep legal key types as-is (except tuples)
            if isinstance(k, tuple):
                if len(k) == 2:
                    ks = f"{k[0]}->{k[1]}"
                else:
                    ks = repr(k)
            elif isinstance(k, (str, int, float, bool)) or k is None:
                ks = k
            else:
                ks = str(k)
            out[ks] = _json_sanitize_keys(v)
        return out

    # list/tuple/set: recurse on elements (convert to list)
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize_keys(x) for x in obj]

    return obj


# ----------------------------
# Convenience: save full result bundle
# ----------------------------
def save_result_bundle(result: dict, out_dir: str, base_name: str = "result", compress: bool = True) -> Dict[str, str]:
    """Save a full result dictionary (including arrays/traces) plus a light JSON summary.

    Writes:
      - {base_name}.pkl or {base_name}.pkl.gz  (authoritative, everything)
      - {base_name}_summary.json              (small, human-readable)
    Returns a dict of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}

    # Summary (avoid huge blobs)
    summary = {}
    for k in ("meta", "reference", "benchmarks", "convergence", "settings"):
        if k in result:
            summary[k] = result[k]
    # Common top-level metrics
    if "equilibrium" in result and "final_state_total" in result["equilibrium"]:
        fs = result["equilibrium"]["final_state_total"]
        for k in ("TSTT", "iterations", "converged"):
            if k in fs:
                summary.setdefault("metrics", {})[k] = fs[k]
    summary_path = os.path.join(out_dir, f"{base_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_keys(summary), f, indent=2, default=_json_fallback)
    written["summary_json"] = summary_path

    pkl_path = os.path.join(out_dir, f"{base_name}.pkl" + (".gz" if compress else ""))
    if compress:
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    written["result_pickle"] = pkl_path
    return written
