"""
Run Sioux Falls (CSV TNDP data) through:
  - UE benchmark
  - SO benchmark
  - 11 Behavioral Equilibrium model variants (3 PV + 4 ARA-family + 4 RRA-family)

This script is designed to be "plug-and-go" for Devin's local folder layout.

You can safely edit the USER PATHS section only.
"""

from __future__ import annotations

import os
import json
import gzip
import pickle
from datetime import datetime
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
import networkx as nx

# --- import solver (same folder) ---
import helper_scripts_v5_4_networkwide_trace as hs


ODPair = Tupzle[int, int]


# ==========================
# USER PATHS (edit here only)
# ==========================
CSV_DIR = r"C:\Users\Devin\OneDrive - University of Connecticut\Info Design\Sioux Falls Attempt\TransportationNetworks\SiouxFalls\CSV-data"
OUT_BASE_DIR = r"C:\Users\Devin\OneDrive - University of Connecticut\Info Design\Sioux Falls Attempt\Attempted Full Runs"

NET_CSV = os.path.join(CSV_DIR, "SiouxFalls_net.csv")
NODE_CSV = os.path.join(CSV_DIR, "SiouxFalls_node.csv")
OD_CSV = os.path.join(CSV_DIR, "SiouxFalls_od.csv")


# ==========================
# Experiment settings
# ==========================
K_PATHS = 20

# Behavioral parameters (edit if desired)
MARKET_PENETRATION = 0.5
CREDIBILITY = 0.5

# Network-wide multiplicative uncertainty in theta (Option A)
PRIOR_THETA = {"dist": "normal", "low": 0.2, "high": 1.8}
SIGNAL_THETA = {"dist": "normal", "low": 0.8, "high": 1.2}

# Pooling method: "parametric" or "linear_pool"
POOLING_METHOD = "linear_pool"

# "Realized" theta multiplier used only for diagnostics/ground truth evaluation (theta_real=1 => base demand)
THETA_REAL = 1.0

# Variance/prospect and risk parameters (defaults are conservative)
BEHAVIORAL_PARAMS = {

    # Prospect+variance parameters (used when risk_model="variance")
    "r_risk": 0.01,          # mean-variance penalty on var(time)
    "alpha_gain": 1.0,      # gain exponent
    "beta_loss": 2.25,       # loss exponent    # loss aversion

    # ARA/RRA parameters
    "a_risk": 1.0,          # ARA exponential slope
    "gamma_risk": 0.5,      # RRA curvature

    # t_ff_scaled reference scalar
    "t_ff_scalar": 2.0,

    # iteration trace controls
    "store_full_trace": False,
    "store_states_each_iter": False,
    "store_theta_diagnostics": False,
}

# Solver configuration
CONFIG = hs.SolverConfig(
    max_iter=750,        # max iterations for all iterative loops (UE, SO, BE)
    tol=1e-5,            # convergence threshold
    sensitivity=0.2,    # softmax sensitivity (higher -> more deterministic)
    grid_points=150,      # belief discretization resolution
)


# ==========================
# Helpers
# ==========================
def build_graph_from_sioux_falls_csv(net_df: pd.DataFrame, node_df: pd.DataFrame) -> nx.DiGraph:
    """
    SiouxFalls_net.csv: columns [LINK, A, B, a0..a4] for polynomial cost t(x)=a0 + a4 x^4 (others often 0).
    We map EXACTLY into a BPR form:
        t(x) = t0 * (1 + alpha * (x/c)^beta)
    with:
        t0 = a0, beta=4, c=1, alpha=a4/a0
    so t(x)=a0 + a4 x^4 exactly (since a0>0 in this dataset).
    """
    G = nx.DiGraph()

    # Add nodes + coordinates (optional)
    for _, r in node_df.iterrows():
        n = int(r["Node"])
        G.add_node(n, x=float(r["X"]), y=float(r["Y"]))

    # Add directed links
    for _, r in net_df.iterrows():
        u = int(r["A"])
        v = int(r["B"])
        a0 = float(r["a0"])
        a4 = float(r["a4"])

        t0 = a0
        cap = 1.0
        beta = 4.0
        alpha = (a4 / a0) if a0 > 0 else 0.0

        G.add_edge(
            u, v,
            free_flow_time=t0,
            capacity=cap,
            alpha=alpha,
            beta=beta,
        )
    return G


def load_od_demands(od_df: pd.DataFrame) -> Dict[ODPair, float]:
    od_demands: Dict[ODPair, float] = {}
    for _, r in od_df.iterrows():
        o = int(r["O"])
        d = int(r["D"])
        q = float(r["Ton"])
        if o == d:
            continue
        if q <= 0:
            continue
        od_demands[(o, d)] = q
    return od_demands


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def save_pickle_gz(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def export_link_table(net: Dict[str, Any], states: Dict[str, Any], out_csv: str) -> None:
    """
    states: dict of named states (e.g., {"UE": ue_state, "SO": so_state, "BE_total": be_total_state, ...})
    Each state should include link_flows, link_times aligned with net["global_links"].
    """
    links = net["global_links"]
    df = pd.DataFrame({"u": [uv[0] for uv in links], "v": [uv[1] for uv in links]})
    for name, st in states.items():
        if st is None:
            continue
        if "link_flows" in st:
            df[f"{name}_x"] = np.asarray(st["link_flows"], dtype=float)
        if "link_times" in st:
            df[f"{name}_t"] = np.asarray(st["link_times"], dtype=float)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


def export_path_table(net: Dict[str, Any], od_demands: Dict[ODPair, float], results: Dict[str, Any], out_csv: str) -> None:
    """
    One row per (OD, path k). Exports final path flows/times and final utilities if available.
    """
    rows: List[Dict[str, Any]] = []
    per_od = net["per_od"]

    # Pull BE final states if present
    eq = results.get("equilibrium", {})
    fs_total = eq.get("final_state_total", None)
    fs_U = eq.get("final_state_uninformed", None)
    fs_I = eq.get("final_state_informed", None)

    util = results.get("utilities", {})
    uU = util.get("final_utility_U_by_od", {})
    uI = util.get("final_utility_I_by_od", {})

    for od in net["ods"]:
        if od not in od_demands:
            continue
        data = per_od[od]
        paths = data["paths"]
        K = int(data["K"])
        for k in range(K):
            rec = {
                "O": int(od[0]),
                "D": int(od[1]),
                "D_od": float(od_demands[od]),
                "k": int(k),
                "path_nodes": "->".join(str(n) for n in paths[k]),
            }

            # times
            if fs_total is not None:
                rec["BE_total_path_flow"] = float(fs_total["path_flows_by_od"][od][k])
                rec["BE_total_path_time"] = float(fs_total["path_times_by_od"][od][k])
            if fs_U is not None:
                rec["BE_U_path_flow"] = float(fs_U["path_flows_by_od"][od][k])
                rec["BE_U_path_time"] = float(fs_U["path_times_by_od"][od][k])
            if fs_I is not None:
                rec["BE_I_path_flow"] = float(fs_I["path_flows_by_od"][od][k])
                rec["BE_I_path_time"] = float(fs_I["path_times_by_od"][od][k])

            if od in uU:
                rec["BE_U_utility"] = float(np.asarray(uU[od])[k])
            if od in uI:
                rec["BE_I_utility"] = float(np.asarray(uI[od])[k])

            rows.append(rec)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


# ==========================
# Main
# ==========================
def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(OUT_BASE_DIR, f"SiouxFalls_full_run_{ts}")
    os.makedirs(out_root, exist_ok=True)

    # Load CSVs
    net_df = pd.read_csv(NET_CSV)
    node_df = pd.read_csv(NODE_CSV)
    od_df = pd.read_csv(OD_CSV)

    # Build graph + OD demand dict
    G = build_graph_from_sioux_falls_csv(net_df, node_df)
    od_demands = load_od_demands(od_df)

    # Build K-path network data (multi-OD)
    od_pairs = sorted(list(od_demands.keys()))
    net = hs.build_multi_od_network_data(G, od_pairs=od_pairs, K=K_PATHS)
    if net is None:
        raise RuntimeError("build_multi_od_network_data returned None. Check connectivity / OD nodes.")

    # Save a tiny metadata file for the run
    meta = {
        "K_PATHS": K_PATHS,
        "MARKET_PENETRATION": MARKET_PENETRATION,
        "CREDIBILITY": CREDIBILITY,
        "PRIOR_THETA": PRIOR_THETA,
        "SIGNAL_THETA": SIGNAL_THETA,
        "POOLING_METHOD": POOLING_METHOD,
        "THETA_REAL": THETA_REAL,
        "CONFIG": {"max_iter": CONFIG.max_iter, "tol": CONFIG.tol, "sensitivity": CONFIG.sensitivity, "grid_points": CONFIG.grid_points},
        "BEHAVIORAL_PARAMS": BEHAVIORAL_PARAMS,
        "paths_built_for_od_pairs": len(od_pairs),
        "total_demand": float(sum(od_demands.values())),
    }
    write_json(os.path.join(out_root, "run_meta.json"), meta)

    # ---------
    # UE + SO baselines (once)
    # ---------
    ue = hs.find_network_ue_multi_od(net, od_demands, config=CONFIG)
    so = hs.find_network_so_multi_od(net, od_demands, config=CONFIG)

    write_json(os.path.join(out_root, "benchmarks_summary.json"), {
        "UE": {k: ue.get(k) for k in ("TSTT", "iterations", "converged", "final_change")},
        "SO": {k: so.get(k) for k in ("TSTT", "iterations", "converged", "final_change")},
    })

    # Export link table for UE/SO
    export_link_table(net, {"UE": ue, "SO": so}, os.path.join(out_root, "UE_SO_link_table.csv"))

    # Precompute reference points ONCE (so BE models don't redo them)
    # t_ff_scaled reference is deterministic
    tau_tff = hs.compute_t_ff_scaled_reference_point_multi_od(net, scalar=float(BEHAVIORAL_PARAMS.get("t_ff_scalar", 1.0)))
    ref_tff = {"tau_by_od": tau_tff, "ref_state": None, "meta": {"method": "t_ff_scaled", "t_ff_scalar": float(BEHAVIORAL_PARAMS.get("t_ff_scalar", 1.0))}}

    # UE reference point (uses theta_real scaling)
    od_demands_ref = {od: float(od_demands[od]) * float(THETA_REAL) for od in od_demands.keys()}
    tau_ue, ref_state_ue = hs.compute_ue_reference_point_multi_od(net, od_demands_ref, config=CONFIG)
    ref_ue = {"tau_by_od": tau_ue, "ref_state": ref_state_ue, "meta": {"method": "ue", "od_demands_ref": od_demands_ref}}

    # Behavioral reference point (computed once using the blended belief)
    blended = hs.update_belief(PRIOR_THETA, SIGNAL_THETA, credibility=float(CREDIBILITY), method=POOLING_METHOD, config=CONFIG)
    tau_be, ref_state_be = hs.compute_behavioral_reference_point_multi_od(
        net=net,
        od_demands_base=od_demands,
        belief_theta=blended,
        market_penetration=float(MARKET_PENETRATION),
        behavioral_params=BEHAVIORAL_PARAMS,
        risk_model="eut",            # compute a common BE reference using risk-neutral EUT
        utility_base="outcome",      # BE reference defines tau (path times) under EUT
        config=CONFIG
    )
    ref_be = {"tau_by_od": tau_be, "ref_state": ref_state_be, "meta": {"method": "behavioral"}}

    # Save references
    save_pickle_gz(os.path.join(out_root, "reference_points.pkl.gz"), {"ue": ref_ue, "t_ff_scaled": ref_tff, "behavioral": ref_be})

    # ---------
    # Define the 11 BE variants
    # ---------
    models = [
        # 3 Prospect+Variance (deviation base) across 3 reference points
        ("Variance_ref=ue",          dict(risk_model="variance", utility_base="deviation", reference_override=ref_ue)),
        ("Variance_ref=behavioral",  dict(risk_model="variance", utility_base="deviation", reference_override=ref_be)),
        ("Variance_ref=t_ff_scaled", dict(risk_model="variance", utility_base="deviation", reference_override=ref_tff)),

        # 4 ARA-family: one is EUT (risk-neutral outcome), plus 3 deviation variants
        ("EUT_outcome",              dict(risk_model="eut",      utility_base="outcome",  reference_override=None)),
        ("ARA_dev_ref=ue",           dict(risk_model="ara",      utility_base="deviation", reference_override=ref_ue)),
        ("ARA_dev_ref=behavioral",   dict(risk_model="ara",      utility_base="deviation", reference_override=ref_be)),
        ("ARA_dev_ref=t_ff_scaled",  dict(risk_model="ara",      utility_base="deviation", reference_override=ref_tff)),

        # 4 RRA-family: outcome + 3 deviation variants
        ("RRA_outcome",              dict(risk_model="rra",      utility_base="outcome",   reference_override=None)),
        ("RRA_dev_ref=ue",           dict(risk_model="rra",      utility_base="deviation", reference_override=ref_ue)),
        ("RRA_dev_ref=behavioral",   dict(risk_model="rra",      utility_base="deviation", reference_override=ref_be)),
        ("RRA_dev_ref=t_ff_scaled",  dict(risk_model="rra",      utility_base="deviation", reference_override=ref_tff)),
    ]

    # Run all models
    summary_rows = []
    for name, opts in models:
        model_dir = os.path.join(out_root, name)
        os.makedirs(model_dir, exist_ok=True)

        # Assemble parameters
        risk_model = opts["risk_model"]
        utility_base = opts["utility_base"]
        reference_override = opts.get("reference_override", None)

        res = hs.find_mixed_strategy_equilibrium_multi_od(
            net=net,
            od_demands_base=od_demands,
            prior_theta_belief=PRIOR_THETA,
            signal_theta_belief=SIGNAL_THETA,
            market_penetration=float(MARKET_PENETRATION),
            credibility=float(CREDIBILITY),
            behavioral_params=BEHAVIORAL_PARAMS,
            reference_method="ue",  # ignored when reference_override is provided
            risk_model=risk_model,
            utility_base=utility_base,
            pooling_method=POOLING_METHOD,
            theta_real=float(THETA_REAL),
            run_benchmarks=False,   # we already computed UE/SO above
            reference_override=reference_override,
            config=CONFIG
        )

        # Save full result bundle
        hs.save_result_bundle(res, out_dir=model_dir, base_name="result", compress=True)

        # Export link table comparing UE/SO vs BE states
        eq = res.get("equilibrium", {})
        st_total = eq.get("final_state_total", None)
        st_U = eq.get("final_state_uninformed", None)
        st_I = eq.get("final_state_informed", None)
        export_link_table(net, {"UE": ue, "SO": so, "BE_total": st_total, "BE_U": st_U, "BE_I": st_I},
                          os.path.join(model_dir, "link_table.csv"))

        # Export path table (final)
        export_path_table(net, od_demands, res, os.path.join(model_dir, "path_table.csv"))

        # Light summary
        metrics = {}
        if st_total is not None:
            metrics = {
                "TSTT": float(st_total.get("TSTT", np.nan)),
                "iterations": int(res.get("convergence", {}).get("iterations", -1)),
                "converged": bool(res.get("convergence", {}).get("converged", False)),
                "final_change": float(res.get("convergence", {}).get("final_change", np.nan)),
            }

        summary_rows.append({"model": name, **metrics})
        write_json(os.path.join(model_dir, "model_meta.json"), {"name": name, "risk_model": risk_model, "utility_base": utility_base})

    # Save overall summary
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_root, "ALL_MODELS_summary.csv"), index=False)

    print(f"Done. Results written to: {out_root}")


if __name__ == "__main__":
    main()
