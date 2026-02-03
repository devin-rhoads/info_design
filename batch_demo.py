import os
import re
import json
import networkx as nx
import pandas as pd

from helper_scripts_v2 import build_network_data, find_mixed_strategy_equilibrium


# -------------------------
# TNTP loaders
# -------------------------
def load_tntp_net(net_path: str) -> nx.DiGraph:
    G = nx.DiGraph()
    with open(net_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("~") or line.startswith("<"):
                continue
            if line.lower().startswith("init"):
                continue

            line = line.rstrip(";")
            parts = re.split(r"\s+", line)
            if len(parts) < 7:
                continue

            u = int(parts[0])
            v = int(parts[1])
            cap = float(parts[2])
            length = float(parts[3])
            fft = float(parts[4])
            alpha = float(parts[5])
            beta = float(parts[6])

            G.add_edge(
                u, v,
                capacity=cap,
                free_flow_time=fft,
                alpha=alpha,
                beta=beta,
                length=length
            )
    return G


def load_tntp_trips(trips_path: str) -> dict[tuple[int, int], float]:
    od = {}
    origin = None
    with open(trips_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("~") or line.startswith("<"):
                continue
            if line.lower().startswith("origin"):
                origin = int(line.split()[1])
                continue

            for chunk in line.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                dest_str, val_str = chunk.split(":")
                dest = int(dest_str.strip())
                demand = float(val_str.strip())
                od[(origin, dest)] = demand
    return od


# -------------------------
# Belief factory (keep simple)
# -------------------------
def make_prior_and_signal(D_st: float, signal_mode: str = "congestion_ahead"):
    """
    Returns (prior_belief, signal_belief) dictionaries compatible with update_belief().
    signal_mode:
      - "congestion_ahead": signal shifts demand higher & tighter
      - "all_clear": signal shifts demand lower & tighter
      - "neutral": signal ~ prior (debug)
    """
    prior = {
        "dist": "beta",
        "low": 0.2 * D_st,
        "high": 1.0 * D_st,
        "a": 2.5,
        "b": 2.5
    }

    if signal_mode == "congestion_ahead":
        signal = {
            "dist": "beta",
            "low": 0.9 * D_st,
            "high": 1.2 * D_st,
            "a": 7.0,   # skew toward high end
            "b": 2.0
        }
    elif signal_mode == "all_clear":
        signal = {
            "dist": "beta",
            "low": 0.4 * D_st,
            "high": 1.2 * D_st,
            "a": 2.0,
            "b": 7.0   # skew toward low end
        }
    elif signal_mode == "neutral":
        signal = dict(prior)
    else:
        raise ValueError(f"Unknown signal_mode: {signal_mode}")

    return prior, signal


# -------------------------
# Batch runner
# -------------------------
def run_batch(
    data_dir: str,
    K: int = 15,
    top_n_od: int = 25,
    min_demand: float = 1.0,
    market_penetration: float = 0.35,
    credibility: float = 0.70,
    signal_mode: str = "congestion_ahead",
    risk_model: str = "ARA",
    utility_base: str = "deviation",
    reference_method: str = "t_ff_scaled",
    run_benchmarks: bool = True,
    out_csv: str = "siouxfalls_batch_results.csv",
):
    net_file = os.path.join(data_dir, "SiouxFalls_net.tntp")
    trips_file = os.path.join(data_dir, "SiouxFalls_trips.tntp")

    if not os.path.exists(net_file):
        raise FileNotFoundError(f"Missing: {net_file}")
    if not os.path.exists(trips_file):
        raise FileNotFoundError(f"Missing: {trips_file}")

    G = load_tntp_net(net_file)
    od = load_tntp_trips(trips_file)

    # pick OD pairs: demand descending, filter tiny/zero
    od_sorted = sorted([(st, d) for st, d in od.items() if d >= min_demand],
                       key=lambda x: x[1], reverse=True)

    od_pick = od_sorted[:top_n_od]

    behavioral_params = {
        "alpha_gain": 0.0,
        "beta_loss": 0.0,
        "r_risk": 0.0005
    }

    rows = []
    cache = {}  # (s,t,K) -> network_data

    for idx, ((s_node, t_node), D_st) in enumerate(od_pick, start=1):
        print(f"[{idx}/{len(od_pick)}] OD=({s_node}->{t_node}) D={D_st}")

        try:
            key = (s_node, t_node, K)
            if key in cache:
                network_data = cache[key]
            else:
                network_data = build_network_data(G, s_node=s_node, t_node=t_node, K=K)
                if network_data is None:
                    print(f"  [SKIP] No feasible K-path set for ({s_node}->{t_node})")
                    continue
                cache[key] = network_data

            prior, signal = make_prior_and_signal(D_st, signal_mode=signal_mode)

            results = find_mixed_strategy_equilibrium(
                total_demand=D_st,
                network_data=network_data,
                prior_belief_demand=prior,
                signal_belief_demand=signal,
                market_penetration=market_penetration,
                credibility=credibility,
                behavioral_params=behavioral_params,
                reference_method=reference_method,
                risk_model=risk_model,
                utility_base=utility_base,
                run_benchmarks=run_benchmarks
            )

            eq = results["equilibrium"]
            qU = eq["qU_vec"]
            qI = eq["qI_vec"]
            final_state = eq["final_state"]

            # quick difference metric between populations
            l1_gap = float(abs(qI - qU).sum())

            # best path by final travel time
            pt = final_state["path_times"]  # pandas Series indexed by path_id
            best_path = int(pt.idxmin())
            best_time = float(pt.loc[best_path])

            UE = results["benchmarks"].get("UE") if run_benchmarks else None
            SO = results["benchmarks"].get("SO") if run_benchmarks else None

            row = {
                "s": s_node,
                "t": t_node,
                "D": float(D_st),
                "K": int(network_data["K"]),
                "market_penetration": float(market_penetration),
                "credibility": float(credibility),
                "signal_mode": signal_mode,

                "TSTT_mixed": float(final_state["TSTT"]),
                "best_path_id": best_path,
                "best_path_time": best_time,
                "qU_best": float(qU[best_path]),
                "qI_best": float(qI[best_path]),
                "L1_gap_qI_qU": l1_gap,

                "converged": bool(results["convergence"]["converged"]),
                "final_change": float(results["convergence"]["final_change"]),
                "iterations": int(eq["iterations"]),

                # store full strategies for later inspection (JSON strings)
                "qU_vec": json.dumps([float(x) for x in qU]),
                "qI_vec": json.dumps([float(x) for x in qI]),
            }

            if UE is not None:
                row["TSTT_UE"] = float(UE["TSTT"])
            else:
                row["TSTT_UE"] = None

            if SO is not None:
                row["TSTT_SO"] = float(SO["TSTT"])
            else:
                row["TSTT_SO"] = None

            rows.append(row)
            def summarize_state(name, state):
                print(f"\n--- {name} ---")
                print("TSTT:", state["TSTT"])
                print("path_times:", {k: float(v) for k, v in state["path_times"].items()})
                print("path_flows:", {k: float(v) for k, v in state["path_flows"].items()})

            # After you compute `results = find_mixed_strategy_equilibrium(...)`:
    
            # Grab the network_data you just built in this loop
            print("\n==============================")
            print(f"DEBUG OD=({s_node}->{t_node}) D={D_st}")
            print("==============================")
            print("K_found:", network_data["K"])

            print("Candidate paths (node lists):")
            for k, pth in enumerate(network_data["paths"]):
                print(k, pth)

            # Benchmarks
            UE = results["benchmarks"].get("UE")
            SO = results["benchmarks"].get("SO")
            if UE is not None:
                summarize_state("UE", UE)
            if SO is not None:
                summarize_state("SO", SO)

            # Mixed final
            be_final = results["equilibrium"]["final_state"]
            summarize_state("MIXED(final_state)", be_final)

            # Flow totals sanity check
            print("\nTotal flow checks:")
            print("UE total flow:", float(UE["path_flows"].sum()) if UE is not None else None)
            print("SO total flow:", float(SO["path_flows"].sum()) if SO is not None else None)
            print("MIXED total flow:", float(be_final["path_flows"].sum()))
            print("inputs total_demand:", results["inputs"]["total_demand"])

        except Exception as e:
            print(f"  [FAIL] ({s_node}->{t_node}) error: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(df.head(10))
    return df


if __name__ == "__main__":
    DATA_DIR = r"C:\Users\Devin\Downloads\TransportationNetworks\SiouxFalls"  # <-- change if needed

    run_batch(
        data_dir=DATA_DIR,
        K=15,
        top_n_od=25,
        min_demand=50.0,            # bump this up to focus on meaningful ODs
        market_penetration=0.50,
        credibility=0.70,
        signal_mode="congestion_ahead",
        out_csv=r"C:\Users\Devin\OneDrive - University of Connecticut\Info Design\Sioux Falls Attempt\siouxfalls_batch_results.csv"
    )
