import re
import os
import networkx as nx

# --- import your existing helpers (no changes needed) ---
from helper_scripts_v2 import build_network_data, find_mixed_strategy_equilibrium


# =========================
# TNTP loaders (minimal)
# =========================

def load_tntp_net(net_path: str) -> nx.DiGraph:
    """
    Reads a TNTP *_net.tntp file and returns a DiGraph with the edge attributes
    your helper expects:
      - free_flow_time
      - capacity
      - alpha
      - beta
    (plus length, optional)
    """
    G = nx.DiGraph()

    with open(net_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # TNTP comment/metadata lines
            if line.startswith("~") or line.startswith("<"):
                continue

            # header line often starts with "Init node"
            if line.lower().startswith("init"):
                continue

            line = line.rstrip(";")
            parts = re.split(r"\s+", line)

            # Typical TNTP format: u v capacity length fft alpha beta ...
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
    """
    Reads a TNTP *_trips.tntp file and returns {(origin, dest): demand}.
    """
    od = {}
    origin = None

    with open(trips_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("~") or line.startswith("<"):
                continue

            if line.lower().startswith("origin"):
                # e.g., "Origin 1"
                origin = int(line.split()[1])
                continue

            # e.g., " 2 : 100; 3 : 50;"
            for chunk in line.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                dest_str, val_str = chunk.split(":")
                dest = int(dest_str.strip())
                demand = float(val_str.strip())
                od[(origin, dest)] = demand

    return od


# =========================
# Demo run (one OD pair)
# =========================

def main():
    # --- point this to your local clone ---
    DATA_DIR = r"C:\Users\Devin\OneDrive - University of Connecticut\Info Design\Sioux Falls Attempt\TransportationNetworks\SiouxFalls"
    NET_FILE = os.path.join(DATA_DIR, "SiouxFalls_net.tntp")
    TRIPS_FILE = os.path.join(DATA_DIR, "SiouxFalls_trips.tntp")

    # Load network + OD table
    G = load_tntp_net(NET_FILE)
    od = load_tntp_trips(TRIPS_FILE)

    # Pick ONE OD pair (change these freely)
    s_node = 1
    t_node = 20

    if (s_node, t_node) not in od:
        # quick fallback: pick the first OD entry that exists
        (s_node, t_node), D_st = next(iter(od.items()))
        print(f"[WARN] (1,20) not found; using first OD pair: ({s_node},{t_node}) demand={D_st}")
    else:
        D_st = od[(s_node, t_node)]

    # Build K candidate paths (free-flow-based) using YOUR helper
    K = 10
    network_data = build_network_data(G, s_node=s_node, t_node=t_node, K=K)
    if network_data is None:
        raise RuntimeError("build_network_data returned None (no feasible paths?)")

    # -----------------------------
    # Beliefs (prior vs signal)
    # -----------------------------
    # NOTE: In YOUR code, beliefs must share the same dist type ("beta" with "beta", etc.)
    # We'll do a simple example:
    # - prior: broad beta around [0.6D, 1.4D]
    # - signal: tighter + shifted upward (phone says congestion ahead -> demand likely higher)
    prior_belief = {
        "dist": "beta",
        "low": 0.6 * D_st,
        "high": 1.4 * D_st,
        "a": 2.5,
        "b": 2.5
    }

    signal_belief = {
        "dist": "beta",
        "low": 0.9 * D_st,
        "high": 1.6 * D_st,
        # skew toward HIGH end (a > b pushes mass toward z=1)
        "a": 6.0,
        "b": 2.0
    }

    # Pop split + credibility (your existing knobs)
    market_penetration = 0.35   # fraction informed (uses blended belief)
    credibility = 0.70          # trust in phone signal

    # Behavioral params for risk_model="variance"
    behavioral_params = {
        "alpha_gain": 0.0,    # start at 0 if you want pure mean+variance
        "beta_loss": 0.0,
        "r_risk": 0.0005      # small variance penalty; tune later
    }

    # Run equilibrium (your helper)
    results = find_mixed_strategy_equilibrium(
        total_demand=D_st,
        network_data=network_data,
        prior_belief_demand=prior_belief,
        signal_belief_demand=signal_belief,
        market_penetration=market_penetration,
        credibility=credibility,
        behavioral_params=behavioral_params,
        reference_method="ue",
        risk_model="variance",
        utility_base="outcome",
        run_benchmarks=True
    )

    eq = results["equilibrium"]
    qU = eq["qU_vec"]
    qI = eq["qI_vec"]
    final_state = eq["final_state"]

    print("\n==============================")
    print("Sioux Falls | ONE OD pair demo")
    print("==============================")
    print(f"OD pair: ({s_node} -> {t_node}), demand D = {D_st:.2f}")
    print(f"K paths: {network_data['K']}")
    print(f"Market penetration p = {market_penetration:.2f}, credibility c = {credibility:.2f}")

    print("\n--- Mixed strategies (route shares) ---")
    print("Uninformed qU:", [float(x) for x in qU])
    print("Informed   qI:", [float(x) for x in qI])

    print("\n--- Final network outputs (under mixed flow) ---")
    print("TSTT:", float(final_state["TSTT"]))
    print("Path times:", {int(k): float(v) for k, v in final_state["path_times"].items()})

    print("\n--- Benchmarks (UE/SO) ---")
    print("UE exists:", results["benchmarks"]["UE"] is not None)
    print("SO exists:", results["benchmarks"]["SO"] is not None)

if __name__ == "__main__":
    main()
