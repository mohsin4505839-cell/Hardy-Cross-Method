"""
Hardy Cross Method - Two Loop Pipe Network Analysis
CE-319 Fluid Mechanics-II | Complex Engineering Problem (CEP)
NED University of Engineering & Technology

Solves for discharge in each pipe of a two-loop pipe network using the
Hardy Cross iterative method, with editable pipe data, loop diagrams at
each iteration, and full iteration tables.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this folder (app.py + requirements.txt) to a GitHub repo
    2. Go to https://share.streamlit.io , connect the repo, deploy
"""

import math
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hardy Cross Pipe Network Solver", layout="wide")

g = 32.2  # ft/s^2

# ----------------------------------------------------------------------
# ---------------------------  DEFAULT DATA  ----------------------------
# ----------------------------------------------------------------------
# Network layout (from CEP problem statement):
#
#        100 cfs                     25 cfs
#           \\                        ^
#            a ----- (1) -----> b ----+
#            | \\                |
#           (3) (2)             (4)
#            |     \\             |
#            v      v            v
#            c ----- (6) -----> e ----> 25 cfs
#            |
#            v
#          50 cfs
#
# Loop I  (a-b-c-a)     : pipes 1 (a->b), 2 (b->c), 3 (a->c)
# Loop II (b-d-e-c-b)   : pipes 4 (b->d), 5 (d->e), 6 (c->e), 2 (b->c, shared)
#
# Node demands (+ve = inflow to network, -ve = outflow/demand)
DEFAULT_NODE_DEMANDS = {"a": 100.0, "b": -25.0, "c": -50.0, "d": 0.0, "e": -25.0}

DEFAULT_PIPES = pd.DataFrame(
    [
        {"Pipe": 1, "From": "a", "To": "b", "f": 0.0037, "L_ft": 1000, "D_in": 6,  "Loop": "I only",  "Q0_cfs": 60.0},
        {"Pipe": 2, "From": "b", "To": "c", "f": 0.0021, "L_ft": 2500, "D_in": 8,  "Loop": "Shared",  "Q0_cfs": 15.0},
        {"Pipe": 3, "From": "a", "To": "c", "f": 0.0064, "L_ft": 5000, "D_in": 10, "Loop": "I only",  "Q0_cfs": 40.0},
        {"Pipe": 4, "From": "b", "To": "d", "f": 0.0037, "L_ft": 1000, "D_in": 6,  "Loop": "II only", "Q0_cfs": 20.0},
        {"Pipe": 5, "From": "d", "To": "e", "f": 0.0021, "L_ft": 2500, "D_in": 8,  "Loop": "II only", "Q0_cfs": 20.0},
        {"Pipe": 6, "From": "c", "To": "e", "f": 0.0064, "L_ft": 5000, "D_in": 10, "Loop": "II only", "Q0_cfs": 5.0},
    ]
)

# Loop membership: +1 if pipe direction agrees with the loop's traversal
# direction (clockwise, as drawn above), -1 if opposite, 0 if not in loop.
LOOP_SIGN = {
    "I":  {1: +1, 2: +1, 3: -1, 4: 0,  5: 0,  6: 0},
    "II": {1: 0,  2: -1, 3: 0,  4: +1, 5: +1, 6: -1},
}

NODE_POS = {"a": (0, 2), "b": (2, 2), "c": (1, 0), "d": (4, 2), "e": (4, 0)}
PIPE_ENDPOINTS = {1: ("a", "b"), 2: ("b", "c"), 3: ("a", "c"),
                   4: ("b", "d"), 5: ("d", "e"), 6: ("c", "e")}

# ----------------------------------------------------------------------
# ---------------------------  CORE MATH  --------------------------------
# ----------------------------------------------------------------------
def compute_K(f, L, D_in, formula):
    """Head-loss coefficient K such that h_f = K * Q^2 (Q in cfs, h_f in ft)."""
    D_ft = D_in / 12.0
    if formula == "Simplified:  K = f*L / D^5":
        return f * L / (D_ft ** 5)
    else:  # Darcy-Weisbach:  h_f = f*(L/D)*V^2/2g , V = Q/A
        A = math.pi * D_ft ** 2 / 4.0
        return f * L / (D_ft * 2 * g * A ** 2)


def hardy_cross_solve(pipes_df, formula, n_exp=2, max_iter=15, tol=1e-4):
    """
    Runs Hardy Cross iterations.
    Returns: results dataframe (final), iteration_log (list of dicts per
    iteration with per-loop correction and per-pipe flow snapshot),
    K values, and convergence flag.
    """
    pipes = pipes_df.copy()
    pipes["K"] = pipes.apply(lambda r: compute_K(r["f"], r["L_ft"], r["D_in"], formula), axis=1)
    Q = {int(r["Pipe"]): float(r["Q0_cfs"]) for _, r in pipes.iterrows()}
    K = {int(r["Pipe"]): float(r["K"]) for _, r in pipes.iterrows()}

    iteration_log = []
    converged = False

    for it in range(1, max_iter + 1):
        delta_Qs = {}
        for loop in ["I", "II"]:
            num = 0.0  # sum of K*Q*|Q|^(n-1)  (signed head loss)
            den = 0.0  # sum of n*K*|Q|^(n-1)
            for pipe_id, sign in LOOP_SIGN[loop].items():
                if sign == 0:
                    continue
                q_signed = sign * Q[pipe_id]
                num += K[pipe_id] * q_signed * abs(q_signed) ** (n_exp - 1)
                den += n_exp * K[pipe_id] * abs(q_signed) ** (n_exp - 1)
            delta_Q = -num / den if den != 0 else 0.0
            delta_Qs[loop] = delta_Q

        # snapshot BEFORE applying correction (for the diagram/table of this iteration)
        snapshot_before = Q.copy()

        # apply corrections (shared pipe 2 gets both loop corrections, signed)
        new_Q = Q.copy()
        for pipe_id, sign in LOOP_SIGN["I"].items():
            if sign != 0:
                new_Q[pipe_id] = new_Q[pipe_id] + sign * delta_Qs["I"]
        for pipe_id, sign in LOOP_SIGN["II"].items():
            if sign != 0:
                new_Q[pipe_id] = new_Q[pipe_id] + sign * delta_Qs["II"]

        iteration_log.append({
            "iteration": it,
            "Q_before": snapshot_before,
            "deltaQ_I": delta_Qs["I"],
            "deltaQ_II": delta_Qs["II"],
            "Q_after": new_Q.copy(),
        })

        max_change = max(abs(new_Q[p] - Q[p]) for p in Q)
        Q = new_Q

        if max_change < tol:
            converged = True
            break

    results = pipes[["Pipe", "From", "To", "f", "L_ft", "D_in", "K"]].copy()
    results["Q_final_cfs"] = results["Pipe"].map(Q)
    results["Velocity_fps"] = results.apply(
        lambda r: r["Q_final_cfs"] / (math.pi * (r["D_in"] / 12.0) ** 2 / 4.0), axis=1
    )
    results["Headloss_ft"] = results.apply(lambda r: r["K"] * abs(r["Q_final_cfs"]) ** (n_exp - 1) * r["Q_final_cfs"], axis=1)

    return results, iteration_log, converged


def verify_node_continuity(Q):
    """Check inflow=outflow at every node given final pipe flows Q (dict pipe->cfs)."""
    net = {n: 0.0 for n in NODE_POS}
    for pipe_id, (frm, to) in PIPE_ENDPOINTS.items():
        net[frm] -= Q[pipe_id]
        net[to] += Q[pipe_id]
    for node, demand in DEFAULT_NODE_DEMANDS.items():
        net[node] += demand
    return net  # should all be ~0


# ----------------------------------------------------------------------
# ---------------------------  DIAGRAM  ----------------------------------
# ----------------------------------------------------------------------
def draw_network(Q, title="Pipe Network"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for pipe_id, (frm, to) in PIPE_ENDPOINTS.items():
        x1, y1 = NODE_POS[frm]
        x2, y2 = NODE_POS[to]
        q = Q[pipe_id]
        # arrow direction follows sign of Q along the defined From->To direction
        if q >= 0:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", lw=2, color="steelblue"))
        else:
            ax.annotate("", xy=(x1, y1), xytext=(x2, y2),
                        arrowprops=dict(arrowstyle="->", lw=2, color="firebrick"))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.12, f"P{pipe_id}\n{q:.2f} cfs", fontsize=8, ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    for node, (x, y) in NODE_POS.items():
        ax.plot(x, y, "ko", markersize=8)
        ax.text(x - 0.15, y - 0.22, node, fontsize=12, fontweight="bold")

    demand_text = {"a": "100 cfs IN", "b": "25 cfs OUT", "c": "50 cfs OUT", "e": "25 cfs OUT"}
    for node, txt in demand_text.items():
        x, y = NODE_POS[node]
        ax.text(x, y + 0.3 if node != "c" else y - 0.35, txt, fontsize=8, color="green", ha="center")

    ax.set_title(title, fontsize=11)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1.2, 3)
    ax.axis("off")
    return fig


# ----------------------------------------------------------------------
# ---------------------------  STREAMLIT UI  ------------------------------
# ----------------------------------------------------------------------
st.title("Pipe Network Analysis — Hardy Cross Method")
st.caption("CE-319 Fluid Mechanics-II | Complex Engineering Problem (CEP) | Two-Loop Pipe Network")

with st.sidebar:
    st.header("Settings")
    formula = st.selectbox(
        "Head-loss formula",
        ["Simplified:  K = f*L / D^5", "Darcy-Weisbach:  K = f*L / (2gDA²)"],
        help="Choose the formula matching your manual calculation convention."
    )
    n_exp = st.number_input("Pipe constant n", value=2, min_value=1, max_value=3, step=1)
    max_iter = st.slider("Max iterations", min_value=3, max_value=30, value=10)
    tol = st.number_input("Convergence tolerance (cfs)", value=0.0001, format="%.5f")
    st.markdown("---")
    st.markdown(
        "**Note:** Edit pipe data (f, L, D, initial flow) in the table below. "
        "Initial flows must satisfy continuity at every node — the app checks this for you."
    )

st.subheader("1️⃣ Pipe Data (editable)")
pipes_df = st.data_editor(
    DEFAULT_PIPES, num_rows="fixed", use_container_width=True,
    column_config={
        "Pipe": st.column_config.NumberColumn(disabled=True),
        "From": st.column_config.TextColumn(disabled=True),
        "To": st.column_config.TextColumn(disabled=True),
        "Loop": st.column_config.TextColumn(disabled=True),
        "f": st.column_config.NumberColumn(format="%.4f"),
        "Q0_cfs": st.column_config.NumberColumn(help="Initial assumed flow (cfs), + = From→To direction"),
    },
)

# continuity check on initial assumption
Q0 = {int(r["Pipe"]): float(r["Q0_cfs"]) for _, r in pipes_df.iterrows()}
node_check = verify_node_continuity(Q0)
bad_nodes = {n: v for n, v in node_check.items() if abs(v) > 1e-6}
if bad_nodes:
    st.error(f"⚠️ Initial flows do NOT satisfy continuity at node(s): {bad_nodes}. "
             f"Adjust Q0_cfs values above (positive = From→To).")
else:
    st.success("✅ Initial flows satisfy continuity at every node.")

run = st.button("▶ Run Hardy Cross Solver", type="primary", disabled=bool(bad_nodes))

if run:
    results, iteration_log, converged = hardy_cross_solve(
        pipes_df, formula, n_exp=n_exp, max_iter=max_iter, tol=tol
    )

    st.subheader("2️⃣ Iteration-by-Iteration Results")
    if converged:
        st.success(f"Converged in {len(iteration_log)} iterations (tolerance = {tol} cfs).")
    else:
        st.warning(f"Did not fully converge within {max_iter} iterations — showing last state. "
                   f"Increase 'Max iterations' in the sidebar if needed.")

    for entry in iteration_log:
        it = entry["iteration"]
        with st.expander(f"Iteration {it}  —  ΔQ(Loop I) = {entry['deltaQ_I']:.4f} cfs, "
                          f"ΔQ(Loop II) = {entry['deltaQ_II']:.4f} cfs", expanded=(it <= 3)):
            col1, col2 = st.columns([1, 1])
            with col1:
                tbl = pd.DataFrame({
                    "Pipe": list(entry["Q_before"].keys()),
                    "Q before (cfs)": list(entry["Q_before"].values()),
                    "Q after (cfs)": [entry["Q_after"][p] for p in entry["Q_before"]],
                })
                st.dataframe(tbl.sort_values("Pipe"), use_container_width=True, hide_index=True)
            with col2:
                fig = draw_network(entry["Q_after"], title=f"Network after Iteration {it}")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    st.subheader("3️⃣ Final Results")
    st.dataframe(
        results.style.format({
            "f": "{:.4f}", "K": "{:.6f}", "Q_final_cfs": "{:.3f}",
            "Velocity_fps": "{:.3f}", "Headloss_ft": "{:.4f}",
        }),
        use_container_width=True, hide_index=True,
    )

    final_Q = {int(r["Pipe"]): r["Q_final_cfs"] for _, r in results.iterrows()}
    final_check = verify_node_continuity(final_Q)
    st.caption(f"Node continuity residuals (should be ≈ 0): "
               f"{ {k: round(v,5) for k,v in final_check.items()} }")

    st.subheader("4️⃣ Final Network Diagram")
    fig_final = draw_network(final_Q, title="Final Converged Flow Distribution")
    st.pyplot(fig_final, use_container_width=True)
    plt.close(fig_final)

    # ---- downloads ----
    st.subheader("5️⃣ Export")
    csv_buf = io.StringIO()
    results.to_csv(csv_buf, index=False)
    st.download_button("⬇ Download Final Results (CSV)", csv_buf.getvalue(),
                        file_name="hardy_cross_final_results.csv", mime="text/csv")

    all_iters_rows = []
    for entry in iteration_log:
        for pipe_id in entry["Q_before"]:
            all_iters_rows.append({
                "Iteration": entry["iteration"], "Pipe": pipe_id,
                "Q_before_cfs": entry["Q_before"][pipe_id],
                "Q_after_cfs": entry["Q_after"][pipe_id],
                "deltaQ_LoopI": entry["deltaQ_I"], "deltaQ_LoopII": entry["deltaQ_II"],
            })
    iter_csv_buf = io.StringIO()
    pd.DataFrame(all_iters_rows).to_csv(iter_csv_buf, index=False)
    st.download_button("⬇ Download Full Iteration Log (CSV)", iter_csv_buf.getvalue(),
                        file_name="hardy_cross_iteration_log.csv", mime="text/csv")

else:
    st.info("👈 Adjust pipe data / settings if needed, then click **Run Hardy Cross Solver**.")
