# Hardy Cross Pipe Network Solver (Streamlit)

CE-319 Fluid Mechanics-II — Complex Engineering Problem (CEP)
Two-loop pipe network discharge calculation using the Hardy Cross method.

## Files
- `app.py` — the Streamlit application
- `requirements.txt` — Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL shown in the terminal (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud (free, no server needed)
1. Create a **new GitHub repository** and upload `app.py` + `requirements.txt`
   (and this README, optional).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your repo, branch (`main`), and set
   **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and give you a public URL (e.g.
   `https://your-app-name.streamlit.app`) — this is the link you can
   submit / put in your CEP report.

## What the app does
1. Shows editable pipe data (f, L, D, assumed initial flow Q0) for all 6 pipes.
2. Checks that your assumed initial flows satisfy continuity at every node
   before letting you run the solver.
3. Runs the **Hardy Cross iteration** (loop head-loss correction ΔQ) for both
   loops simultaneously, for as many iterations as needed to converge
   (minimum 3, as required by the CEP — shown regardless of convergence).
4. For every iteration, shows a table of Q before/after and a **network
   diagram** with arrows/labels for that iteration (satisfies the CEP
   requirement to "draw diagram at each iteration").
5. Gives final converged discharge, velocity, and head loss per pipe, plus a
   continuity residual check (should be ~0 at every node).
6. Lets you download the final results and the full iteration log as CSV —
   useful to paste into your Excel verification sheet.

## Matching this to your manual (Excel) calculation
- In the sidebar, choose the **head-loss formula**:
  - `K = f*L/D^5` (simplified form commonly used in Hardy Cross textbook
    problems given a bare "friction factor f")
  - Darcy–Weisbach `K = f*L/(2gDA²)` (if your manual solution used the
    standard Darcy-Weisbach head-loss equation)
- Make sure this matches whatever formula you used by hand, or your code
  and manual results will not agree within the 10% target in the rubric.
- Adjust the **initial assumed flows (Q0)** in the pipe table to match
  the same starting assumption you used in your manual Hardy Cross trial —
  Hardy Cross should converge to the same final answer regardless of a
  reasonable starting guess, but matching it makes it easier to compare
  iteration-by-iteration with your handwritten work.

## Verifying against EPANET
Build the same network in EPANET (same node demands, pipe lengths/diameters,
and the same friction factor/roughness convention), run it, and compare the
final discharges in each pipe to this app's "Final Results" table. Record
your 1-minute video showing both results side by side as required by the CEP.
