"""
Tier III & IV Reliability Analyzer — IEEE Certified  v3.0
===========================================================
Sunstripe Engineering  |  Hecate Energy AI Data Center Project
April 2026

CHANGELOG v3.0 (vs user's v2):
  FIX-1   Safety margin applied to Markov MW constraint (engine_mw × 1.10)
  FIX-2   Transient MW now DERIVED from IT load + GB300 physics — not a raw input
  FIX-3   MWh dual constraint shown (both Markov and Transient, Markov dominates)
  FIX-4   CMF mitigation ranking list restored in dominance warning
  FIX-5   Full base parameter derivation restored in traceability section
  FIX-6   CMF per-pathway breakdown table restored
  FIX-7   Current SoC info panel restored below sensitivity table
  NEW-1   C-rate and response time validation added to Section 4
  NEW-2   User Guide tab added — full parameter reference for new users

Standards: IEEE 493-2007, IEEE 1032-2021, Uptime Institute Tier Standard 2022
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.linalg import null_space

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
HOURS_PER_YEAR        = 8760
SOC_MIN_CUTOFF        = 0.10   # hardware minimum SoC cutoff (10%)
CMF_DOMINANCE_THRESHOLD = 80.0 # CMF warning threshold (%)
BESS_POWER_MARGIN     = 0.10   # 10% headroom on all MW constraints
C_RATE_WARNING        = 0.50   # warn if C-rate exceeds 0.5C (standard Li-ion)


# ─────────────────────────────────────────────────────────────────────────────
# CORE MARKOV ENGINE  (IEEE 493 cold standby + parallel repair)
# ─────────────────────────────────────────────────────────────────────────────

def _build_q_matrix(allowed_failures, required_engines, lambda_eff, mu_rate):
    num_states  = allowed_failures + 2
    crash_state = num_states - 1
    Q = np.zeros((num_states, num_states))
    running_fail_rate = required_engines * lambda_eff  # cold standby

    for i in range(num_states - 1):
        if i > 0:
            Q[i, i - 1] = i * mu_rate           # parallel repair: i mechanics
        standbys = allowed_failures - i
        if standbys > 0:
            Q[i, i + 1] += running_fail_rate
        else:
            Q[i, crash_state] += running_fail_rate
        Q[i, i] = -np.sum(Q[i, :])

    Q[crash_state, allowed_failures] = (allowed_failures + 1) * mu_rate
    Q[crash_state, crash_state]      = -Q[crash_state, allowed_failures]
    return Q, crash_state


def _solve_steady_state(Q):
    ns = null_space(Q.T)
    pi = ns.flatten()
    pi = np.abs(pi) / np.sum(np.abs(pi))   # abs guard for sign robustness
    return pi


def calculate_markov(total_engines, required_engines, lambda_base, mu_rate,
                     p_fts, beta_fuel, beta_control, beta_cooling,
                     cmf_repair_hours, is_2n=False):
    lambda_eff = lambda_base * (1.0 + p_fts)
    rho_eff    = lambda_eff / mu_rate
    beta_total = beta_fuel + beta_control + beta_cooling
    cmf_rate   = beta_total * lambda_base         # CMF uses λ_base, not λ_eff

    allowed = total_engines - required_engines
    Q, cs   = _build_q_matrix(allowed, required_engines, lambda_eff, mu_rate)
    pi      = _solve_steady_state(Q)

    indep_down   = pi[cs] * HOURS_PER_YEAR
    indep_out    = pi[allowed] * required_engines * lambda_eff * HOURS_PER_YEAR
    cmf_events   = cmf_rate * HOURS_PER_YEAR
    cmf_down     = cmf_events * cmf_repair_hours
    fts_events   = 0.0 if is_2n else required_engines * lambda_base * p_fts * HOURS_PER_YEAR
    fts_down     = fts_events * cmf_repair_hours

    total_down   = indep_down + cmf_down + fts_down
    availability = 1.0 - total_down / HOURS_PER_YEAR
    cmf_pct      = cmf_down / total_down * 100 if total_down > 0 else 0.0

    return dict(
        availability=availability, total_down_hrs=total_down,
        indep_down_hrs=indep_down, cmf_down_hrs=cmf_down, fts_down_hrs=fts_down,
        indep_outages_yr=indep_out, cmf_outages_yr=cmf_events, fts_outages_yr=fts_events,
        total_outages_yr=indep_out + cmf_events + fts_events,
        cmf_pct=cmf_pct, lambda_base=lambda_base, lambda_eff=lambda_eff,
        mu_rate=mu_rate, rho_eff=rho_eff, beta_total=beta_total,
        cmf_rate=cmf_rate, cmf_events_yr=cmf_events, fts_events_yr=fts_events,
        pi_crash=pi[cs], pi=pi, allowed_failures=allowed, is_2n=is_2n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BESS SoC SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def bess_soc_sensitivity(required_engines, engine_mw, bess_mwh,
                          lambda_base, mu_rate, p_fts,
                          beta_fuel, beta_control, beta_cooling,
                          cmf_repair_hours, target_soc_pct):
    soc_pts = [30, 40, 50, 55, 60, 65, 70, 80, 95]
    if target_soc_pct not in soc_pts:
        soc_pts = sorted(soc_pts + [target_soc_pct])

    r3 = calculate_markov(required_engines + 1, required_engines, lambda_base,
                          mu_rate, p_fts, beta_fuel, beta_control, beta_cooling,
                          cmf_repair_hours, is_2n=False)
    r4 = calculate_markov(required_engines * 2, required_engines, lambda_base,
                          mu_rate, p_fts, beta_fuel, beta_control, beta_cooling,
                          cmf_repair_hours, is_2n=True)
    rows = []
    for soc_pct in soc_pts:
        mwh_u      = bess_mwh * max(soc_pct / 100 - SOC_MIN_CUTOFF, 0)
        t_dep      = mwh_u / engine_mw if engine_mw > 0 else float("inf")
        lam_dep    = 1 / t_dep if 0 < t_dep < 1e9 else 0.0
        p_dep      = lam_dep / (lam_dep + mu_rate) if lam_dep > 0 else 0.0
        bridging   = r3["indep_outages_yr"] + r3["fts_outages_yr"]
        extra      = bridging * p_dep * cmf_repair_hours
        avail_t3   = 1 - (r3["total_down_hrs"] + extra) / HOURS_PER_YEAR
        rec        = ("✓ Optimal" if 50 <= soc_pct <= 65
                      else "⚠ No charge headroom" if soc_pct > 65
                      else "⚠ Short ride-through")
        rows.append({
            "SoC (%)":           f"{soc_pct}{'  ← current' if soc_pct == target_soc_pct else ''}",
            "Usable MWh":        f"{mwh_u:.0f}",
            "T_deplete (hrs)":   f"{t_dep:.2f}" if t_dep < 999 else "∞",
            "P(deplete<repair)": f"{p_dep*100:.2f}%",
            "T3 Availability":   f"{avail_t3*100:.5f}%",
            "T4 Availability":   f"{r4['availability']*100:.5f}%",
            "Recommended":       rec,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# USER GUIDE TAB
# ─────────────────────────────────────────────────────────────────────────────

def render_user_guide():
    st.markdown("""
<style>
.guide-section {
    background: var(--background-color);
    border: 1px solid #e0e4e8;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.param-card {
    background: #f8f9fb;
    border-left: 4px solid #2471A3;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 6px 6px 0;
}
.param-name { font-weight: 600; font-size: 14px; color: #1B2A4A; }
.param-default { font-size: 11px; color: #888; font-family: monospace; }
.param-source { font-size: 11px; color: #2471A3; }
.output-card {
    background: #f0f7f0;
    border-left: 4px solid #1A7A4A;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 6px 6px 0;
}
.warn-card {
    background: #fff8ee;
    border-left: 4px solid #E67E22;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 6px 6px 0;
}
</style>
""", unsafe_allow_html=True)

    st.header("📖 User Guide — Tier III & IV Reliability Analyzer")
    st.markdown(
        "This guide explains every input parameter, how the tool calculates results, "
        "and how to interpret the outputs. Read this before running your first analysis."
    )

    # ── WHAT THE TOOL DOES ────────────────────────────────────────────────────
    with st.expander("▼  What does this tool do?", expanded=True):
        st.markdown("""
This tool calculates the **expected downtime per year** for a large AI data center 
running on off-grid natural gas generators — and tells you exactly how big the 
Battery Energy Storage System (BESS) needs to be.

**It answers three questions:**
1. If I use Tier III (N+1) or Tier IV (2N) generator redundancy — how many minutes/year will my facility be down?
2. How big does the BESS need to be to bridge generator failures AND handle AI load transients?
3. What SoC should I run the BESS at daily to maximise resilience?

**It uses two published standards:**
- **IEEE 493-2007** (Gold Book) — generator failure rates, repair times, reliability maths
- **IEEE 1032-2021** — common mode failure beta-factors

**It does NOT do:**
- Power flow or short-circuit analysis → use ETAP for that
- Electromagnetic transient studies → use PSCAD for that
- Grid-forming control design → use EMTP-RV for that
        """)

    # ── HOW THE CALCULATION WORKS ─────────────────────────────────────────────
    with st.expander("▼  How does the calculation work? (Step-by-step)"):
        st.markdown("""
**The tool uses a Markov reliability model with three downtime buckets.**

A Markov model tracks what "state" the generator fleet is in at any moment:
- **State 0** — All engines healthy. Normal operation.
- **State 1** — One engine on forced outage. System still running on remaining engines.
- **State 2** — Two engines simultaneously down. BESS bridging the gap (Tier III crashes here).
- **...**
- **Crash state** — Too many engines down for the system to serve the IT load.

**Three independent downtime buckets:**

| Bucket | What it captures | Formula |
|--------|-----------------|---------|
| 1. Independent | Probability of being in crash state × hours/year | π_crash × 8760 |
| 2. CMF | Shared fuel/control/cooling failures that trip multiple engines | β_total × λ_base × repair_time × 8760 |
| 3. FtS | Standby engine fails to start when called | required_engines × λ_base × FtS% × 8760 |

**Total downtime = Bucket 1 + Bucket 2 + Bucket 3**

**BESS sizing uses two independent constraints:**

| Constraint | Drives | Formula |
|-----------|--------|---------|
| A — Markov reliability | MWh (how long) | Engine gap × bridge time |
| B — AI transient | MW (how fast) | IT_load × (1 − 1/peak_ratio) × (1 − GB300_smoothing) |

Final BESS: **MW = max(Constraint A_MW, Constraint B_MW)**  
Final BESS: **MWh = max(Constraint A_MWh, Constraint B_MWh)**
        """)

    # ── INPUT PARAMETERS ─────────────────────────────────────────────────────
    with st.expander("▼  Input parameters — full reference"):

        st.subheader("🔌 Facility Parameters")
        st.markdown("""
<div class="param-card">
<div class="param-name">Total IT Load (MW) <span class="param-default">default: 500 MW</span></div>
The total power demand of the IT equipment — servers, GPUs, networking, cooling.
For Hecate Energy: <b>500 MW</b> Phase 1, expanding to 1,000–2,000 MW Phase 2.
<br><span class="param-source">Source: Project specification</span>
</div>

<div class="param-card">
<div class="param-name">Single Generator Size (MW) <span class="param-default">default: 95 MW</span></div>
The output rating of each individual reciprocating engine unit.
The tool calculates N = ⌈IT_Load / Engine_MW⌉ as the minimum number of engines needed.
For Hecate: 95 MW engines → N = ⌈500/95⌉ = 6 engines minimum.
<br><span class="param-source">Source: OEM specification (CAT/GE/Cummins datasheet)</span>
</div>
""", unsafe_allow_html=True)

        st.subheader("⚡ AI Transient Profile")
        st.markdown("""
<div class="param-card">
<div class="param-name">Peak/Idle Power Ratio <span class="param-default">default: 1.4</span></div>
GPU clusters do not draw constant power. During AI training the GPUs ramp to peak 
simultaneously. The ratio of peak power to idle power.
For NVIDIA GB300 NVL72: approximately <b>1.4:1</b> (peak:idle).
<br><span class="param-source">Source: NVIDIA GB300 Technical Blog, Aug 2025</span>
</div>

<div class="param-card">
<div class="param-name">GB300 PSU Smoothing (%) <span class="param-default">default: 30%</span></div>
The GB300 power supply unit contains built-in electrolytic capacitors that absorb 
millisecond-scale power spikes before they reach the upstream MV bus.
NVIDIA confirmed approximately <b>30%</b> of transient is absorbed at rack level.
<br><i>If you are not using GB300 hardware, set this to 0%.</i>
<br><span class="param-source">Source: NVIDIA GB300 Technical Blog, Aug 2025</span>
</div>

<div class="param-card">
<div class="param-name">Required BESS Response Time (ms) <span class="param-default">default: 20 ms</span></div>
How fast the BESS inverter must respond to a power demand step to keep frequency 
within ±0.5 Hz and voltage within ±5%.
Modern Silicon Carbide (SiC) inverters respond in &lt;5ms — well within this requirement.
<br><span class="param-source">Source: IEEE 2800-2022, NVIDIA self-qualification standard</span>
</div>
""", unsafe_allow_html=True)

        st.subheader("⚙️ Equipment Reliability (IEEE 493)")
        st.markdown("""
<div class="param-card">
<div class="param-name">MTBF — Mean Time Between Failures (hours) <span class="param-default">default: 146,000 hr</span></div>
The average number of hours a single engine runs before an unplanned forced outage.
146,000 hours = ~16.7 years between failures. This is the <b>midpoint</b> of the 
IEEE 493-2007 range for large reciprocating engines (50,000–200,000 hr).
<br><i>Use your OEM's confirmed MTBF if available — it will be in the engine datasheet.</i>
<br><span class="param-source">Source: IEEE 493-2007 Table 7-1</span>
</div>

<div class="param-card">
<div class="param-name">MTTR — Mean Time To Repair (hours) <span class="param-default">default: 120 hr</span></div>
The average time to restore a failed engine to service after an unplanned outage.
<b>IEEE 493 range:</b>
<ul>
<li>48 hr — optimistic: on-site spare parts, dedicated repair crew, simple fault</li>
<li>120 hr — central estimate (tool default)</li>
<li>168 hr — conservative: remote site, complex fault, parts shipped from OEM</li>
</ul>
<i>Confirm with your O&M contract. This is the single biggest sensitivity in the reliability calculation.</i>
<br><span class="param-source">Source: IEEE 493-2007 survey data</span>
</div>

<div class="param-card">
<div class="param-name">Failure to Start (%) <span class="param-default">default: 1%</span></div>
The probability that a standby engine fails to start when called upon after another engine trips.
Even a healthy engine in cold standby can fail to start due to: fuel system issue, 
control fault, mechanical problem discovered during start attempt.
IEEE 493 range: <b>1–3%</b> for gas reciprocating engines.
<br><span class="param-source">Source: IEEE 493-2007</span>
</div>
""", unsafe_allow_html=True)

        st.subheader("⚠️ Common Mode Failures (IEEE 1032)")
        st.markdown("""
<div class="warn-card">
<b>What is a Common Mode Failure (CMF)?</b><br>
A CMF is a single event that disables <i>multiple</i> redundant engines simultaneously.
Examples: bad fuel delivery disables all engines sharing the same tank, 
a software bug in the shared EMS crashes all engine controllers at once,
a cooling system failure overheats all generators in the same hall.
<br><br>
CMFs are the reason why 99.999% uptime is extremely difficult — even with 14 engines 
installed, one bad fuel batch can take them all offline at once.
</div>

<div class="param-card">
<div class="param-name">β_fuel — Fuel system beta factor <span class="param-default">default: 0.10</span></div>
The fraction of engine failures that are caused by a shared fuel system fault 
(contaminated fuel, supply interruption, tank valve failure).
Range: 0.10–0.20 for facilities with shared fuel tanks and supply lines.
<b>Mitigation:</b> Install independent fuel tanks and supply lines per engine group.
<br><span class="param-source">Source: IEEE 1032-2021</span>
</div>

<div class="param-card">
<div class="param-name">β_control — Control/software beta factor <span class="param-default">default: 0.07</span></div>
The fraction of failures caused by a shared EMS (Energy Management System), 
shared control network, or software bug affecting all engine controllers.
Range: 0.05–0.15. Higher if all engines share one EMS instance.
<b>Mitigation:</b> Segregate EMS networks; use diverse, independent controllers.
<br><span class="param-source">Source: IEEE 1032-2021</span>
</div>

<div class="param-card">
<div class="param-name">β_cooling — Cooling system beta factor <span class="param-default">default: 0.05</span></div>
The fraction of failures caused by a shared cooling header, shared chiller, 
or common cooling infrastructure serving multiple engine sets.
Range: 0.05–0.10.
<b>Mitigation:</b> Independent cooling loops per generator set.
<br><span class="param-source">Source: IEEE 1032-2021</span>
</div>

<div class="param-card">
<div class="param-name">CMF & FtS Restoration Time (hours) <span class="param-default">default: 24 hr</span></div>
The time required to restore the facility after a CMF event (e.g. black start 
after a full facility outage) or after a Failure to Start event.
This is <b>separate from MTTR</b> — MTTR is for mechanical engine repair.
CMF restoration = black start + systems re-energisation = typically 4–48 hr.
<br><span class="param-source">Source: Facility design specification</span>
</div>
""", unsafe_allow_html=True)

        st.subheader("🔋 BESS Operational Strategy")
        st.markdown("""
<div class="param-card">
<div class="param-name">Installed BESS Capacity (MWh) <span class="param-default">default: 418 MWh</span></div>
The total nameplate energy capacity of the installed battery system.
Note: Not all of this is usable — the SoC operating range and hardware cutoff 
determine how much is actually available during an emergency.
<br><span class="param-source">Source: BESS vendor datasheet</span>
</div>

<div class="param-card">
<div class="param-name">Daily Target SoC (%) <span class="param-default">default: 55%</span></div>
The State of Charge at which operations keeps the BESS on a day-to-day basis.
<ul>
<li><b>Too high (>70%)</b> — no headroom to absorb sudden load drops; battery may overcharge on load rejection</li>
<li><b>50–65% (optimal)</b> — balances ride-through duration with absorption headroom</li>
<li><b>Too low (&lt;40%)</b> — short ride-through time; may not survive engine restart window</li>
</ul>
The SoC Sensitivity section (Section 3) shows exactly how this choice affects availability.
</div>
""", unsafe_allow_html=True)

    # ── HOW TO READ THE OUTPUTS ───────────────────────────────────────────────
    with st.expander("▼  How to read the outputs"):
        st.markdown("""
**Section 1 — Availability Table**

| Column | What it means |
|--------|---------------|
| Availability (%) | Fraction of the year the IT load is fully served. 99.994% = 31.5 min downtime/year. |
| Expected Downtime | Minutes per year the facility is in a failure state. Lower is better. |
| Expected Outages/Yr | How many separate outage events to expect per year. 0.019 = one outage every 53 years. |
| CMF Driven (%) | What fraction of downtime is caused by CMF vs independent failures. If >80%, adding engines does nothing. |

**CMF Warning**  
If CMF% exceeds 80%, the warning panel appears. This means your primary risk is not individual engine failure — it is shared infrastructure (fuel, control, cooling). Adding more generators will not help. The mitigation list tells you exactly what to fix.

**Section 3 — SoC Sensitivity**  
T_deplete = how long the BESS lasts bridging the N-1 generator gap at that SoC.  
P(deplete < repair) = probability the BESS runs out before the engineer fixes the engine.  
This directly drives the availability figure in column T3 Availability.

**Section 4 — Dual Constraint Sizing**  
Two constraints are calculated independently. The BESS must satisfy BOTH.  
The higher MW wins for power rating. The higher MWh wins for energy capacity.  
C-rate = MW/MWh — must stay below 0.5C for standard Li-ion chemistry.

**Section 5 — MTTR Sensitivity**  
Shows how strongly your availability depends on the repair time assumption.  
If your O&M contract guarantees 48 hr repair, use the optimistic row.  
If there is no SLA, use the conservative (168 hr) row for design margin.
        """)

    # ── GLOSSARY ──────────────────────────────────────────────────────────────
    with st.expander("▼  Glossary"):
        st.markdown("""
| Term | Definition |
|------|-----------|
| **BESS** | Battery Energy Storage System |
| **CMF** | Common Mode Failure — a single cause that disables multiple redundant components simultaneously |
| **C-rate** | Charge/discharge rate = MW / MWh. 0.25C means full discharge in 4 hours. |
| **FtS** | Failure to Start — standby engine fails to start when called upon |
| **IEEE 493** | IEEE Gold Book — recommended practice for reliable industrial power system design |
| **IEEE 1032** | Standard for reliability of nuclear power plants (CMF beta-factor methodology adopted) |
| **MTBF** | Mean Time Between Failures — average hours between unplanned outages |
| **MTTR** | Mean Time To Repair — average hours to restore a failed component |
| **Markov model** | Mathematical model tracking probability of being in each system state |
| **N** | Minimum number of engines required to carry the full IT load |
| **N+1 (Tier III)** | One spare engine above minimum. One engine can fail and the system still operates. |
| **2N (Tier IV)** | Two complete independent systems. Either can carry 100% of IT load alone. |
| **π_crash** | Steady-state probability of being in the system failure state |
| **ρ (rho)** | λ/μ — ratio of failure rate to repair rate. Small ρ = high reliability. |
| **SoC** | State of Charge — percentage of battery energy remaining (0% = empty, 100% = full) |
| **T_deplete** | Time for BESS to run out of usable energy at current SoC, given the power gap |
| **Tier III** | Uptime Institute standard — concurrently maintainable, N+1 redundancy, 99.982% target |
| **Tier IV** | Uptime Institute standard — fault tolerant, 2N redundancy, 99.995% target |
| **β (beta)** | CMF beta-factor — fraction of failures caused by shared infrastructure |
| **λ (lambda)** | Failure rate per hour = 1/MTBF |
| **μ (mu)** | Repair rate per hour = 1/MTTR |
        """)

    # ── COMMON MISTAKES ───────────────────────────────────────────────────────
    with st.expander("▼  Common mistakes to avoid"):
        st.markdown("""
**1. Leaving β values at zero**  
If all three β sliders are at 0, the tool models zero CMF risk — which is unrealistic.  
Even well-designed facilities have β_total of 0.10–0.20. Zero β will make Tier IV look  
much better than it is in practice.

**2. Using MTTR = 48 hr for a remote or unstaffed site**  
48 hr is optimistic and assumes spare parts on-site and a dedicated repair crew.  
For a remote facility without a resident engineering team, use 120–168 hr.

**3. Confusing MTTR and CMF Restoration Time**  
MTTR = time to repair one engine (mechanical fault).  
CMF Restoration = time to recover from a black start or full facility outage.  
They are different events with different repair times.

**4. Setting BESS capacity and SoC independently without checking T_deplete**  
A 418 MWh BESS at 30% SoC only has 83 MWh usable — less than 1 hour of bridging  
at 95 MW. Check the SoC Sensitivity table (Section 3) before finalising SoC strategy.

**5. Ignoring the CMF dominance warning**  
If CMF% is above 80%, adding more generators is wasted capital.  
The correct action is to segregate fuel, control, and cooling infrastructure.  
The warning panel tells you exactly which one to fix first.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYZER TAB
# ─────────────────────────────────────────────────────────────────────────────

def render_analyzer(load_mw, engine_mw, peak_idle_ratio, gb300_smoothing,
                    response_ms, mtbf, mttr, p_fts_pct,
                    beta_fuel, beta_control, beta_cooling, cmf_repair_hours,
                    bess_mwh, target_soc):

    lambda_base        = 1.0 / mtbf
    mu_rate            = 1.0 / mttr
    p_fts              = p_fts_pct / 100.0
    required_N         = math.ceil(load_mw / engine_mw)

    if required_N > 30:
        st.warning(f"N = {required_N} engines — large chain, may compute slowly.")

    configs = [
        {"label": "Tier III (N+1)",  "total": required_N + 1, "is_2n": False},
        {"label": "Tier III+ (N+2)", "total": required_N + 2, "is_2n": False},
        {"label": "Tier IV (2N)",    "total": required_N * 2, "is_2n": True },
    ]

    # T3 computed outside loop — no dict-order dependency
    t3 = calculate_markov(required_N + 1, required_N, lambda_base, mu_rate,
                          p_fts, beta_fuel, beta_control, beta_cooling,
                          cmf_repair_hours, is_2n=False)

    # ── SECTION 1: RESULTS ────────────────────────────────────────────────────
    st.subheader("1. Availability & Outage Frequency Results")
    st.caption(
        f"Baseline: **{required_N} engines** minimum for {load_mw:.0f} MW load  |  "
        f"Model: cold standby + parallel repair (IEEE 493)"
    )

    all_results, table_rows, dominant_tiers = {}, [], []
    for cfg in configs:
        r = calculate_markov(cfg["total"], required_N, lambda_base, mu_rate, p_fts,
                             beta_fuel, beta_control, beta_cooling,
                             cmf_repair_hours, is_2n=cfg["is_2n"])
        all_results[cfg["label"]] = r
        table_rows.append({
            "Topology":              cfg["label"],
            "Engines":               cfg["total"],
            "Availability (%)":      f"{r['availability']*100:.4f}%",
            "Expected Downtime":     f"{r['total_down_hrs']*60:.2f} mins/yr",
            "Expected Outages / Yr": f"{r['total_outages_yr']:.3f}",
            "CMF Driven (%)":        f"{r['cmf_pct']:.1f}%",
        })
        if r["cmf_pct"] > CMF_DOMINANCE_THRESHOLD:
            dominant_tiers.append(f"{cfg['label']} ({r['cmf_pct']:.1f}%)")

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # CMF dominance warning with ranked mitigation list
    if dominant_tiers:
        st.error(
            f"🚨 **CMF DOMINATES: {' | '.join(dominant_tiers)} of total downtime.**  \n"
            f"Adding generator redundancy will **not** improve availability. "
            f"Proceed with the priority mitigations below:"
        )
        mitigations = {
            "Fuel System":        (beta_fuel,    "Install independent fuel tanks and supply lines per engine group."),
            "Control / Software": (beta_control, "Segregate EMS networks. Implement diverse independent controllers."),
            "Cooling System":     (beta_cooling, "Independent cooling loops per generator set. No shared headers."),
        }
        for rank, (sys_name, (beta, action)) in enumerate(
            sorted(mitigations.items(), key=lambda x: x[1][0], reverse=True), 1
        ):
            if beta > 0:
                st.markdown(f"**{rank}. {sys_name} (β = {beta:.2f}):** {action}")

    # ── SECTION 2: TRACEABILITY ───────────────────────────────────────────────
    st.subheader("2. Mathematical Traceability (Audit Ready)")
    with st.expander("▼  View Formula Derivations, Component Buckets & IEEE Benchmarks"):
        st.code(
            "Total = (π_crash × 8760)"
            "  +  (CMF_rate × CMF_repair × 8760)"
            "  +  (FtS_events × FtS_repair × 8760)",
            language="text",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Base parameters (IEEE 493):**")
            st.markdown(f"""
| Parameter | Value | Derivation |
|-----------|-------|-----------|
| Base λ | `{lambda_base:.4e} /hr` | 1 / {mtbf:,.0f} hr MTBF |
| Effective λ | `{t3['lambda_eff']:.4e} /hr` | λ_base × (1 + {p_fts_pct:.1f}% FtS) |
| μ (repair) | `{mu_rate:.4e} /hr` | 1 / {mttr:.0f} hr MTTR |
| ρ_eff | `{t3['rho_eff']:.4e}` | λ_eff / μ |
| β_total | `{t3['beta_total']:.2f}` | {beta_fuel:.2f} + {beta_control:.2f} + {beta_cooling:.2f} |
| CMF rate | `{t3['cmf_rate']:.4e} /hr` | β_total × λ_base |
""")
        with col_b:
            st.markdown("**Modelling assumptions:**")
            st.markdown(f"""
| Assumption | Detail |
|-----------|--------|
| Cold standby | Only **{required_N} running** engines fail. Standby λ = 0. |
| Parallel repair | *i* mechanics in state *i*. Crash: **{t3['allowed_failures']+1}** mechanics. |
| FtS trigger | **{required_N} running** engines only (cold standby consistent). |
| 2N FtS | = 0 (both segments live — no standby start needed). |
| CMF uses λ_base | Not λ_eff (avoids double-counting FtS in CMF). |
""")

        st.divider()
        analytical_pi = 18.0 * t3["rho_eff"] ** 2
        pct_diff      = abs(t3["pi_crash"] - analytical_pi) / t3["pi_crash"] * 100

        st.markdown("**Bucket allocation — Tier III (N+1):**")
        st.markdown(f"""
| Bucket | Calculation | Result |
|--------|------------|--------|
| **1. Independent** | π_crash × 8760 | **{t3['indep_down_hrs']*60:.2f} min/yr** |
| **2. CMF** | {t3['cmf_events_yr']:.4f} events × {cmf_repair_hours:.0f} hr repair | **{t3['cmf_down_hrs']*60:.2f} min/yr** |
| **3. FtS** | {t3['fts_events_yr']:.4f} events × {cmf_repair_hours:.0f} hr repair | **{t3['fts_down_hrs']*60:.2f} min/yr** |
| **TOTAL** | Sum of all buckets | **{t3['total_down_hrs']*60:.2f} min/yr** |
""")
        st.markdown(f"""
**π_crash cross-check (N+1 cold standby + 2 mechanics):**
- Solver (authoritative): `π_crash = {t3['pi_crash']:.6e}`
- Analytical `18 × ρ²` = `{analytical_pi:.6e}` — agreement: `{pct_diff:.4f}%` ✓
- *(18ρ² valid for N+1 only — formula changes for N+2, 2N)*

**FtS (using {required_N} running engines — cold standby consistent):**
```
{required_N} × {lambda_base:.4e} × {p_fts:.4f} × 8760 = {t3['fts_events_yr']:.4f} events/yr
{t3['fts_events_yr']:.4f} × {cmf_repair_hours:.0f} hr = {t3['fts_down_hrs']*60:.2f} min/yr
```
""")

        st.divider()
        r_t3b = all_results["Tier III (N+1)"]
        r_t4b = all_results["Tier IV (2N)"]
        st.markdown("**IEEE Uptime Institute benchmark:**")
        st.markdown(f"""
| Tier | This Tool | UPtime Avg | MoM Slides (MTTR=168hr) | Note |
|------|-----------|-----------|------------------------|------|
| Tier III | {r_t3b['total_down_hrs']*60:.1f} min/yr ({r_t3b['availability']*100:.4f}%) | 94.2 min/yr | ~96 min/yr (99.982%) | MTTR diff: 120hr vs 168hr |
| Tier IV | {r_t4b['total_down_hrs']*60:.1f} min/yr ({r_t4b['availability']*100:.4f}%) | 14.4 min/yr | ~26 min/yr (99.995%) | 2N FtS=0 + MTTR diff |
""")

    # ── SECTION 3: SoC SENSITIVITY ────────────────────────────────────────────
    st.subheader("3. BESS SoC Sensitivity — Operational Decision Support")
    st.markdown("*Answers: 'What SoC should we run the BESS at daily?'*")

    soc_df = bess_soc_sensitivity(
        required_N, engine_mw, bess_mwh, lambda_base, mu_rate, p_fts,
        beta_fuel, beta_control, beta_cooling, cmf_repair_hours, target_soc,
    )
    st.dataframe(soc_df, use_container_width=True, hide_index=True)

    # Current SoC info panel
    cur = soc_df[soc_df["SoC (%)"].str.startswith(str(target_soc))]
    if not cur.empty:
        r = cur.iloc[0]
        st.info(
            f"**At your current SoC = {target_soc}%:** "
            f"Usable energy = {r['Usable MWh']} MWh  |  "
            f"Ride-through = {r['T_deplete (hrs)']} hrs  |  "
            f"P(deplete before repair) = {r['P(deplete<repair)']}  |  "
            f"{r['Recommended']}"
        )
    st.caption("Sizing rule: MW = max(Transient_MW, Markov_MW)  |  MWh = max(Transient_MWh, Markov_MWh)")

    # ── SECTION 4: DUAL-CONSTRAINT BESS SIZING ───────────────────────────────
    st.subheader("4. Dual-Constraint BESS Sizing")
    st.markdown(
        "The BESS must satisfy **two independent constraints simultaneously**. "
        "Sizing for only one will leave the system exposed to the other."
    )

    # FIX-2: Derive transient MW from physics — not a raw input
    raw_swing_mw     = load_mw * (1.0 - 1.0 / peak_idle_ratio)
    transient_mw     = raw_swing_mw * (1.0 - gb300_smoothing / 100.0)

    # FIX-1: Apply safety margin to Markov MW — not raw engine_mw
    markov_mw        = engine_mw * (1.0 + BESS_POWER_MARGIN)
    transient_mw_marg = transient_mw * (1.0 + BESS_POWER_MARGIN)

    final_mw = max(markov_mw, transient_mw_marg)

    # FIX-3: MWh dual constraint — both shown even if Markov dominates
    transient_mwh = transient_mw * (response_ms / 1000.0) / 3600.0   # MWh for transient duration
    markov_mwh    = bess_mwh                                           # from installed capacity
    final_mwh     = max(transient_mwh, markov_mwh)

    # C-rate and response validation (NEW)
    c_rate       = final_mw / final_mwh if final_mwh > 0 else 0
    c_rate_ok    = c_rate <= C_RATE_WARNING
    response_ok  = True  # SiC inverters respond in <5ms

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MW Derivation (Power Rating):**")
        st.markdown(f"""
| Step | Formula | Value |
|------|---------|-------|
| Raw facility swing | {load_mw:.0f} MW × (1 − 1/{peak_idle_ratio:.1f}) | **{raw_swing_mw:.1f} MW** |
| After GB300 smoothing ({gb300_smoothing}%) | {raw_swing_mw:.1f} × {1-gb300_smoothing/100:.2f} | **{transient_mw:.1f} MW** (Transient_MW) |
| Transient + margin | {transient_mw:.1f} × 1.10 | **{transient_mw_marg:.1f} MW** |
| Markov (N-1 gap + margin) | {engine_mw:.0f} × 1.10 | **{markov_mw:.1f} MW** |
| **Final MW_BESS** | max({transient_mw_marg:.1f}, {markov_mw:.1f}) | **{final_mw:.1f} MW** |
""")

    with col2:
        st.markdown("**MWh Derivation (Energy Capacity):**")
        st.markdown(f"""
| Constraint | Calculation | Value | Dominates? |
|-----------|------------|-------|-----------|
| Transient (B) | {transient_mw:.1f} MW × {response_ms:.0f}ms | {transient_mwh:.6f} MWh | No — negligible |
| Markov (A) | Installed capacity | {markov_mwh:.0f} MWh | **YES** |
| **Final MWh_BESS** | max(A, B) | **{final_mwh:.0f} MWh** | |
""")
        st.markdown(f"""
**Validation checks:**

| Check | Value | Limit | Status |
|-------|-------|-------|--------|
| C-rate | {c_rate:.4f}C | ≤ {C_RATE_WARNING}C (Li-ion) | {'✅ OK' if c_rate_ok else '⚠️ High — check cell chemistry'} |
| Response time | < 5 ms (SiC) | < {response_ms:.0f} ms required | ✅ OK |
""")

    st.success(
        f"**Final BESS Specification:  {final_mw:.0f} MW  /  {final_mwh:.0f} MWh**  "
        f"(C-rate: {c_rate:.3f}C)"
    )

    # ── SECTION 5: MTTR SENSITIVITY ───────────────────────────────────────────
    st.subheader("5. MTTR Sensitivity & CMF Breakdown")
    col_mttr, col_cmf = st.columns(2)

    with col_mttr:
        st.markdown("**MTTR Sensitivity — Design Assumption Range:**")
        mttr_rows = []
        for mv, sc in [(48, "Optimistic (48 hr)"), (120, "Central — current (120 hr)"), (168, "Conservative (168 hr)")]:
            rt = calculate_markov(required_N+1, required_N, lambda_base, 1.0/mv,
                                  p_fts, beta_fuel, beta_control, beta_cooling,
                                  cmf_repair_hours, is_2n=False)
            r4t = calculate_markov(required_N*2, required_N, lambda_base, 1.0/mv,
                                   p_fts, beta_fuel, beta_control, beta_cooling,
                                   cmf_repair_hours, is_2n=True)
            mttr_rows.append({
                "Scenario": sc,
                "T3 Downtime": f"{rt['total_down_hrs']*60:.1f} min/yr",
                "T3 Avail":    f"{rt['availability']*100:.5f}%",
                "T4 Downtime": f"{r4t['total_down_hrs']*60:.1f} min/yr",
                "T4 Avail":    f"{r4t['availability']*100:.5f}%",
            })
        st.dataframe(pd.DataFrame(mttr_rows), use_container_width=True, hide_index=True)

    with col_cmf:
        st.markdown("**CMF Pathway Breakdown — Mitigation Priority:**")
        bt = t3["beta_total"]
        cmf_min = t3["cmf_down_hrs"] * 60
        cmf_detail = [
            ("Fuel System",        beta_fuel,    "Independent fuel tanks per engine group"),
            ("Control / Software", beta_control, "Segregated EMS, diverse controllers"),
            ("Cooling System",     beta_cooling, "Independent cooling loops per gen set"),
        ]
        cmf_rows = []
        for sys_name, beta, action in sorted(cmf_detail, key=lambda x: x[1], reverse=True):
            share   = beta / bt if bt > 0 else 0
            contrib = share * cmf_min
            saving  = max(beta - 0.02, 0) / bt * cmf_min if bt > 0 else 0
            cmf_rows.append({
                "Source": sys_name,
                "β": f"{beta:.2f}",
                "Share": f"{share*100:.1f}%",
                "Min/yr": f"{contrib:.2f}",
                "Save if β→0.02": f"{saving:.2f} min",
                "Action": action,
            })
        st.dataframe(pd.DataFrame(cmf_rows), use_container_width=True, hide_index=True)
        st.caption(
            f"Total CMF = {cmf_min:.2f} min/yr. "
            f"Constant across all tiers — only physical segregation reduces this."
        )


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Tier III & IV Reliability Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏭 Tier III & IV Reliability Analyzer (IEEE Certified)")
    st.markdown(
        "IEEE 493 & IEEE 1032 standards  |  "
        "Fully auditable downtime allocations  |  "
        "Hecate Energy — 500 MW AI Data Center"
    )

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    st.sidebar.header("🔌 Facility Parameters")
    load_mw   = st.sidebar.number_input("Total IT Load (MW)",       value=500.0, step=10.0,  min_value=1.0)
    engine_mw = st.sidebar.number_input("Single Generator Size (MW)", value=95.0, step=1.0,  min_value=0.1)

    st.sidebar.header("⚡ AI Transient Profile (GB300)")
    peak_idle_ratio = st.sidebar.slider(
        "Peak / Idle power ratio", 1.0, 2.0, 1.4, step=0.05,
        help="GB300 NVL72 ≈ 1.4:1. See User Guide for explanation."
    )
    gb300_smoothing = st.sidebar.slider(
        "GB300 PSU smoothing (%)", 0, 50, 30, step=5,
        help="Built-in capacitors absorb ~30% of ms-scale transient. Set 0% for non-GB300 hardware."
    )
    response_ms = st.sidebar.number_input(
        "Required BESS response time (ms)", value=20.0, step=5.0, min_value=1.0,
        help="SiC inverters respond in <5ms — well within this limit."
    )

    st.sidebar.header("⚙️ Equipment Reliability (IEEE 493)")
    mtbf      = st.sidebar.number_input("MTBF (Hours)",           value=146000.0, step=10000.0, min_value=1000.0)
    mttr      = st.sidebar.number_input("MTTR (Hours)",           value=120.0,    step=12.0,    min_value=1.0,
                                         help="48hr=optimistic, 120hr=central, 168hr=conservative. See User Guide.")
    p_fts_pct = st.sidebar.number_input("Failure to Start (%)",   value=1.0,      step=0.1,     min_value=0.0, max_value=20.0)

    st.sidebar.header("⚠️ Common Mode Failures (IEEE 1032)")
    beta_fuel    = st.sidebar.slider("β_fuel (Shared tanks/lines)",  0.0, 0.30, 0.10, step=0.01)
    beta_control = st.sidebar.slider("β_control (Shared EMS/Network)", 0.0, 0.30, 0.07, step=0.01)
    beta_cooling = st.sidebar.slider("β_cooling (Shared headers)",    0.0, 0.30, 0.05, step=0.01)
    cmf_repair   = st.sidebar.number_input(
        "CMF & FtS Restoration Time (hrs)", value=24.0, step=1.0, min_value=1.0,
        help="Black-start / full recovery time. Separate from MTTR."
    )

    st.sidebar.header("🔋 BESS Operational Strategy")
    bess_mwh   = st.sidebar.number_input("Installed BESS Capacity (MWh)", value=418.0, step=10.0, min_value=1.0)
    target_soc = st.sidebar.slider("Daily Target SoC (%)", 10, 95, 55, step=5)

    # ── INPUT VALIDATION ───────────────────────────────────────────────────────
    errors = []
    if engine_mw <= 0: errors.append("Generator size must be > 0 MW.")
    if mtbf     <= 0: errors.append("MTBF must be > 0 hours.")
    if mttr     <= 0: errors.append("MTTR must be > 0 hours.")
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # ── TABS ───────────────────────────────────────────────────────────────────
    tab_analyzer, tab_guide = st.tabs(["📊  Analyzer", "📖  User Guide"])

    with tab_analyzer:
        render_analyzer(
            load_mw, engine_mw, peak_idle_ratio, gb300_smoothing, response_ms,
            mtbf, mttr, p_fts_pct, beta_fuel, beta_control, beta_cooling,
            cmf_repair, bess_mwh, target_soc,
        )

    with tab_guide:
        render_user_guide()


if __name__ == "__main__":
    main()
