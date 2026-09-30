"""
SD2900 - Multistage gravity-turn ascent simulation
====================================================

Purpose
-------
Given a launch-vehicle definition (stage by stage), a payload mass, and a
target circular orbit (altitude + inclination), this script determines
whether a gravity-turn ascent exists that reaches that orbit (horizontal
flight path angle, correct altitude, correct circular speed) at cutoff of
the final stage, and if so, computes it.

Boundary value problem (Rocket Dynamics lecture, slides 18 and 25)
-----------------------------------------------------------------
    final conditions : gamma = 0,  H = H*,  V = V*

  * gamma = 0 is enforced automatically by the linear tangent steering law
    on the last stage (slide 25, hint 5: "gamma is not a variable for the
    last stage if the steering law is used").
  * H = H* and V = V* need TWO free parameters. They are:
        delta   - pitch-kick angle at the end of the vertical rise [deg]
        t_cut   - engine cutoff time of the last stage, expressed as the
                  propellant left in its tanks at cutoff (prop_left) [kg]
  * t1 (duration of the vertical rise, hint 1) is an INPUT, not an unknown.
    Why: the kick is given while the vehicle is still slow, and the gravity
    turn that follows only "sees" one combination of (t1, delta). Every
    (t1, delta) pair that reaches H* gives the same trajectory (same final
    V, same losses), i.e. t1 and delta are not independent unknowns -- with
    (t1, delta) as unknowns the Jacobian is singular and the final speed
    cannot be matched. Choose t1 as a few seconds of vertical flight (hint 1)
    and let the solver find delta.
  * Engine cutoff (ENGINE_CUTOFF = True): the last stage is shut down when the
    orbit is reached; the propellant still on board is an OUTPUT and is
    checked against the required reserve PROP_RESERVE_REQUIRED_KG (e.g. for
    the later orbit-lowering manoeuvres). ENGINE_CUTOFF = False: the last
    stage burns to depletion and only delta is free (1 unknown, 2 targets ->
    best fit, the mismatch is reported as over/undershoot).

Course simplifications used (all from the SD2900 lectures)
----------------------------------------------------------
  Atmosphere, exponential single layer (Rocket Dynamics slide 4-5):
      rho(H) = RHO0 * exp(-H / H0),  RHO0 = 1.225 kg/m^3,  H0 = 8.4 km
  Gravity, inverse-square law:
      g(H) = G0 * (R_E / (R_E + H))**2
  Drag, constant Cd (slide 7):
      D = 0.5 * rho * Cd * A * V**2
  Round Earth, LVLH frame, Coriolis term H_dot*X_dot/(R+H) kept (slides 28-31),
  state u = [X, Xd, H, Hd, m]:
      Xd_dot = (T - D) cos(gamma) / m - Xd*Hd / (R_E + H)
      Hd_dot = (T - D) sin(gamma) / m - (g - Xd**2 / (R_E + H))
      m_dot  = -beta   (constant burn rate, slide 22)
  Gravity turn, thrust along velocity, alpha ~ 0, L ~ 0 (slide 15):
      cos(gamma) = Xd / V,  sin(gamma) = Hd / V
  Linear tangent steering law, flat-Earth form (slide 23; the curvature-
  corrected version is struck out on that slide and is NOT used):
      tan(gamma(t)) = tan(gamma0) * (1 - t / t_cutoff)
  ECEF treated as inertial in the dynamics (slide 19). Earth rotation enters
  only as a velocity-budget credit on the target speed (slide 10 "dV_rot",
  Earth Satellite Operations slide 16):
      V* = V_circ(H*) - OMEGA_E * R_E * cos(i)
  (the eastward surface speed projected on the orbit direction equals
  OMEGA_E*R_E*cos(i) for any launch latitude L <= i; a launch window only
  exists for L <= i, slide 16).
  Circular orbit speed (slide 20):
      V_circ(H) = sqrt(g(H) * (R_E + H))
  Velocity budget identity (slide 9 / Rocket Performance slide 25):
      V* = dV_thrust + dV_grav + dV_air
  which holds EXACTLY in this model, so it is printed as an integration
  consistency check, not as a margin.

Known limitation (discuss it in the report)
-------------------------------------------
Because gamma is prescribed on the last stage, the model does not check that
the thrust can actually turn the velocity vector that fast. After the run the
script evaluates the normal acceleration the steering law requires,
      a_n = V*gamma_dot + (g - V**2/(R_E+H)) * cos(gamma)   (slide 22, h2 eq.)
and compares it with the available T/m. If a_n > T/m anywhere, the last-stage
trajectory is not flyable with thrust alone (diagnosis NOT_FLYABLE, see ENFORCE_STEERING_LIMIT) (a low-thrust upper stage such as
Fregat would need a coast arc + circularisation burn instead). If it is
flyable, a steering-loss estimate  integral of T/m * (1 - cos(alpha)) dt  with
sin(alpha) = a_n / (T/m) is reported (slide 10 "dV_steering").
"""

import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import least_squares, brentq
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

# ---------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------
R_EARTH = 6378.137e3            # m, mean Earth radius
G0 = 9.80665                # m/s^2, sea-level gravitational acceleration
RHO0 = 1.225                # kg/m^3, sea-level atmospheric density
H0_SCALE = 8.4e3            # m, atmospheric scale height (slide 4/5)
OMEGA_EARTH = 7.2921159e-5  # rad/s, Earth rotation rate
CD = 0.3                    # drag coefficient (typical for a slender body)

# Pitch-kick search range, in log10(delta [deg]). The kick that reaches a
# given orbit can be very small (1e-3 deg or less for a short t1), so the
# search is done on a log scale.
LOG_DELTA_MIN = -6.0
LOG_DELTA_MAX = math.log10(45.0)

# Largest fraction of the last-stage propellant the cutoff may leave unburned
RESERVE_FRAC_MAX = 0.95

# Steering-feasibility gate (post-check, does not change the trajectory).
# The tangent law prescribes gamma(t); the transverse force needed to follow it
# comes from thrust at an angle alpha to the velocity (slide 22, h2 equation with
# T*sin(alpha) added). If max a_n/(T/m) > STEER_RATIO_LIMIT the law cannot be
# followed and the result is labelled NOT_FLYABLE instead of OK.
# Set ENFORCE_STEERING_LIMIT = False to get the pure course result (orbit matched
# = OK); the ratio is still computed and written to the Excel.
ENFORCE_STEERING_LIMIT = True
STEER_RATIO_LIMIT = 1.0

# Solver start-point strategy: True -> 1-D scan of the kick angle (on a log
# scale) that brackets H = H_target before least_squares; False -> start from
# delta_guess. Single switch used by solve_gravity_turn, run_database_to_excel
# and the driver.
USE_GRID_PRESEARCH = True


# ---------------------------------------------------------------------
# 2. ENVIRONMENT MODELS
# ---------------------------------------------------------------------
def gravity(H):
    """g(H) = g0 * (R/(R+H))**2 -- inverse-square law."""
    return G0 * (R_EARTH / (R_EARTH + H)) ** 2


def density(H):
    """Exponential atmosphere, single layer (slide 4). Adequate here
    since drag is negligible above ~70-80 km regardless of the model."""
    H = max(H, 0.0)
    return RHO0 * np.exp(-H / H0_SCALE)


def drag_force(V, H, Cd, A):
    """D = 0.5 * rho * Cd * A * V^2 (slide 7)."""
    return 0.5 * density(H) * Cd * A * V ** 2


def v_circular(H):
    """Circular orbital speed at altitude H (slide 20)."""
    return np.sqrt(gravity(H) * (R_EARTH + H))


def v_rotation_credit(inclination_deg):
    """Earth-rotation speed credit along the orbit direction [m/s].
    For a launch latitude L <= i the launch azimuth satisfies
    sin(Az) = cos(i)/cos(L), so the eastward surface speed OMEGA*R*cos(L)
    projected on the flight direction is OMEGA*R*cos(i), independent of L.
    Negative for retrograde orbits (i > 90 deg): rotation then costs speed."""
    return OMEGA_EARTH * R_EARTH * np.cos(np.deg2rad(inclination_deg))


def v_target_effective(H, inclination_deg):
    """Target speed relative to the rotating Earth (ECEF treated as
    inertial in the dynamics, slide 19): V_circ - OMEGA*R*cos(i)."""
    return v_circular(H) - v_rotation_credit(inclination_deg)


def check_launch_window(launch_latitude_deg, inclination_deg):
    """Earth Satellite Operations slide 16: direct launch into an orbit of
    inclination i is only possible from latitude |L| <= i (prograde) or
    |L| <= 180 - i (retrograde)."""
    i_eff = inclination_deg if inclination_deg <= 90.0 else 180.0 - inclination_deg
    if abs(launch_latitude_deg) > i_eff + 1e-9:
        raise ValueError(f"No launch window: launch latitude {launch_latitude_deg:.1f} deg "
                         f"> orbit inclination {inclination_deg:.1f} deg (slide 16). "
                         f"Pick a site at lower latitude or plan a plane change.")


# ---------------------------------------------------------------------
# 3. EQUATIONS OF MOTION
# ---------------------------------------------------------------------
def tangent_law(t, gamma0, t_burnout):
    """Flat-Earth linear tangent law (slide 23): gamma(t) and gamma_dot(t)."""
    gamma = np.arctan(np.tan(gamma0) * (1.0 - t / t_burnout))
    gamma_dot = -np.tan(gamma0) / t_burnout * np.cos(gamma) ** 2
    return gamma, gamma_dot


def rhs(t, u, T, beta, Cd, A, mode, gamma0=None, t_burnout=None):
    """
    Right-hand side of the ODE system, state u = [X, Xd, H, Hd, m, dVg, dVd]
    dVg, dVd are bookkeeping states accumulating gravity and drag losses
    (not required for the trajectory itself, useful for the DeltaV budget).

    mode:
      'vertical' - phase 1, gamma = 90 deg by construction
      'free'     - natural gravity turn, gamma from the velocity vector
      'tangent'  - last stage, gamma prescribed by the linear tangent law
                   (t is the time since last-stage ignition)
    """
    X, Xd, H, Hd, m, dVg, dVd = u
    V = np.sqrt(Xd ** 2 + Hd ** 2)

    if mode == 'vertical':
        cosg, sing = 0.0, 1.0
    elif mode == 'free':
        if V < 1e-3:
            cosg, sing = 0.0, 1.0
        else:
            cosg, sing = Xd / V, Hd / V
    elif mode == 'tangent':
        gamma, gamma_dot = tangent_law(t, gamma0, t_burnout)
        cosg, sing = np.cos(gamma), np.sin(gamma)
    else:
        raise ValueError("unknown mode")

    D = drag_force(V, H, Cd, A)
    g = gravity(H)

    if mode == 'tangent':
        # gamma is prescribed (slide 25 hint 5): only the speed equation
        # (slide 22, h1) is integrated, the direction follows the law.
        V_dot = (T - D) / m - g * sing
        Xdd = V_dot * cosg - V * gamma_dot * sing
        Hdd = V_dot * sing + V * gamma_dot * cosg
    else:
        # round-Earth LVLH equations (slides 30-31)
        Xdd = (T - D) * cosg / m - Xd * Hd / (R_EARTH + H)
        Hdd = (T - D) * sing / m - (g - Xd ** 2 / (R_EARTH + H))

    return [Xd, Xdd, Hd, Hdd, -beta, g * sing, D / m]


# ---------------------------------------------------------------------
# 4. STAGE MASS STACKING
# ---------------------------------------------------------------------
def stage_ignition_masses(stages, payload_mass):
    """
    Returns the ignition (total) mass of the vehicle at the start of
    each stage, stacking from the top (payload) down (Rocket Performance).
    """
    m_above = payload_mass
    m0 = [None] * len(stages)
    for i in reversed(range(len(stages))):
        m0[i] = stages[i]['m_dry'] + stages[i]['m_prop'] + m_above
        m_above = m0[i]
    return m0


# ---------------------------------------------------------------------
# 4a. ENGINE CUTOFF -- burn plan of the LAST stage
# ---------------------------------------------------------------------
def last_stage_burn_plan(stages, prop_left_kg=0.0):
    """
    Burn plan of the last stage when `prop_left_kg` of its propellant is
    still on board at engine cutoff.

    beta is constant (m_prop / t_burn), so burning (m_prop - prop_left) takes
        t_burn_eff = (m_prop - prop_left) / beta = t_burn - prop_left / beta

    prop_left_kg = 0.0 -> no cutoff, the stage burns to depletion.
    """
    last = stages[-1]
    if prop_left_kg < 0.0:
        raise ValueError(f"prop_left_kg must be >= 0 (got {prop_left_kg})")
    if prop_left_kg >= last['m_prop']:
        raise ValueError(f"prop_left_kg ({prop_left_kg:.1f} kg) must be smaller than the "
                         f"last-stage propellant load ({last['m_prop']:.1f} kg)")
    beta = last['m_prop'] / last['t_burn']
    m_burned = last['m_prop'] - prop_left_kg
    return dict(m_burned=m_burned, t_burn_eff=m_burned / beta,
                t_burn_nominal=last['t_burn'], beta=beta,
                prop_left_kg=prop_left_kg)


# ---------------------------------------------------------------------
# 4b. PER-STAGE IDEAL (VACUUM, NO-LOSS) DELTA-V -- ROCKET EQUATION
# ---------------------------------------------------------------------
def stage_ideal_deltav(stages, payload_mass, prop_left_kg=0.0):
    """
    Ideal (no gravity/drag losses) DeltaV each stage supplies, from the
    rocket equation (Rocket Performance lecture):
        dV_k = Veff_k * ln(m0_k / mf_k)
    where Veff_k = T_k / beta_k (consistent with the same T and beta used
    in the ODE). Structural ratio eps_k = ms/(ms+mp) and payload ratio
    pi_k = m0,k+1 / m0,k as defined in the Rocket Performance lecture.

    With an engine cutoff, `prop_left_kg` of the LAST stage is not burned,
    so its mf (and therefore its dV) uses only the propellant actually burned.
    """
    m0 = stage_ignition_masses(stages, payload_mass)
    out = []
    dv_total = 0.0
    for i, s in enumerate(stages):
        beta = s['m_prop'] / s['t_burn']
        Veff_sim = s['T'] / beta
        Isp_sim = Veff_sim / G0

        # Thrust sanity check (requires 'Isp' in database)
        T_check = beta * s['Isp'] * G0 if 'Isp' in s else None

        prop_left = prop_left_kg if i == len(stages) - 1 else 0.0
        m_prop_burned = s['m_prop'] - prop_left

        mf = m0[i] - m_prop_burned
        dv = Veff_sim * np.log(m0[i] / mf)
        dv_total += dv

        structural_ratio = s['m_dry'] / (s['m_dry'] + s['m_prop'])
        m_above = m0[i] - (s['m_dry'] + s['m_prop'])
        payload_ratio = m_above / m0[i]
        mass_ratio = m0[i] / mf

        out.append(dict(stage=i, m0=m0[i], mf=mf, Veff=Veff_sim, dV=dv,
                        Isp=Isp_sim, T_check=T_check,
                        structural_ratio=structural_ratio,
                        payload_ratio=payload_ratio, mass_ratio=mass_ratio,
                        m_prop_burned=m_prop_burned, prop_left=prop_left))
    return out, dv_total


# ---------------------------------------------------------------------
# 4c. SAFETY EVENTS -- stop a bad shooting guess cleanly instead of
#     letting it integrate into numerical garbage (ballistic dive through
#     the ground, or a runaway). The truncated result is still a continuous
#     function of the unknowns, so the optimizer keeps a usable gradient.
# ---------------------------------------------------------------------
def _ground_event(t, u, *args):
    return u[2] + 50e3  # triggers if H drops below -50 km
_ground_event.terminal = True
_ground_event.direction = -1


def _speed_event(t, u, *args):
    return np.hypot(u[1], u[3]) - 20_000.0  # triggers above 20 km/s
_speed_event.terminal = True
_speed_event.direction = 1

_SAFETY_EVENTS = [_ground_event, _speed_event]
_ODE_OPTS = dict(events=_SAFETY_EVENTS, max_step=0.5, rtol=1e-8, atol=1e-8)


def _max_q(sol):
    """Maximum dynamic pressure q = rho V^2 / 2 over the solver output points."""
    V = np.hypot(sol.y[1], sol.y[3])
    rho = RHO0 * np.exp(-np.maximum(sol.y[2], 0.0) / H0_SCALE)
    return float(np.max(0.5 * rho * V ** 2))


def steering_diagnostic(sol, T, gamma0, t_burnout):
    """
    Post-processing check of the prescribed-gamma last stage.
    Normal acceleration the tangent law requires (slide 22, h2 equation):
        a_n = V*gamma_dot + (g - V^2/(R_E+H)) * cos(gamma)
    versus the available T/m. Returns (ratio_max, steering_loss or None).
    Steering loss (only if a_n <= T/m everywhere):
        dV_steer = integral T/m * (1 - cos(alpha)) dt,  sin(alpha) = a_n / (T/m)
    """
    t = sol.t
    H, V, m = sol.y[2], np.hypot(sol.y[1], sol.y[3]), sol.y[4]
    gamma, gamma_dot = tangent_law(t, gamma0, t_burnout)
    a_n = V * gamma_dot + (gravity(H) - V ** 2 / (R_EARTH + H)) * np.cos(gamma)
    a_T = T / m
    ratio = np.abs(a_n) / a_T
    ratio_max = float(np.max(ratio))
    if ratio_max > 1.0:
        return ratio_max, None
    loss = float(np.trapezoid(a_T * (1.0 - np.sqrt(1.0 - ratio ** 2)), t)) \
        if hasattr(np, "trapezoid") else float(np.trapz(a_T * (1.0 - np.sqrt(1.0 - ratio ** 2)), t))
    return ratio_max, loss


# ---------------------------------------------------------------------
# 5. FULL ASCENT SIMULATION FOR A GIVEN (t1, delta, prop_left)
# ---------------------------------------------------------------------
def simulate_ascent(t1, delta_deg, stages, payload_mass, return_traj=False,
                    prop_left_kg=0.0):
    """
    Runs the complete phased ascent (slide 25 hints):
      vertical rise (t1) -> kick (delta_deg) -> free gravity turn for
      stages[0:-1] -> tangent-steering law for stages[-1], cut off with
      `prop_left_kg` of propellant still on board (0.0 = burn to depletion).

    Returns a dict with final H, V, gamma, max dynamic pressure, propellant
    left / cutoff time, steering diagnostic and (optionally) the full history.
    """
    plan = last_stage_burn_plan(stages, prop_left_kg)   # validates prop_left
    if t1 <= 0.0 or t1 >= stages[0]['t_burn'] or (len(stages) == 1 and t1 >= plan['t_burn_eff']):
        raise ValueError(f"t1 = {t1:.2f} s must be > 0 and shorter than the first burn")
    m0 = stage_ignition_masses(stages, payload_mass)
    beta = [s['m_prop'] / s['t_burn'] for s in stages]
    stage_times = []        # (stage_index, t_start, t_end) -- for reporting
    stage_bookkeeping = []  # (V, dVg, dVd) snapshots at each stage boundary
    t_hist, u_hist = [0.0], []
    max_q = 0.0

    def _stage_deltav_actual(bk):
        """dV each stage actually had to supply: (V_end - V_start) + losses.
        Per-stage analogue of V* = dV_thrust + dV_grav + dV_air (slide 9);
        it is an identity of this model (V_dot = (T-D)/m - g sin(gamma) in
        every phase), so it must equal the rocket-equation dV of the stage."""
        out = []
        for i in range(len(bk) - 1):
            V0, dVg0, dVd0 = bk[i]
            V1, dVg1, dVd1 = bk[i + 1]
            out.append(dict(stage=i, V_start=V0, V_end=V1,
                            dV_gravity_loss=dVg1 - dVg0, dV_drag_loss=dVd1 - dVd0,
                            dV_needed_actual=(V1 - V0) + (dVg1 - dVg0) + (dVd1 - dVd0)))
        return out

    def _append(sol, t_offset):
        if return_traj:
            t_hist.extend(list(t_offset + sol.t[1:]))
            u_hist.extend([sol.y[:, k] for k in range(1, sol.y.shape[1])])

    def _finish(u, aborted, t_cut=None, steer=(None, None)):
        X, Xd, H, Hd, m, dVg, dVd = u
        bk = list(stage_bookkeeping)
        if aborted:
            bk.append((np.hypot(Xd, Hd), dVg, dVd))
        prop_left = None if aborted else m - payload_mass - stages[-1]['m_dry']
        if prop_left is not None and abs(prop_left - prop_left_kg) > 1.0:
            raise RuntimeError("propellant bookkeeping inconsistency")
        Veff_last = stages[-1]['T'] / beta[-1]
        reserve_dV = (Veff_last * np.log(m / (m - prop_left))
                      if prop_left is not None and prop_left > 1e-9 else (None if aborted else 0.0))
        result = dict(H=H, V=np.hypot(Xd, Hd), gamma_deg=np.rad2deg(np.arctan2(Hd, Xd)),
                      m_final=m, dV_gravity_loss=dVg, dV_drag_loss=dVd,
                      max_dynamic_pressure=max_q, stage_times=stage_times, aborted=aborted,
                      stage_deltav_actual=_stage_deltav_actual(bk), t1=t1, delta_deg=delta_deg,
                      prop_left_kg=prop_left, reserve_dV_ms=reserve_dV, t_cutoff_s=t_cut,
                      steer_ratio_max=steer[0], steering_loss_est_ms=steer[1])
        if return_traj:
            result['t'] = np.array(t_hist)
            result['u'] = np.array(u_hist).T
        return result

    # --- Phase 1: vertical rise --------------------------------------
    u = [0.0, 0.0, 0.0, 0.0, m0[0], 0.0, 0.0]
    u_hist.append(np.array(u))
    stage_bookkeeping.append((0.0, 0.0, 0.0))
    sol = solve_ivp(rhs, [0, t1], u, args=(stages[0]['T'], beta[0],
                    stages[0]['Cd'], stages[0]['A'], 'vertical'), **_ODE_OPTS)
    _append(sol, 0.0)
    max_q = max(max_q, _max_q(sol))
    if sol.status == 1:
        return _finish(sol.y[:, -1], aborted=True)
    u = sol.y[:, -1].tolist()
    t_now = t1

    # --- Phase 2: pitch kick (instantaneous) --------------------------
    V = np.hypot(u[1], u[3])
    delta = np.deg2rad(delta_deg)
    u[1] = V * np.sin(delta)   # Xd
    u[3] = V * np.cos(delta)   # Hd

    # --- Phase 3: free gravity turn for all stages except the last ---
    # Stage 0 already burned for t1 during the vertical rise.
    for i in range(len(stages) - 1):
        t_stage_start = t_now - (t1 if i == 0 else 0.0)
        t_remaining = stages[i]['t_burn'] - (t1 if i == 0 else 0.0)
        sol = solve_ivp(rhs, [0, t_remaining], u,
                        args=(stages[i]['T'], beta[i], stages[i]['Cd'],
                              stages[i]['A'], 'free'), **_ODE_OPTS)
        _append(sol, t_now)
        max_q = max(max_q, _max_q(sol))
        if sol.status == 1:
            return _finish(sol.y[:, -1], aborted=True)
        u = sol.y[:, -1].tolist()
        t_now += t_remaining
        stage_bookkeeping.append((np.hypot(u[1], u[3]), u[5], u[6]))
        stage_times.append((i, t_stage_start, t_now))
        u[4] -= stages[i]['m_dry']                    # separation
        if abs(u[4] - m0[i + 1]) > 1.0:
            raise RuntimeError("mass stacking inconsistency")

    # --- Phase 4: tangent-law steering on the final stage -------------
    last = stages[-1]
    gamma0 = np.arctan2(u[3], u[1])                   # continuity
    t_last_start = t_now - (t1 if len(stages) == 1 else 0.0)
    t_remaining_last = plan['t_burn_eff'] - (t1 if len(stages) == 1 else 0.0)
    sol = solve_ivp(rhs, [0, t_remaining_last], u,
                    args=(last['T'], beta[-1], last['Cd'], last['A'],
                          'tangent', gamma0, t_remaining_last), **_ODE_OPTS)
    _append(sol, t_now)
    max_q = max(max_q, _max_q(sol))
    if sol.status == 1:
        return _finish(sol.y[:, -1], aborted=True)
    t_now += t_remaining_last
    stage_bookkeeping.append((np.hypot(sol.y[1, -1], sol.y[3, -1]), sol.y[5, -1], sol.y[6, -1]))
    stage_times.append((len(stages) - 1, t_last_start, t_now))
    steer = steering_diagnostic(sol, last['T'], gamma0, t_remaining_last)
    return _finish(sol.y[:, -1], aborted=False, t_cut=t_now, steer=steer)


# ---------------------------------------------------------------------
# 6. SHOOTING METHOD: SOLVE FOR (delta, prop_left) THAT HIT THE ORBIT
# ---------------------------------------------------------------------
def _unpack(x, engine_cutoff, m_prop_last, prop_left_fixed=0.0):
    """Solver variables -> physical: x[0] = log10(delta [deg]),
    x[1] = propellant left at cutoff / last-stage propellant load."""
    delta = 10.0 ** x[0]
    prop_left = x[1] * m_prop_last if engine_cutoff else prop_left_fixed
    return delta, prop_left


def residuals(x, t1, stages, payload_mass, H_target, V_target, engine_cutoff):
    """
    Normalized residuals [dH/H*, dV/V*], dimensionless so that H and V have
    the same weight. A trajectory that trips a safety event still returns a
    real (if extreme) H, V, so the residual stays continuous.
    """
    delta, prop_left = _unpack(x, engine_cutoff, stages[-1]['m_prop'])
    res = simulate_ascent(t1, delta, stages, payload_mass, prop_left_kg=prop_left)
    return [(res['H'] - H_target) / H_target, (res['V'] - V_target) / V_target]


def _kick_presearch(t1, stages, payload_mass, H_target, prop_left, n=60):
    """
    1-D scan of log10(delta) at fixed prop_left. Small kicks overshoot the
    target altitude, large kicks make the vehicle fall back (safety abort).
    The last "overshoot -> undershoot" sign change brackets H = H_target and
    is refined with Brent's method. Returns log10(delta) (or the best scan
    point if no bracket exists).
    """
    def dH(ld):
        r = simulate_ascent(t1, 10.0 ** ld, stages, payload_mass, prop_left_kg=prop_left)
        return (r['H'] - H_target) / H_target

    lds = np.linspace(LOG_DELTA_MIN, LOG_DELTA_MAX, n)
    vals = [dH(ld) for ld in lds]
    bracket = None
    for k in range(n - 1):
        if vals[k] > 0.0 and vals[k + 1] <= 0.0:
            bracket = (lds[k], lds[k + 1])
    if bracket is None:
        return lds[int(np.argmin(np.abs(vals)))], False
    return brentq(dH, *bracket, xtol=1e-10), True


def solve_gravity_turn(stages, payload_mass, H_target, inclination_deg=71.0,
                       launch_latitude_deg=71.0, t1=10.0, delta_guess=1.0,
                       engine_cutoff=True, prop_reserve_required_kg=0.0,
                       tol=1e-4, lower_tol=-1e-3, upper_tol=0.03, upper_limit=0.15,
                       use_grid_presearch=None, name=None, verbose=True):
    """
    Solves the boundary value problem for a given launcher / payload / orbit.
    Returns (converged, t1, delta_deg, result_dict); result_dict also carries
    'diagnosis' and 'explanation'.

    engine_cutoff=True : unknowns (delta, prop_left) -> exact H*, V* match;
                         prop_left is then checked against
                         prop_reserve_required_kg.
    engine_cutoff=False: burn to depletion, unknown delta only -> best fit.
    tol                : |dH/H*| and |dV/V*| accepted as "orbit reached"
                         (1e-4 = 85 m and 0.7 m/s at 851 km).
    lower_tol/upper_tol/upper_limit: over/undershoot classes, used only
                         without engine cutoff (see diagnose_convergence).
    """
    if use_grid_presearch is None:
        use_grid_presearch = USE_GRID_PRESEARCH
    label = name if name is not None else 'launcher'
    fail = (False, None, None, None)

    try:
        check_launch_window(launch_latitude_deg, inclination_deg)
        last_stage_burn_plan(stages, prop_reserve_required_kg if engine_cutoff else 0.0)
    except ValueError as err:
        print(f"--- {label} --- ABORT: {err}")
        return fail

    m0_total = payload_mass + sum(s['m_dry'] + s['m_prop'] for s in stages)
    twr_liftoff = stages[0]['T'] / (m0_total * G0)
    if twr_liftoff < 1.0:
        print(f"--- {label} --- ABORT: liftoff TWR is {twr_liftoff:.2f}, vehicle cannot lift off.")
        return fail
    if not (0.0 < t1 < stages[0]['t_burn']):
        print(f"--- {label} --- ABORT: t1 = {t1} s must lie inside the first burn.")
        return fail

    V_target = v_target_effective(H_target, inclination_deg)
    m_prop_last = stages[-1]['m_prop']
    prop_left0 = prop_reserve_required_kg if engine_cutoff else 0.0

    # --- start point -------------------------------------------------
    if use_grid_presearch:
        ld0, bracketed = _kick_presearch(t1, stages, payload_mass, H_target, prop_left0)
        if verbose:
            print(f"Kick pre-search ({'bracketed' if bracketed else 'no bracket'}): "
                  f"delta = {10 ** ld0:.4g} deg for t1 = {t1:.2f} s")
    else:
        ld0 = math.log10(max(delta_guess, 10 ** LOG_DELTA_MIN))

    # --- local solve -------------------------------------------------
    if engine_cutoff:
        x0 = [ld0, prop_left0 / m_prop_last]
        lb, ub = [LOG_DELTA_MIN, 0.0], [LOG_DELTA_MAX, RESERVE_FRAC_MAX]
        fun = lambda x: residuals(x, t1, stages, payload_mass, H_target, V_target, True)
    else:
        x0 = [ld0]
        lb, ub = [LOG_DELTA_MIN], [LOG_DELTA_MAX]
        fun = lambda x: residuals([x[0], 0.0], t1, stages, payload_mass, H_target, V_target, False)
    x0 = np.clip(x0, lb, ub)
    opt = least_squares(fun, x0, bounds=(lb, ub), x_scale='jac',
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
    delta_deg, prop_left = _unpack(list(opt.x) + [0.0], engine_cutoff, m_prop_last)

    res = simulate_ascent(t1, delta_deg, stages, payload_mass, return_traj=True,
                          prop_left_kg=prop_left)
    diagnosis, explanation = diagnose_convergence(
        res, H_target, V_target, engine_cutoff=engine_cutoff,
        prop_reserve_required_kg=prop_reserve_required_kg,
        at_zero_reserve=engine_cutoff and opt.x[1] <= 1e-9,
        tol=tol, lower_tol=lower_tol, upper_tol=upper_tol, upper_limit=upper_limit)
    converged = diagnosis in ("OK", "VIABLE_OVERSHOOT")
    res.update(diagnosis=diagnosis, explanation=explanation, V_target=V_target,
               inclination_deg=inclination_deg, engine_cutoff=engine_cutoff,
               prop_reserve_required_kg=prop_reserve_required_kg,
               solver_message=opt.message.strip(), residuals=list(opt.fun))

    if verbose:
        _print_report(label, converged, res, stages, payload_mass, H_target)
    return converged, t1, delta_deg, res


def _print_report(label, converged, res, stages, payload_mass, H_target):
    plan = last_stage_burn_plan(stages, res['prop_left_kg'] or 0.0)
    stage_dv, dv_total = stage_ideal_deltav(stages, payload_mass, res['prop_left_kg'] or 0.0)
    print()
    print(f"--- {label} ---")
    print(f"converged      : {converged} ({res['diagnosis']})")
    print(f"explanation    : {res['explanation']}")
    print(f"solver message : {res['solver_message']}")
    print(f"residuals      : dH/H_target={res['residuals'][0]:+.2e}  dV/V_target={res['residuals'][-1]:+.2e}")
    print(f"t1 (input)     : {res['t1']:8.2f} s")
    print(f"kick angle     : {res['delta_deg']:10.4g} deg")
    print(f"final H        : {res['H']/1e3:8.2f} km  (target {H_target/1e3:.2f} km)")
    print(f"final V        : {res['V']:8.1f} m/s  (target {res['V_target']:.1f} m/s = V_circ "
          f"{v_circular(H_target):.1f} - rotation credit {v_rotation_credit(res['inclination_deg']):.1f}, "
          f"i = {res['inclination_deg']:.1f} deg)")
    print(f"final gamma    : {res['gamma_deg']:8.3f} deg (target 0 deg)")
    print(f"max dyn. press.: {res['max_dynamic_pressure']/1e3:8.2f} kPa")
    print(f"gravity loss   : {res['dV_gravity_loss']:8.1f} m/s")
    print(f"drag loss      : {res['dV_drag_loss']:8.1f} m/s")
    if res.get('steer_ratio_max') is not None:
        feas = ("FLYABLE, steering loss est. "
                f"{res['steering_loss_est_ms']:.1f} m/s" if res['steering_loss_est_ms'] is not None
                else "NOT FLYABLE with thrust alone (needs coast arc + circularisation)")
        print(f"last-stage turn: required normal accel. up to {res['steer_ratio_max']:.2f} x T/m -> {feas}")
    print()
    print("--- Engine cutoff / propellant (last stage) ---")
    if res['prop_left_kg'] is None:
        print("trajectory aborted -> no propellant bookkeeping")
    else:
        m_prop_last = stages[-1]['m_prop']
        t_nom_end = res['t_cutoff_s'] + res['prop_left_kg'] / plan['beta']
        print(f"engine cutoff  : {'ENABLED (cutoff time solved for)' if res['engine_cutoff'] else 'disabled (burn to depletion)'}")
        print(f"last-stage prop: loaded {m_prop_last:10.1f} kg | burned {m_prop_last - res['prop_left_kg']:10.1f} kg")
        print(f"PROPELLANT LEFT: {res['prop_left_kg']:10.1f} kg  ({100 * res['prop_left_kg'] / m_prop_last:.2f} % of load)"
              f"  | required {res['prop_reserve_required_kg']:.1f} kg"
              f"  | margin {res['prop_left_kg'] - res['prop_reserve_required_kg']:+.1f} kg")
        print(f"cutoff time    : {res['t_cutoff_s']:8.1f} s  (depletion would be at {t_nom_end:.1f} s)")
        print(f"dV left on board (rocket eq., same Veff): {res['reserve_dV_ms']:8.1f} m/s")
    print()
    print("--- Stage Values ---")
    for (i, t_start, t_end), sd, sda in zip(res['stage_times'], stage_dv, res['stage_deltav_actual']):
        print(f"Stage {i+1} | Burns: [{t_start:6.1f}, {t_end:6.1f}] s")
        print(f"Masses:   m0={sd['m0']:9.1f} kg | mf={sd['mf']:9.1f} kg | "
              f"prop burned={sd['m_prop_burned']:9.1f} kg | prop left={sd['prop_left']:8.1f} kg")
        print(f"dV_ideal={sd['dV']:8.1f} m/s = V gain {sda['V_end'] - sda['V_start']:7.1f} "
              f"({sda['V_start']:7.1f}->{sda['V_end']:7.1f}) + grav loss {sda['dV_gravity_loss']:6.1f} "
              f"+ drag loss {sda['dV_drag_loss']:6.1f}   [check {sd['dV'] - sda['dV_needed_actual']:+.2f}]")
        print(f"Ratios: Struct={sd['structural_ratio']:.3f} | Payload={sd['payload_ratio']:.3f} | Mass(R)={sd['mass_ratio']:.3f}")
        print(f"Engine: Isp={sd['Isp']:.1f} s | Veff={sd['Veff']:.1f} m/s")
        if sd['T_check'] is not None:
            diff = stages[i]['T'] - sd['T_check']
            print(f"Thrust: Simulated={stages[i]['T']/1000:.1f} kN | Checked={sd['T_check']/1000:.1f} kN | Diff={diff/1000:+.1f} kN")
        print()
    # Velocity budget V* = dV_thrust + dV_grav + dV_air (slide 9, signs as losses):
    #   dV supplied - gravity loss - drag loss = dV needed (= V*).
    # Holds exactly in this model, so a non-zero value only measures the
    # integration error / a missed target. (Steering loss is NOT part of it.)
    V_star = res['V_target']
    print(f"TOTAL ideal DeltaV supplied (rocket eq.) : {dv_total:8.1f} m/s")
    print(f"dV needed to reach orbit (V*)             : {V_star:8.1f} m/s")
    print(f"dV supplied - drag - gravity - dV needed  : "
          f"{dv_total - res['dV_drag_loss'] - res['dV_gravity_loss'] - V_star:8.2f} m/s (should be ~0)")
    # Overall payload ratio of the multistage rocket (Rocket Performance lecture):
    #   Lambda = m*/m0,1 = prod_k lambda_k, lambda_k = m0,k+1/m0,k
    Lam = float(np.prod([sd['payload_ratio'] for sd in stage_dv]))
    print(f"overall payload ratio Lambda = prod(lambda_k) = m*/m0,1 : {Lam:.5f}  "
          f"({payload_mass:.1f} kg / {stage_dv[0]['m0']:.1f} kg lift-off mass)")
    print()


# ---------------------------------------------------------------------
# 6b. DIAGNOSIS
# ---------------------------------------------------------------------
def _diagnose_orbit(res, H_target, V_target, engine_cutoff=True,
                         prop_reserve_required_kg=0.0, at_zero_reserve=False,
                         tol=1e-4, lower_tol=-1e-3, upper_tol=0.03, upper_limit=0.15):
    """
    Classifies a result. Returns (label, explanation), label one of
      "OK"                 orbit reached (|dH|,|dV| <= tol) and, with engine
                           cutoff, propellant left >= required reserve
      "RESERVE_SHORTFALL"  orbit reached but less propellant left than the
                           required reserve -> reduce payload / bigger stage
      "UNDERSHOOT"         orbit not reached even when burning everything
                           -> not enough delta-V
      "VIABLE_OVERSHOOT"   (no cutoff only) up to upper_limit above target
      "MASSIVE_OVERSHOOT"  (no cutoff only) more than upper_limit above
      "MIXED"              H and V off in different directions
      "SAFETY_ABORT"       trajectory hit a safety event
    Without engine cutoff the asymmetric band lower_tol..upper_tol counts
    as "OK" (burning to depletion cannot hit V* exactly).
    """
    if res is None:
        return "SAFETY_ABORT", "Solver never ran (e.g. liftoff TWR < 1)."
    if res.get('aborted'):
        return ("SAFETY_ABORT",
                "Trajectory hit a safety event (ground impact or runaway) before "
                "cutoff -- the kick sent it off a physically sane path.")

    dH = (res['H'] - H_target) / H_target
    dV = (res['V'] - V_target) / V_target

    if engine_cutoff:
        if abs(dH) <= tol and abs(dV) <= tol:
            left = res['prop_left_kg']
            if left + 1e-6 >= prop_reserve_required_kg:
                return "OK", (f"Orbit reached; {left:.1f} kg propellant left "
                              f"(required {prop_reserve_required_kg:.1f} kg).")
            return ("RESERVE_SHORTFALL",
                    f"Orbit reached but only {left:.1f} kg propellant left, "
                    f"{prop_reserve_required_kg - left:.1f} kg short of the required "
                    f"{prop_reserve_required_kg:.1f} kg -> reduce payload or enlarge the last stage.")
        if dV < 0 and at_zero_reserve:
            return ("UNDERSHOOT",
                    f"Even burning all propellant: H {dH*100:+.2f}%, V {dV*100:+.2f}% "
                    f"-> insufficient delta-V (more propellant, bigger stage, extra stage).")
        return ("MIXED",
                f"Solver could not match the orbit: H {dH*100:+.2f}%, V {dV*100:+.2f}% "
                f"-- check t1 / kick search range.")

    if (lower_tol <= dH <= upper_tol) and (lower_tol <= dV <= upper_tol):
        return "OK", "Within tolerance of the target orbit."
    if dV > upper_tol and dH >= lower_tol:
        if dH > upper_limit or dV > upper_limit:
            return "MASSIVE_OVERSHOOT", f"Overpowered: dV={dV*100:+.1f}%, dH={dH*100:+.1f}%."
        return "VIABLE_OVERSHOOT", f"Viable with reserve: dV={dV*100:+.1f}%, dH={dH*100:+.1f}% -> use engine cutoff."
    if dV < lower_tol and dH <= upper_tol:
        return ("UNDERSHOOT",
                f"Best trajectory falls short: H {dH*100:+.1f}%, V {dV*100:+.1f}% "
                f"-> insufficient delta-V.")
    return ("MIXED", f"H {dH*100:+.1f}% and V {dV*100:+.1f}% off target in different directions.")


def diagnose_convergence(res, H_target, V_target, engine_cutoff=True,
                         prop_reserve_required_kg=0.0, at_zero_reserve=False,
                         tol=1e-4, lower_tol=-1e-3, upper_tol=0.03, upper_limit=0.15):
    """
    Orbit classification (see _diagnose_orbit) plus a steering-feasibility gate.

    Extra label:
      "NOT_FLYABLE"  the orbit is matched numerically (label would be OK,
                     VIABLE_OVERSHOOT or RESERVE_SHORTFALL) but the tangent law
                     needs a normal acceleration above STEER_RATIO_LIMIT * T/m
                     somewhere on the last stage, so thrust alone cannot follow it.
    Failure labels (UNDERSHOOT, MIXED, SAFETY_ABORT, ...) are left unchanged.
    """
    label, explanation = _diagnose_orbit(
        res, H_target, V_target, engine_cutoff=engine_cutoff,
        prop_reserve_required_kg=prop_reserve_required_kg, at_zero_reserve=at_zero_reserve,
        tol=tol, lower_tol=lower_tol, upper_tol=upper_tol, upper_limit=upper_limit)
    ratio = None if res is None else res.get('steer_ratio_max')
    if (ENFORCE_STEERING_LIMIT and ratio is not None and ratio > STEER_RATIO_LIMIT
            and label in ("OK", "VIABLE_OVERSHOOT", "RESERVE_SHORTFALL")):
        return ("NOT_FLYABLE",
                f"Orbit matched numerically ({label}), but the last-stage tangent law needs "
                f"a_n up to {ratio:.2f} x T/m (limit {STEER_RATIO_LIMIT:.2f}): thrust alone "
                f"cannot turn the velocity vector that fast. {explanation}")
    return label, explanation


# ---------------------------------------------------------------------
# 6c. EXCEL EXPORT -- run every launcher and dump results into a
#     spreadsheet (one summary row per launcher, one row per stage).
# ---------------------------------------------------------------------
def run_database_to_excel(launcher_database, payload_mass, H_target, inclination_deg=71.0,
                          launch_latitude_deg=71.0, filename="gravity_turn_results.xlsx",
                          t1=10.0, delta_guess=1.0, engine_cutoff=True,
                          prop_reserve_required_kg=0.0, use_grid_presearch=None, **tolerances):
    """
    Runs solve_gravity_turn() for every launcher and writes an .xlsx with
    sheets 'Summary' and 'Stage Details'. The diagnosis written to the sheet
    is the one computed by solve_gravity_turn (same tolerances).
    Returns (summary_rows, stage_rows, all_results); all_results maps
    launcher name -> dict(converged=..., res=...) for plotting.
    """
    if use_grid_presearch is None:
        use_grid_presearch = USE_GRID_PRESEARCH
    V_target = v_target_effective(H_target, inclination_deg)
    summary_rows, stage_rows, all_results = [], [], {}

    for name, stages in launcher_database.items():
        converged, t1_used, delta_deg, res = solve_gravity_turn(
            stages, payload_mass, H_target, inclination_deg=inclination_deg,
            launch_latitude_deg=launch_latitude_deg, t1=t1, delta_guess=delta_guess,
            engine_cutoff=engine_cutoff, prop_reserve_required_kg=prop_reserve_required_kg,
            use_grid_presearch=use_grid_presearch, name=name, **tolerances)
        all_results[name] = dict(converged=converged, res=res)

        if res is None:
            summary_rows.append(dict(launcher=name, converged=False, target_H_km=H_target / 1e3,
                                     target_V_ms=V_target, prop_reserve_required_kg=prop_reserve_required_kg,
                                     note="Aborted before simulating (see terminal: TWR, launch window, reserve, t1)"))
            continue

        left = res['prop_left_kg'] or 0.0
        stage_dv, dv_total = stage_ideal_deltav(stages, payload_mass, left)
        Lam = float(np.prod([sd['payload_ratio'] for sd in stage_dv]))
        summary_rows.append(dict(
            launcher=name, converged=converged, diagnosis=res['diagnosis'],
            t1_s=t1_used, delta_deg=delta_deg,
            final_H_km=res['H'] / 1e3, target_H_km=H_target / 1e3,
            final_V_ms=res['V'], target_V_ms=V_target, inclination_deg=inclination_deg,
            final_gamma_deg=res['gamma_deg'], max_q_kPa=res['max_dynamic_pressure'] / 1e3,
            gravity_loss_ms=res['dV_gravity_loss'], drag_loss_ms=res['dV_drag_loss'],
            steer_ratio_max=res['steer_ratio_max'], steering_loss_ms=res['steering_loss_est_ms'],
            dV_supplied_ms=dv_total,
            # dV needed = speed that must be gained to reach the target orbit
            # (V* = V_circ - rotation credit). Budget: supplied - drag - gravity = needed.
            dV_needed_ms=V_target,
            dV_needed_check_ms=(dv_total - res['dV_drag_loss'] - res['dV_gravity_loss']) - V_target,
            m0_liftoff_kg=stage_dv[0]['m0'], overall_payload_ratio=Lam,
            prop_reserve_required_kg=prop_reserve_required_kg, prop_left_kg=res['prop_left_kg'],
            prop_margin_kg=(res['prop_left_kg'] - prop_reserve_required_kg
                            if res['prop_left_kg'] is not None else None),
            t_cutoff_s=res['t_cutoff_s'], reserve_dV_ms=res['reserve_dV_ms'],
            note="" if res['diagnosis'] == "OK" else res['explanation']))

        for (i, t_start, t_end), sd, sda in zip(res['stage_times'], stage_dv, res['stage_deltav_actual']):
            T_check = sd['T_check']
            stage_rows.append(dict(
                launcher=name, stage=i + 1, t_start_s=t_start, t_end_s=t_end,
                m0_kg=sd['m0'], mf_kg=sd['mf'], Veff_ms=sd['Veff'], Isp_s=sd['Isp'],
                dV_ideal_ms=sd['dV'], V_start_ms=sda['V_start'], V_end_ms=sda['V_end'],
                gravity_loss_ms=sda['dV_gravity_loss'], drag_loss_ms=sda['dV_drag_loss'],
                identity_check_ms=sd['dV'] - sda['dV_needed_actual'],
                structural_ratio=sd['structural_ratio'],
                payload_ratio=sd['payload_ratio'], mass_ratio=sd['mass_ratio'],
                thrust_diff_kN=(stages[i]['T'] - T_check) / 1e3 if T_check is not None else None,
                prop_burned_kg=sd['m_prop_burned'], prop_left_kg=sd['prop_left']))

    _write_results_workbook(summary_rows, stage_rows, payload_mass, H_target, filename,
                            inclination_deg, t1, engine_cutoff, prop_reserve_required_kg)
    return summary_rows, stage_rows, all_results


def _style_sheet(ws, headers, rows, key_order):
    """Header row + data rows: bold header, frozen header, autofilter,
    column widths sized to content."""
    header_font = Font(name="Arial", bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    body_font = Font(name="Arial")

    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for r, row in enumerate(rows, start=2):
        for c, key in enumerate(key_order, start=1):
            val = row.get(key)
            if isinstance(val, np.generic):
                val = val.item()
            cell = ws.cell(row=r, column=c, value=val)
            cell.font = body_font
            if isinstance(val, float):
                cell.number_format = "0.000"

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(len(rows) + 1, 1)}"

    for c, h in enumerate(headers, start=1):
        max_len = max([len(str(h))] + [len(str(row.get(key_order[c - 1], ""))) for row in rows]) if rows else len(str(h))
        ws.column_dimensions[get_column_letter(c)].width = min(max(max_len + 2, 10), 32)


def _write_results_workbook(summary_rows, stage_rows, payload_mass, H_target, filename,
                            inclination_deg, t1, engine_cutoff, prop_reserve_required_kg):
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Summary"
    summary_cols = [
        ("Launcher", "launcher"), ("Converged", "converged"), ("Diagnosis", "diagnosis"),
        ("t1 (s, input)", "t1_s"), ("Kick angle (deg)", "delta_deg"),
        ("Final H (km)", "final_H_km"), ("Target H (km)", "target_H_km"),
        ("Final V (m/s)", "final_V_ms"), ("Target V (m/s)", "target_V_ms"),
        ("Inclination (deg)", "inclination_deg"), ("Final gamma (deg)", "final_gamma_deg"),
        ("Max dyn. pressure (kPa)", "max_q_kPa"), ("Gravity loss (m/s)", "gravity_loss_ms"),
        ("Drag loss (m/s)", "drag_loss_ms"),
        ("Last stage: max a_n/(T/m) (>1 = not flyable)", "steer_ratio_max"),
        ("Steering loss est. (m/s)", "steering_loss_ms"),
        ("dV supplied, rocket eq. (m/s)", "dV_supplied_ms"),
        ("dV needed to reach orbit, V* (m/s)", "dV_needed_ms"),
        ("dV supplied - drag - gravity - dV needed (m/s, ~0)", "dV_needed_check_ms"),
        ("Lift-off mass m0,1 (kg)", "m0_liftoff_kg"),
        ("Overall payload ratio (Lambda = m*/m0,1 = prod lambda_k)", "overall_payload_ratio"),
        ("Prop. reserve required (kg)", "prop_reserve_required_kg"),
        ("Prop. left at cutoff (kg)", "prop_left_kg"), ("Prop. margin (kg)", "prop_margin_kg"),
        ("Cutoff time (s)", "t_cutoff_s"), ("dV left on board (m/s)", "reserve_dV_ms"),
        ("Notes", "note"),
    ]
    _style_sheet(ws1, [c[0] for c in summary_cols], summary_rows, [c[1] for c in summary_cols])
    note_row = len(summary_rows) + 3
    ws1.cell(row=note_row, column=1,
             value=(f"Payload = {payload_mass:.1f} kg | Target: {H_target/1e3:.1f} km circular, "
                    f"i = {inclination_deg:.1f} deg | t1 = {t1:.1f} s | Engine cutoff: "
                    f"{'ON (cutoff time solved), required reserve ' + format(prop_reserve_required_kg, '.1f') + ' kg' if engine_cutoff else 'OFF'}")
             ).font = Font(name="Arial", italic=True, size=9)

    ws2 = wb.create_sheet("Stage Details")
    stage_cols = [
        ("Launcher", "launcher"), ("Stage", "stage"), ("t start (s)", "t_start_s"),
        ("t end (s)", "t_end_s"), ("m0 (kg)", "m0_kg"), ("mf (kg)", "mf_kg"),
        ("Veff (m/s)", "Veff_ms"), ("Isp (s)", "Isp_s"), ("dV ideal (m/s)", "dV_ideal_ms"),
        ("V start (m/s)", "V_start_ms"), ("V end (m/s)", "V_end_ms"),
        ("Gravity loss (m/s)", "gravity_loss_ms"), ("Drag loss (m/s)", "drag_loss_ms"),
        ("Identity check (m/s, ~0)", "identity_check_ms"),
        ("Structural ratio", "structural_ratio"), ("Payload ratio", "payload_ratio"),
        ("Mass ratio", "mass_ratio"), ("Thrust diff vs Isp check (kN)", "thrust_diff_kN"),
        ("Prop. burned (kg)", "prop_burned_kg"), ("Prop. left (kg)", "prop_left_kg"),
    ]
    _style_sheet(ws2, [c[0] for c in stage_cols], stage_rows, [c[1] for c in stage_cols])
    wb.save(filename)


# ---------------------------------------------------------------------
# 6d. PLOTS
# ---------------------------------------------------------------------
def plot_trajectory(res, H_target=None, V_target=None, title_label="Trajectory",
                    save_path=None, show=True):
    """
    4-panel dashboard, all versus time: altitude, ground range, speed,
    flight path angle. Requires return_traj=True. Ground range uses
    dX0/dt = R_E/(R_E+H) * V cos(gamma) (slide 22), since X itself is the
    arc length measured at altitude H.
    save_path : if given, the figure is saved there (overwritten).
    """
    if 't' not in res or 'u' not in res:
        print("No trajectory data to plot. Make sure return_traj=True was used.")
        return

    t = res['t']
    Xd, H, Hd = res['u'][1, :], res['u'][2, :], res['u'][3, :]
    V = np.hypot(Xd, Hd)
    gamma = np.where(V > 1.0, np.rad2deg(np.arctan2(Hd, Xd)), np.nan)  # undefined at rest
    gamma[t <= res.get('t1', 0.0)] = 90.0                               # vertical rise
    X_ground = cumulative_trapezoid(R_EARTH / (R_EARTH + H) * Xd, t, initial=0.0) / 1e3

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Ascent Profile: {title_label}", fontsize=16, fontweight='bold')

    # 1. Altitude vs time
    axs[0, 0].plot(t, H / 1e3, 'b-', linewidth=2)
    if H_target:
        axs[0, 0].axhline(H_target / 1e3, color='r', linestyle='--', label=f'Target H ({H_target/1e3:.1f} km)')
        axs[0, 0].legend()
    axs[0, 0].set_title("Altitude vs Time")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Altitude (km)")

    # 2. Trajectory Profile (Altitude vs Downrange)
    axs[0, 1].plot(X_ground, H / 1e3, 'g-', linewidth=2)
    if H_target:
        axs[0, 1].axhline(H_target / 1e3, color='r', linestyle='--', label=f'Target H ({H_target/1e3:.1f} km)')
        axs[0, 1].legend()
    axs[0, 1].set_title("Altitude vs Downrange Distance")
    axs[0, 1].set_xlabel("Downrange (km)")
    axs[0, 1].set_ylabel("Altitude (km)")
    axs[0, 1].grid(True)

    # 3. Speed vs time
    axs[1, 0].plot(t, V, 'r-', linewidth=2)
    if V_target:
        axs[1, 0].axhline(V_target, color='k', linestyle='--', label=f'Target V ({V_target:.1f} m/s)')
        axs[1, 0].legend()
    axs[1, 0].set_title("Speed vs Time")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Speed (m/s)")

    # 4. Flight path angle vs time
    axs[1, 1].plot(t, gamma, 'm-', linewidth=2)
    axs[1, 1].axhline(0, color='k', linestyle='--', label="Target gamma (0 deg)")
    axs[1, 1].legend()
    axs[1, 1].set_title("Flight Path Angle (γ) vs Time")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Angle (degrees)")

    # Staging / cutoff events on every panel
    for ax in axs.flat:
        ax.grid(True)
        for (i, t_start, t_end) in res.get('stage_times', []):
            if t_start > 0:
                ax.axvline(t_start, color='gray', linestyle=':', alpha=0.7)
        if res.get('t_cutoff_s'):
            ax.axvline(res['t_cutoff_s'], color='gray', linestyle=':', alpha=0.7)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)     # must come BEFORE plt.show()
    if show:
        plt.show()


# ---------------------------------------------------------------------
# 7. LAUNCHER DATABASE
#    Needed values for simulation: T (N), m_prop (kg), m_dry (kg), t_burn (s), Cd, A (m^2)
#    Isp (s) is needed only for crosschecking the thrust value against the ideal rocket equation.
# ---------------------------------------------------------------------

# Scale factor applied to ALL the propellant in the database (1.0 = nominal values).
# It is applied at ONLY one place (in make_stage / make_booster_core_stages) : do not
# put it in the database entries, otherwise it would be applied twice.


def calc_A(diameter):
    """Calculate the reference area from the maximum diameter in meters."""
    return math.pi * (diameter / 2)**2


def calc_t_burn(m_prop, Isp, T):
    return m_prop * Isp * G0 / T


def calc_Isp_parallel(engines):
    """
    Calculate the effective Isp of multiple engines
    burning simultaneously.
    """
    T_total = sum(engine["T"] for engine in engines)
    mdot_total = sum(engine["T"] / (engine["Isp"] * G0) for engine in engines)
    Isp_eff = T_total / (mdot_total * G0)
    return Isp_eff


def calc_parallel_phase(engines, tol=0.10):
    """
    Combine several engines burning simultaneously
    into one equivalent propulsion phase.

    engines : list of dict(T, m_prop, Isp) -- ONE dict per engine/booster.
              m_prop is the propellant that engine burns DURING THIS PHASE.

    Returns the TUPLE (T_total, m_prop_total, Isp_eff, t_burn).

    The equivalent engine has T_total = sum(T_i), m_prop_total = sum(m_prop_i)
    and the thrust-weighted Isp_eff, so that beta = m_prop_total / t_burn and
    Veff = T_total / beta = Isp_eff * G0 stay consistent, as the simulator expects.

    The lumping assumes all engines run dry at the same instant. A warning is
    raised if their individual burn times differ by more than `tol` (fraction
    of the phase duration).
    """
    T_total = sum(engine["T"] for engine in engines)
    m_prop_total = sum(engine["m_prop"] for engine in engines)
    Isp_eff = calc_Isp_parallel(engines)
    t_burn = calc_t_burn(m_prop_total, Isp_eff, T_total)

    t_each = [calc_t_burn(e["m_prop"], e["Isp"], e["T"]) for e in engines]
    if max(t_each) - min(t_each) > tol * t_burn:
        warnings.warn(
            f"Parallel phase: individual burn times range from {min(t_each):.1f} s to "
            f"{max(t_each):.1f} s (lumped phase = {t_burn:.1f} s). The lumped model assumes "
            f"all engines deplete together -- check m_prop / T / Isp of the engines.",
            stacklevel=3)
    return T_total, m_prop_total, Isp_eff, t_burn


def calc_t_burn_parallel(engines):
    """Burn time [s] only: the 4th element of calc_parallel_phase()."""
    return calc_parallel_phase(engines)[3]


def make_stage(T, m_prop, m_dry, Isp, diameter, Cd=CD):
    """
    Build one stage dict. t_burn is DERIVED from (m_prop, Isp, T), so the three
    can never disagree
    """
    return dict(T=T, m_prop=m_prop, m_dry=m_dry,
                t_burn=calc_t_burn(m_prop, Isp, T),
                Cd=Cd, A=calc_A(diameter), Isp=Isp)


def make_booster_core_stages(boosters, core, m_dry_boosters, m_dry_core,
                             core_diameter, booster_diameter, t_sep=None, Cd=CD):
    """
    Build the TWO sequential stage dicts of a launcher whose boosters and core
    burn together, the boosters are then dropped, and the core keeps burning.
    Returns [phase_1, phase_2]; use it as  *make_booster_core_stages(...)  in a list.

    boosters : list of dict(T, m_prop, Isp), one per booster.
               m_prop = TOTAL propellant of that booster.
    core     : dict(T, m_prop, Isp) -- the core as ONE equivalent engine (several
               identical engines: sum their T). m_prop = TOTAL core propellant.
    m_dry_boosters : dry mass of ALL boosters together (dropped after phase 1)
    m_dry_core     : dry mass of the core (carried by phase 2, hence by phase 1 too)

    Phase 1 lasts until the boosters run dry:  t_sep = min_i(m_prop_i / mdot_i).
    Every engine burns mdot_i * t_sep in phase 1, so all engines burn for exactly
    the same time and the lumped equivalent engine is exact. Nothing is typed twice:
      - core propellant burned in phase 1 = mdot_core * t_sep
      - phase 2 m_prop = core total - burned in phase 1 (never double counted)
    t_sep can be imposed (e.g. a published separation time); it must then be
    <= the booster depletion time, and any booster propellant not burned by t_sep
    is jettisoned with the boosters (added to phase-1 m_dry).

    Drag area of phase 1 = core cross-section + n * booster cross-section
    (first-order estimate: no shielding between the bodies).
    """
    boosters = [dict(b, m_prop=b["m_prop"]) for b in boosters]
    core = dict(core, m_prop=core["m_prop"])
    mdot = lambda e: e["T"] / (e["Isp"] * G0)

    t_dry = min(b["m_prop"] / mdot(b) for b in boosters)      # first booster runs dry
    if t_sep is None:
        t_sep = t_dry
    elif t_sep > t_dry * (1.0 + 1e-9):
        raise ValueError(f"t_sep = {t_sep:.1f} s is longer than the booster burn time "
                         f"({t_dry:.1f} s): the boosters would run out of propellant first")

    burned = [dict(e, m_prop=mdot(e) * t_sep) for e in boosters + [core]]   # phase-1 use
    booster_left = sum(b["m_prop"] for b in boosters) - sum(e["m_prop"] for e in burned[:-1])
    core_burned = burned[-1]["m_prop"]
    if core_burned >= core["m_prop"]:
        raise ValueError(f"the core would run dry ({core['m_prop']:.0f} kg) before the boosters "
                         f"are jettisoned (needs {core_burned:.0f} kg)")

    T, m_prop, Isp, t_burn = calc_parallel_phase(burned)
    phase_1 = dict(T=T, m_prop=m_prop, m_dry=m_dry_boosters + booster_left, t_burn=t_burn,
                   Cd=Cd, A=calc_A(core_diameter) + len(boosters) * calc_A(booster_diameter),
                   Isp=Isp)

    m_prop_2 = core["m_prop"] - core_burned                    # what is LEFT in the core
    phase_2 = dict(T=core["T"], m_prop=m_prop_2, m_dry=m_dry_core,
                   t_burn=calc_t_burn(m_prop_2, core["Isp"], core["T"]),
                   Cd=Cd, A=calc_A(core_diameter), Isp=core["Isp"])
    return [phase_1, phase_2]


LAUNCHER_DATABASE = {
# ============================================================
# SOYUZ 2.1b
# Booster and core: mean of the published sea-level and vacuum values.
# ============================================================
"Soyuz_2_1b": [
    *make_booster_core_stages(
        boosters=[dict(T=929900.0, m_prop=0.7723*39160.0, Isp=290.5)] * 4,
        core=dict(T=891350.0, m_prop=90100.0, Isp=287.0),
        m_dry_boosters=4 * 3784.0,
        m_dry_core=6545.0,
        core_diameter=2.95,
        booster_diameter=2.68),

    # Block I (RD-0124, 2.1b column; vacuum only)
    make_stage(T=294300.0, m_prop=25400.0, m_dry=2355.0, Isp=359.0, diameter=2.66),
    
    # Stage 4: Fregat-M (restartable; vacuum only)
    make_stage(T=19850.0, m_prop=0.4162*6650.0, m_dry=1035.0, Isp=333.2, diameter=3.8),
],

}


# ---------------------------------------------------------------------
# 8. EXAMPLE DRIVER -- fill in your real payload mass and target orbit
# ---------------------------------------------------------------------
if __name__ == "__main__":

    PAYLOAD_MASS = 3587          # kg

    H_TARGET = 851e3             # m, parking-orbit altitude
    TARGET_INCLINATION = 71.0    # deg, orbit inclination (sets the Earth-rotation credit)
    LAUNCH_LATITUDE = 71.0       # deg, launch-site latitude (only checked: must be <= inclination)

    # Vertical-rise duration (slide 25 hint 1): a short INPUT, not an unknown.
    # Any value of a few seconds gives the same trajectory; the solver finds the kick.
    T1_VERTICAL = 10.0           # s

    # --- ENGINE CUTOFF ------------------------------------------------
    ENGINE_CUTOFF = True         # True -> the solver finds the cutoff time of the last stage
    PROP_RESERVE_REQUIRED_KG = 40  # kg that must still be on board at cutoff (checked)
                                    # 198.98 Falcon 9 Block 5, 33.87 Vega-C, 119.95; 39.66 (with Fregat-M) Soyuz 2.1b, 111.95 H3-24
    OUTPUT_FILE = "gravity_turn_results.xlsx"

    # 1. Solve EVERY launcher first, then write the workbook
    summary_rows, stage_rows, all_results = run_database_to_excel(
        LAUNCHER_DATABASE, PAYLOAD_MASS, H_TARGET, inclination_deg=TARGET_INCLINATION,
        launch_latitude_deg=LAUNCH_LATITUDE, filename=OUTPUT_FILE, t1=T1_VERTICAL,
        engine_cutoff=ENGINE_CUTOFF, prop_reserve_required_kg=PROP_RESERVE_REQUIRED_KG,
        use_grid_presearch=USE_GRID_PRESEARCH)
    print(f"Database results written to {OUTPUT_FILE}\n")

    # 2. Then plot: one 4-panel window (and one PNG) per launcher, no re-simulation
    plot = True                  # Set to False to skip the plots
    if plot:
        V_target = v_target_effective(H_TARGET, TARGET_INCLINATION)
        for name, r in all_results.items():
            res = r['res']
            if res is None or res.get('aborted'):
                print(f"{name}: simulation aborted, no full trajectory to plot.")
                continue
            label = name if r['converged'] else f"{name} ({res['diagnosis']})"
            plot_trajectory(res, H_target=H_TARGET, V_target=V_target, title_label=label,
                            save_path=f"gravity_turn_{name}.png", show=False)
        plt.show()               # opens all the figures together, once everything is drawn
