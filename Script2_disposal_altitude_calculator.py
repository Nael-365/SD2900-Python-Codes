"""
Disposal orbit altitude calculator
-----------------------------------
Finds the disposal (circular) altitude that satisfies a target orbital
decay lifetime (e.g. 25-year post-mission disposal rule), using the
piecewise exponential-atmosphere decay model (Wiesel, Spaceflight
Dynamics, Eq. 3.51) with the US Standard Atmosphere 1976 table.

Method:
  1. The atmosphere is divided into altitude bands (as tabulated).
  2. Within each band the density is exponential (slide 25: logarithmic
     variation of the density): rho(H) = rho0 * exp(-(H - H_initial)/H_scale),
     with rho0 = density AT THE TOP of the band (H_initial; logarithmically
     interpolated from the table if the start altitude is not tabulated) and
     H_scale = tabulated scale height at the BOTTOM of the band (slide 25 remark:
     most accurate to use the scale height at the lower altitude).
  3. Decay time across a band is computed with Wiesel Eq. (3.51), written with
     rho0 = density at H_initial (see wiesel_dt for the exact form).
  4. Total decay time from any starting altitude down to a chosen
     "reentry complete" altitude is the SUM of the band contributions.
  5. A bisection search finds the highest disposal altitude whose total
     decay time is <= the target lifetime (with your chosen margin).

ASSUMPTIONS (state/justify these in your report):
  - Object assumed to decay on a near-circular orbit at each step
    (Wiesel Eq. 3.51 is for a single-altitude, not an eccentric orbit).
  - Atmosphere table only goes down to 150 km; below that this script
    treats 150 km as "decay complete" (reentry interface proxy). If you
    want a more accurate answer, extend the table down to ~100-120 km
    using a standard atmosphere reference.
  - C_d is a user input; a commonly cited placeholder for a tumbling,
    non-streamlined object is ~2.2, but you should justify your own
    value/source in the report.
  - Uses a static (non-solar-cycle-dependent) atmosphere table -> this
    is a nominal estimate, not a guarantee (real density varies with
    solar activity, as discussed).
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# US Standard Atmosphere 1976 table (from course slide, Table 4.2)
# (altitude_km, density_kg_m3, scale_height_km)
# ---------------------------------------------------------------------
ATMOSPHERE_TABLE = [
    (0,    1.225,     8.4345),
    (150,  2.076e-9,  23.380),
    (200,  2.541e-10, 36.183),
    (250,  6.073e-11, 44.924),
    (300,  1.916e-11, 51.193),
    (350,  7.014e-12, 55.832),
    (400,  2.803e-12, 59.678),
    (450,  1.184e-12, 63.644),
    (500,  5.215e-13, 68.785),
    (550,  2.384e-13, 76.427),
    (600,  1.137e-13, 88.244),
    (650,  5.712e-14, 105.992),
    (700,  3.070e-14, 130.630),
    (750,  1.788e-14, 161.074),
    (800,  1.136e-14, 193.862),
    (850,  7.824e-15, 224.737),
    (900,  5.759e-15, 250.894),
    (950,  4.453e-15, 271.754),
    (1000, 3.561e-15, 288.203),
]

MU_E = 3.986004418e14   # m^3/s^2
R_E = 6378137.0          # m
MU_KM = MU_E / 1e9        # km^3/s^2
R_E_KM = R_E / 1000.0      # km
G0 = 9.80665              # m/s^2, standard gravity for Isp definition
SECONDS_PER_YEAR = 365.25 * 24 * 3600

def calc_A(diameter):
    """Calculate the reference area from the maximum diameter in meters."""
    return math.pi * (diameter / 2)**2

# ---------------------------------------------------------------------
# Launcher database (upper stage / stage to be disposed of)
#   m_dry          : stage dry mass [kg]  -> B* = Cd*A/m, and final mass of the burn
#   Cd, A          : drag coefficient [-], cross-sectional area [m^2]
#   Isp            : deorbit-burn specific impulse [s]
# ---------------------------------------------------------------------
LAUNCHERS = {
    "Soyuz_2_1b_FregatM": dict(m_dry=1035.0, Cd=2.2, A=calc_A(3.8), Isp=333.2),
}


def density_at(h_km, table=ATMOSPHERE_TABLE):
    """Density [kg/m^3] at h_km: logarithmic interpolation between the tabulated
    points (slide 25: a logarithmic variation of the density is assumed within a
    band). Equals the table value at tabulated altitudes."""
    alts = [a for a, _, _ in table]
    ln_rho = [math.log(r) for _, r, _ in table]
    return math.exp(float(np.interp(h_km, alts, ln_rho)))


def scale_height_at(h_km, table=ATMOSPHERE_TABLE):
    """Tabulated scale height [m] at h_km (linear interpolation between points)."""
    alts = [a for a, _, _ in table]
    hs = [h for _, _, h in table]
    return float(np.interp(h_km, alts, hs)) * 1000.0


def band_params(h_top_km, h_bot_km, table=ATMOSPHERE_TABLE):
    """(rho0 [kg/m^3], H_scale [m]) for the band [h_bot, h_top]:
      rho0    = density at the TOP of the band (= H_initial of the band),
      H_scale = tabulated scale height at the BOTTOM of the band (slide 25)."""
    return density_at(h_top_km, table), scale_height_at(h_bot_km, table)


def wiesel_dt(h_initial_m, h_m, rho0, h_scale_m, b_star):
    """Wiesel Eq. (3.51): decay time [s] to go from h_initial down to h
    (both in meters), inside ONE band with exponential atmosphere.

    rho0 [kg/m^3] is the density AT h_initial (slide 24), H_scale [m] the band's
    scale height, B* = Cd*A/m [m^2/kg]. With rho(H) = rho0*exp(-(H-h_initial)/H_scale)
    and dh/dt = -B* rho sqrt(mu R_E):

        dt = H_scale / (B* rho0 sqrt(mu R_E)) * (1 - exp(-(h_initial - h)/H_scale))

    This is the printed form  -H_scale/(B* rho_ref sqrt(mu R_E)) [e^(h/H_scale) -
    e^(h_initial/H_scale)]  with rho_ref = rho0 * exp(h_initial/H_scale), i.e. the
    band's exponential extrapolated to h = 0. The printed form must NOT be fed the
    local density directly (that overestimates the decay time by exp(h_initial/H_scale))."""
    prefactor = h_scale_m / (b_star * rho0 * math.sqrt(MU_E * R_E))
    return prefactor * (1.0 - math.exp(-(h_initial_m - h_m) / h_scale_m))


def total_decay_time_years(h_start_km, h_end_km, b_star, table=ATMOSPHERE_TABLE):
    """Sum decay time band-by-band from h_start_km down to h_end_km."""
    if h_end_km < table[1][0]:
        raise ValueError(
            f"h_end_km = {h_end_km} km is below the lowest tabulated band "
            f"({table[1][0]} km): extend the atmosphere table first.")
    # altitudes present in the table, restricted to [h_end_km, h_start_km],
    # used as the band boundaries
    boundaries = [alt for alt, _, _ in table if h_end_km <= alt <= h_start_km]
    if not boundaries or boundaries[0] > h_end_km:
        boundaries = [h_end_km] + boundaries
    if boundaries[-1] < h_start_km:
        boundaries = boundaries + [h_start_km]
    boundaries = sorted(set(boundaries))

    total_seconds = 0.0
    # walk downward through consecutive boundary pairs
    for i in range(len(boundaries) - 1, 0, -1):
        h_top_km = boundaries[i]
        h_bot_km = boundaries[i - 1]
        rho0, h_scale_m = band_params(h_top_km, h_bot_km, table)
        dt = wiesel_dt(
            h_initial_m=h_top_km * 1000.0,
            h_m=h_bot_km * 1000.0,
            rho0=rho0,
            h_scale_m=h_scale_m,
            b_star=b_star,
        )
        total_seconds += dt

    return total_seconds / SECONDS_PER_YEAR


def find_disposal_altitude(
    operational_alt_km,
    reentry_alt_km,
    b_star,
    target_lifetime_years,
    tol_km=0.5,
):
    """Bisection search: find the HIGHEST disposal altitude (cheapest burn)
    whose total decay time to reentry_alt_km is <= target_lifetime_years."""
    lo, hi = reentry_alt_km, operational_alt_km

    # sanity check: even the lowest candidate must satisfy the constraint
    t_lo = total_decay_time_years(lo, reentry_alt_km, b_star)
    if t_lo > target_lifetime_years:
        raise ValueError(
            f"Even the reentry-interface altitude ({lo} km) does not satisfy "
            f"the target lifetime with this B*. Check inputs."
        )

    while hi - lo > tol_km:
        mid = 0.5 * (lo + hi)
        t_mid = total_decay_time_years(mid, reentry_alt_km, b_star)
        if t_mid <= target_lifetime_years:
            lo = mid   # still compliant -> can try higher (cheaper)
        else:
            hi = mid   # not compliant -> must go lower
    return lo, total_decay_time_years(lo, reentry_alt_km, b_star)


def hohmann_delta_v(r1_km, r2_km, mu_km=MU_KM):
    """Two-impulse Hohmann transfer between two circular, coplanar orbits
    of radii r1_km (start) and r2_km (target). Works for both raising
    (r2 > r1) and lowering (r2 < r1) - same equations, magnitudes only.
    Returns (dv1, dv2, dv_total [km/s], transfer_time [s])."""
    a_t = (r1_km + r2_km) / 2.0  # transfer ellipse semimajor axis

    v1_circ = math.sqrt(mu_km / r1_km)
    v2_circ = math.sqrt(mu_km / r2_km)

    # speed on the transfer ellipse at r1 and at r2 (vis-viva)
    v1_transfer = math.sqrt(mu_km * (2.0 / r1_km - 1.0 / a_t))
    v2_transfer = math.sqrt(mu_km * (2.0 / r2_km - 1.0 / a_t))

    # Signed delta-V: positive = prograde (speed up), negative = retrograde (slow down).
    # Burn 1: leave circular orbit at r1, enter transfer ellipse
    #   (negative/retrograde if r2 < r1, i.e. lowering the orbit)
    dv1 = v1_transfer - v1_circ

    # Burn 2: leave transfer ellipse at r2, circularize
    #   (negative/retrograde if lowering, since transfer speed > local circular speed)
    dv2 = v2_circ - v2_transfer

    # Propellant/budget purposes always need magnitudes (fuel doesn't care
    # about direction) - sum the magnitudes, not the signed values.
    dv_total = abs(dv1) + abs(dv2)
    transfer_time_s = math.pi * math.sqrt(a_t ** 3 / mu_km)

    return dv1, dv2, dv_total, transfer_time_s


def propellant_mass(delta_v_km_s, isp_s, m_final_kg, g0=G0):
    """Tsiolkovsky rocket equation solved for the propellant needed so that
    the stage ENDS the burn at m_final_kg (here: its dry mass).
    m0 = m_final * exp(dv / (Isp*g0))  ->  m_prop = m0 - m_final.
    Returns (propellant_mass_kg, initial_mass_kg)."""
    delta_v_m_s = delta_v_km_s * 1000.0
    m0_kg = m_final_kg * math.exp(delta_v_m_s / (isp_s * g0))
    return m0_kg - m_final_kg, m0_kg


def plot_orbits(r1_km, r2_km, name="", save_path="orbit_transfer.png"):
    """Plot the original circular orbit, the disposal circular orbit, and
    the Hohmann transfer ellipse between them, with Earth for scale.
    Departure (from r1) is placed at angle 0; arrival (at r2) at angle pi."""
    theta = np.linspace(0, 2 * np.pi, 400)

    # the two circular orbits
    x1, y1 = r1_km * np.cos(theta), r1_km * np.sin(theta)
    x2, y2 = r2_km * np.cos(theta), r2_km * np.sin(theta)

    # transfer ellipse: focus at origin, departure point (r1) at theta = 0
    a_t = (r1_km + r2_km) / 2.0
    e_t = abs(r1_km - r2_km) / (r1_km + r2_km)
    if r1_km >= r2_km:
        # departure at apogee (theta=0), arrival at perigee (theta=pi)
        r_transfer = a_t * (1 - e_t ** 2) / (1 - e_t * np.cos(theta))
    else:
        # departure at perigee (theta=0), arrival at apogee (theta=pi)
        r_transfer = a_t * (1 - e_t ** 2) / (1 + e_t * np.cos(theta))
    x_t, y_t = r_transfer * np.cos(theta), r_transfer * np.sin(theta)

    fig, ax = plt.subplots(figsize=(7, 7))

    earth = plt.Circle((0, 0), R_E_KM, color="tab:blue", alpha=0.3, label="Earth")
    ax.add_patch(earth)

    ax.plot(x1, y1, "--", color="tab:green",
            label=f"Original orbit ({r1_km - R_E_KM:.0f} km alt)")
    ax.plot(x2, y2, "--", color="tab:red",
            label=f"Disposal orbit ({r2_km - R_E_KM:.0f} km alt)")
    ax.plot(x_t, y_t, "-", color="tab:orange", label="Hohmann transfer orbit")

    # mark the two burn points
    ax.plot(r1_km, 0, "o", color="tab:green", markersize=8, label="Burn 1 (departure)")
    ax.plot(-r2_km, 0, "o", color="tab:red", markersize=8, label="Burn 2 (arrival)")

    ax.set_aspect("equal")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_title(f"Deorbit Hohmann Transfer - {name}" if name else "Deorbit Hohmann Transfer")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nOrbit plot saved to {save_path}")


def analyse_launcher(name, launcher, operational_alt_km, target_years,
                     margin_fraction, reentry_alt_km):
    """Run the full disposal analysis for one launcher from the database."""
    mass = launcher["m_dry"]   # stage dry mass [kg] (= final mass of the burn)
    area = launcher["A"]       # cross-sectional area [m^2]
    cd = launcher["Cd"]        # drag coefficient [-]
    isp = launcher["Isp"]      # deorbit-burn Isp [s]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    b_star = cd * area / mass  # m^2/kg
    effective_target_years = target_years * (1 - margin_fraction)

    print(f"B* = {b_star:.3e} m^2/kg")
    print(f"Effective target lifetime (with margin): {effective_target_years:.2f} years\n")

    # First, check decay time from the CURRENT operational altitude,
    # to show why disposal is/isn't needed at all.
    t_current = total_decay_time_years(operational_alt_km, reentry_alt_km, b_star)
    print(f"Decay time from current {operational_alt_km:.0f} km orbit: {t_current:,.1f} years")

    if t_current <= effective_target_years:
        print("-> Natural decay already satisfies the target. No deorbit burn required.")
        return

    disposal_alt, decay_years = find_disposal_altitude(
        operational_alt_km, reentry_alt_km, b_star, effective_target_years
    )

    print(f"\n-> Required disposal altitude: {disposal_alt:.1f} km")
    print(f"-> Estimated decay time from disposal altitude: {decay_years:.2f} years")
    print(f"-> Altitude drop required: {operational_alt_km - disposal_alt:.1f} km")

    # ---------------------------------------------------------------
    # Hohmann transfer delta-V and propellant mass for the deorbit burn
    # (circular r1 -> circular r2, same plane; no inclination/RAAN change
    # needed since only the orbit's SIZE changes, not its shape/orientation)
    # ---------------------------------------------------------------
    r1_km = R_E_KM + operational_alt_km
    r2_km = R_E_KM + disposal_alt

    dv1, dv2, dv_total, transfer_time_s = hohmann_delta_v(r1_km, r2_km)
    # stage ends the burn at its dry mass -> all remaining propellant is used
    m_prop, m0_burn = propellant_mass(dv_total, isp, mass)

    print("\n--- Hohmann deorbit transfer ---")
    print(f"Burn 1 (leave {operational_alt_km:.0f} km circular orbit): "
          f"dv1 = {dv1 * 1000:+.2f} m/s (negative = retrograde)")
    print(f"Burn 2 (circularize at {disposal_alt:.1f} km):          "
          f"dv2 = {dv2 * 1000:+.2f} m/s (negative = retrograde)")
    print(f"Total delta-V (magnitude, for propellant budget): "
          f"{dv_total * 1000:.2f} m/s ({dv_total:.4f} km/s)")
    print(f"Transfer time: {transfer_time_s / 60:.1f} minutes "
          f"({transfer_time_s / 3600:.2f} hours)")

    print("\n--- Propellant required (Tsiolkovsky rocket equation) ---")
    print(f"Isp = {isp:.0f} s, final mass (= dry mass) = {mass:.1f} kg")
    print(f"Propellant mass required: {m_prop:.2f} kg "
          f"({100 * m_prop / m0_burn:.1f}% of initial mass)")
    print(f"Initial mass at start of burn: {m0_burn:.2f} kg")

    plot_orbits(r1_km, r2_km, name=name, save_path=f"orbit_transfer_{name}.png")


def main():
    print("=== Disposal Altitude Calculator ===\n")

    # --- Inputs that are NOT in the launcher database (shared by all launchers) ---
    operational_alt_km = 851            #Current operational altitude [km]
    target_years = 25.0                 #Target decay lifetime [years]
    margin_fraction = 0.2               #Safety margin fraction, 0-1
    reentry_alt_km = 150.0              #Reentry-interface / 'decay complete' altitude [km]

    for name, launcher in LAUNCHERS.items():
        analyse_launcher(name, launcher, operational_alt_km, target_years,
                         margin_fraction, reentry_alt_km)

    plt.show()  # show all orbit plots at once, after the text results


if __name__ == "__main__":
    main()
