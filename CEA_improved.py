"""Merlin 1D / NASA CEA: three equivalence ratios and presentation-ready charts.

Scientific scope:
  - Quasi-1D ideal chemical-equilibrium rocket performance (NASA CEA).
  - Fixed chamber pressure + fixed geometric throat/exit areas across phi cases.
  - Rocket CEA's finite-area chamber option (FAC) is enabled.
  - Altitude curves assume attached flow and sea-level-standard atmosphere.
  - Stoichiometric O/F is estimated from NASA RP-1 empirical C:H=1:1.95.
  - Values below for engine geometry and reference performance are assumptions
    supplied with the original project code, NOT verified Merlin 1D dimensions.

Outputs are produced ONLY after a successful real CEA solve. No fictitious
or substitute thermochemical values are generated.

Verified reference for the CEA API:
https://nasa.github.io/cea/examples/rocket/example9.html (finite-area combustor)
https://nasa.github.io/cea/examples/rocket/example12.html (frozen at throat)
https://nasa.github.io/cea/interfaces/python_api.html (Isp units, fractions)
"""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cea
except ImportError as exc:
    raise SystemExit(
        "NASA CEA Python binding was not found. Install the NASA 'cea' package "
        "compatible with your Python environment, then rerun this script. "
        "No chemistry results can be generated without that solver."
    ) from exc

# ----------------------- editable design assumptions -----------------------
PC_BAR = 97.0
O_F_DESIGN = 2.36
G0 = 9.80665
T_FUEL_K, T_OX_K = 298.15, 90.19
ISP_VAC_REF_S = 311.0                  # original project's reference, not a model output
THRUST_VAC_REF_N = 914_000.0           # original project's reference, not a model output
THRUST_SL_PUBLISHED_N = 845_000.0      # SpaceX Falcon Payload User's Guide (2025)

RT, RE, RC = 0.1275, 0.5100, 0.3045  # m; original model assumptions
LC = 0.50
THETA_CONV = math.radians(30)
THETA_START = math.radians(25)
THETA_EXIT = math.radians(10)
RU, RD, RF = 1.5 * RT, 0.382 * RT, 0.10
L_NOZZLE = 1.14

# NASA CEA standard RP-1 empirical formula: C1 H1.95 (approximation).
# C H_1.95 + (1 + 1.95/4) O2 -> CO2 + (1.95/2) H2O
C_H_RATIO = 1.95
M_C, M_H, M_O = 12.011, 1.008, 15.999
OF_STOICH = (1 + C_H_RATIO / 4) * (2 * M_O) / (M_C + C_H_RATIO * M_H)

# Three regimes; rich is selected to INCLUDE the original engine design point.
PHI_CASES = [
    ("Lean", 0.85, "#1976d2"),
    ("Stoichiometric", 1.00, "#de7a00"),
    ("Rich / design", OF_STOICH / O_F_DESIGN, "#bd3477"),
]

BASE = Path(__file__).resolve().parent
OUT = BASE / "merlin_cea_results"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.titlesize': 14, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.facecolor': 'white', 'legend.frameon': False,
})

A_T = math.pi * RT**2
A_E = math.pi * RE**2
EPS_EXIT = A_E / A_T
AC_AT = (RC / RT)**2

# ----------------------------- nozzle geometry -----------------------------
def build_geometry():
    p = np.linspace(THETA_CONV, 0.0, 32)
    xu = -RU * np.sin(p)
    yu = RT + RU * (1 - np.cos(p))
    x_a, y_a = xu[0], yu[0]
    x_corner = x_a - (RC - y_a) / np.tan(THETA_CONV)
    x_f0 = x_corner - RF * np.tan(THETA_CONV / 2)
    p = np.linspace(0.0, THETA_CONV, 25)
    xf = x_f0 + RF * np.sin(p)
    yf = RC - RF + RF * np.cos(p)
    xc = np.linspace(x_f0 - LC, x_f0, 50)
    yc = np.full_like(xc, RC)
    xcon = np.linspace(xf[-1], x_a, 20)
    ycon = np.linspace(yf[-1], y_a, 20)

    p = np.linspace(0.0, THETA_START, 32)
    xd = RD * np.sin(p)
    yd = RT + RD * (1 - np.cos(p))
    x0, y0 = xd[-1], yd[-1]
    tn, te = np.tan(THETA_START), np.tan(THETA_EXIT)
    p1x = (RE - y0 + tn * x0 - te * L_NOZZLE) / (tn - te)
    p1y = y0 + tn * (p1x - x0)
    if not x0 < p1x < L_NOZZLE:
        raise ValueError('Bezier control point is outside the divergent nozzle.')
    tb = np.linspace(0, 1, 150)
    xb = (1 - tb)**2 * x0 + 2 * (1 - tb) * tb * p1x + tb**2 * L_NOZZLE
    yb = (1 - tb)**2 * y0 + 2 * (1 - tb) * tb * p1y + tb**2 * RE

    x = np.concatenate([xc, xf[1:], xcon[1:], xu[1:], xd[1:], xb[1:]])
    r = np.concatenate([yc, yf[1:], ycon[1:], yu[1:], yd[1:], yb[1:]])
    if np.any(np.diff(x) <= 0):
        raise ValueError('Nozzle geometry must be strictly monotonic in x.')
    throat_i = int(np.argmin(r))
    if not np.isclose(x[throat_i], 0, atol=1e-9):
        raise AssertionError('The geometric throat must lie at x = 0.')
    return x, r

X_FULL, R_FULL = build_geometry()
X_NOZ = X_FULL[X_FULL >= 0]
EPS_NOZ = (R_FULL[X_FULL >= 0] / RT)**2
EPS_CEA = np.linspace(1.001, EPS_EXIT, 60)
V_CHAMBER = np.trapezoid(np.pi * R_FULL[X_FULL <= 0]**2, X_FULL[X_FULL <= 0])
L_STAR = V_CHAMBER / A_T

# ------------------------------ NASA CEA run -------------------------------
REACT_NAMES = ["RP-1", "O2(L)"]
FUEL_W = np.array([1.0, 0.0])
OXID_W = np.array([0.0, 1.0])
T_REAC = np.array([T_FUEL_K, T_OX_K])


def run_cea(of_ratio: float, area_ratios: np.ndarray, freeze_at_throat: bool = False):
    """Finite-area combustor: injector, chamber, throat, supersonic points.

    For the FAC layout, the throat is station 3 (n_frz=3, 1-based CEA
    station index). Unlike the old code, this does not pass 'frozen=' to
    RocketSolver, because NASA CEA defines freezing via the solve n_frz option.
    """
    reac = cea.Mixture(REACT_NAMES)
    prod = cea.Mixture(REACT_NAMES, products_from_reactants=True)
    weights = np.ascontiguousarray(reac.of_ratio_to_weights(OXID_W, FUEL_W, of_ratio), dtype=np.float64)
    hc = float(reac.calc_property(cea.ENTHALPY, weights, T_REAC) / cea.R)
    solver = cea.RocketSolver(prod, reactants=reac)
    solution = cea.RocketSolution(solver)
    kw = dict(supar=np.asarray(area_ratios, dtype=np.float64), ac_at=float(AC_AT),
              iac=False, hc=hc)
    if freeze_at_throat:
        kw['n_frz'] = 3
    solver.solve(solution, weights, float(PC_BAR), **kw)
    if hasattr(solution, 'converged') and not solution.converged:
        raise RuntimeError(f'NASA CEA did not converge for O/F={of_ratio:.5f}')
    return solution


def arr(sol, name):
    val = np.asarray(getattr(sol, name), dtype=float).reshape(-1)
    if not np.all(np.isfinite(val)):
        raise ValueError(f'Non-finite values from NASA CEA for: {name}')
    return val


def station_indexes(sol, n_nozzle):
    # For this solve there are no extra pressure-ratio/subsonic stations.
    n = arr(sol, 'T').size
    i_throat = n - n_nozzle - 1
    i_chamber = i_throat - 1
    if i_chamber < 0:
        raise ValueError(f'Unexpected CEA station count {n} for {n_nozzle} exit points.')
    # The CEA Mach number at the throat should be ~1.0.
    m = arr(sol, 'Mach')
    if not (0.90 <= m[i_throat] <= 1.10):
        raise ValueError(f'CEA throat station ambiguous (index={i_throat}, M={m[i_throat]:.4f}).')
    return i_chamber, i_throat, n - 1


def noz_values(sol, name):
    v = arr(sol, name)
    return v[-len(EPS_CEA):]


def species_series(sol, species: str, key='mole_fractions'):
    """Access a species by canonical name, accounting for CEA's * markers."""
    dictionary = getattr(sol, key)
    canonical = lambda s: re.sub(r'^[* ]+', '', s).strip().upper()
    wanted = canonical(species)
    for name, val in dictionary.items():
        if canonical(name) == wanted:
            a = np.asarray(val, dtype=float).reshape(-1)
            if a.size != arr(sol, 'T').size:
                raise ValueError(f'Unexpected length of {key}[{name!r}].')
            return a
    return None


@dataclass
class Case:
    name: str
    phi: float
    color: str
    of: float
    sol: object
    c: int
    t: int
    e: int
    mdot: float
    isp_vac_s: float
    thrust_vac_n: float


def case_from_cea(name, phi, color):
    of = OF_STOICH / phi
    sol = run_cea(of, EPS_CEA)
    c, t, e = station_indexes(sol, len(EPS_CEA))
    cstar = arr(sol, 'c_star')[e]  # m/s, ideal CEA characteristic velocity
    if cstar <= 0:
        raise ValueError('CEA c* must be positive.')
    mdot = PC_BAR * 1e5 * A_T / cstar
    isp_vac_s = arr(sol, 'Isp_vacuum')[e] / G0  # NASA CEA returns m/s!
    return Case(name, phi, color, of, sol, c, t, e, mdot, isp_vac_s,
                mdot * G0 * isp_vac_s)


# ------------------------------- plotting ----------------------------------
def finish(fig, filename):
    fig.savefig(OUT / filename, dpi=240, bbox_inches='tight')
    plt.close(fig)


def plot_geometry():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(X_FULL, R_FULL, c='#252b36', lw=2.2)
    ax.plot(X_FULL, -R_FULL, c='#252b36', lw=2.2)
    ax.fill_between(X_FULL, -R_FULL, R_FULL, color='#e8eef4', alpha=0.85)
    ax.axvline(0, c='#d95e37', ls='--', lw=1.6, label='Throat x=0')
    ax.set(aspect='equal', xlabel='Axial position from throat, x (m)',
           ylabel='Radius (m)', title='Assumed geometry: chamber + converging/diverging nozzle')
    ax.grid(alpha=0.13)
    ax.legend(loc='upper left')
    fig.text(0.02, 0.01, 'Dimensions are assumptions from the input code; not confirmed proprietary Merlin dimensions.',
             fontsize=9, color='#555555')
    finish(fig, '01_geometry_assumed.png')


def plot_thermo(cases):
    for var, ylab, title, filename in [
        ('T', 'Static temperature (K)', 'Temperature: three equivalence ratios', '02_temperature_three_phi.png'),
        ('P', 'Static pressure (bar)', 'Pressure: three equivalence ratios', '03_pressure_three_phi.png'),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.7))
        for case in cases:
            yvals = np.r_[arr(case.sol, var)[case.t], noz_values(case.sol, var)]
            xvals = np.r_[0, np.interp(EPS_CEA, EPS_NOZ, X_NOZ)]
            ax.plot(xvals, yvals, lw=2.5, color=case.color,
                    label=f'{case.name}: phi={case.phi:.2f}, O/F={case.of:.2f}')
            ax.scatter([xvals[0], xvals[-1]], [yvals[0], yvals[-1]], color=case.color, s=26, zorder=4)
        ax.set(xlabel='Axial distance downstream of throat, x (m)', ylabel=ylab, title=title)
        ax.grid(alpha=0.25, ls=':')
        ax.legend(loc='best', fontsize=10)
        fig.text(0.02, 0.015, 'CEA finite-area chamber; nozzle stations interpolated from CEA area-ratio solutions.',
                 fontsize=9, color='#555555')
        finish(fig, filename)



def plot_station_thermo(cases):
    """Actual CEA stations rather than an assumed chamber-to-throat interpolation."""
    stations = ['Chamber', 'Throat', 'Exit']
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, key, label in [(axs[0], 'T', 'Static temperature (K)'),
                           (axs[1], 'P', 'Static pressure (bar)')]:
        for case in cases:
            v = arr(case.sol, key)
            y = [v[case.c], v[case.t], v[case.e]]
            ax.plot(stations, y, marker='o', lw=2.3, ms=7, color=case.color,
                    label=f'{case.name}, phi={case.phi:.2f}')
        ax.set(xlabel='CEA station', ylabel=label)
        ax.grid(alpha=.2, ls=':')
        ax.legend(fontsize=8.5)
    fig.suptitle('Chamber / throat / exit: comparison at three phi values', y=.99, fontsize=14)
    fig.tight_layout(rect=[0,.01,1,.96])
    finish(fig, '03b_chamber_throat_exit_three_phi.png')


def plot_species_numeric_table(design):
    """Readable actual numerical X at chamber/throat/exit for the chosen design case."""
    names = ['CO2', 'H2O', 'CO', 'H2', 'OH', 'H', 'O', 'O2']
    rows = []
    for sp in names:
        v = species_series(design.sol, sp)
        if v is None:
            rows.append(['not available']*4)
        else:
            rows.append([f'{v[design.c]:.4g}', f'{v[design.t]:.4g}',
                         f'{v[design.e]:.4g}', f'{(v[design.e]-v[design.c]):+.3g}'])
    fig, ax = plt.subplots(figsize=(10.6, 5.5))
    ax.axis('off')
    tb = ax.table(cellText=rows, rowLabels=names,
                  colLabels=['Chamber X', 'Throat X', 'Exit X', 'Exit minus chamber'],
                  cellLoc='center', loc='center', colWidths=[.2,.2,.2,.27])
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1,1.9)
    for (r,c), cell in tb.get_celld().items():
        cell.set_edgecolor('#d9e1e9')
        if r==0:
            cell.set_facecolor('#263a55');cell.set_text_props(color='white',weight='bold')
        elif r%2==0:
            cell.set_facecolor('#eff4f8')
    ax.set_title(f'Chemical species: absolute mole fractions (design phi={design.phi:.2f})', pad=18)
    fig.text(.1,.015, 'X = species mole fraction (dimensionless), not chemical production rate. Values are CEA equilibrium results.',
             fontsize=9, color='#444444')
    finish(fig,'08b_species_numeric_station_table.png')

def plot_frozen(design):
    try:
        fr = run_cea(design.of, EPS_CEA, freeze_at_throat=True)
        ic, it, ie = station_indexes(fr, len(EPS_CEA))
        x = np.r_[0, np.interp(EPS_CEA, EPS_NOZ, X_NOZ)]
        shift = np.r_[arr(design.sol, 'T')[design.t], noz_values(design.sol, 'T')]
        froz = np.r_[arr(fr, 'T')[it], noz_values(fr, 'T')]
        if np.allclose(shift, froz, atol=0.25, rtol=0):
            print('[Warning] Frozen and shifting curves are identical; check n_frz semantics and CEA solver.')
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(x, shift, color='#2471a3', lw=2.7, label='Shifting equilibrium (CEA)')
        ax.plot(x, froz, color='#d96b26', lw=2.7, ls='--', label='Frozen composition from throat (CEA)')
        ax.set(xlabel='Axial distance from throat (m)', ylabel='Static temperature (K)',
               title=f'Frozen vs shifting, design rich mixture (phi={design.phi:.2f})')
        ax.grid(alpha=0.25, ls=':'); ax.legend()
        finish(fig, '04_frozen_vs_shifting.png')
    except Exception as exc:
        print(f'[Warning] Genuine CEA frozen solve failed: {exc}\n'
              'The Frozen chart is OMITTED; no fake or isentropic substitute has been drawn.')


def plot_species(design):
    groups = [
        (['CO2', 'H2O', 'CO', 'H2'], '05_major_species_absolute.png',
         'Major species mole fractions (absolute values)', False),
        (['OH', 'H', 'O', 'O2'], '06_radicals_absolute_log.png',
         'Minor species / radicals (logarithmic mole fraction)', True),
    ]
    x = np.r_[0, np.interp(EPS_CEA, EPS_NOZ, X_NOZ)]
    for names, file, title, log in groups:
        fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for ax, s in zip(axs.flat, names):
            vals = species_series(design.sol, s)
            if vals is None:
                ax.text(.5, .5, f'{s}: unavailable in CEA products', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9)
            else:
                y = np.r_[vals[design.t], vals[-len(EPS_CEA):]]
                if log:
                    good = y > 0
                    ax.plot(x[good], y[good], color='#a44891', lw=2.2)
                    ax.set_yscale('log')
                    if np.any(good):
                        ax.scatter([x[good][0], x[good][-1]], [y[good][0], y[good][-1]],
                                   c='#a44891', s=20)
                else:
                    ax.plot(x, y, color='#16689a', lw=2.2)
                    ax.scatter([x[0], x[-1]], [y[0], y[-1]], c='#16689a', s=20)
                ax.set_title(f'{s}: chamber={vals[design.c]:.3g}; exit={vals[design.e]:.3g}', fontsize=11)
                ax.set_ylabel('Mole fraction X' + (' (log)' if log else ''))
            ax.grid(alpha=.2, ls=':')
            ax.set_xlabel('x (m)')
        fig.suptitle(title + ' — rich/design condition', fontsize=15, y=.99)
        fig.tight_layout(rect=[0, .01, 1, .96])
        finish(fig, file)

    fig, axs = plt.subplots(1, 2, figsize=(12.3, 5.4))
    for ax, species_names, title, color in [
        (axs[0], ['CO2', 'H2O', 'CO', 'H2'], 'Major species: net change from chamber', '#165c98'),
        (axs[1], ['OH', 'H', 'O', 'O2'], 'Minor species: net change from chamber', '#9a3d80'),
    ]:
        for s in species_names:
            vals = species_series(design.sol, s)
            if vals is None:
                continue
            ax.plot(x, np.r_[vals[design.t], vals[-len(EPS_CEA):]] - vals[design.c],
                    lw=2, label=s)
        ax.axhline(0, color='#333333', lw=.8)
        ax.set(xlabel='Nozzle axial position x (m)', ylabel='Delta mole fraction from chamber', title=title)
        ax.grid(alpha=.2, ls=':'); ax.legend(ncol=2)
    fig.tight_layout()
    finish(fig, '07_species_net_changes_separated.png')

    species = ['CO2', 'H2O', 'CO', 'H2']
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    xpos = np.arange(len(species))
    width = .24
    for shift, index, desc, color in [(-width, design.c, 'Chamber', '#426b9a'),
                                      (0, design.t, 'Throat', '#e3a046'),
                                      (width, design.e, 'Exit', '#a54c86')]:
        vals = []
        for s in species:
            data = species_series(design.sol, s)
            vals.append(float(data[index]) if data is not None else np.nan)
        ax.bar(xpos + shift, vals, width, label=desc, color=color)
    ax.set(xticks=xpos, xticklabels=species, ylabel='Mole fraction (dimensionless)',
           title='Major species: absolute mole fraction at three engine stations')
    ax.grid(axis='y', alpha=.2); ax.legend()
    finish(fig, '08_species_chamber_throat_exit_bars.png')


def plot_parametric():
    # Rich conditions to slightly lean: both stoichiometric and original O/F=2.36 visible.
    of_grid = np.unique(np.r_[np.linspace(2.0, 4.25, 31), O_F_DESIGN, OF_STOICH])
    records = []
    for of in of_grid:
        try:
            s = run_cea(float(of), np.array([EPS_EXIT], dtype=float))
            c, t, e = station_indexes(s, 1)
            records.append(dict(of=float(of), phi=float(OF_STOICH / of),
                                isp_vac_s=float(arr(s, 'Isp_vacuum')[e] / G0),
                                chamber_T_K=float(arr(s, 'T')[c])))
        except Exception as exc:
            print(f'[Warning] Parametric CEA solve skipped for O/F={of:.4g}: {exc}')
    if not records:
        raise RuntimeError('No valid CEA results for parametric study.')
    write_csv('parametric_OF.csv', records)
    for key, ylabel, title, filename, col in [
        ('isp_vac_s', 'Ideal vacuum specific impulse (s)', 'Isp vs O/F',
         '09_isp_vs_of.png', '#3874ad'),
        ('chamber_T_K', 'Chamber static temperature (K)', 'Chamber temperature vs O/F',
         '10_chamber_temperature_vs_of.png', '#d56a33'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.plot([r['of'] for r in records], [r[key] for r in records], '.-', c=col, lw=2.4)
        ax.axvline(O_F_DESIGN, ls='--', c='#ab437e', lw=1.8, label=f'Design O/F={O_F_DESIGN:.2f}')
        ax.axvline(OF_STOICH, ls=':', c='#222222', lw=1.8, label=f'Stoichiometric O/F={OF_STOICH:.2f}')
        ax.set(xlabel='Oxidizer-to-fuel mass ratio O/F', ylabel=ylabel, title=title)
        ax.grid(alpha=.2); ax.legend()
        finish(fig, filename)


def atmosphere_pressure_pa(h):
    # Simplified ISA 1976 up to 50 km, dry static atmosphere.
    R = 287.053
    if h < 11000:
        return 101325 * ((288.15 - .0065*h)/288.15)**(G0/(R*.0065))
    p11 = 22632.06
    if h < 20000:
        return p11 * math.exp(-G0*(h-11000)/(R*216.65))
    p20 = p11 * math.exp(-G0*9000/(R*216.65))
    if h < 32000:
        return p20 * ((216.65+.001*(h-20000))/216.65)**(-G0/(R*.001))
    p32 = p20 * (228.65/216.65)**(-G0/(R*.001))
    if h < 47000:
        return p32 * ((228.65+.0028*(h-32000))/228.65)**(-G0/(R*.0028))
    p47 = p32 * (270.65/228.65)**(-G0/(R*.0028))
    return p47 * math.exp(-G0*(h-47000)/(R*270.65))


def plot_altitude(cases):
    altitude_km = np.linspace(0, 50, 251)
    p_ambient = np.array([atmosphere_pressure_pa(h*1000) for h in altitude_km])
    records = []
    for type_, ylabel, title, fname in [
        ('thrust', 'Ideal modeled thrust (kN)', 'Ideal thrust vs altitude: three equivalence ratios',
         '11_thrust_vs_altitude_three_phi.png'),
        ('isp', 'Ideal specific impulse (s)', 'Specific impulse vs altitude: three equivalence ratios',
         '12_isp_vs_altitude_three_phi.png'),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5.9))
        for case in cases:
            thrust = case.thrust_vac_n - A_E*p_ambient
            isp = thrust / (case.mdot * G0)
            y = thrust/1000 if type_=='thrust' else isp
            ax.plot(altitude_km, y, lw=2.6, c=case.color,
                    label=f'{case.name}, phi={case.phi:.2f}, O/F={case.of:.2f}')
            if type_ == 'thrust':
                ax.scatter([0, 50], [y[0], y[-1]], color=case.color, s=24, zorder=4)
        ax.set(xlabel='Altitude above sea level (km)', ylabel=ylabel, title=title)
        ax.grid(alpha=.2, ls=':'); ax.legend(fontsize=9)
        if type_ == 'thrust':
            ax.scatter([0], [THRUST_SL_PUBLISHED_N/1000], s=75, c='black', marker='x', lw=2,
                       label='SpaceX published sea-level thrust (845 kN)')
            ax.legend(fontsize=9)
        fig.text(.02, .01, 'Assumptions: fixed Pc and areas, quasi-1D ideal performance, attached flow (no nozzle separation).',
                 fontsize=9, color='#555555')
        finish(fig, fname)
    for case in cases:
        t = case.thrust_vac_n - A_E*p_ambient
        for h, pa, thrust in zip(altitude_km, p_ambient, t):
            records.append(dict(case=case.name, phi=case.phi, of=case.of, altitude_km=float(h),
                                ambient_pressure_pa=float(pa), modeled_thrust_kN=float(thrust/1000),
                                modeled_isp_s=float(thrust/(case.mdot*G0))))
    write_csv('altitude_three_phi.csv', records)


def write_csv(filename, rows):
    if not rows: return
    with (OUT / filename).open('w', newline='', encoding='utf-8-sig') as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def export_species(cases):
    entries = []
    for case in cases:
        for species in ['CO2', 'H2O', 'CO', 'H2', 'OH', 'H', 'O', 'O2']:
            x = species_series(case.sol, species, 'mole_fractions')
            y = species_series(case.sol, species, 'mass_fractions')
            if x is None and y is None:
                print(f'[Notice] {species} not present in CEA product data, case={case.name}.')
                continue
            for st, ix in [('chamber', case.c), ('throat', case.t), ('exit', case.e)]:
                entries.append(dict(case=case.name, phi=float(case.phi), of=float(case.of),
                                    species=species, station=st,
                                    mole_fraction=float(x[ix]) if x is not None else '',
                                    mass_fraction=float(y[ix]) if y is not None else '',
                                    species_mass_flow_kg_s=float(y[ix]*case.mdot) if y is not None else ''))
    write_csv('chemical_species_stations.csv', entries)

    # Net change of species mass flux from chamber to exit is not a reaction-rate measurement.
    net = []
    for case in cases:
        for species in ['CO2', 'H2O', 'CO', 'H2', 'OH', 'H', 'O', 'O2']:
            y = species_series(case.sol, species, 'mass_fractions')
            if y is not None:
                net.append(dict(case=case.name, phi=float(case.phi), of=float(case.of),
                                species=species,
                                chamber_massflow_kg_s=float(case.mdot*y[case.c]),
                                exit_massflow_kg_s=float(case.mdot*y[case.e]),
                                net_change_kg_s=float(case.mdot*(y[case.e]-y[case.c]))))
    write_csv('chemical_species_net_massflow.csv', net)


def export_case_summary(cases):
    recs = []
    for case in cases:
        T = arr(case.sol, 'T'); P = arr(case.sol, 'P')
        pexit = P[case.e]
        thrust_sl = case.thrust_vac_n - A_E*101325.0
        recs.append(dict(case=case.name, phi=float(case.phi), of=float(case.of),
                         chamber_temperature_K=float(T[case.c]), throat_temperature_K=float(T[case.t]),
                         exit_temperature_K=float(T[case.e]),
                         chamber_pressure_bar=float(P[case.c]), throat_pressure_bar=float(P[case.t]),
                         exit_pressure_bar=float(pexit), cstar_m_s=float(arr(case.sol,'c_star')[case.e]),
                         mdot_ideal_kg_s=float(case.mdot), isp_vac_ideal_s=float(case.isp_vac_s),
                         thrust_vac_ideal_kN=float(case.thrust_vac_n/1000),
                         thrust_sea_ideal_kN=float(thrust_sl/1000),
                         sea_level_Pexit_over_Pambient=float(pexit*1e5/101325.0)))
    write_csv('case_summary.csv', recs)
    for rec in recs:
        print(f"{rec['case']:<16} phi={rec['phi']:.3f} O/F={rec['of']:.3f} "
              f"Tc={rec['chamber_temperature_K']:.0f} K, "
              f"Isp_vac={rec['isp_vac_ideal_s']:.1f} s, "
              f"F_vac={rec['thrust_vac_ideal_kN']:.1f} kN, "
              f"F_SL={rec['thrust_sea_ideal_kN']:.1f} kN")


def main():
    print('NASA CEA Merlin 1D classroom model — not manufacturer geometry')
    print(f'Assumed geometry: Ae/At={EPS_EXIT:.3f}, Ac/At={AC_AT:.3f}, L*={L_STAR:.3f} m')
    print(f'RP-1 empirical stoichiometry: OF_st={OF_STOICH:.4f} (C1 H{C_H_RATIO:.2f})')
    print(f'Original design O/F={O_F_DESIGN:.2f} corresponds to phi={OF_STOICH/O_F_DESIGN:.3f}')
    cases = [case_from_cea(*definition) for definition in PHI_CASES]
    plot_geometry()
    plot_thermo(cases)
    plot_station_thermo(cases)
    plot_frozen(cases[2])
    plot_species(cases[2])
    plot_species_numeric_table(cases[2])
    plot_parametric()
    plot_altitude(cases)
    export_species(cases)
    export_case_summary(cases)
    with (OUT / 'README_results.txt').open('w', encoding='utf-8') as f:
        f.write('Results generated using the installed NASA CEA backend.\n'
                'Three phi values: lean 0.85, stoichiometric 1.0, rich = OF_st/2.36.\n'
                'RP-1 empirical formula C1H1.95, so OF_st is an approximation, not a measured composition.\n'
                'Model assumes fixed pressure and nozzle dimensions; mdot from Pc*At/cstar (CEA).\n'
                'Thrust = mdot*g0*Isp_vac - P_ambient*Ae.\n'
                'No flow separation, losses, throttling, transient operation or off-design shock modelling.\n'
                'CEA finite area chamber: iac=False, ac_at=Ac/At; frozen throat n_frz=3.\n'
                'Species plots: X is mole fraction (dimensionless), NOT kinetic production rate.\n'
                'Species mass flow = mdot*Y_species, and net change is outlet minus chamber mass flux.\n'
                'Actual Merlin geometrical parameters in source code were assumed rather than sourced.\n'
                'The black sea-level 845 kN marker is SpaceX published reference, not a modeled datum.\n'
                'SpaceX: https://www.spacex.com/assets/media/falcon-users-guide-2025-05-09.pdf\n'
                'NASA CEA API: https://nasa.github.io/cea/interfaces/python_api.html\n'
                'NASA CEA FAC: https://nasa.github.io/cea/examples/rocket/example9.html\n'
                'NASA CEA Frozen: https://nasa.github.io/cea/examples/rocket/example12.html\n')
    print(f'\nAll valid CEA-derived images and CSV tables saved to: {OUT}')


if __name__ == '__main__':
    main()
