"""Quantitative data-vs-model comparison for Rogers (2025), Section 4.

Evaluates the observed young-planet sample against the gas-dwarf vs water-world
occurrence predictions in two sample-matched age bins (<=100 Myr and 100-700 Myr),
and tabulates the project's 'Our Work' targets relative to the radius valley.

Used by data_vs_model_results.ipynb. All model code mirrors
rogers2025_gasdwarf_waterworld.ipynb (CGS units).
"""
import os
import numpy as np
import pandas as pd
import astropy.units as u
from scipy import integrate, optimize
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator

# --- constants (CGS) ---
G = 6.674e-8; kB = 1.381e-16; mH = 1.6726e-24; sigma = 5.670e-5
Mearth = 5.972e27; Rearth = 6.371e8; Myr = 3.156e13; Msun = 1.989e33; Lsun = 3.828e33
_GLx, _GLw = np.polynomial.legendre.leggauss(80)

_HERE = os.path.dirname(os.path.abspath(__file__))
NEXSCI_CSV = os.path.join(_HERE, '..', 'data', 'nexsci_ps.csv')
HOSTS_CSV = ('/ut2/jerome/github/research/project/TOI-6715/data/'
             'young_validated_planets_TOIs - validated_hosts.csv')

# sample-matched age bins (Myr): label -> (epoch_key, lo, hi)
AGE_BINS = [('<=100 Myr', 'le100', 0, 100), ('100-1000 Myr', 'mid', 100, 1000)]

# 'Our Work' targets: name, period [d], radius [R_earth], radius err [R_earth]
OUR_WORK = [
    # dict(name='TIC*1445 b', P=3.272052, R=5.7, Rerr=0.7, age=(10,30)),
    dict(name='TOI-6715 b', P=2.86, R=4.5, Rerr=0.3, age=(30,50)),
    # dict(name='TOI-5973 b', P=3.16, R=8.5, Rerr=0.1, age=(10,50)),
    # dict(name='TOI-6940 b', P=0.3052608, R=1.4, Rerr=0.17),
    # dict(name='TOI-5169 b', P=8.0387972, R=2.25604, Rerr=0.174709),
    # dict(name='TOI-5734 b',   P=6.1841876, R=2.1, Rerr=0.12),
]


def core_radius(Mc_e, kind):
    return Rearth * (1.30 if kind == 'water' else 1.0) * Mc_e ** 0.27


def vaneylen_valley_radius(P):
    """Radius-valley location [R_earth] at period P [d], Van Eylen et al. 2018."""
    return 10 ** (-0.09 * np.log10(P) + 0.37)


def _radius_track(Mc_e, Teq, mu, gamma, kappa, kind, mu_c, X, ages, gamma_c=4 / 3):
    Mc = Mc_e * Mearth; Rc = core_radius(Mc_e, kind); Menv = X * Mc
    cs2 = kB * Teq / (mu * mH); nab = (gamma - 1) / gamma
    RBp = G * Mc / cs2; RB = G * Mc / (2 * cs2); p = 1 / (gamma - 1)

    def props(Rr):
        r = 0.5 * (Rr - Rc) * _GLx + 0.5 * (Rr + Rc); jac = 0.5 * (Rr - Rc)
        b = np.maximum(1 + nab * (RBp / Rr) * (Rr / r - 1), 1e-30); f = b ** p
        A = np.sum(_GLw * 4 * np.pi * r * r * f) * jac; rho = Menv / A
        e = kB * Teq * b / (mu * mH * (gamma - 1)) - G * Mc / r
        B = np.sum(_GLw * 4 * np.pi * r * r * f * e) * jac
        Tc = Teq * (1 + nab * (RBp / Rr) * (Rr / Rc - 1))
        Ep = rho * B + Mc * kB * Tc / (mu_c * mH * (gamma_c - 1))
        L = 64 * np.pi * nab * G * Mc * sigma * Teq ** 4 / (3 * cs2 * kappa * rho)
        return Ep, rho, L

    def tau(Rr):
        Ep, rho, L = props(Rr); return abs(Ep) / L / Myr

    lo, hi = Rc * 1.02, RB * 0.999
    try:
        if tau(lo) < ages.min():
            return np.full_like(ages, Rc / Rearth)
        R0 = optimize.brentq(lambda R: tau(R) - ages.min(), lo, hi)
    except Exception:
        return np.full_like(ages, Rc / Rearth)

    def dEpdR(Rr):
        h = Rr * 1e-4; return (props(Rr + h)[0] - props(Rr - h)[0]) / (2 * h)

    sol = integrate.solve_ivp(lambda t, y: [-props(y[0])[2] * Myr / dEpdR(y[0])],
                              [ages.min(), ages.max()], [R0], t_eval=ages,
                              rtol=1e-4, atol=1e4, max_step=500)

    def Rtr(Rr, rho):
        def tc(r):
            rr = rho * np.exp((G * Mc / cs2) * (1 / r - 1 / Rr)); H = cs2 * r * r / (G * Mc)
            return kappa * rr * np.sqrt(2 * np.pi * r * H)
        if tc(Rr) < 1:
            return Rr
        try:
            return optimize.brentq(lambda r: tc(r) - 1, Rr, 10 * Rr)
        except Exception:
            return Rr

    out = np.array([Rtr(Rr, props(Rr)[1]) / Rearth for Rr in sol.y[0]])
    if len(out) < len(ages):
        out = np.concatenate([out, np.full(len(ages) - len(out), out[-1])])
    return np.maximum(out, Rc / Rearth)


_Mc = np.array([1, 2, 3, 5, 8, 13, 20.])
_X = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
_Te = np.array([300, 550, 800, 1100.])
_ag = np.logspace(0, 4, 32)


def _build_interp(mu, gamma, kind, mu_c, kappa=0.1):
    tab = np.zeros((len(_Mc), len(_X), len(_Te), len(_ag)))
    for i, M in enumerate(_Mc):
        for j, X in enumerate(_X):
            for k, T in enumerate(_Te):
                tab[i, j, k] = np.log10(_radius_track(M, T, mu, gamma, kappa, kind, mu_c, X, _ag))
    it = RegularGridInterpolator((np.log10(_Mc), np.log10(_X), _Te, np.log10(_ag)),
                                 tab, bounds_error=False, fill_value=None)
    return lambda M, X, T, a: 10 ** float(it([np.log10(np.clip(M, 1, 20)),
                                              np.log10(np.clip(X, 1e-4, .3)),
                                              np.clip(T, 300, 1100), np.log10(np.clip(a, 1, 1e4))]))


def _teq_orbit(P, Ms):
    a = (G * Ms * (P * 86400) ** 2 / (4 * np.pi ** 2)) ** (1 / 3)
    Lb = Lsun * (Ms / Msun) ** 4
    return (Lb / (16 * sigma * np.pi * a * a)) ** 0.25, a


def _lxuv(t, Ms):
    Ls = 10 ** -3.5 * Lsun * (Ms / Msun) ** 4
    return Ls if t < 100 else Ls * (t / 100) ** -1.5


def _draw_period(n, P0, k1, k2, rng):
    P = np.logspace(0, 2, 500); w = 1 / ((P / P0) ** (-k1) + (P / P0) ** (-k2)); w /= w.sum()
    return rng.choice(P, n, p=w)


def _tobs(epoch, td, rng):
    return rng.uniform(td, 100) if epoch == 'le100' else rng.uniform(100, 700)


def load_planets():
    """Matched young-planet sample with host age midpoint (Myr)."""
    df = pd.read_csv(NEXSCI_CSV, low_memory=False)
    df['pl_rade'] = df['pl_radj'] * u.Rjup.to(u.Rearth)
    idx = ((df.pl_orbper > 0) & (df.pl_rade > 0) & (df.st_mass > 0) &
           (df.discoverymethod.isin(['Transit', 'Radial Velocity'])) & (df['default_flag'] == 1))
    df = df[idx].copy()
    df['hostname'] = df['hostname'].str.lower().str.replace(' ', '')
    h = (pd.read_csv(HOSTS_CSV, comment='#', skiprows=[1]).dropna(subset=['Host'])
         .iloc[:68].reset_index(drop=True))
    h['hostname'] = h.Host.str.lower().str.replace(' ', '')
    h['Alo'] = h.Age.apply(lambda x: float(str(x).split('-')[0]))
    h['Ahi'] = h.Age.apply(lambda x: float(str(x).split('-')[1]))
    dp = df[df.hostname.isin(h.hostname)].copy()
    dp['amid'] = dp.hostname.map(dict(zip(h.hostname, (h.Alo + h.Ahi) / 2)))
    dp['alo'] = dp.hostname.map(dict(zip(h.hostname, h.Alo)))
    dp['ahi'] = dp.hostname.map(dict(zip(h.hostname, h.Ahi)))
    # symmetric radius uncertainty [R_earth] from the Jupiter-radius errors
    re1 = df.get('pl_radjerr1'); re2 = df.get('pl_radjerr2')
    if re1 is not None and re2 is not None:
        dp['pl_rade_err'] = (0.5 * (df['pl_radjerr1'].abs() + df['pl_radjerr2'].abs())
                             * u.Rjup.to(u.Rearth)).reindex(dp.index)
    # equilibrium temperature: prefer catalog pl_eqt, else compute from stellar
    # Teff/radius and a (a from Kepler's 3rd law: period+st_mass, both complete)
    Rsun = 6.957e10
    a_cm = (G * dp.st_mass.values * Msun * (dp.pl_orbper.values * 86400.0) ** 2
            / (4 * np.pi ** 2)) ** (1 / 3)
    teq_calc = dp.st_teff.values * np.sqrt(dp.st_rad.values * Rsun / (2 * a_cm))  # albedo=0
    dp['teq_eff'] = dp['pl_eqt'].where(dp['pl_eqt'].notna(), other=teq_calc)
    return dp


def synthesize(seed=42, N=2000):
    """Build the radius interpolators and synthesise gas-dwarf / water-world
    populations for both age bins. Returns pops[(model, epoch)] = (P, R)."""
    Rgd = _build_interp(2.35, 7 / 5, 'earth', 60.)
    Rww = _build_interp(18., 4 / 3, 'water', 27.)
    rng = np.random.default_rng(seed)

    def gd_pop(epoch):
        M = np.clip(np.exp(rng.normal(np.log(4), .35, N)), 1, 20)
        X0 = np.clip(np.exp(rng.normal(np.log(.02), .5, N)), 1e-3, .3)
        P = _draw_period(N, 5.5, 2.3, 0, rng)
        Ms = np.clip(rng.normal(1, .3, N), .5, 1.6) * Msun
        R = np.empty(N)
        for i in range(N):
            td = rng.uniform(3, 10); to = _tobs(epoch, td, rng)
            Mc = M[i] * Mearth; Teq, a = _teq_orbit(P[i], Ms[i]); Me = X0[i] * Mc
            ts = np.logspace(np.log10(td), np.log10(to), 50); stripped = False
            for n in range(len(ts) - 1):
                if Me / Mc < 1e-4:
                    R[i] = core_radius(M[i], 'earth') / Rearth; stripped = True; break
                Rp = Rgd(M[i], Me / Mc, Teq, ts[n]) * Rearth
                Me -= .15 * (_lxuv(ts[n], Ms[i]) / (4 * np.pi * a * a)) * (np.pi * Rp ** 3 / (G * Mc)) * (ts[n + 1] - ts[n]) * Myr
            if not stripped:
                R[i] = core_radius(M[i], 'earth') / Rearth if Me / Mc < 1e-4 else Rgd(M[i], Me / Mc, Teq, to)
        return P, R

    def ww_pop(epoch):
        n_se = int(.35 * N); n_sn = N - n_se
        Ms = np.clip(rng.normal(1, .3, N), .5, 1.6) * Msun
        Mse = np.clip(10 ** rng.normal(np.log10(3), .35, n_se), .5, 20)
        Pse = _draw_period(n_se, 4.5, 2, -.5, rng); Rse = core_radius(Mse, 'earth') / Rearth
        Msn = np.clip(10 ** rng.normal(np.log10(10), .30, n_sn), 1, 20)
        Psn = _draw_period(n_sn, 8.5, 2.5, .1, rng)
        Xsn = 10 ** rng.uniform(np.log10(1e-5), np.log10(.1), n_sn); Rsn = np.empty(n_sn)
        for i in range(n_sn):
            td = rng.uniform(3, 10); Te, _ = _teq_orbit(Psn[i], Ms[i + n_se])
            Rsn[i] = Rww(Msn[i], Xsn[i], Te, _tobs(epoch, td, rng))
        return np.concatenate([Pse, Psn]), np.concatenate([Rse, Rsn])

    pops = {}
    for _, epoch, _, _ in AGE_BINS:
        pops[('gd', epoch)] = gd_pop(epoch)
        pops[('ww', epoch)] = ww_pop(epoch)
    return pops


def compare(pops, dp):
    """Per-bin data-vs-model summary. Returns a DataFrame; favored hypothesis is
    by KDE likelihood ratio in log(P)-log(R) over planets in the panel window."""
    rows = []
    for label, epoch, lo, hi in AGE_BINS:
        Pg, Rg = pops[('gd', epoch)]; Pw, Rw = pops[('ww', epoch)]
        sub = dp[(dp.amid >= lo) & (dp.amid < hi) & (dp.pl_orbper < 100) & (dp.pl_rade < 10)]
        Pd, Rd = sub.pl_orbper.values, sub.pl_rade.values
        kg = gaussian_kde(np.vstack([np.log10(Pg), np.log10(Rg)]))
        kw = gaussian_kde(np.vstack([np.log10(Pw), np.log10(Rw)]))
        pts = np.vstack([np.log10(Pd), np.log10(Rd)])
        dlnL = float(np.sum(np.log(np.clip(kg(pts), 1e-300, None)) -
                            np.log(np.clip(kw(pts), 1e-300, None))))
        rows.append(dict(bin=label, N_data=len(Pd),
                         med_R_data=np.median(Rd), med_R_gd=np.median(Rg), med_R_ww=np.median(Rw),
                         frac_gt3_data=np.mean(Rd > 3), frac_gt3_gd=np.mean(Rg > 3),
                         frac_gt3_ww=np.mean(Rw > 3), dlnL_gd_minus_ww=dlnL,
                         favored='gas dwarf' if dlnL > 0 else 'water world'))
    return pd.DataFrame(rows).round(2)


def targets_table():
    """'Our Work' targets with radius-valley context and plot-window flags."""
    rows = []
    for t in OUR_WORK:
        Rv = vaneylen_valley_radius(t['P'])
        rows.append(dict(name=t['name'], P_day=round(t['P'], 3), R_earth=round(t['R'], 2),
                         valley_R=round(Rv, 2), R_over_valley=round(t['R'] / Rv, 2),
                         in_panel=bool(1 <= t['P'] <= 100 and 1 <= t['R'] <= 10)))
    return pd.DataFrame(rows)
