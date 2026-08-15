#!/usr/bin/env python
"""Plot transit radius as a function of age: Rogers (2025) gas-dwarf vs
water-world contraction tracks, with the observed young-planet sample overlaid
at their host ages.

Model tracks come from the radius interpolators in data_vs_model.py (fixed
Teq, envelope fraction X), swept over age. Data points are the matched young
planets (radius vs host-age midpoint, with age and radius error bars).

Usage:
    python plot_radius_vs_age_rogers25_model.py
    python plot_radius_vs_age_rogers25_model.py --teq 800 --masses 4 6 --gd_x 0.02 --ww_x 0.05 --rmax 9
"""
import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

import data_vs_model as dvm


def load_rcparams(path='../rcParams.txt'):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) != 2 or row[0] == 'name':
                continue
            name, value = row
            for cast in (int, float):
                try:
                    value = cast(value); break
                except ValueError:
                    pass
            try:
                plt.rcParams[name] = value
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    # shared: Teq is an external boundary condition (same star+orbit), masses are
    # the controlled comparison variable -- both physically identical across scenarios
    ap.add_argument('--teq', type=float, default=800.0, help='equilibrium temperature [K] (shared)')
    ap.add_argument('--masses', type=float, nargs='+', default=[2, 5],
                    help='core masses [M_earth] for the tracks (shared: same-core comparison)')
    # gas-dwarf (orange) physics  -- solar H2/He envelope on a rocky core
    ap.add_argument('--gd_x', type=float, default=0.01,
                    help='gas dwarf H2/He envelope mass fraction Menv/Mc (~1-2%)')
    ap.add_argument('--gd_kappa', type=float, default=0.1,
                    help='gas dwarf envelope opacity [cm^2/g] (H/He, fairly transparent)')
    ap.add_argument('--gd_mu', type=float, default=2.35,
                    help='gas dwarf envelope mean molecular weight [amu] (solar H2/He)')
    ap.add_argument('--gd_gamma', type=float, default=7 / 5,
                    help='gas dwarf envelope adiabatic index (diatomic H2)')
    ap.add_argument('--gd_mu_c', type=float, default=60.0, help='gas dwarf core mean molecular weight [amu]')
    ap.add_argument('--gd_kind', choices=['earth', 'water'], default='earth',
                    help='gas dwarf core composition / mass-radius relation')
    # water-world (blue) physics  -- steam envelope on a water-rich core
    ap.add_argument('--ww_x', type=float, default=0.05,
                    help='water world steam envelope mass fraction Menv/Mc')
    ap.add_argument('--ww_kappa', type=float, default=1.0,
                    help='water world envelope opacity [cm^2/g] (steam is far more opaque than H/He)')
    ap.add_argument('--ww_mu', type=float, default=18.0,
                    help='water world envelope mean molecular weight [amu] (H2O steam)')
    ap.add_argument('--ww_gamma', type=float, default=4 / 3,
                    help='water world envelope adiabatic index (triatomic steam)')
    ap.add_argument('--ww_mu_c', type=float, default=27.0, help='water world core mean molecular weight [amu]')
    ap.add_argument('--ww_kind', choices=['earth', 'water'], default='water',
                    help='water world core composition / mass-radius relation')
    ap.add_argument('--color', choices=['teq', 'period'], default='teq',
                    help='physical quantity to colour the data by (Teq controls radius; '
                         'period is the 100%%-complete proxy)')
    ap.add_argument('--rmax', type=float, default=9.0, help='radius axis upper limit')
    ap.add_argument('--logr', action='store_true', help='log-scale the radius axis')
    ap.add_argument('--xmin', type=float, default=10.0, help='age axis lower limit [Myr]')
    ap.add_argument('--xmax', type=float, default=2000.0, help='age axis upper limit [Myr]')
    ap.add_argument('--out', default='../figures/rogers2025_radius_vs_age.png')
    args = ap.parse_args()

    load_rcparams()

    # --- model contraction tracks via the radius interpolators ---
    # both tracks independently tunable via --gd_* (gas dwarf) and --ww_* (water world)
    Rgd = dvm._build_interp(args.gd_mu, args.gd_gamma, args.gd_kind, args.gd_mu_c, kappa=args.gd_kappa)
    Rww = dvm._build_interp(args.ww_mu, args.ww_gamma, args.ww_kind, args.ww_mu_c, kappa=args.ww_kappa)
    ages = np.logspace(1, np.log10(2000), 120)            # 1 Myr - 5 Gyr

    fig, ax = plt.subplots(figsize=(9, 7))
    styles = ['-', '--', ':', '-.']
    for Mc, ls in zip(args.masses, styles):
        Rg = [Rgd(Mc, args.gd_x, args.teq, t) for t in ages]
        Rw = [Rww(Mc, args.ww_x, args.teq, t) for t in ages]
        ax.plot(ages, Rg, color='darkorange', ls=ls, lw=2.5,
                label=f'gas dwarf, {Mc:.0f} ' + r'$M_\oplus$')
        ax.plot(ages, Rw, color='royalblue', ls=ls, lw=2.5,
                label=f'water world, {Mc:.0f} ' + r'$M_\oplus$')

    # --- observed young planets: radius vs host age ---
    # colour by a physically-motivated quantity (equilibrium temperature by
    # default): Teq sets the envelope scale height H ~ Teq/mu and hence the radius.
    # Colouring by age would be redundant with the x-axis.
    dp = dvm.load_planets()
    sel = (dp.amid > 0) & (dp.pl_rade > 0) & (dp.pl_rade < args.rmax)
    d = dp[sel]
    amid = d.amid.values
    R = d.pl_rade.values
    xerr = np.vstack([np.clip(amid - d.alo.values, 0, None),
                      np.clip(d.ahi.values - amid, 0, None)])
    yerr = d.pl_rade_err.values if 'pl_rade_err' in d else None

    cinfo = {'teq': ('teq_eff', r'$T_{\rm eq}$ [K]', 'plasma'),
             'period': ('pl_orbper', 'Orbital period [days]', 'viridis')}
    # light grey error bars for everyone (so colour encodes only the points)
    ax.errorbar(amid, R, xerr=xerr, yerr=yerr, fmt='none',
                ecolor='0.75', elinewidth=0.7, capsize=0, zorder=8)
    col, clabel, cmap = cinfo[args.color]
    cval = d[col].values
    ok = np.isfinite(cval)
    norm = (plt.matplotlib.colors.LogNorm() if args.color == 'period'
            else plt.matplotlib.colors.Normalize())
    sc = ax.scatter(amid[ok], R[ok], c=cval[ok], cmap=cmap, norm=norm,
                    s=42, edgecolors='w', lw=0.5, zorder=10)
    if (~ok).any():    # points without the colour quantity -> grey
        ax.scatter(amid[~ok], R[~ok], c='0.6', s=42, edgecolors='w', lw=0.5,
                   zorder=10, label='no ' + args.color)
    fig.colorbar(sc, ax=ax, label=clabel, pad=0.02)

    # --- 'Our Work' targets (stars) coloured on the same scale as the data ---
    ow_key = {'teq': 'teq', 'period': 'P'}[args.color]   # period always present; teq optional
    for t in dvm.OUR_WORK:
        lo, hi = t['age']
        mid = 0.5 * (lo + hi)
        v = t.get(ow_key)
        v = float(v) if v is not None else np.nan
        fc = sc.cmap(sc.norm(v)) if np.isfinite(v) else '0.6'   # grey if quantity missing
        ax.errorbar(mid, t['R'], xerr=[[mid - lo], [hi - mid]], yerr=t.get('Rerr'),
                    fmt='none', ecolor='k', elinewidth=1.0, capsize=0, zorder=15)
        ax.scatter(mid, t['R'], marker='*', s=320, color=fc, edgecolors='k',
                   lw=1.0, zorder=16)
    # legend proxy for the marker shape (colour now encodes the chosen quantity)
    ax.scatter([], [], marker='*', s=200, facecolor='none', edgecolors='k',
               lw=1.0, label='Our Work')
        # ax.annotate(t['name'], (mid, t['R']), xytext=(6, 5), textcoords='offset points',
        #             fontsize=8, zorder=16)

    ax.set_xscale('log')
    if args.logr:
        ax.set_yscale('log')
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(1, args.rmax)
    ax.set_xlabel('Age [Myr]')
    ax.set_ylabel(r'Planet radius [$R_\oplus$]')
    ax.set_title(
        rf'$T_{{\rm eq}}={args.teq:.0f}$ K' + '\n'
        rf'GD: $\mu={args.gd_mu:g},\ X={args.gd_x:g},\ \kappa={args.gd_kappa:g}$    '
        rf'WW: $\mu={args.ww_mu:g},\ X={args.ww_x:g},\ \kappa={args.ww_kappa:g}$',
        fontsize=11)
    ax.legend(frameon=False, fontsize=11, ncol=1)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print('saved', args.out)


if __name__ == '__main__':
    main()
