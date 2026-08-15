#!/usr/bin/env python
"""Animate the Rogers (2025) gas-dwarf vs water-world occurrence prediction as a
function of age (10 Myr -> 1 Gyr), with the observed young-planet sample overlaid
in a moving age window.

Two panels (gas dwarf | water world): each shows the period-radius KDE of the
synthetic population *observed at the current age*, the radius valley + Neptune
desert, and the matched young planets whose host age sits near that age.

Model and data matching reuse data_vs_model.py (mirrors
rogers2025_gasdwarf_waterworld.ipynb). Per-planet radius tracks are precomputed
once, so rendering each frame is cheap.

Usage:
    python animate_gasdwarf_waterworld.py                 # default mp4 (gif fallback)
    python animate_gasdwarf_waterworld.py --gif --frames 50 --n 1500 --fps 12
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.ticker import ScalarFormatter, NullLocator
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

import data_vs_model as dvm

# constants (CGS) from the shared module
G, sigma = dvm.G, dvm.sigma
Mearth, Rearth, Myr, Msun = dvm.Mearth, dvm.Rearth, dvm.Myr, dvm.Msun
core_radius, teq_orbit, lxuv = dvm.core_radius, dvm._teq_orbit, dvm._lxuv
draw_period, valley_R = dvm._draw_period, dvm.vaneylen_valley_radius


# ----------------------------------------------------------------------
# styling
# ----------------------------------------------------------------------
def load_rcparams(path='../rcParams.txt'):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) != 2 or row[0] == 'name':
                continue
            name, value = row
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            try:
                plt.rcParams[name] = value
            except Exception:
                pass


# ----------------------------------------------------------------------
# radius-track precompute (one full evolution per planet, sampled at frame ages)
# ----------------------------------------------------------------------
def precompute_gas_dwarf(Rgd, ages, rng, N, n_sub=250):
    """Gas dwarfs with XUV photoevaporation. Returns P[N], R[N, n_frames]."""
    M = np.clip(np.exp(rng.normal(np.log(4.0), 0.35, N)), 1, 20)
    X0 = np.clip(np.exp(rng.normal(np.log(0.02), 0.5, N)), 1e-3, 0.3)
    P = draw_period(N, 5.5, 2.3, 0.0, rng)
    Ms = np.clip(rng.normal(1.0, 0.3, N), 0.5, 1.6) * Msun
    td = rng.uniform(3, 10, N)
    R = np.empty((N, len(ages)))
    for i in range(N):
        Mc = M[i] * Mearth
        Teq, a = teq_orbit(P[i], Ms[i])
        fine = np.logspace(np.log10(td[i]), np.log10(ages.max()), n_sub)
        Menv = X0[i] * Mc
        Marr = np.empty(n_sub)
        Marr[0] = Menv
        for n in range(n_sub - 1):
            X = Menv / Mc
            if X > 1e-4:
                Rp = Rgd(M[i], X, Teq, fine[n]) * Rearth
                Menv -= 0.15 * (lxuv(fine[n], Ms[i]) / (4 * np.pi * a * a)) * \
                    (np.pi * Rp ** 3 / (G * Mc)) * (fine[n + 1] - fine[n]) * Myr
                Menv = max(Menv, 0.0)
            Marr[n + 1] = Menv
        Mof = interp1d(fine, Marr, bounds_error=False, fill_value=(Marr[0], Marr[-1]))
        for f, t in enumerate(ages):
            X = Mof(t) / Mc
            R[i, f] = core_radius(M[i], 'earth') / Rearth if X < 1e-4 \
                else Rgd(M[i], np.clip(X, 1e-4, 0.3), Teq, t)
    return P, R


def precompute_water_world(Rww, ages, rng, N):
    """35% rocky super-Earths (static) + 65% steam sub-Neptunes (cool/contract,
    no mass loss). Returns P[N], R[N, n_frames]."""
    n_se = int(0.35 * N)
    n_sn = N - n_se
    Ms = np.clip(rng.normal(1.0, 0.3, N), 0.5, 1.6) * Msun
    Mse = np.clip(10 ** rng.normal(np.log10(3.0), 0.35, n_se), 0.5, 20)
    Pse = draw_period(n_se, 4.5, 2.0, -0.5, rng)
    Rse = core_radius(Mse, 'earth') / Rearth
    Msn = np.clip(10 ** rng.normal(np.log10(10.0), 0.30, n_sn), 1, 20)
    Psn = draw_period(n_sn, 8.5, 2.5, 0.1, rng)
    Xsn = 10 ** rng.uniform(np.log10(1e-5), np.log10(0.1), n_sn)
    R = np.empty((N, len(ages)))
    R[:n_se, :] = Rse[:, None]                       # rocky cores: time-independent
    for i in range(n_sn):
        Te, _ = teq_orbit(Psn[i], Ms[n_se + i])
        R[n_se + i, :] = [Rww(Msn[i], Xsn[i], Te, t) for t in ages]
    return np.concatenate([Pse, Psn]), R


# ----------------------------------------------------------------------
# overlays
# ----------------------------------------------------------------------
def overplot_desert_and_valley(ax):
    xl = np.linspace(-0.30, np.log10(2.97), 100)
    xu = np.linspace(np.log10(2.97), 0.120, 100)
    x = np.concatenate([np.tile(-0.30, 100), xl, xu, np.tile(0.12, 100)])
    y = np.concatenate([np.linspace(-0.07, 0.191, 100), 0.55 * xl + 0.36,
                        -0.43 * xu + 1.14, np.linspace(1.0884, 1.20, 100)])
    ax.plot(10 ** x, 10 ** y, 'k--', lw=2, alpha=0.9, zorder=5)     # Neptune desert
    xv = np.linspace(0, 2, 100)
    ax.plot(10 ** xv, valley_R(10 ** xv), 'k:', lw=1.3, alpha=0.85, zorder=5)  # valley


def style_axes(ax, title):
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1, 100); ax.set_ylim(1, 10)
    ax.set_title(title, fontsize=18)
    ax.set_yticks([1, 3, 6, 10])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Orbital period [days]')


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=45)
    ap.add_argument('--n', type=int, default=1500, help='planets per population')
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--window', type=float, default=1.8,
                    help='multiplicative half-width of the data age window')
    ap.add_argument('--gif', action='store_true', help='force GIF output')
    ap.add_argument('--out', default='../figures/rogers2025_evolution')
    ap.add_argument('--dpi', type=int, default=130)
    args = ap.parse_args()

    load_rcparams()
    ages = np.logspace(np.log10(10), np.log10(1000), args.frames)   # Myr

    print('building radius interpolators ...')
    Rgd = dvm._build_interp(2.35, 7 / 5, 'earth', 60.0)
    Rww = dvm._build_interp(18.0, 4 / 3, 'water', 27.0)
    rng = np.random.default_rng(args.seed)
    print('precomputing radius tracks ...')
    Pg, Rg = precompute_gas_dwarf(Rgd, ages, rng, args.n)
    Pw, Rw = precompute_water_world(Rww, ages, rng, args.n)

    dp = dvm.load_planets()
    Pd_all, Rd_all, amid = dp.pl_orbper.values, dp.pl_rade.values, dp.amid.values
    inwin = (Pd_all >= 1) & (Pd_all <= 100) & (Rd_all >= 1) & (Rd_all <= 10)
    Pd_all, Rd_all, amid = Pd_all[inwin], Rd_all[inwin], amid[inwin]
    ow = dvm.targets_table()                                        # 'Our Work'

    xs = np.linspace(0, 2, 90)
    ys = np.linspace(0, np.log10(10), 90)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.vstack([XX.ravel(), YY.ravel()])
    levels = np.linspace(0, 1, 11)

    fig, (axg, axw) = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.88, wspace=0.1)
    
    def draw_kde(ax, P, R, cmap):
        Z = gaussian_kde(np.vstack([np.log10(P), np.log10(R)]))(grid).reshape(XX.shape)
        Z /= Z.max()
        return ax.contourf(10 ** XX, 10 ** YY, Z, levels=levels, cmap=cmap, zmin=0, zmax=0.5)

    def update(f):
        age = ages[f]
        lo, hi = age / args.window, age * args.window
        sel = (amid >= lo) & (amid < hi)
        for ax, P, R, cmap, name in [(axg, Pg, Rg[:, f], 'Greens', 'Gas dwarf'),
                                     (axw, Pw, Rw[:, f], 'Purples', 'Water world')]:
            ax.clear()
            axg.set_ylabel(r'Planet radius [R$_\oplus$]')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(1, 100); ax.set_ylim(1, 10)
            draw_kde(ax, P, R, cmap)
            overplot_desert_and_valley(ax)
            ax.scatter(Pd_all, Rd_all, s=10, c='0.6', alpha=0.35, zorder=8)  # all data, faint
            if sel.any():
                c = np.where(amid[sel] < 100, 'C3', 'C0')
                ax.scatter(Pd_all[sel], Rd_all[sel], s=55, c=c, edgecolors='w',
                           lw=0.8, zorder=12)
            if age<60:
                ax.scatter(ow.P_day, ow.R_earth, marker='*', s=180, c='C3',
                           edgecolors='w', lw=0.6, zorder=14)
            style_axes(ax, f'{name} : {int(round(age, -1))} Myr')
        #fig.suptitle(f'Rogers (2025) gas dwarf vs water world  —  age = {age:.0f} Myr '
        #             f'(data within {lo:.0f}–{hi:.0f} Myr)', fontsize=13)
        return []

    anim = FuncAnimation(fig, update, frames=len(ages), blit=False)

    use_gif = args.gif
    if not use_gif:
        try:
            out = args.out + '.mp4'
            anim.save(out, writer=FFMpegWriter(fps=args.fps, bitrate=2400), dpi=args.dpi)
            print('saved', out)
            return
        except Exception as e:
            print('ffmpeg unavailable (%s); falling back to GIF' % e)
    out = args.out + '.gif'
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print('saved', out)


if __name__ == '__main__':
    main()
