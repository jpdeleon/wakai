#!/usr/bin/env python
"""Plot planet radius versus age using the Lee (2022) primordial-gap model.

The model is transcribed from ``lee2022_radius_period_gap.ipynb``.  A planet's
primordial H/He mass fraction is set by late-stage, dust-free gas accretion and
the maximally cooled isothermal cap.  Its radius is then evaluated over time
with the Owen & Wu (2017) radiative-convective-boundary prescription used by
Lee & Connors (2021/2022).

Lee & Connors calibrate tau_KH=100 Myr to a 100 Myr planet and tau_KH=300 Myr
to a 1 Gyr planet.  Between (and beyond) those points this script uses a power
law in age.  This is an approximate sequence of cooling states, not a
self-consistent evolutionary integration.

Usage:
    python plot_radius_vs_age_lee22_model.py
    python plot_radius_vs_age_lee22_model.py --period 10 --masses 2 5 8 --rmax 6
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import data_vs_model as dvm


# Physical constants (cgs), matching lee2022_radius_period_gap.ipynb.
G = 6.674e-8
kB = 1.380649e-16
mH = 1.6726e-24
SIGMA_SB = 5.6704e-5
MEARTH = 5.972e27
REARTH = 6.371e8
MSUN = 1.989e33
RSUN = 6.957e10
AU = 1.496e13
YR = 3.1557e7
TEFF_SUN = 5772.0

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rcparams(path=os.path.join(HERE, '..', 'rcParams.txt')):
    """Load the project's two-column matplotlib configuration if present."""
    if not os.path.exists(path):
        return
    with open(path) as stream:
        for row in csv.reader(stream):
            if len(row) != 2 or row[0] == 'name':
                continue
            name, value = row
            for cast in (int, float):
                try:
                    value = cast(value)
                    break
                except ValueError:
                    pass
            try:
                plt.rcParams[name] = value
            except Exception:
                pass


class Lee22Model:
    """Fiducial dust-free late-stage accretion and radius model."""

    def __init__(self, f_r=0.1, f_t=2.0, f_dep=1e-3, t_acc_myr=1.0,
                 metallicity=0.02, mu=2.37, gamma=7 / 5,
                 rho_core_ratio=1.0, stellar_mass=1.0):
        self.f_r = f_r
        self.f_t = f_t
        self.f_dep = f_dep
        self.t_acc = t_acc_myr * 1e6 * YR
        self.metallicity = metallicity
        self.mu = mu
        self.gamma = gamma
        self.nabla_ad = (gamma - 1) / gamma
        self.integral_exponent = 1 / (gamma - 1)
        self.rho_core_ratio = rho_core_ratio
        self.mstar = stellar_mass * MSUN

        # Rogers et al. (2010) dust-free opacity fit used by Lee & Connors.
        self.opacity_c = -7.32
        self.opacity_alpha = 0.68
        self.opacity_beta = 0.45

    def semimajor_axis(self, period_days):
        return (G * self.mstar * (period_days * 86400.0) ** 2
                / (4 * np.pi ** 2)) ** (1 / 3)

    def disk_temperature(self, a_cm):
        return 1000.0 * self.f_t * (a_cm / (0.1 * AU)) ** (-3 / 7)

    def gas_surface_density(self, a_cm):
        return 1.3e5 * self.f_dep * (a_cm / (0.2 * AU)) ** (-1.6)

    def dustfree_gcr(self, core_mass, a_cm):
        """Analytic dust-free envelope-to-core mass ratio."""
        disk_temp = self.disk_temperature(a_cm)
        sigma_gas = self.gas_surface_density(a_cm)
        return (0.25 * self.f_r
                * (core_mass / (5 * MEARTH)) ** 1.1
                * (self.t_acc / (1e3 * YR)) ** 0.4
                * (200 / disk_temp) ** 1.5
                * (sigma_gas / 4e5) ** 0.12
                * (0.02 / self.metallicity) ** 0.4
                * (self.mu / 2.37) ** 2.2)

    def isothermal_envelope_mass(self, core_mass, a_cm):
        """Maximum mass of a fully cooled isothermal envelope [g]."""
        disk_temp = self.disk_temperature(a_cm)
        sound_speed_sq = kB * disk_temp / (self.mu * mH)
        sound_speed = np.sqrt(sound_speed_sq)
        omega = np.sqrt(G * self.mstar / a_cm ** 3)
        rho_disk = self.gas_surface_density(a_cm) * omega / sound_speed
        core_radius = REARTH * (core_mass / MEARTH) ** 0.25
        bondi_radius = G * core_mass / sound_speed_sq
        hill_radius = a_cm * (core_mass / (3 * self.mstar)) ** (1 / 3)
        outer_radius = self.f_r * min(bondi_radius, hill_radius)
        if outer_radius <= core_radius:
            return 0.0
        b = G * core_mass / sound_speed_sq

        def integrand(radius):
            return radius ** 2 * np.exp(b * (1 / radius - 1 / outer_radius))

        value, _ = quad(integrand, core_radius, outer_radius, limit=200)
        return 4 * np.pi * rho_disk * value

    def envelope_fraction(self, core_mass, a_cm):
        accreted_mass = self.dustfree_gcr(core_mass, a_cm) * core_mass
        return min(accreted_mass, self.isothermal_envelope_mass(core_mass, a_cm)) / core_mass

    @staticmethod
    def cooling_time(age_myr):
        """Lee-calibrated Kelvin-Helmholtz time [s] for a planet age [Myr]."""
        exponent = np.log(3.0) / np.log(10.0)
        return 100e6 * YR * (age_myr / 100.0) ** exponent

    def structure_integral(self, order, radius_ratio):
        lower = 1 / radius_ratio

        def integrand(u):
            return u ** order * (1 / u - 1) ** self.integral_exponent

        value, _ = quad(integrand, lower, 1, limit=200)
        return value

    def planet_radius(self, core_mass, a_cm, gcr, age_myr):
        """Photometric radius [cm] at the requested cooling age."""
        core_radius = (REARTH * (core_mass / MEARTH) ** 0.25
                       * self.rho_core_ratio ** (-1 / 3))
        teq = TEFF_SUN * np.sqrt(RSUN / a_cm)
        sound_speed_sq = kB * teq / (self.mu * mH)
        tau_kh = self.cooling_time(age_myr)

        # Equation 12 in Lee & Connors (2021), with its normalization scaled
        # from their stated values at tau_KH=100 and 300 Myr.
        threshold_exponent = np.log(7 / 4.4) / np.log(3)
        gcr_min = (4.4e-5 * (tau_kh / (100e6 * YR)) ** threshold_exponent
                   * (core_mass / MEARTH) ** 0.74
                   * (a_cm / (0.42 * AU)) ** 0.44)
        if gcr <= gcr_min:
            return core_radius
        envelope_mass = gcr * core_mass

        def rho_from_mass(radius_ratio):
            i2 = self.structure_integral(2, radius_ratio)
            return (envelope_mass / (4 * np.pi)
                    * radius_ratio ** (-3 + self.integral_exponent)
                    * core_radius ** (-3 + self.integral_exponent)
                    * (self.nabla_ad * G * core_mass / sound_speed_sq)
                    ** (1 / (1 - self.gamma)) / i2)

        def rho_power_from_luminosity(radius_ratio):
            i2 = self.structure_integral(2, radius_ratio)
            i1 = self.structure_integral(1, radius_ratio)
            return (64 * np.pi * SIGMA_SB * self.mu * mH / (3 * kB)
                    * self.nabla_ad * 10 ** (-self.opacity_c)
                    * (self.mu * mH / kB) ** self.opacity_alpha
                    * teq ** (3 - self.opacity_alpha - self.opacity_beta)
                    * (i2 / i1) * (tau_kh / envelope_mass)
                    * radius_ratio * core_radius)

        def residual(radius_ratio):
            return ((1 + self.opacity_alpha) * np.log(rho_from_mass(radius_ratio))
                    - np.log(rho_power_from_luminosity(radius_ratio)))

        try:
            radius_ratio = brentq(residual, 1.0001, 200, maxiter=200)
        except ValueError:
            return core_radius
        if radius_ratio < 1.05:
            return core_radius

        rcb_radius = radius_ratio * core_radius
        rho_rcb = rho_from_mass(radius_ratio)
        opacity = (10 ** self.opacity_c * rho_rcb ** self.opacity_alpha
                   * (kB / (self.mu * mH)) ** self.opacity_alpha
                   * teq ** (self.opacity_alpha + self.opacity_beta))
        gravity = G * core_mass / rcb_radius ** 2
        rho_ph = ((2 / 3) * self.mu * mH * gravity
                  / (kB * teq * opacity))
        return (rcb_radius + np.log(rho_rcb / rho_ph)
                * kB * teq / (self.mu * mH * gravity))


def add_observations(ax, fig, args):
    """Overlay the same young-planet sample used by the Rogers age plot."""
    planets = dvm.load_planets()
    selected = ((planets.amid > 0) & (planets.pl_rade > 0)
                & (planets.pl_rade < args.rmax))
    data = planets[selected]
    ages = data.amid.values
    radii = data.pl_rade.values
    xerr = np.vstack([np.clip(ages - data.alo.values, 0, None),
                      np.clip(data.ahi.values - ages, 0, None)])
    yerr = data.pl_rade_err.values if 'pl_rade_err' in data else None
    ax.errorbar(ages, radii, xerr=xerr, yerr=yerr, fmt='none', ecolor='0.75',
                elinewidth=0.7, capsize=0, zorder=8)

    color_info = {
        'teq': ('teq_eff', r'$T_{\rm eq}$ [K]', 'plasma'),
        'period': ('pl_orbper', 'Orbital period [days]', 'viridis'),
    }
    column, label, cmap = color_info[args.color]
    values = data[column].values
    finite = np.isfinite(values)
    norm = (plt.matplotlib.colors.LogNorm() if args.color == 'period'
            else plt.matplotlib.colors.Normalize())
    scatter = ax.scatter(ages[finite], radii[finite], c=values[finite], cmap=cmap,
                         norm=norm, s=42, edgecolors='w', lw=0.5, zorder=10)
    if (~finite).any():
        ax.scatter(ages[~finite], radii[~finite], c='0.6', s=42,
                   edgecolors='w', lw=0.5, zorder=10,
                   label='no ' + args.color)
    fig.colorbar(scatter, ax=ax, label=label, pad=0.02)

    our_work_key = {'teq': 'teq', 'period': 'P'}[args.color]
    for target in dvm.OUR_WORK:
        age_lo, age_hi = target['age']
        age_mid = 0.5 * (age_lo + age_hi)
        value = target.get(our_work_key)
        value = float(value) if value is not None else np.nan
        facecolor = (scatter.cmap(scatter.norm(value))
                     if np.isfinite(value) else '0.6')
        ax.errorbar(age_mid, target['R'],
                    xerr=[[age_mid - age_lo], [age_hi - age_mid]],
                    yerr=target.get('Rerr'), fmt='none', ecolor='k',
                    elinewidth=1, capsize=0, zorder=15)
        ax.scatter(age_mid, target['R'], marker='*', s=320, color=facecolor,
                   edgecolors='k', lw=1, zorder=16)
    ax.scatter([], [], marker='*', s=200, facecolor='none', edgecolors='k',
               lw=1, label='Our Work')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', type=float, default=10.0,
                        help='orbital period [days] shared by all tracks')
    parser.add_argument('--masses', type=float, nargs='+', default=[2, 5],
                        help='core masses [M_earth] for the tracks')
    parser.add_argument('--stellar_mass', type=float, default=1.0,
                        help='host-star mass [M_sun]')
    parser.add_argument('--f_dep', type=float, default=1e-3,
                        help='disk gas depletion factor')
    parser.add_argument('--f_r', type=float, default=0.1,
                        help='outer-radius truncation factor')
    parser.add_argument('--f_t', type=float, default=2.0,
                        help='disk temperature coefficient')
    parser.add_argument('--t_acc', type=float, default=1.0,
                        help='gas accretion duration [Myr]')
    parser.add_argument('--color', choices=['teq', 'period'], default='teq',
                        help='quantity used to color the observed planets')
    parser.add_argument('--xmin', type=float, default=10.0,
                        help='age axis lower limit [Myr]')
    parser.add_argument('--xmax', type=float, default=2000.0,
                        help='age axis upper limit [Myr]')
    parser.add_argument('--rmax', type=float, default=9.0,
                        help='radius axis upper limit [R_earth]')
    parser.add_argument('--logr', action='store_true',
                        help='use a logarithmic radius axis')
    parser.add_argument('--out', default=os.path.join(HERE, '..', 'figures',
                                                      'lee2022_radius_vs_age.png'))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.period <= 0 or args.xmin <= 0 or args.xmax <= args.xmin:
        raise ValueError('period and xmin must be positive, and xmax must exceed xmin')
    if any(mass <= 0 for mass in args.masses):
        raise ValueError('all core masses must be positive')

    load_rcparams()
    model = Lee22Model(f_r=args.f_r, f_t=args.f_t, f_dep=args.f_dep,
                       t_acc_myr=args.t_acc, stellar_mass=args.stellar_mass)
    semimajor_axis = model.semimajor_axis(args.period)
    ages = np.logspace(np.log10(args.xmin), np.log10(args.xmax), 120)

    fig, ax = plt.subplots(figsize=(9, 7))
    styles = ['-', '--', ':', '-.']
    colors = plt.cm.Oranges(np.linspace(0.55, 0.95, len(args.masses)))
    for index, (mass, color) in enumerate(zip(args.masses, colors)):
        core_mass = mass * MEARTH
        gcr = model.envelope_fraction(core_mass, semimajor_axis)
        radii = np.array([
            model.planet_radius(core_mass, semimajor_axis, gcr, age) / REARTH
            for age in ages
        ])
        ax.plot(ages, radii, color=color, ls=styles[index % len(styles)], lw=2.5,
                label=(rf'Lee+22, {mass:g} $M_\oplus$'
                       + rf', $X={gcr:.1e}$'))

    add_observations(ax, fig, args)

    ax.set_xscale('log')
    if args.logr:
        ax.set_yscale('log')
    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(1, args.rmax)
    ax.set_xlabel('Age [Myr]')
    ax.set_ylabel(r'Planet radius [$R_\oplus$]')
    ax.set_title(
        rf'Lee et al. (2022): $P={args.period:g}$ d, '
        rf'$M_\star={args.stellar_mass:g}\,M_\odot$' + '\n'
        + rf'$f_{{\rm dep}}={args.f_dep:g},\ f_R={args.f_r:g},\ '
        + rf'f_T={args.f_t:g},\ t_{{\rm acc}}={args.t_acc:g}$ Myr',
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', args.out)


if __name__ == '__main__':
    main()
