"""Run the madys-code cells of ext_code_madys_corot3b.ipynb, ignoring the rogue `d` cell.

star_list is built from the host star only (the `d`-based cluster members are skipped).
"""
import madys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

print("madys version:", madys.MADYS_VERSION)

gaiaDR3id = 3762515188088861184

# --- ignore rogue `d`: star_list is just the host (cell 6 appends the host anyway) ---
star_list = [f"Gaia DR3 {gaiaDR3id}"]
print("star_list:", star_list)

# --- madys.SampleObject pipeline ---
p = madys.SampleObject(star_list, id_type="DR3", ext_map="leike")
print("\nphot_table:")
print(p.phot_table)
print("\nphot_table shape:", p.phot_table.to_pandas().shape)
print("abs_phot shape:", p.abs_phot.shape)
print("abs_phot_err shape:", p.abs_phot_err.shape)

target = pd.DataFrame(p.abs_phot, columns=p.filters).squeeze()
print("\ntarget (absolute photometry):")
print(target)

# --- CMD with isochrones (uses `p`; notebook's `p2` is undefined) ---
model = "parsec2"
x, y = "Gbp-Grp", "G"
plot_ages = [10, 20, 30, 100]
plot_masses = [0.3, 0.5, 0.7, 0.8, 1.0, 1.5, 2]

fig, ax = p.CMD(
    x, y, model,
    xlim=[-0.3, 4.1], ylim=[14, -2],
    close=False,
    plot_ages=plot_ages, plot_masses=plot_masses,
    label_points=False, alpha=0.5, s=8, figsize=(8, 8),
)
xx = target.Gbp - target.Grp
yy = target.G
ax.plot(xx, yy, marker=r"$\star$", mec="k", c="r", ms=25, label="host", zorder=10)
ax.legend(loc="lower left")
ax.set_ylabel(r"$G$")
ax.set_xlabel(r"$G_{BP} - G_{RP}$")
ax.set_title(f"CoRoT-3b host | {model.upper()} isochrones")

out = "_madys_cmd_corot3b_host.png"
fig.savefig(out, dpi=150)
print("\nCMD saved to:", out)
print("DONE")
