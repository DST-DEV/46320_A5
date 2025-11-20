"""Create blade design files.

This script creates/updates the following files for your design:
 * _master/[YOUR_DESIGN_NAME].htc
 * data/[YOUR_DESIGN_NAME]_ae.dat
 * data/[YOUR_DESIGN_NAME]_Blade_st.dat
"""
from collections.abc import Sequence
from pathlib import Path
import warnings

import lacbox
from lacbox.io import load_ae, load_st, load_oper, save_ae, save_st, \
    save_oper, load_pc, load_c2def, save_c2def
from lacbox.rotor_design import get_design_functions, single_point_design, \
    scale_ST_data
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import fsolve
from scipy.signal import find_peaks
import xarray as xr

from myteampack import MyHTC
import scivis

# %% Preparations
DESIGN_NAME = "IEC_Ya_Later"
# Note that files starting with the design name should be in the .gitignore
SHOW_PLOT = False  # Flag for showing plots
EXP_FLD = "plots"
SAVEFIG = False  # Flag for exporting figures to a file
LATEX = False  # Flag for using latex text interpretation
FIG_SCALE = .75  # Scaling factor for figures

plasma_custom = [
    "#0D0887",  # indigo
    "#7E03A8",  # purple
    "#CB4679",  # pink
    "#F89441",  # orange
    "#FFB300",  # strong orange (L* ≈ 80)
]
cmap_custom = LinearSegmentedColormap.from_list("plasma_custom", plasma_custom)

# Filepaths
ROOT = Path(__file__).parent
if not Path(ROOT, EXP_FLD).is_dir():
    Path(ROOT, EXP_FLD).mkdir()
ORIG_HTC = ROOT / '_master' / 'dtu_10mw.htc'  # Original htc
ORIG_AE = ROOT / 'data' / 'DTU_10MW_RWT_ae.dat'  # Original AE file
ORIG_BLADE_ST = ROOT / 'data' / 'DTU_10MW_RWT_Blade_st.dat'  # Original blade ST
TARGET_HTC = ROOT / '_master' / f'{DESIGN_NAME}.htc'  # Scaled htc
TARGET_AE = ROOT / 'data' / f'{DESIGN_NAME}_ae.dat'  # Scaled AE file
TARGET_BLADE_ST = ROOT / 'data' / f'{DESIGN_NAME}_Blade_st.dat'  # Scaled blade ST

# Blade design settings
R_HUB = 2.8  # hub radius. cannot be changed. [m]
L_BLD_ORIG = 86.37 # original blade length, excluding coning but including static prebend [m]
R_ORIG = R_HUB + L_BLD_ORIG
ID_POLY = 2  # design polynomial identifier [-]
B = 3  # Number of blades [-]
RHO = 1.225  # Air density [kg/m^3]
TI_ORIG = .16  # Original turbulence intensity [-]
TI = .14  # Turbulence intensity [-]
V_RTD_ORIG = 11.4  # Original rated wind speed [m/s]
GEN_SPEED_MAX_ORIG = 480  # Original maximum generator speed [rpm]
GEAR_RATIO_ORIG = 50  # Original gear ratio [-]

# load the DTU 10 MW files
r_ORIG, c_ORIG, tc_ORIG, pcset_ORIG = load_ae(ORIG_AE, unpack=True)
t_ORIG = tc_ORIG*c_ORIG/100

# %% Rotor radius scaling
# Define your new rotor radius and calculate your rotor scale factor.
# Calculate your new BLADE length and use s-hat, your blade length, and the hub
# length to define the array of rotor span points (from the HUB CENTER) to
# evaluate for your design.
# Note that final point must be less than R -- suggest moving last point 0.5 m
# inboard.
# Calculate also your BLADE LENGTH scale factor.


# Define a function that represents the system of equations
# The function should return a list or array of the values of the equations
# when they are set to zero.
def rotor_equilibrium(vars: Sequence) -> list:
    """
    System of equations representing the equilibrium of scaled vs unscaled
    rated rotor power and design generator torque

    Parameters
    ----------
    vars : Sequence
        Scaled rotor radius and rated wind speed.

    Returns
    -------
    list
        Residual of the equilibrium equations.

    """
    Ry, Vy = vars
    eq1 = R_ORIG**2*V_RTD_ORIG**3-Ry**2*Vy**3
    eq2 = R_ORIG**3*V_RTD_ORIG**2*(1+2*TI_ORIG)**2-Ry**3*Vy**2*(1+2*TI)**2
    return [eq1, eq2]


# Use fsolve to find the solution for the scaled rotor radius and rated wind
# speed
solution = fsolve(rotor_equilibrium, [R_ORIG, V_RTD_ORIG])
R, V_RTD = solution
print(f'New R is {R}, and the new V is {V_RTD}')

# Calculate scale factors and radial stations
l_bld = R - R_HUB  # Blade length [m]
sc_bld = (R-R_HUB)/L_BLD_ORIG  # Blade scaling factor [-]
sc_R = R/R_ORIG  # Rotor scaling factor [-]

r = r_ORIG*sc_bld  # Scaled radial section z-positions
if r[-1] >= l_bld-.5:
    r[-1] = l_bld-.5

s_hat = r/np.max(r)  # Normalized radial blade section z-positions

# Define chord root and max chord from DTU 10 MW files
c_root = c_ORIG[0]
c_max = np.max(c_ORIG)

# Compute absolute thickness, and cap it at the root so it doesn't exceed the
# root chord
c = c_ORIG*sc_bld
c[0] = c_root
t = tc_ORIG*c/100
t[t > c_root] = c_root

# Plot thickness
if SHOW_PLOT:
    colors = cmap_custom(np.linspace(0, 1, 4))
    scivis.plot_line(s_hat, np.vstack([t, t_ORIG]),
                     ax_labels=[r"\hat{s}", "t"], ax_units=[None, "m"],
                     plt_labels=["Scaled model", "Original model"],
                     ax_show_minor_ticks=True,
                     ax_lims=[(0, 1), None],
                     colors=[colors[0], colors[2]], linestyles="-",
                     latex=LATEX, profile="partsize", scale=FIG_SCALE,
                     overflow=False,
                     fname="t_vs_s", exp_fld=EXP_FLD, savefig=SAVEFIG)

# %% Aerodynamic parameter adjustments
# Choose Design aerodynamic values
# Load airfoil performance characteristics (Only for -90 < AoA < 90)
pc_path = Path(ROOT, "data", "DTU_10MW_RWT_pc.dat")
pc_data = load_pc(pc_path)

aoa = pc_data[0]["aoa_deg"]

aoa_lims = (-90, 90)
mask = (aoa > aoa_lims[0]-.5) & (aoa < aoa_lims[1]+.5)
C_l = np.vstack([data_i["cl"] for data_i in pc_data])[0:4, mask]
C_d = np.vstack([data_i["cd"] for data_i in pc_data])[0:4, mask]
aoa = aoa[mask]
N_AIRFOILS = 4

airfoil_names = [data_i["comment"].split()[0].replace("\"", "")
                 for data_i in pc_data][0:4]
t_airfoils = [float(name.split("-")[-1])/10 for name in airfoil_names]

# Determine design AoA as the point of maximum glide ratio.
# Stability should be ensured by keeping the design lift coefficient ~0.4
# below its global maximum (onset of stall region)
aoa_enhanced = np.hstack([aoa[aoa < -5], np.arange(-5, 15.01, .1),
                          aoa[aoa > 15]]) # Higher resolution in relevant region
C_l_enhanced = np.vstack([np.interp(aoa_enhanced, aoa, C_l[i])
                          for i in range(N_AIRFOILS)])
C_d_enhanced = np.vstack([np.interp(aoa_enhanced, aoa, C_d[i])
                          for i in range(N_AIRFOILS)])

# Find maximum glide ratio
C_ld_enhanced = C_l_enhanced/C_d_enhanced  # Glide ratio [-]
idx_max_glide = np.argmax(C_ld_enhanced, axis=1)
C_ld_max = np.max(C_ld_enhanced, axis=1)
aoa_max_glide = aoa_enhanced[idx_max_glide]  # AoA of maximum glide ratio [deg]
C_l_max_glide = C_l_enhanced[range(N_AIRFOILS),
                             idx_max_glide]  # Lift at maximum glide ratio
C_d_max_glide = C_d_enhanced[range(N_AIRFOILS),
                             idx_max_glide]  # Drag at maximum glide ratio

# Find gurney flap peak in C_l curve of the 48% airfoil
C_l_peaks_48, _ = find_peaks(C_l_enhanced[-1], height=0)
idx_des_48 = C_l_peaks_48[-2]

aoa_des = aoa_max_glide
C_l_des = C_l_max_glide
C_d_des = C_d_max_glide
C_ld_des = C_ld_max

# Determine threshold value
C_l_thres = np.max(C_l_enhanced,
                   axis=1)-.4  # Stability threshold lift coefficient
aoa_stall = np.empty(N_AIRFOILS)
aoa_thres = np.empty(N_AIRFOILS)
C_d_thres = np.empty(N_AIRFOILS)
C_ld_thres = np.empty(N_AIRFOILS)
for i in range(N_AIRFOILS):
    C_l_peaks, _ = find_peaks(C_l_enhanced[i], height=0)
    aoa_stall[i] = aoa_enhanced[C_l_peaks[-1]]

    mask_aoa_limited = (aoa_enhanced > 0) & (aoa_enhanced < aoa_stall[i])
    aoa_thres[i] = np.interp(C_l_thres[i], C_l_enhanced[i, mask_aoa_limited],
                             aoa_enhanced[mask_aoa_limited])
    C_d_thres[i] = np.interp(aoa_thres[i], aoa_enhanced, C_d_enhanced[i])
    C_ld_thres[i] = np.interp(aoa_thres[i], aoa_enhanced, C_ld_enhanced[i])

# Enforce stability condition
ex_limit = .1  # Maximum accepted exceedance of the stability threshold
aoa_des_corr = aoa_des.copy()
C_l_des_corr = C_l_des.copy()
C_d_des_corr = C_d_des.copy()
C_ld_des_corr = C_ld_des.copy()
if any(C_l_des-C_l_thres > ex_limit):
    warnings.warn("Point of maximum glide ratio exceeds stability limit")

    idx_ex = np.argwhere(C_l_des-C_l_thres > ex_limit)
    aoa_des_corr[idx_ex] = aoa_thres[idx_ex]
    C_l_des_corr[idx_ex] = C_l_thres[idx_ex]
    C_d_des_corr[idx_ex] = C_d_thres[idx_ex]
    C_ld_des_corr[idx_ex] = C_ld_thres[idx_ex]

# Plot polar and lift curve for each airfoil
if SHOW_PLOT:
    aoa_ticks = np.arange(aoa_lims[0], aoa_lims[-1]+.5, 15)
    rc_profile = scivis.rcparams._prepare_rcparams(latex=LATEX,
                                                   profile="halfsize")
    colors = cmap_custom(np.linspace(0, 1, 4))
    for i in range(N_AIRFOILS):
        with mpl.rc_context(rc_profile):
            fig, ax = plt.subplots(1, 2, figsize=(32, 8))

            ax1, ax2 = ax

            fig, ax1, _ = scivis.plot_line(
                C_d_enhanced[i], C_l_enhanced[i], ax=ax1,
                ax_labels=["C_d", "C_l"], plt_labels=["_"],
                ax_show_minor_ticks=True,
                latex=LATEX, profile="halfsize",
                overflow=False, savefig=False)
            ax1.axvline(C_d_des[i], ls="--", c=colors[0], lw=1.8,
                        label="Design")
            ax1.axhline(C_l_des[i], ls="--", c=colors[0], lw=1.8,
                        label="_")
            if not C_l_des[i] == C_l_des_corr[i]:
                ax1.axvline(C_d_des_corr[i], ls="-.", c=colors[1], lw=1.8,
                            label="Design corrected")
                ax1.axhline(C_l_des_corr[i], ls="-.", c=colors[1], lw=1.8,
                            label="_")
            ax1.axvline(C_d_thres[i], ls=":", c=colors[2], lw=2,
                        label="Stability threshold")
            ax1.axhline(C_l_thres[i], ls=":", c=colors[2], lw=2,
                        label="_")
            thres_rng_rect = mpl.patches.Rectangle(
                (ax1.get_xlim()[0], C_l_thres[i]-ex_limit),
                ax1.get_xlim()[1] - ax1.get_xlim()[0], 2*ex_limit,
                ec='none', fc=colors[2])
            thres_rng_rect.set_alpha(0.25)
            ax1.add_patch(thres_rng_rect)

            fig, ax1, _ = scivis.plot_line(
                aoa_enhanced, C_l_enhanced[i], ax=ax2,
                ax_labels=[r"\alpha", "C_l"], ax_units=["deg", None],
                plt_labels=["_"],
                ax_lims=[aoa_lims, None],
                ax_ticks=[aoa_ticks, None], ax_show_minor_ticks=True,
                latex=LATEX, profile="halfsize",
                overflow=False, savefig=False)
            ax2.axvline(aoa_des[i], ls="--", c=colors[0], lw=1.8, label="_")
            ax2.axhline(C_l_des[i], ls="--", c=colors[0], lw=1.8, label="_")
            if not C_l_des[i] == C_l_des_corr[i]:
                ax2.axvline(aoa_des_corr[i], ls="-.", c=colors[1], lw=1.8,
                            label="_")
                ax2.axhline(C_l_des_corr[i], ls="-.", c=colors[1], lw=1.8,
                            label="_")
            ax2.axvline(aoa_thres[i], ls=":", c=colors[2], lw=2, label="_")
            ax2.axhline(C_l_thres[i], ls=":", c=colors[2], lw=2, label="_")
            thres_rng_rect = mpl.patches.Rectangle(
                (ax2.get_xlim()[0], C_l_thres[i]-ex_limit),
                ax2.get_xlim()[1] - ax2.get_xlim()[0], 2*ex_limit,
                ec='none', fc=colors[2])
            thres_rng_rect.set_alpha(0.25)
            ax2.add_patch(thres_rng_rect)

            fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0),
                       ncols=2 if C_l_des[i] == C_l_des_corr[i] else 3)

            if SAVEFIG:
                fig.savefig(Path(ROOT, EXP_FLD,
                                 "Aero_coeffs_" + airfoil_names[i] + ".svg"))

# Get Cl, Cd and AoA design functions using: get_design_functions
N_DES_FUNC = 3
des_funcs = [get_design_functions(i+1) for i in range(N_DES_FUNC)]
des_func_keys = ["cl_des", "cd_des", "aoa_des", "tc_vals", "cl_vals",
                 "cd_vals", "aoa_vals"]
des_funcs = {des_func_keys[i]: [des_funcs[0][i],
                                des_funcs[1][i],
                                des_funcs[2][i]]
             for i in range(7)}

# Plot the C_l curve including the design functions and the preliminary design
if SHOW_PLOT:
    colors = cmap_custom(np.linspace(0, 1, N_AIRFOILS))

    fig, ax, fpath = scivis.plot_line(
        aoa, C_l,
        ax_labels=[r"\alpha", "C_l"], ax_units=[r"\degree", None],
        plt_labels=airfoil_names,
        ax_show_minor_ticks=True,
        ax_lims=[[0, 20], None],
        colors=colors, linewidths=1.7, linestyles="-",
        latex=LATEX, profile="partsize", scale=FIG_SCALE,
        overflow=False, savefig=False)
    plt.show()

    # Plot the three design function & the preliminary design
    markers = ("+", "x", "d", ".")
    aoa_funcs = np.stack([des_funcs["aoa_vals"][i] for i in range(3)])
    Cl_funcs = np.stack([des_funcs["cl_vals"][i] for i in range(3)])

    for j in range(N_AIRFOILS):
        for i in range(N_DES_FUNC):
            # Plot the three design function
            ax.scatter(aoa_funcs[i, j], Cl_funcs[i, j],
                       color=colors[j], marker=markers[i], s=100)

        # Plot preliminary design point
        ax.scatter(aoa_max_glide[j], C_l_max_glide[j], marker=markers[-1],
                   s=150, color=colors[j])

    # Add a second legend for the markers
    rc_profile = scivis.rcparams._prepare_rcparams(latex=LATEX,
                                                   profile="partsize",
                                                   scale=FIG_SCALE,)
    with mpl.rc_context(rc_profile):
        ax.add_artist(ax.get_legend())

        legend_handles = [
            Line2D([0], [0], color='black', marker=markers[i], linestyle='',
                   markersize=8, label=f"Design Function {i+1}")
            for i in range(N_DES_FUNC)
        ]
        legend_handles.append(
            Line2D([0], [0], color='black', marker=markers[-1], linestyle='',
                   markersize=10, label="Preliminary Design"))

        ax.legend(handles=legend_handles, loc="lower center")

    if SAVEFIG:
        fig.savefig(Path(ROOT, EXP_FLD,
                         "C_l_vs_alpha_design_functions_all.svg"))

# Chosen design values: Design curve 1
idx_des = 0
aoa_des_func = des_funcs["aoa_des"][idx_des]
cl_des_func = des_funcs["cl_des"][idx_des]
cd_des_func = des_funcs["cd_des"][idx_des]

# Plot chosen Design curve over relative thickness
if SHOW_PLOT:
    tcr_plot = np.arange(0, 100, 1)
    aoa_des_plot = aoa_des_func(tcr_plot)
    cl_des_plot = cl_des_func(tcr_plot)
    cd_des_plot = cd_des_func(tcr_plot)

    # Plot C_l vs tcr
    fig, ax, fpath = scivis.plot_line(
        tcr_plot, cl_des_plot,
        ax_labels=[r"t/r", "C_l"],
        ax_lims=[None, [0, (np.max(C_l_max_glide)//.1+1)/10]],
        autoscale_y=False,
        ax_show_minor_ticks=True,
        latex=LATEX, profile="partsize", scale=FIG_SCALE,
        overflow=False, savefig=False)

    ax.scatter(t_airfoils, C_l_des_corr, color="k", marker="+", s=150)

    if SAVEFIG:
        fig.savefig(Path(ROOT, EXP_FLD, "C_l_vs_tcr_design.svg"))

    # Plot C_l/C_d vs tcr
    fig, ax, fpath = scivis.plot_line(
        tcr_plot, cl_des_plot/cd_des_plot,
        ax_labels=[r"t/r", "C_{l}/C_{d}"],
        ax_lims=[None, [0, np.ceil(np.max(C_ld_max))]],
        autoscale_y=False,
        ax_show_minor_ticks=True,
        latex=LATEX, profile="partsize", scale=FIG_SCALE,
        overflow=False, savefig=False)

    ax.scatter(t_airfoils, C_ld_des_corr, color="k", marker="+", s=150)

    if SAVEFIG:
        fig.savefig(Path(ROOT, EXP_FLD, "C_ld_vs_tcr_design.svg"))

    # Plot AoA vs tcr
    fig, ax, fpath = scivis.plot_line(
        tcr_plot, aoa_des_plot,
        ax_labels=[r"t/r", r"\alpha"], ax_units=[None, "deg"],
        ax_lims=[None, [0, np.ceil(np.max(aoa_des_plot))]],
        autoscale_y=False,
        ax_show_minor_ticks=True,
        latex=LATEX, profile="partsize", scale=FIG_SCALE,
        overflow=False, savefig=False)

    ax.scatter(t_airfoils, aoa_des_corr, color="k", marker="+", s=150)

    if SAVEFIG:
        fig.savefig(Path(ROOT, EXP_FLD, "Alpha_vs_tcr_design.svg"))


# Determine optimal TSR
# Use single single_point_design to get CP for varying TSR (using a range from
# TSR=5.5-10 should be sufficient)
tsr = np.arange(5.5, 10.01, .1)
ds_shape = (len(r), len(tsr), N_DES_FUNC)

ds_spo = xr.Dataset(
    {
        "chord": (["r", "tsr", "func"], np.empty(ds_shape)),
        "tc": (["r", "tsr", "func"], np.empty(ds_shape)),
        "twist": (["r", "tsr", "func"], np.empty(ds_shape)),
        "cl": (["r", "tsr", "func"], np.empty(ds_shape)),
        "cd": (["r", "tsr", "func"], np.empty(ds_shape)),
        "aoa": (["r", "tsr", "func"], np.empty(ds_shape)),
        "a": (["r", "tsr", "func"], np.empty(ds_shape)),
        "CLT": (["r", "tsr", "func"], np.empty(ds_shape)),
        "CLP": (["r", "tsr", "func"], np.empty(ds_shape)),
        "CT": (["r", "tsr", "func"], np.empty(ds_shape)),
        "CP": (["r", "tsr", "func"], np.empty(ds_shape)),
    },
    coords={
        "r": r,
        "tsr": tsr,
        "func": [1, 2, 3],
    },
)

for tsr_i in tsr:
    for func in range(N_DES_FUNC):
        blade_design = single_point_design(
            r+R_HUB, t, tsr_i, R,
            des_funcs["cl_des"][func], des_funcs["cd_des"][func],
            des_funcs["aoa_des"][func],
            c_root, c_max, B
        )

        for key, value in blade_design.items():
            ds_spo[key].loc[dict(tsr=tsr_i, func=func+1)] = value

# Plot: show how CP varies with TSR. Indicated your chosen design-point
C_P_max = np.max(ds_spo["CP"].sel(r=0, func=idx_des+1).values)
idx_CPmax = np.argwhere(ds_spo["CP"].sel(r=0, func=idx_des+1).values == C_P_max).item()
TSR_max = round(tsr[idx_CPmax], 1)

TS_MAX = 90  # Maximum allowed tip speed [m/s]
if GEN_SPEED_MAX_ORIG/60 * (2*np.pi*R) / GEAR_RATIO_ORIG > TS_MAX:
    GEN_SPEED_MAX = TS_MAX / (2*np.pi*R) * GEAR_RATIO_ORIG * 60
    GEN_SPEED_MAX = GEN_SPEED_MAX // 0.01 / 100  # Round to two decimal points
else:
    GEN_SPEED_MAX = GEN_SPEED_MAX_ORIG

if True:
    fig, ax, fpath = scivis.plot_line(
        tsr, ds_spo["CT"].sel(r=0).values.T,
        ax_labels=[r"TSR", "C_T"],
        plt_labels=[f"Design Function {i+1}" for i in range(N_DES_FUNC)],
        ax_show_minor_ticks=True,
        ax_lims=[(tsr[0], tsr[-1]), None],
        cmap=cmap_custom,
        linewidths=1.7, linestyles="-",
        latex=LATEX, profile="partsize", scale=FIG_SCALE,
        overflow=False, savefig=False)

    ax.axvline(TSR_max, ls="--", c="k", lw=2)
    ax.axhline(C_P_max, ls="--", c="k", lw=2)

    if SAVEFIG:
        fig.savefig(Path(ROOT, EXP_FLD, "C_P_vs_TSR.svg"))

# Get your new rotor design using: single_point_design
rotor_design = {}
for key in ds_spo.keys():
    rotor_design[key] = ds_spo[key][:, idx_CPmax, idx_des+1].values

np.save("array_aoa.npy", ds_spo['aoa'].sel(tsr=7,func=1,method='nearest').values)
np.save("array_cl.npy", ds_spo['cl'].sel(tsr=7,func=1,method='nearest').values)
np.save("array_cd.npy", ds_spo['cd'].sel(tsr=7,func=1,method='nearest').values)
np.save("array_r.npy", r)
# %% Scale structural data
# Scale the structural data using: scale_ST_data
st_data_flex = load_st(ORIG_BLADE_ST, 0, 0)
st_data_stiff = load_st(ORIG_BLADE_ST, 0, 1)
st_data_flex_scaled = scale_ST_data(st_data_flex, sc_bld)
st_data_stiff_scaled = scale_ST_data(st_data_stiff, sc_bld)

# Define new ae_data and st_data (two sets: rigid and flexible)
ae_data_scaled = np.vstack([r, rotor_design["chord"],
                            rotor_design["tc"], np.ones_like(r)]).T

# Save your ae file and st file
save_ae(TARGET_AE, ae_data_scaled)
save_st(TARGET_BLADE_ST, [st_data_flex_scaled, st_data_stiff_scaled])
# Note: Export deactivated bc it removes the column names


# Load the blade1's c2_def block from the DTU 10 MW htc file.
# Scale (x, y, z) by your blade scale factor and update the twist. (Note that<
# you will need to interpolate your twist to the different stations in c2_def.
# And don't forget the correct sign for the twist!)
def signif(x: ArrayLike, p: int) -> np.ndarray:
    """
    Rounds each element in a numpy array to a specified number of significant
    digits.
    Taken from https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy

    Parameters
    ----------
    x : ArrayLike
        Input array.
    p : int
        Number of significant digits.

    Returns
    -------
    np.ndarray
        Numpy array with the rounded values.

    """
    if not isinstance(p, int):
        raise TypeError("Number of significant digits must be an integer")

    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


bld_c2def_ORIG = load_c2def(ORIG_HTC, bodyname="blade1")
bld_c2def_scaled = bld_c2def_ORIG.copy()
bld_c2def_scaled[:, 0:3] = bld_c2def_scaled[:, 0:3]*sc_bld
bld_c2def_scaled[:, 3] = np.interp(bld_c2def_scaled[:, 2], r,
                                   -rotor_design["twist"])
bld_c2def_scaled = signif(bld_c2def_scaled, 6)

if SHOW_PLOT:
    colors = cmap_custom(np.linspace(0, 1, 4))

    # Plot final twist
    fig, ax, fpath = scivis.plot_line(
        bld_c2def_scaled[:, 2]/l_bld,
        np.vstack([bld_c2def_ORIG[:, 3], bld_c2def_scaled[:, 3]]),
        ax_labels=[r"\hat{s}", r"\beta"],
        plt_labels=["Original", "Interpolated scaled"],
        ax_show_minor_ticks=True,
        colors=[colors[0], colors[2]], linestyles="-",
        latex=LATEX, profile="partsize", scale=FIG_SCALE,
        overflow=False,
        exp_fld=EXP_FLD, fname="beta_vs_s_final", savefig=SAVEFIG)

    fig, ax, fpath = scivis.plot_line(
        s_hat, np.vstack([c_ORIG, rotor_design["chord"]]),
        ax_labels=[r"\hat{s}", "c"],
        plt_labels=["Original", "Scaled"],
        ax_show_minor_ticks=True,
        colors=[colors[0], colors[2]], linestyles="-",
        # cmap=cmap_custom, linestyles="-",
        latex=LATEX, profile="partsize", scale=FIG_SCALE,
        overflow=False,
        exp_fld=EXP_FLD, fname="c_vs_s_final", savefig=SAVEFIG)

    fig, ax, fpath = scivis.plot_line(
        s_hat, np.vstack([tc_ORIG, rotor_design["tc"]]),
        ax_labels=[r"\hat{s}", "t/c"],
        plt_labels=["Original", "Scaled"],
        ax_show_minor_ticks=True,
        colors=[colors[0], colors[2]], linestyles="-",
        # cmap=cmap_custom, linestyles="-",
        latex=LATEX, profile="partsize", scale=FIG_SCALE,
        overflow=False,
        exp_fld=EXP_FLD, fname="tc_vs_s_final", savefig=SAVEFIG)


# %% Create new HTC file
# Set the blade1 c2_def block using: htc.set_main_body_c2_def_axis
# Update the blade ST-filename using:
# htc.new_htc_structure.main_body(name="blade1").timoschenko_input.filename
# Update the AE-filename using: htc.aero.ae_filename
# Update operational data for generator-speed and optimal TSR using:
# htc.hawcstab2.operational_data
# save the htc file using: htc._update_name_and_save(..., name=DESIGN_NAME).

htc = lacbox.htc.HTCFile(ORIG_HTC, ORIG_HTC.parent)
htc.new_htc_structure.main_body(name="blade1").timoschenko_input.filename = \
    "./" + str(TARGET_BLADE_ST.relative_to(ROOT))
htc.aero.ae_filename = "./" + str(TARGET_AE.relative_to(ROOT))
htc.hawcstab2.operational_data.opt_lambda.values = [TSR_max]
htc.hawcstab2.operational_data.genspeed.values[1] = GEN_SPEED_MAX
htc.save(TARGET_HTC)

htc = MyHTC(TARGET_HTC)
htc.set_main_body_c2_def_axis(*bld_c2def_scaled.T, mbdy_name="blade1")
htc._update_name_and_save(TARGET_HTC.parent, ".htc", name=DESIGN_NAME)
