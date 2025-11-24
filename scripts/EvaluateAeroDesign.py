from pathlib import Path
import json

from lacbox.io import load_ind, load_pwr, load_oper
import matplotlib as mpl
import numpy as np

import scivis

# %% User inputs
DESIGN_NAME = "IEC_What_You_Did_There"
DESIGN_NAME_OLD = "IEC_Ya_Later"
DTU_NAME = "dtu_10mw"

eval_1wsp = False
eval_multitsr = True
eval_rigid = False
eval_flex = False

save_plots = False
exp_fld = "plots/aero_eval"  # Figure export path( relative to project root)

colors_2line = ['#0d0887', '#e97a54']
colors_3line = ["#0D0887", "#CB4679", "#FFB300"]

# %% File path preparation
ROOT = Path(__file__).parent.parent
RES_DIR = ROOT / "hawc_files" / "res" / "hawc2s"
DATA_DIR = ROOT / "hawc_files" / "data"

SAVE_PLOTS_PATH  = ROOT / exp_fld
SAVE_PLOTS_PATH.mkdir(parents=True, exist_ok=True)

OPT_PATH_RIGID_DTU = DATA_DIR / f"{DTU_NAME}_rigid.opt"
OPT_PATH_FLEX_DTU = DATA_DIR / f"{DTU_NAME}_flex.opt"

OPT_PATH_RIGID_A5 = RES_DIR / f"{DESIGN_NAME}_hawc2s_rigid.opt"
OPT_PATH_FLEX_A5 = RES_DIR / f"{DESIGN_NAME}_hawc2s_flex.opt"
IND_PATH_A5 = RES_DIR / f"{DESIGN_NAME}_hawc2s_1wsp_u8000.ind"
PWR_MULTITSR_PATH_A5 = RES_DIR / f"{DESIGN_NAME}_hawc2s_multitsr.pwr"
PWR_RIGID_PATH_A5 = RES_DIR / f"{DESIGN_NAME}_hawc2s_rigid.pwr"
PWR_FLEX_PATH_A5 = RES_DIR / f"{DESIGN_NAME}_hawc2s_flex.pwr"
AERO_PATH_A5 = DATA_DIR / f"{DESIGN_NAME}_aero_params.npz"
DES_PARAMS_PATH_A5 = DATA_DIR / f"{DESIGN_NAME}_params.json"

OPT_PATH_RIGID_A4 = RES_DIR / f"{DESIGN_NAME_OLD}_hawc2s_rigid.opt"
OPT_PATH_FLEX_A4 = RES_DIR / f"{DESIGN_NAME_OLD}_hawc2s_flex.opt"
IND_PATH_A4 = RES_DIR / f"{DESIGN_NAME_OLD}_hawc2s_1wsp_u8000.ind"
PWR_MULTITSR_PATH_A4 = RES_DIR / f"{DESIGN_NAME_OLD}_hawc2s_multitsr.pwr"
PWR_RIGID_PATH_A4 = RES_DIR / f"{DESIGN_NAME_OLD}_hawc2s_rigid.pwr"
PWR_FLEX_PATH_A4 = RES_DIR / f"{DESIGN_NAME_OLD}_hawc2s_flex.pwr"
AERO_PATH_A4 = DATA_DIR / f"{DESIGN_NAME_OLD}_aero_params.npz"
DES_PARAMS_PATH_A4 = DATA_DIR / f"{DESIGN_NAME_OLD}_params.json"

# Read design parameters from a_make_blade_design.py
aero_params_A5 = dict(np.load(AERO_PATH_A5))
aero_params_A4 = dict(np.load(AERO_PATH_A4))

with open(DES_PARAMS_PATH_A5, 'r') as file: des_params_A4 = json.load(file)
with open(DES_PARAMS_PATH_A4, 'r') as file: des_params_A5 = json.load(file)

# DTU 10 MW design parameters (from documentation)
des_params_dtu = {"R": 178.3/2, "V_rtd": 11.4, "tsr_des": 8,
                  "omega_gen_rtd_rpm": 480,
                  "omega_rot_rtd_rads": 9.6*2*np.pi/60}


# %% Plot Aerodynamic parameters
with open(DES_PARAMS_PATH_A5, "r") as file:
    des_params_A5 = json.load(file)

if eval_1wsp:
    # Load induction results
    ind_A4 = load_ind(IND_PATH_A4)
    ind_A5 = load_ind(IND_PATH_A5)

    s_m_ind_A4 = ind_A4['s_m']
    s_m_ind_A5 = ind_A5['s_m']
    s_m_des_A5 = aero_params_A5["r"]

    # Plot C_l, C_l/C_d and AoA
    aero_params_A5["C_ld"] = aero_params_A5["C_l"] / aero_params_A5["C_d"]
    ind_A5["Cld"] = ind_A5["Cl"] / ind_A5["Cd"]
    ind_A5["aoa_deg"] = np.rad2deg(ind_A5["aoa_rad"])
    params_compare_1wsp = {"C_l":["C_l", "Cl"], "C_l/C_d": ["C_ld", "Cld"],
                           r"\alpha": ["aoa_deg", "aoa_deg"]}
    units_dict = {r"\alpha": "deg"}
    for ylabel, keys in params_compare_1wsp.items():
        fig, ax, _ = scivis.plot_line(s_m_des_A5, aero_params_A5[keys[0]],
                                      colors=colors_2line[0],
                                      ax_labels=["s", ylabel],
                                      ax_units=["m", units_dict.get(ylabel)],
                                      plt_labels=["Design Rotor"],
                                      show_legend=False)
        scivis.plot_line(s_m_ind_A5, ind_A5[keys[1]], ax=ax,
                         colors=colors_2line[1], linestyles="-",
                         plt_labels=["HAWC2S Results"],
                         ax_show_grid_minor=True,
                         show_legend=True
                         )

        if save_plots:
            fig.savefig(SAVE_PLOTS_PATH
                        / (f"{ylabel.replace('\\', '').replace('/', '')}"
                           + "_1wsp_vs_design.svg"))

    # Compare axial induction, local C_p and local C_T of A5 vs A4
    params_A5_A4_1wsp = {"a": "a", "C_P": "CP", "C_T": "CT"}
    for ylabel, key in params_A5_A4_1wsp.items():
        fig, ax, _ = scivis.plot_line(s_m_ind_A5, ind_A5[key],
                                      colors=colors_2line[0],
                                      ax_labels=["s", ylabel],
                                      plt_labels=["Design Rotor V2"],
                                      show_legend=False)
        scivis.plot_line(s_m_ind_A4, ind_A4[key], ax=ax,
                         colors=colors_2line[1], linestyles="-",
                         plt_labels=["Design Rotor V1"],
                         ax_show_grid_minor=True,
                         show_legend=True
                         )

        if save_plots:
            fig.savefig(SAVE_PLOTS_PATH / f"{ylabel}_1wsp_vs_A1.svg")

    del fig, ax

if eval_multitsr:
    # Load power results
    pwr_multitsr_A4 = load_pwr(PWR_MULTITSR_PATH_A4)
    pwr_multitsr_A5 = load_pwr(PWR_MULTITSR_PATH_A5)

    # Calculate tsr values
    tsr_A4 = np.round(pwr_multitsr_A4["Speed_rpm"]*2*np.pi/60*des_params_A4["R"]
                      / pwr_multitsr_A4["V_ms"], 2)
    tsr_A4 = np.round(tsr_A4/.05, 0)*.05  # Round to 0.05 precision

    tsr_A5 = np.round(pwr_multitsr_A5["Speed_rpm"]*2*np.pi/60*des_params_A5["R"]
                      / pwr_multitsr_A5["V_ms"], 2)
    tsr_A5 = np.round(tsr_A5/.05, 0)*.05  # Round to 0.05 precision

    # Plot C_p and C_t vs tsr
    rc_profile = scivis.rcparams._prepare_rcparams(profile="partsize",
                                                   scale=.7)
    with mpl.rc_context(rc_profile):
        fig, ax_l = scivis.subplots(profile="partsize", scale=.7,
                                    figsize=(20,8))
        ax_r = ax_l.twinx()
        ax_r.zorder = 2

        ax_l.plot(tsr_A5, pwr_multitsr_A5["Cp"], label="$C_p$ - Design V2",
                  ls="-", c=colors_2line[0], marker=".", zorder=3)
        ax_l.plot(tsr_A4, pwr_multitsr_A4["Cp"], label="$C_p$ - Design V1",
                  ls="-", c=colors_2line[0], marker="x", ms=10, zorder=3)
        ax_r.plot(tsr_A5, pwr_multitsr_A5["Ct"], label="$C_T$ - Design V2",
                  ls="-", c=colors_2line[1], marker=".", zorder=3)
        ax_r.plot(tsr_A4, pwr_multitsr_A4["Ct"], label="$C_T$ - Design V1",
                  ls="-", c=colors_2line[1], marker="x", ms=10, zorder=3)

        tsr_opt_A4 = tsr_A4[np.argmax(pwr_multitsr_A4["Cp"])]
        tsr_opt_A5 = tsr_A5[np.argmax(pwr_multitsr_A5["Cp"])]

        ax_l.axvline(tsr_opt_A4, ls="-.", lw=1.2, c="0.2", zorder=2)
        ax_l.axvline(tsr_opt_A5, ls="-.", lw=1.2, c="0.2", zorder=2)

        ax_l.set_xlabel(r"$\lambda$")
        ax_l.set_ylabel(r"$C_p$")
        ax_r.set_ylabel(r"$C_T$")

        ax_l.set_xticks(np.arange(tsr_A5[0], tsr_A5[-1]+.1, .5))

        ax_l.minorticks_on()
        ax_r.minorticks_on()
        ax_l.grid(which="both", zorder=1)
        ax_r.grid(which="both", visible=False)

        h_l, l_l = ax_l.get_legend_handles_labels()
        h_r, l_r = ax_r.get_legend_handles_labels()
        ax_l.legend(h_l+h_r, l_l+l_r, ncols=1, loc="center left",
                    bbox_to_anchor=(1.15, .5))

        # Change color of axes
        ax_l.tick_params(axis='y', which = "both", colors=colors_2line[0])
        ax_l.yaxis.label.set_color(colors_2line[0])
        ax_l.spines['left'].set_color(colors_2line[0])
        ax_r.tick_params(axis='y', which = "both", colors=colors_2line[1])
        ax_r.yaxis.label.set_color(colors_2line[1])
        ax_r.spines['right'].set_color(colors_2line[1])

        if save_plots:
            fig.savefig(SAVE_PLOTS_PATH / "C_P_multitsr.svg")

    del rc_profile, fig, ax_l, ax_r, h_l, h_r, l_l, l_r

if eval_rigid:
    # Load operational data
    opt_rigid_dtu = load_oper(OPT_PATH_RIGID_DTU)
    opt_rigid_A4 = load_oper(OPT_PATH_RIGID_A4)
    opt_rigid_A5 = load_oper(OPT_PATH_RIGID_A5)

    plot_params_rigid = {r"\theta_p": "pitch_deg",
                         r"\omega": "rotor_speed_rpm",
                         "P_{el}": "power_kw",
                         r"T": "thrust_kn"}
    plot_units_rigid = {r"\theta_p": ["deg", 1],
                        r"\omega": ["rpm", 1],
                        "P_{el}": ["MW", 1e-3],
                        r"T": ["MN", 1e-3]}

    for ylabel, key in plot_params_rigid.items():
        unit = plot_units_rigid[ylabel][0]
        scale_factor = plot_units_rigid[ylabel][1]

        fig, ax, _ = scivis.plot_line(opt_rigid_dtu["ws_ms"],
                                      np.vstack([opt_rigid_A5[key],
                                                 opt_rigid_A4[key],
                                                 opt_rigid_dtu[key]]
                                                )*scale_factor,
                                      colors=colors_3line, linestyles="-",
                                      ax_labels=["V_0", ylabel],
                                      ax_units=["m/s", unit],
                                      plt_labels=["Design Rotor V2",
                                                  "Design Rotor V1",
                                                  "DTU 10 MW"],
                                      ax_show_grid_minor=True
                                      )

    if save_plots:
        fig.savefig(SAVE_PLOTS_PATH / f"{ylabel.replace('\\', '')}_rigid.svg")

if eval_flex:
    # Load operational data
    opt_flex_dtu = load_oper(OPT_PATH_FLEX_DTU)
    opt_flex_A4 = load_oper(OPT_PATH_FLEX_A4)
    opt_flex_A5 = load_oper(OPT_PATH_FLEX_A5)

    # Calculate C_P and C_T from operational data
    def C_p (P_el_kW, R_m, s_ms_m, rho=1.225):
        return P_el_kW*1e3 / (0.5 * rho * np.pi*R_m**2 * s_ms_m**3)
    def C_t (T_kN, R_m, s_ms_m, rho=1.225):
        return T_kN*1e3 / (0.5 * rho * np.pi*R_m**2 * s_ms_m**2)

    opt_flex_A4["Cp"] = C_p(opt_flex_A4["power_kw"], des_params_A4["R"],
                            opt_flex_A4["ws_ms"])
    opt_flex_A5["Cp"] = C_p(opt_flex_A5["power_kw"], des_params_A5["R"],
                            opt_flex_A5["ws_ms"])
    opt_flex_dtu["Cp"] = C_p(opt_flex_dtu["power_kw"], des_params_dtu["R"],
                            opt_flex_dtu["ws_ms"])

    opt_flex_A4["Ct"] = C_t(opt_flex_A4["power_kw"], des_params_A4["R"],
                            opt_flex_A4["ws_ms"])
    opt_flex_A5["Ct"] = C_t(opt_flex_A5["power_kw"], des_params_A5["R"],
                            opt_flex_A5["ws_ms"])
    opt_flex_dtu["Ct"] = C_t(opt_flex_dtu["power_kw"], des_params_dtu["R"],
                             opt_flex_dtu["ws_ms"])

    plot_params_flex = {r"\theta_p": "pitch_deg",
                         "P_{el}": "power_kw",
                         r"T": "thrust_kn",
                         "C_p": "Cp",
                         "C_t": "Ct"}
    plot_units_flex = {r"\theta_p": ["deg", 1],
                        "P_{el}": ["MW", 1e-3],
                        r"T": ["MN", 1e-3],
                        "C_p": ["", 1],
                        "C_t": ["", 1]}

    for ylabel, key in plot_params_flex.items():
        unit = plot_units_flex[ylabel][0]
        scale_factor = plot_units_flex[ylabel][1]

        fig, ax, _ = scivis.plot_line(opt_flex_dtu["ws_ms"],
                                      np.vstack([opt_flex_A5[key],
                                                 opt_flex_A4[key],
                                                 opt_flex_dtu[key]]
                                                )*scale_factor,
                                      colors=colors_3line, linestyles="-",
                                      ax_labels=["V_0", ylabel],
                                      ax_units=["m/s", unit],
                                      plt_labels=["Design Rotor V2",
                                                  "Design Rotor V1",
                                                  "DTU 10 MW"],
                                      ax_show_grid_minor=True
                                      )

    if save_plots:
        fig.savefig(SAVE_PLOTS_PATH / f"{ylabel.replace('\\', '')}_flex.svg")
