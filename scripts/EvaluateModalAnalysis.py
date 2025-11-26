from pathlib import Path
import json

from lacbox.io import load_cmb, load_amp, load_oper
import matplotlib as mpl
from matplotlib.markers import MarkerStyle
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import numpy as np

import scivis

# %% User inputs
DESIGN_NAME = "IEC_What_You_Did_There"

eval_structural = True  # Whether the structural modal simulation should be analysed
eval_aeroelastic = True  # Whether the aeroelastic modal simulation should be analysed
eval_amp = True # Whether the modal amplitudes should be analysed

save_plots = False
exp_fld = "plots/aero_eval"  # Figure export path( relative to project root)

color_dict = {'Tower S2S': "#3F35E3", 'Tower FA': '#0172cf',
              '1st flap': '#D1200E', '1st edge':'#44a942',
              '2nd flap':'#F5620B'}
marker_dict = {'FW': {"marker": '>', "ms": 9},
               'BW': {"marker": '<', "ms": 9},
               'sym': {"marker": '.', "ms": 13}}
lw = 2.5

# %% File path preparation
ROOT = Path(__file__).parent.parent
RES_DIR = ROOT / "hawc_files" / "res" / "hawc2s"
DATA_DIR = ROOT / "hawc_files" / "data"

CMB_PATH_STRUCTURAL = RES_DIR / f"{DESIGN_NAME}_hawc2s_modal_structural_struc.cmb"
CMB_PATH_AEROELASTIC = RES_DIR / f"{DESIGN_NAME}_hawc2s_modal_aeroelastic.cmb"

AMP_PATH_STRUCTURAL = RES_DIR / f"{DESIGN_NAME}_hawc2s_modal_structural_struc.amp"
AMP_PATH_AEROELASTIC = RES_DIR / f"{DESIGN_NAME}_hawc2s_modal_aeroelastic.amp"

OPT_PATH_FLEX_A5 = RES_DIR / f"{DESIGN_NAME}_hawc2s_flex.opt"

DES_PARAMS_PATH_A5 = DATA_DIR / f"{DESIGN_NAME}_params.json"
with open(DES_PARAMS_PATH_A5, 'r') as file: des_params_A4 = json.load(file)

SAVE_PLOTS_PATH  = ROOT / exp_fld
SAVE_PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# %% Aeroelastic mode shapes
rc_profile = scivis.rcparams._prepare_rcparams(profile="partsize", scale=.7)
rc_profile["axes.labelsize"] = 23
rc_profile["axes.labelpad"] = 5
rc_profile["xtick.labelsize"] = 20
rc_profile["ytick.labelsize"] = 20
rc_profile["font.size"] = 20
rc_profile["legend.fontsize"] = 18

if eval_aeroelastic:
    wsp, freq, zeta = load_cmb(CMB_PATH_AEROELASTIC,'aeroelastic')
    amp = load_amp(AMP_PATH_AEROELASTIC)
    nmodes = np.size(freq, axis=1)
    modenames = ['1st FA tower', '1st S2S tower',
                 '1st BW flap', '1st FW flap', '1st sym. flap',
                  '1st BW edge', '1st FW edge', '2nd BW flap',
                  '2nd FW flap', '2nd sym. flap', '1st sym. edge']
    opt_flex_A5 = load_oper(OPT_PATH_FLEX_A5)


    def modecolor(modename):
        c = "k"  # Default color
        if 'tower' in modename:
            if 'FA'  in modename:
                c =  color_dict['Tower FA']
            elif 'S2S' in modename:
                c =  color_dict['Tower S2S']
        elif 'flap' in modename:
            if '1st' in modename:
                c =  color_dict['1st flap']
            elif '2nd' in modename:
                c =  color_dict['2nd flap']
        elif 'edge' in modename and '1st' in modename:
            c =  color_dict['1st edge']

        return c

    def typemarker(modename):
        if 'tower' in modename or 'sym' in modename:
            return marker_dict['sym']
        elif 'BW' in modename:
            return marker_dict['BW']
        elif 'FW' in modename:
            return marker_dict['FW']
        else:
            return {"marker": None}

    # Plot C_p and C_t vs tsr
    with mpl.rc_context(rc_profile):
        fig, ax = mpl.pyplot.subplots(nrows=1, ncols=2, figsize=(20, 8))

        xticks = np.arange(wsp[0], wsp[-1]+1, 2)
        ax[0].set_xticks(xticks)
        ax[0].set_xlim([wsp[0]-.5, wsp[-1]+.5])
        ax[1].set_xticks(xticks)
        ax[1].set_xlim([wsp[0]-.5, wsp[-1]+.5])

        ax[0].set_ylim([-.05, 1.95])
        ax[1].set_ylim([-2.5, 78])

        for i, label in enumerate(modenames):
            # Determine whether marker should be rotated
            marker = typemarker(label)
            rotate_marker=False
            if marker["marker"] in [">", "<"]:
                rotate_marker = True
                marker = {"marker": None}

            color = modecolor(label)
            # Plot modal data
            ax[0].plot(wsp, freq[:, i], ls="-", lw=lw, c=color, **marker,
                       zorder=3)
            ax[1].plot(wsp, zeta[:, i], ls="-", lw=lw, c=color, **marker,
                       zorder=3)

            if rotate_marker:
                # Determine gradients
                dy_freq = np.gradient(freq[:, i])
                dy_zeta = np.gradient(zeta[:, i])
                dx = np.gradient(wsp)

                angles_freq = np.rad2deg(np.arctan2(dy_freq, dx))
                angles_zeta = np.rad2deg(np.arctan2(dy_zeta, dx))
                if label == '1st FW flap':
                    pass

                # Plot rotated markers
                marker = typemarker(label)
                if marker["marker"] == "<":
                    angles_freq -= 180
                    angles_zeta -= 180

                # Plot rotated markers
                for j in range(len(wsp)):
                    t_freq = Affine2D().rotate_deg(angles_freq[j])
                    t_zeta = Affine2D().rotate_deg(angles_zeta[j])
                    ax[0].plot(wsp[j], freq[j, i],
                               marker=MarkerStyle('>', 'full', t_freq),
                               ms=marker["ms"], c=color, ls='None', zorder=3)
                    ax[1].plot(wsp[j], zeta[j, i],
                               marker=MarkerStyle('>', 'full', t_zeta),
                               ms=marker["ms"], c=color, ls='None', zorder=3)

        ax[0].plot(opt_flex_A5["ws_ms"], opt_flex_A5["rotor_speed_rpm"]/60,
                   ls="-", lw=lw, c='k', zorder=2)
        ax[0].plot(opt_flex_A5["ws_ms"], 3*opt_flex_A5["rotor_speed_rpm"]/60,
                   ls="-", lw=lw, c='k', zorder=2, label="_")
        ax[0].plot(opt_flex_A5["ws_ms"], 6*opt_flex_A5["rotor_speed_rpm"]/60,
                   ls="-", lw=lw, c='k', zorder=2, label="_")

        # Adjust labels, ticks and grid
        ax[0].set_xlabel(r'$V_0\:[m/s]$')
        ax[0].set_ylabel(r'Damped nat. frequencies [Hz]')
        ax[0].grid(which="both", visible="True", zorder=0)
        ax[0].minorticks_on()
        ax[0].tick_params(zorder=4)

        ax[1].set_xlabel(r'$V_0\:[m/s]$')
        ax[1].set_ylabel(r'Modal damping [% critical]')
        ax[1].grid(which="both", visible="True", zorder=0)
        ax[1].set_xticks(xticks)
        ax[1].minorticks_on()
        ax[1].tick_params(zorder=4)

        marker_handles = [Line2D([], [], c="black", ls="",
                                 marker=marker["marker"], ms=marker["ms"]*1.2,
                                 label=whirl_name)
                          for whirl_name, marker in marker_dict.items()]

        color_handles = [Line2D([], [], c=col, ls="-", lw=2, label=col_name)
                         for col_name, col in color_dict.items()]
        color_handles.append(Line2D([], [], c="k", ls="-", lw=2,
                                    label="Forcing frequencies (1P, 3P, 6P)"))

        fig.legend(handles=marker_handles, loc='lower center',
                   ncols=len(marker_handles), bbox_to_anchor=(0.5, 1.08))
        fig.legend(handles=color_handles, loc='lower center',
                   ncols=len(color_handles), bbox_to_anchor=(0.53, 1))
