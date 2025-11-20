"""Make HAWC2S files from a master htc file.

This file should create the following htc files:
 * _hawc2s_1wsp.htc
 * _hawc2s_multitsr.htc
 * _hawc2s_rigid.htc
 * _hawc2s_flex.htc
 * _hawc2s_ctrltune_fX_dY_C[T/P].htc (7 files)
 * ...and more?

We recommend saving the HAWC2S files in a dedicated subfolder. If you
do this, note that you will need to move the htc before running HAWCStab2.

Requires myteampack (which requires lacbox).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from myteampack import MyHTC
from lacbox.io import load_ind, load_pwr

#%% plot commands
if False:
    #size
    mpl.rcParams['figure.figsize'] = (14,7)

    #font size of label, title, and legend
    mpl.rcParams['font.size'] = 40
    mpl.rcParams['xtick.labelsize'] = 30
    mpl.rcParams['ytick.labelsize'] = 30
    mpl.rcParams['axes.labelsize'] = 40
    mpl.rcParams['axes.titlesize'] = 40
    mpl.rcParams['legend.fontsize'] = 25

    #Lines and markers
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 9
    mpl.rcParams['scatter.marker'] = "+"
    plt_marker = "d"

    #Latex font
    plt.rcParams['font.family'] = 'serif'  # Simula il font di LaTeX
    plt.rcParams['mathtext.fontset'] = 'cm'  # Usa Computer Modern per la matematica

    #Export
    mpl.rcParams['savefig.bbox'] = "tight"




#%% flexible htc file generation
from lacbox.io import load_st, load_pwr
DESIGN_NAME = "IEC_Ya_Later"
DTU_NAME = 'dtu_10mw'
ROOT = Path(__file__).parent  # define root as folder where this script is located
MASTER_FILE = ROOT / '_master' / f'{DESIGN_NAME}.htc'  # your master htc file
TARGET_DIR = ROOT / 'htc_hawc2s'  # where to save the htc files this script will make
FLEX_PWR_PATH = ROOT / 'res_hawc2s' / 'IEC_Ya_Later_hawc2s_flex.pwr'
RIGID_PWR_PATH = ROOT / 'res_hawc2s' / 'IEC_Ya_Later_hawc2s_rigid.pwr'
GENSPEED = (0, 403.1)  # minimum and maximum generator speed [rpm]
IND_PATH = ROOT / 'res_hawc2s'

if True: #Make HTC
    OPT_TSR = 7
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=False,
                    append='_hawc2s_flex',
                    opt_path=f'./data/{DESIGN_NAME}_flex.opt',
                    compute_steady_states=True,
                    save_power=True,
                    save_induction=False,
                    minipitch=0,
                    opt_lambda=OPT_TSR,
                    genspeed=GENSPEED)

ws_des, pitch_des, omega_des, power_des, thrust_des = np.loadtxt(ROOT / 'data' / f'{DESIGN_NAME}_rigid.opt', skiprows=1, unpack=True)
ws_dtu, pitch_dtu, omega_dtu, power_dtu, thrust_dtu = np.loadtxt(ROOT / 'data' / f'{DTU_NAME}_rigid.opt', skiprows=1, unpack=True)
ws_des_flex, pitch_des_flex, omega_des_flex, power_des_flex, thrust_des_flex = np.loadtxt(ROOT / 'data' / f'{DESIGN_NAME}_flex.opt', skiprows=1, unpack=True)
ws_dtu_flex, pitch_dtu_flex, omega_dtu_flex, power_dtu_flex, thrust_dtu_flex = np.loadtxt(ROOT / 'data' / f'{DTU_NAME}_flex.opt', skiprows=1, unpack=True)

Design_st_path = ROOT / 'data' / f'{DESIGN_NAME}_Blade_st.dat'
Design_st_data = load_st(Design_st_path)
DTU_st_path = ROOT / 'data' / 'DTU_10MW_RWT_Blade_st.dat'
DTU_st_data = load_st(DTU_st_path)
subset_Design = Design_st_data[0][0]
subset_DTU = DTU_st_data[0][0]

flex_pwr_data = load_pwr(FLEX_PWR_PATH)
pwr_P_flex = flex_pwr_data['P_kW']
pwr_ws_flex = flex_pwr_data['V_ms']
CP_flex = flex_pwr_data['Cp']
CT_flex = flex_pwr_data['Ct']

rigid_pwr_data = load_pwr(RIGID_PWR_PATH)
ws_rigid = rigid_pwr_data['V_ms']
CP_rigid = rigid_pwr_data['Cp']
CT_rigid = rigid_pwr_data['Ct']

dtu_rigid_pwr_data = load_pwr(ROOT / 'res_hawc2s' / 'dtu_10mw_hawc2s_rigid.pwr')
ws_dtu_rigid = dtu_rigid_pwr_data['V_ms']
CP_dtu_rigid = dtu_rigid_pwr_data['Cp']
CT_dtu_rigid = dtu_rigid_pwr_data['Ct']

dtu_flec_pwr_data = load_pwr(ROOT / 'res_hawc2s' / 'dtu_10mw_hawc2s_flex.pwr')
ws_dtu_flex = dtu_flec_pwr_data['V_ms']
CP_dtu_flex = dtu_flec_pwr_data['Cp']
CT_dtu_flex = dtu_flec_pwr_data['Ct']

if False: #Plots
    # plot pitch for our design and DTU's one

    fig,ax = plt.subplots(1,1)
    ax.plot(ws_des, pitch_des, color='g',linestyle='-',label='Rigid design' )
    ax.plot(ws_des_flex,pitch_des_flex, color='g',linestyle='--',label='Flexible design')
    ax.plot(ws_dtu, pitch_dtu, color='k',linestyle='-',label='DTU 10 MW' )
    ax.plot(ws_dtu_flex,pitch_dtu_flex, color='k',linestyle='--',label='DTU 10 MW Flexible')
    ax.set_xlabel(r'$V_0\: [m/s]$')
    ax.set_ylabel(r'$\theta_P\: [^\circ]$')
    ax.legend( loc='upper left')
    ax.set_xlim(min(ws_des), max(ws_des))
    ax.grid()

    fig,ax = plt.subplots(1,1)
    ax.plot(subset_Design['s'], subset_Design['m'], color='k',linestyle='-',label='Design mass' )
    ax.plot(subset_DTU['s'], subset_DTU['m'], color='r',linestyle='--',label='DTU 10 MW mass' )
    ax.set_xlabel('s [m]')
    ax.set_ylabel(r'$m \: [kg/m]$')
    ax.legend(loc='upper right')
    ax.grid()

    fig,ax = plt.subplots(1,1)
    ax.plot(subset_Design['s'], subset_Design['I_x'], color='k',linestyle='-',label='Design Ix' )
    ax.plot(subset_Design['s'], subset_Design['I_y'], color='r',linestyle='-',label='Deisgn Iy' )
    ax.plot(subset_DTU['s'], subset_DTU['I_x'], color='k',linestyle='--',label='DTU 10 MW Ix' )
    ax.plot(subset_DTU['s'], subset_DTU['I_y'], color='r',linestyle='--',label='DTU 10 MW Iy' )
    ax.set_xlabel('s [m]')
    ax.set_ylabel(r'$I \: [m^4]$')
    ax.legend(loc='upper right')
    ax.grid()

    fig,ax = plt.subplots(1,1)
    ax.plot(ws_des, CP_rigid, color='g', linestyle='-', label='Rigid design')
    ax.plot(ws_des_flex, CP_flex, color='g', linestyle='--', label='Flexible design')
    ax.plot(ws_dtu, CP_dtu_rigid, color='k', linestyle='-', label='DTU 10 MW Rigid')
    ax.plot(ws_dtu_flex, CP_dtu_flex, color='k', linestyle='--', label='DTU 10 MW Flexible')
    ax.set_xlabel('Wind Speed [m/s]')
    ax.set_ylabel(r'$CP \: [-]$')
    ax.legend(loc='upper right')
    ax.grid()

    fig,ax = plt.subplots(1,1)
    ax.plot(ws_des, CT_rigid, color='g', linestyle='-', label='Rigid design')
    ax.plot(ws_des_flex, CT_flex, color='g', linestyle='--', label='Flexible design')
    ax.plot(ws_dtu, CT_dtu_rigid, color='k', linestyle='-', label='DTU 10 MW Rigid')
    ax.plot(ws_dtu_flex, CT_dtu_flex, color='k', linestyle='--', label='DTU 10 MW Flexible')
    ax.set_xlabel('Wind speed [m/s]')
    ax.set_ylabel(r'$CT \: [-]$')
    ax.legend(loc='upper right')
    ax.grid()

    fig,ax = plt.subplots(1,1)
    ax.plot(ws_des, power_des/1e3, color='g', linestyle='-', label='Rigid design')
    ax.plot(ws_des_flex, power_des_flex/1e3, color='g', linestyle='--', label='Flexible design')
    ax.plot(ws_dtu, power_dtu/1e3, color='k', linestyle='-', label='DTU 10 MW Rigid')
    ax.plot(ws_dtu_flex, power_dtu_flex/1e3, color='k', linestyle='--', label='DTU 10 MW Flexible')
    ax.set_xlabel(r'$V_0\: [m/s]$')
    ax.set_ylabel(r'$P\: [MW]$')
    ax.legend(loc='lower right')
    ax.set_xlim(min(ws_des_flex), max(ws_des_flex))
    ax.grid()

    fig,ax = plt.subplots(1,1)
    ax.plot(ws_des, thrust_des/1e3, color='g', linestyle='-', label='Rigid design')
    ax.plot(ws_des_flex, thrust_des_flex/1e3, color='g', linestyle='--', label='Flexible design')
    ax.plot(ws_dtu, thrust_dtu/1e3, color='k', linestyle='-', label='DTU 10 MW Rigid')
    ax.plot(ws_dtu_flex, thrust_dtu_flex/1e3, color='k', linestyle='--', label='DTU 10 MW Flexible')
    ax.set_xlabel(r'$V_0\: [m/s]$')
    ax.set_ylabel(r'$T\: [MN]$')
    ax.legend(loc='upper right')
    ax.set_xlim(min(ws_des_flex), max(ws_des_flex))
    ax.grid()

    plt.show()

#%%
from lacbox.io import load_cmb, load_amp
from lacbox.vis import plot_amp

wsp, freq, zeta = load_cmb(ROOT / 'res_hawc2s' / 'IEC_Ya_Later_aeromodes.cmb','aeroelastic')
amp = load_amp(ROOT / 'res_hawc2s' / 'aero.amp')
nmodes = np.size(freq,axis=1)
modenames = ['1st FA tower', '1st STS tower', '1st BW flap', '1st FW flap', '1st sym. flap',
              '1st BW edge', '1st FW edge', '2nd BW flap', '2nd FW flap', '2nd sym. flap',
              '1st sym. edge']

omega = np.vstack([omega_des_flex * i / 60 for i in [1, 3, 6]]).T

def modecolor(modename):
    if 'tower' in modename:
        return 'tab:blue'
    elif 'flap' in modename and '1st' in modename:
        return 'tab:orange'
    elif 'edge' in modename:
        return 'tab:green'
    elif 'flap' in modename and '2nd' in modename:
        return 'tab:red'
    else:
        return 'k'

def marker(modename):
    if 'Tower' in modename:
        return '--'
    elif 'BW' in modename:
        return '<'
    elif 'FW' in modename:
        return '>'
    elif 'sym' in modename:
        return 'x'
    elif 'fore' in modename:
        return 's'
    else:
        return 'o'

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

for i in range(nmodes):
    axs[0].plot(wsp, freq[:, i], marker=marker(modenames[i]), color=modecolor(modenames[i]), label=modenames[i])
    axs[1].plot(wsp, zeta[:, i], marker=marker(modenames[i]), color=modecolor(modenames[i]))

axs[0].plot(wsp, omega, color='k')
axs[0].set(xlabel='Wind speed [m/s]', ylabel='Damped nat. frequencies [Hz]')
axs[0].grid()

axs[1].set(xlabel='Wind speed [m/s]', ylabel='Modal damping [% critical]')
axs[1].grid()

fig.legend(loc='upper center', ncols=6, labels=[*modenames, "Forcing frequencies (1P, 3P, 6P)"], bbox_to_anchor=(0.5, 1))
fig.tight_layout(rect=[0, 0, 1, 0.8])  # Adjust layout to make space for the legend at the top

wsp=11
#fig, ax = plot_amp(amp, modenames, wsp)

plt.show()
