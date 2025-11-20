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
#size
mpl.rcParams['figure.figsize'] = (16,8)

#font size of label, title, and legend
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 35
mpl.rcParams['ytick.labelsize'] = 35
mpl.rcParams['axes.labelsize'] = 50
mpl.rcParams['axes.titlesize'] = 30
mpl.rcParams['legend.fontsize'] = 45

#Lines and markers
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 9
mpl.rcParams['scatter.marker'] = "+"
plt_marker = "d"

#Latex font
plt.rcParams['font.family'] = 'serif'  # Simula il font di LaTeX
plt.rcParams['mathtext.fontset'] = 'cm'  # Usa Computer Modern per la matematica

#Export
mpl.rcParams['savefig.bbox'] = "tight"

#%% file names and paths
DESIGN_NAME = "IEC_Ya_Later"
DTU_NAME = 'dtu_10mw'
ROOT = Path(__file__).parent  # define root as folder where this script is located
MASTER_FILE = ROOT / '_master' / f'{DESIGN_NAME}.htc'  # your master htc file
TARGET_DIR = ROOT / 'htc_hawc2s'  # where to save the htc files this script will make
PWR_PATH = ROOT / 'res_hawc2s' / f'{DESIGN_NAME}_hawc2s_multitsr.pwr'
PWR_PATH_DTU = ROOT / 'res_hawc2s' / f'{DTU_NAME}_hawc2s_multitsr.pwr'
# We have to keep the rotational speed to the opt(TSR)*rated_ws/tip_radius
GENSPEED = (0, 403.1)  # minimum and maximum generator speed [rpm]
IND_PATH = ROOT / 'res_hawc2s'
#%% 1-wsp and multiTSR htc file generation
# make rigid hawc2s file for single-wsp opt file
htc = MyHTC(MASTER_FILE)
htc.make_hawc2s(TARGET_DIR,
                rigid=True,
                append='_hawc2s_1wsp',
                opt_path=f'./data/{DESIGN_NAME}_1wsp.opt',
                compute_steady_states=True,
                save_power=False,
                save_induction=True,
                genspeed=GENSPEED)

# make rigid hawc2s file for multi-tsr opt file
htc = MyHTC(MASTER_FILE)
htc.make_hawc2s(TARGET_DIR,
                rigid=True,
                append='_hawc2s_multitsr',
                opt_path=f'./data/{DESIGN_NAME}_multitsr.opt',
                compute_steady_states=True,
                save_power=True,
                genspeed=GENSPEED)

# saving the ind file
OPT_TSR = 7
ind_data = {}
complete_name = IND_PATH /f"{DESIGN_NAME}_hawc2s_1wsp_u8000.ind"
ind_data[OPT_TSR]= load_ind(complete_name)
# extracting information from the .ind files for different TSR
r = ind_data[OPT_TSR]['s_m']
a= ind_data[OPT_TSR]['a']
alpha = ind_data[OPT_TSR]['aoa_rad']
cl = ind_data[OPT_TSR]['Cl']
glide = ind_data[OPT_TSR]['Cl']/ind_data[OPT_TSR]['Cd']
cp = ind_data[OPT_TSR]['CP']
ct = ind_data[OPT_TSR]['CT']
alpha_loaded = np.load("array_aoa.npy")
cl_loaded = np.load("array_cl.npy")
cd_loaded = np.load("array_cd.npy")
r_loaded = np.load("array_r.npy")

# plot Cl, Cl/Cd, and AoA along the blade
fig,ax = plt.subplots(1,1)
ax.plot(r, cl, color='k',linestyle='-',label='HAWC2S results' )
ax.plot(r_loaded, cl_loaded, color='k',linestyle='--',label='design rotor')
ax.set_xlabel(r'$s\: [m]$')
ax.set_ylabel(r'$C_l$')
# ax.set_title('Lift coefficient along the Blade for different TSR', fontsize=20)
ax.legend(loc='lower right')
ax.set_xlim(min(r), max(r))
ax.grid()
plt.savefig(f"plots/local_cl.pdf")

fig,ax = plt.subplots(1,1)
ax.plot(r, glide, color='k',linestyle='-',label='HAWC2S results' )
ax.plot(r_loaded, cl_loaded/cd_loaded, color='k',linestyle='--',label='design rotor')
ax.set_xlabel(r'$s\: [m]$')
ax.set_ylabel(r'$C_l/C_d$')
# ax.set_title('Lift coefficient along the Blade for different TSR', fontsize=20)
ax.legend(loc='lower right')
ax.set_xlim(min(r), max(r))
ax.grid()
plt.savefig(f"plots/local_glide.pdf")

fig,ax = plt.subplots(1,1)
ax.plot(r, np.rad2deg(alpha), color='k',linestyle='-',label='HAWC2S results' )
ax.plot(r_loaded, alpha_loaded, color='k',linestyle='--',label='design rotor')
ax.set_xlabel(r'$s\: [m]$')
ax.set_ylabel(r'$\alpha$')
# ax.set_title('Lift coefficient along the Blade for different TSR', fontsize=20)
ax.legend(loc='upper right')
ax.set_xlim(min(r), max(r))
ax.grid()
plt.savefig(f"plots/local_aoa.pdf")

fig,ax = plt.subplots(1,1)
ax.plot(r, a, color='k',linestyle='-' )
ax.set_xlabel(r'$s\: [m]$')
ax.set_ylabel(r'a [-]')
ax.set_xlim(min(r), max(r))
ax.grid()
plt.savefig(f"plots/local_axial_induction.pdf")

fig,ax = plt.subplots(1,1)
ax.plot(r, cp, color='k',linestyle='-' )
ax.set_xlabel(r'$s\: [m]$')
ax.set_ylabel(r'$Local\: C_p\: [-]$')
ax.set_xlim(min(r), max(r))
ax.grid()
plt.savefig(f"plots/local_cp.pdf")

fig,ax = plt.subplots(1,1)
ax.plot(r, ct, color='k',linestyle='-' )
ax.set_xlabel(r'$s\: [m]$')
ax.set_ylabel(r'$Local\: C_T\:[-]$')
ax.set_xlim(min(r), max(r))
ax.grid()
plt.savefig(f"plots/local_ct.pdf")

# saving the pwr data in a dict
pwr_data = load_pwr(PWR_PATH)
pwr_data_dtu = load_pwr(PWR_PATH_DTU)
TSR = [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
cp_pwr = pwr_data['Cp']
ct_pwr = pwr_data['Ct']

# plot CP and CT versus TSR
fig,ax1 = plt.subplots(1,1, figsize=(15,10))
ax2 = ax1.twinx()
ax1.plot(TSR, ct_pwr, color='k', label='Ct', marker='o',linestyle='-')
ax2.plot(TSR, cp_pwr, color='gray', label='Cp', marker='x',linestyle='--')
ax1.set_xlabel('TSR')
ax1.set_ylabel(r'$C_T$', color='k')
ax2.set_ylabel(r'$C_p$', color='gray')
# ax1.set_title('Rotor thrust and power coefficients for different TSRs', fontsize=20)
#ax.legend(fontsize=20, loc='upper left')
ax1.set_xlim(min(TSR), max(TSR))
ax1.grid()
plt.savefig(f"plots/cp_ct_vs_tsr.pdf")
#%% Rigid htc file generation
# rigid hawc2s file
htc = MyHTC(MASTER_FILE)
htc.make_hawc2s(TARGET_DIR,
                rigid=True,
                append='_hawc2s_rigid',
                opt_path=f'./data/{DESIGN_NAME}_rigid.opt',
                compute_steady_states=False,
                save_power=True,
                save_induction=False,
                compute_optimal_pitch_angle=True,
                minipitch=0,
                opt_lambda=OPT_TSR,
                genspeed=GENSPEED)

ws_des, pitch_des, omega_des, power_des, thrust_des = np.loadtxt(f'./res_hawc2s/{DESIGN_NAME}_hawc2s_rigid.opt',skiprows=1,unpack=True)
ws, pitch, omega, power, thrust = np.loadtxt(f'./data/{DTU_NAME}_rigid.opt',skiprows=1,unpack=True)
# saving the pwr data in a dict
pwr_data = load_pwr(f'./res_hawc2s/{DESIGN_NAME}_hawc2s_rigid.pwr')
Cp_des = pwr_data['Cp']
Cp = power*1e3/(0.5*1.225*np.pi*89.17**2*ws**3)
CT_des = pwr_data['Ct']
CT = thrust*1e3/(0.5*1.225*np.pi*89.17**2*ws**2)

# plot power and thrust for our design and DTU's one
fig,ax = plt.subplots(1,1)
ax.plot(ws_des, power_des/1e3, color='k',linestyle='-',label='design rotor' )
ax.plot(ws, power/1e3, color='k',linestyle='--',label='DTU 10 MW' )
ax.set_xlabel(r'$V_0\: [m/s]$')
ax.set_ylabel(r'$P\: [MW]$')
ax.legend(loc='lower right')
ax.set_xlim(min(ws_des), max(ws_des))
ax.grid()
plt.savefig(f"plots/design_power.pdf")

# thrust
fig,ax = plt.subplots(1,1)
ax.plot(ws_des, thrust_des/1e3, color='k',linestyle='-',label='design rotor' )
ax.plot(ws, thrust/1e3, color='k',linestyle='--',label='DTU 10 MW' )
ax.set_xlabel(r'$V_0\: [m/s]$')
ax.set_ylabel(r'$T\: [MN]$')
ax.legend(loc='upper right')
ax.set_xlim(min(ws_des), max(ws_des))
ax.grid()
plt.savefig(f"plots/design_thrust.pdf")

# plot Cp
fig,ax = plt.subplots(1,1)
ax.plot(ws_des, Cp_des, color='k',linestyle='-',label='design rotor' )
ax.plot(ws, Cp, color='k',linestyle='--',label='DTU 10 MW' )
ax.set_xlabel(r'$V_0\: [m/s]$')
ax.set_ylabel(r'$C_p\: [-]$')
ax.legend(loc='upper right')
ax.set_xlim(min(ws_des), max(ws_des))
ax.grid()
plt.savefig(f"plots/design_cp.pdf")

# plot ct
fig,ax = plt.subplots(1,1)
ax.plot(ws_des, CT_des, color='k',linestyle='-',label='design rotor' )
ax.plot(ws, CT, color='k',linestyle='--',label='DTU 10 MW' )
ax.set_xlabel(r'$V_0\: [m/s]$')
ax.set_ylabel(r'$C_T\: [-]$')
ax.legend(loc='upper right')
ax.set_xlim(min(ws_des), max(ws_des))
ax.grid()
plt.savefig(f"plots/design_ct.pdf")

# pitch
fig,ax = plt.subplots(1,1)
ax.plot(ws_des, pitch_des, color='k',linestyle='-',label='design rotor' )
ax.plot(ws, pitch, color='k',linestyle='--',label='DTU 10 MW' )
ax.set_xlabel(r'$V_0\: [m/s]$')
ax.set_ylabel(r'$\theta_p\: [^\circ]$')
ax.legend(loc='lower right')
ax.set_xlim(min(ws_des), max(ws_des))
ax.grid()
plt.savefig(f"plots/design_pitch.pdf")

#rot speed
fig,ax = plt.subplots(1,1)
ax.plot(ws_des, omega_des, color='k',linestyle='-',label='design rotor' )
ax.plot(ws, omega, color='k',linestyle='--',label='DTU 10 MW' )
ax.set_xlabel(r'$V_0\: [m/s]$')
ax.set_ylabel(r'$\omega\: [rpm]$')
ax.legend(loc='lower right')
ax.set_xlim(min(ws_des), max(ws_des))
ax.grid()
plt.savefig(f"plots/design_omega.pdf")

#tip speed
fig,ax = plt.subplots(1,1)
ax.plot(ws_des, omega_des*pwr_data['Tip_z_m'][0]*np.pi/30, color='k',linestyle='-',label='design rotor' )
ax.plot(ws, omega*pwr_data_dtu['Tip_z_m'][0]*np.pi/30, color='k',linestyle='--',label='DTU 10 MW' )
ax.set_xlabel(r'$V_0\: [m/s]$')
ax.set_ylabel(r'$Tip\:speed\: [m/s]$')
ax.legend(loc='lower right')
ax.set_xlim(min(ws_des), max(ws_des))
ax.grid()
plt.savefig(f"plots/design_tip_speed.pdf")

# flexible hawc2s file

# controller tuning hawc2s files

# ...and more?
plt.show(block=True)
