import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from myteampack import MyHTC
from lacbox.io import load_ind, load_pwr, load_ctrl_txt

DESIGN_NAME = "IEC_Ya_Later"
DTU_NAME = 'dtu_10mw'
ROOT = Path(__file__).parent  # define root as folder where this script is located
MASTER_FILE = ROOT / '_master' / f'{DESIGN_NAME}.htc'  # your master htc file
TARGET_DIR = ROOT / 'htc_hawc2s_ctrl'  # where to save the htc files this script will make
FLEX_PWR_PATH = ROOT / 'res_hawc2s' / 'IEC_Ya_Later_hawc2s_flex.pwr'
RIGID_PWR_PATH = ROOT / 'res_hawc2s' / 'IEC_Ya_Later_hawc2s_rigid.pwr'
GENSPEED = (0, 403.1)  # minimum and maximum generator speed [rpm]
IND_PATH = ROOT / 'res_hawc2s'

if True: #Make HTC
    OPT_TSR = 7
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s_ctrltune(TARGET_DIR,
                            rigid=False,

                            append='_hawc2s_flex_ctrltune_CP_f0.05_Z0.6',
                            opt_path=f'./data/{DESIGN_NAME}_flex.opt',
                            compute_steady_states=True,
                            compute_controller_input = True,
                            save_power=True,
                            save_induction=False,
                            minipitch=0,
                            opt_lambda=OPT_TSR,
                            genspeed=GENSPEED,
                            partial_load=(0.05, 0.7),
                            full_load=(0.01, 0.7),
                            gain_scheduling=2, 
                            constant_power=1
                            )
    
if False: #Part 1 calculations
    pwr_dict = load_pwr(ROOT / 'res_hawc2s' / 'IEC_Ya_Later_hawc2s_flex.pwr')

    ctrltune_dict = load_ctrl_txt(ROOT / 'res_hawc2s' / 'IEC_Ya_Later_hawc2s_flex_ctrltune_ctrl_tuning.txt')

    theta = ctrltune_dict['aero_gains'].index.to_numpy()
    dqdt = ctrltune_dict['aero_gains']['dq/dtheta_kNm/deg'].to_numpy()*1000
    dqdo = ctrltune_dict['aero_gains']['dq/domega_kNm/(rad/s)'].to_numpy()*1000
    fit_dqdt = np.polyfit(theta,dqdt,2)
    fitted = np.polyval(fit_dqdt,theta)

    plt.figure()
    plt.plot(theta,dqdt)
    plt.plot(theta,fitted)



    I_rotor = pwr_dict['J_DT_kgm2'][0]
    I_Gen = 0
    eta = 1e4 / pwr_dict['P_kW'][-1]  # Efficiency
    n = 50
    ctrl_f = 0.05 * 2 * np.pi
    zeta = 0.7
    lambda_opt = 7
    CP_max = 0.4657
    R = 92.52

    Kopt = eta * 1.225 * np.pi * R**5 * CP_max / (2 * lambda_opt**3)
    KP_g = eta * (2*(I_rotor + n**2*I_Gen) * ctrl_f * zeta)
    KI_g = eta * (I_rotor + n**2 * I_Gen) * ctrl_f**2 
    KP_p = (2 * zeta * ctrl_f * (I_rotor + n**2 * I_Gen) - 1/eta * dqdo[0]) / ( - dqdt[0]*180/np.pi)
    KI_p = ctrl_f**2 * (I_rotor) / ( - dqdt[0]*180/np.pi)
    KK1 = 1/(fit_dqdt[1]/fit_dqdt[-1])
    KK2 = 1/(fit_dqdt[0]/fit_dqdt[-1])
    aero_gain = fit_dqdt[-1] + fit_dqdt[1] * theta[0] + fit_dqdt[0] * theta[0]**2

