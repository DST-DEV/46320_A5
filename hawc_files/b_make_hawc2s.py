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
from pathlib import Path
import json
import shutil

from myteampack import MyHTC
import numpy as np

from lacbox.io import save_oper

# %% User inputs
DESIGN_NAME = "IEC_What_You_Did_There"
DTU_NAME = 'dtu_10mw'
tsr_oper = -1  # Operational tsr [-1 if design tsr should be used]

create_tsr_analysis = True  # Whether 1wsp and multitsr files should be created
create_rigid = True  # Whether rigid simulation files should be created
create_flex = True  # Whether flexible simulation files should be created

# %% Path preparations and file loading
ROOT = Path(__file__).parent
MASTER_FILE = ROOT / '_master' / f'{DESIGN_NAME}.htc'
TARGET_DIR = ROOT / 'htc' / 'hawc2s'  # HTC save folder
OPT_FILE_RIGID_TEMPLATE = ROOT / 'data' / f'{DTU_NAME}_rigid.opt'
OPT_FILE_RIGID = ROOT / 'data' / f'{DESIGN_NAME}_rigid.opt'
OPT_FILE_FLEX_TEMPLATE = ROOT / 'data' / f'{DTU_NAME}_flex.opt'
OPT_FILE_FLEX = ROOT / 'data' / f'{DESIGN_NAME}_flex.opt'

# We have to keep the rotational speed to the opt(TSR)*rated_ws/tip_radius
with open(ROOT / "data" / f"{DESIGN_NAME}_params.json", 'r') as file:
    des_params = json.load(file)
GENSPEED = (0, des_params["omega_gen_rtd_rpm"])  # minimum and maximum generator speed [rpm]
if tsr_oper == -1:
    tsr_oper = des_params["tsr_des"]


# %% 1wsp and multi tsr analysis files

if create_tsr_analysis:
    # Create opt files
    tsrs = np.arange(5, 10.1, .2)
    N_tsr = len(tsrs)
    wsp = 8  # Arbitrary below rated wind speed [m/s]
    omega_rot_rpm = tsrs*wsp/des_params["R"] * 60 / (2*np.pi)  # Rotor speed [rpm]
    wsps = np.arange(N_tsr)*0.001 + wsp
    theta = np.zeros(N_tsr)

    save_oper(ROOT / "data" / f"{DESIGN_NAME}_1wsp.opt",
              {"ws_ms": [wsp], "pitch_deg": [0],
               "rotor_speed_rpm": [des_params["tsr_des"] * wsp / des_params["R"]
               * 60 / (2*np.pi)]})
    save_oper(ROOT / "data" / f"{DESIGN_NAME}_multitsr.opt",
              {"ws_ms": wsps, "pitch_deg": theta,
               "rotor_speed_rpm":omega_rot_rpm})

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


#%% Rigid htc file generation
if create_rigid:
    shutil.copy(OPT_FILE_RIGID_TEMPLATE, OPT_FILE_RIGID)  # Create dummy opt file
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
                    opt_lambda=tsr_oper,
                    genspeed=GENSPEED)

#%% Flexible htc file generation
if create_flex:
    shutil.copy(OPT_FILE_FLEX_TEMPLATE, OPT_FILE_FLEX)  # Create dummy opt file
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=False,
                    append='_hawc2s_flex',
                    opt_path=f'./data/{DESIGN_NAME}_flex.opt',
                    compute_steady_states=False,
                    save_power=True,
                    save_induction=False,
                    compute_optimal_pitch_angle=True,
                    minipitch=0,
                    opt_lambda=tsr_oper,
                    genspeed=GENSPEED)
