"""Make HAWC2S files from a master htc file.

This file should create the following htc files:
 * _hawc2s_1wsp.htc
 * _hawc2s_multitsr.htc
 * _hawc2s_rigid.htc
 * _hawc2s_flex.htc
 * _hawc2s_ctrltune_fX_dY_C[T/P].htc
 * ...and more?

We recommend saving the HAWC2S files in a dedicated subfolder. If you
do this, note that you will need to move the htc before running HAWCStab2.

Requires myteampack (which requires lacbox).
"""
from pathlib import Path
import json
import shutil

from lacbox.io import save_oper
import numpy as np

from myteampack import MyHTC

# %% User inputs
DESIGN_NAME = "IEC_What_You_Did_There"
DTU_NAME = 'dtu_10mw'
tsr_oper = -1  # Operational tsr [-1 if design tsr should be used]
nat_freq = np.arange(.04, .06, .002)
damp_ratio = np.arange(.6, .9, .05)

create_tsr_analysis = False  # Whether 1wsp and multitsr files should be created
create_rigid = False  # Whether rigid simulation files should be created
create_flex = False  # Whether flexible simulation files should be created
create_modal_struct = False  # Whether the structural modal analysis files should be created
create_modal_aeroelastic = False  # Whether the structural modal analysis files should be created
save_modal_amp = True  # Whether the modal amplitudes should be saved
create_ctrl_tuning = True  # Whether controller turning files should be created

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

# %% Structural modal analysis
if create_modal_struct:
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=False,
                    append='_hawc2s_modal_structural',
                    opt_path=f'./res/hawc2s/{DESIGN_NAME}_hawc2s_flex.opt',
                    compute_structural_modal_analysis=True,
                    save_modal_amplitude=save_modal_amp,
                    minipitch=0,
                    opt_lambda=tsr_oper,
                    genspeed=GENSPEED)

# %% Aeroelastic modal analysis
if create_modal_aeroelastic:
    htc = MyHTC(MASTER_FILE)
    htc.make_hawc2s(TARGET_DIR,
                    rigid=False,
                    append='_hawc2s_modal_aeroelastic',
                    opt_path=f'./res/hawc2s/{DESIGN_NAME}_hawc2s_flex.opt',
                    compute_steady_states=True,
                    compute_stability_analysis=True,
                    save_modal_amplitude=save_modal_amp,
                    minipitch=0,
                    opt_lambda=tsr_oper,
                    genspeed=GENSPEED)

if create_ctrl_tuning:
    ctrl_dir = TARGET_DIR / "ctrltune"

    # Clear the current folder
    if ctrl_dir.is_dir():
        print(f'! Folder {ctrl_dir} exists: deleting contents. !')
        shutil.rmtree(ctrl_dir) # delete the folder
        ctrl_dir.mkdir(parents=True)  # make an empty folder
    elif not ctrl_dir.is_dir():  # if the folder doesn't exists
        ctrl_dir.mkdir(parents=True)  # make the folder

    for f in nat_freq:
        for d in damp_ratio:
            htc = MyHTC(MASTER_FILE)
            fname = f"_hawc2s_ctrltune_CT_f{f:5.3f}_Z{d:4.2f}"
            htc.make_hawc2s_ctrltune(
                TARGET_DIR / "ctrltune", append=fname,
                output_folder="res/hawc2s/ctrltune",
                opt_path=f'./res/hawc2s/{DESIGN_NAME}_hawc2s_flex.opt',
                rigid=False,
                compute_steady_states=True,
                compute_controller_input = True,
                save_power=True,
                save_induction=False,
                minipitch=0, opt_lambda=des_params["tsr_des"],
                genspeed=GENSPEED,
                partial_load=(0.05, 0.7), full_load=(f, d),
                gain_scheduling=2, constant_power=0)
