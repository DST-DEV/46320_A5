"""Make steady or turbulent htc files for cluster.

Steady wind settings:
    * No shear, tower shadow method is 0, turb_format is 0

Turbulence settings:
    * Power-law shear with 0.2 exponent, tower shadow method is 3,
      turb_format is 1, etc.
"""
from pathlib import Path
import random

import numpy as np

from lacbox.htc import _clean_directory
from lacbox.io import load_ctrl_txt, load_oper
from myteampack import MyHTC

# Select what to create
create_steady = False
create_turb = True
tow_rot_excl = True  # whether to use the rotational speed exclusion zone to mitigate tower resonance
# low_storm_control = False  # Whether the wsp for the storm control should be lowered
model = "IEC_Ya_Later"  # "dtu_10mw" or "IEC_Ya_Later"
special_case = "tower_exclusion_v6"

# define folders
ROOT = Path(__file__).parent
DESIGN_NAME = model
MASTER_FILE = ROOT / '_master' / f'{DESIGN_NAME}.htc'  # your master htc file
OPT_PATH = Path('data') / f'{DESIGN_NAME}_flex.opt'
DEL_HTC_DIR = True  # delete htc directory if it already exists?

# settings for both steady and turbulence
WSPS = range(5, 25)

# ---------- STEADY WIND ----------
if create_steady:
    STEADY_HTC_DIR = Path(f'./htc/steady/{DESIGN_NAME}/')  # where to save the step-wind files
    RES_DIR = Path(f'./res/steady/{DESIGN_NAME}/')
    CASES = ['tilt', 'notilt', 'notiltrigid', 'notiltnodragrigid']  # ['tilt', 'notilt', 'notiltrigid', 'notiltnodragrigid']
    TIME_START = 200
    TIME_STOP = 400

    # clean the top-level htc directory if requested
    _clean_directory(STEADY_HTC_DIR, DEL_HTC_DIR)

    # make the steady wind files
    for case in CASES:
        TILT = None  # default: don't change tilt
        RIGID = False  # default: flexible blades and tower
        WITHDRAG = True
        if 'notilt' in case:
            TILT = 0
        if 'rigid' in case:
            RIGID = True
        if 'nodrag' in case:
            WITHDRAG = False
        # generate the files
        for wsp in WSPS:
            append = f'_steady_{case}_{wsp:04.1f}'  # fstring black magic! zero-fill with 1 decimal: e.g., '_steady_05.0'
            htc = MyHTC(MASTER_FILE)
            htc.make_steady(STEADY_HTC_DIR, wsp, append, opt_path=OPT_PATH, resdir=RES_DIR,
                            tilt=TILT, subfolder=case, rigid=RIGID, withdrag=WITHDRAG,
                            time_start=TIME_START, time_stop=TIME_STOP)

# ---------- TURBULENT WIND ----------

if create_turb:
    TURB_HTC_DIR = Path(f'./htc/turb/{DESIGN_NAME}/')  # where to save the step-wind files
    RES_DIR = Path(f'./res/turb/{DESIGN_NAME}/')
    CASES = ['tca', 'tcb']
    TI_REF = {"tca": .16, "tcb": .14}  # Turbulence intensities for IEC A & B
    TIME_START = 100
    TIME_STOP = 700
    START_SEED = 42  # initialize the random-number generator for reproducability
    NSEEDS = 6

    if model == "IEC_Ya_Later":
        # Rated speed
        rpm_rtd_HSS = 403.1  # HSS rated rotational speed [rpm]
        rpm_min_HSS = 0  # HSS minimum rotational speed [rpm]
        n_gear = 50  # Gear ratio
        omega_rtd_LSS = rpm_rtd_HSS/n_gear*2*np.pi/60
        omega_min_LSS = rpm_min_HSS/n_gear*2*np.pi/60

        # Load final controller tuning (C1 - omega=0.05, zeta=0.7, constant power)
        ctrl_path = Path(ROOT, "control", DESIGN_NAME +
                         "_hawc2s_flex_ctrltune_CP_f0.05_Z0.7_ctrl_tuning.txt")
        ctrl_tuning_file = load_ctrl_txt(ctrl_path)

        # Tower resonance exclusion zone
        if tow_rot_excl:
            op_data = load_oper(OPT_PATH)
            op_data["rotor_speed_rad/s"] = \
                op_data["rotor_speed_rpm"]*2*np.pi/60
            op_data["GenTrq_kNm"] =  op_data["power_kw"] \
                / op_data["rotor_speed_rad/s"]

            mask_below_rated = op_data["power_kw"]<1e4

            omega_L_Hz = .15/3  # Lower exclusion zone rotational speed limit [Hz]
            omega_H_Hz = .3/3  # Upper exclusion zone rotational speed limit [Hz]

            omega_L_rads = omega_L_Hz*2*np.pi
            omega_H_rads = omega_H_Hz*2*np.pi


            Q_gL_opt, Q_gH_opt = np.interp(
                np.array([omega_L_Hz, omega_H_Hz])*60,
                op_data["rotor_speed_rpm"][mask_below_rated],
                op_data["GenTrq_kNm"][mask_below_rated]*1e3)

            Q_gL = 1.07*Q_gH_opt
            Q_gH = .93*Q_gL_opt

        # Default values
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
    import warnings

    omega_L_rads = 0.39
    omega_H_rads = 0.75

    Q_gL = 9.5e6
    Q_gH = 3e6

    warnings.warn("TOWER EXCLUSION ZONE OVERWRITTEN WITH EXAMPLE VALUES!")
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

    random.seed(START_SEED)

    # clean the top-level htc directory if requested
    _clean_directory(TURB_HTC_DIR / special_case, DEL_HTC_DIR)

    # Make the turbulent wind files
    for idx_seed in range(6):
        for wsp in WSPS:
            sim_seed = random.randrange(int(2**16))
            for tc in CASES:
                htc = MyHTC(MASTER_FILE)

                if model == "IEC_Ya_Later":
                    # update controller block in htc file
                    htc._update_ctrl_params(ctrl_tuning_file,
                                            min_rot_speed=omega_min_LSS,
                                            rated_rot_speed=omega_rtd_LSS)

                    # Update rotor diameter in controller dll
                    r_hub = htc.new_htc_structure.main_body__4.c2_def.sec__2.values[-2]
                    l_bld = htc.new_htc_structure.main_body__7.c2_def.sec__27.values[-2]
                    r_rotor = r_hub + l_bld
                    htc.dll.type2_dll__1.init.constant__47 = [47, 2*r_rotor]

                    # if low_storm_control:
                    #     # Alter storm control
                    #     htc.dll.type2_dll__1.init.constant__43 = [43, 22]
                    #     htc.dll.type2_dll__1.init.constant__44 = [44, 22]


                    if tow_rot_excl:
                        htc.dll.type2_dll__1.init.constant__2 = [2, 0]  # Remove minimum rotor speed
                        htc.make_tower_exclusion_zone(omega_L=omega_L_rads,
                                                      omega_H=omega_H_rads,
                                                      Q_gL=Q_gL, Q_gH=Q_gH,
                                                      tau=25)

                subfolder = special_case + "/" + tc if special_case else tc
                htc.make_turb(wsp=wsp, ti=TI_REF[tc]*(0.75*wsp+5.6)/wsp,
                              seed=sim_seed,
                              append=f"_turb_{tc}_{wsp:04.1f}_{sim_seed:d}",
                              save_dir=TURB_HTC_DIR,
                              subfolder=subfolder,
                              opt_path=OPT_PATH, resdir=RES_DIR,
                              dy=190, dz=190,
                              rigid=False, withdrag=True, tilt=None,
                              time_start=TIME_START, time_stop=TIME_STOP)
