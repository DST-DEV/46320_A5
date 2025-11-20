"""Use control tuning files to make step HAWC2 files.

This file should create the following htc files:
 * _fX_dY_C[T/P].htc (X files)

Note that you need to create some functions in myteampack that should
be called in this script.
"""
from collections import namedtuple
from pathlib import Path

import numpy as np

from lacbox.io import load_ctrl_txt
from myteampack import MyHTC

# define folders
ROOT = Path(__file__).parent
DESIGN_NAME = "IEC_Ya_Later"
MASTER_FILE = ROOT / '_master' / f'{DESIGN_NAME}.htc'  # your master htc file
RES_DIR = ROOT / Path('./res_hawc2s_ctrl/')  # location of _ctrl_tunint.txt files
HAWC2_HTC_DIR = ROOT / Path('./htc_hawc2/htc_step/')  # where to save the step-wind files

# Rated speed
rpm_HSS = 403.1  # HSS rated rotational speed [rpm]
n_gear = 50  # Gear ratio
omega_rtd_LSS = rpm_HSS/n_gear*2*np.pi/60

# step-wind settings
CUTIN, CUTOUT = 4, 25
DT, TSTART = 100, 100

# get a list of all _ctrl_tuning.txt files...
ctrl_tuning_file = namedtuple("ctrl_tuning_file", ["omega", "zeta", "params"])
# omega = [.05, .01, .1]*2
# zeta = [.7]*6
# CTCP = ["CP"]*3 + ["CT"]*3

omega = [.05]
zeta = [.7]
CTCP = ["CP"]

rpm_HSS = 403.1   # HSS rated rotational speed [rpm]
n_gear = 50
omega_rtd_LSS = rpm_HSS/n_gear*2*np.pi/60

ctrl_basepath = DESIGN_NAME + \
    "_hawc2s_flex_ctrltune_{CPCT}_f{omega:g}_Z{zeta:g}_ctrl_tuning.txt"
ctrl_tuning_files = [
    ctrl_tuning_file(o, z,
                     load_ctrl_txt(Path(RES_DIR, ctrl_basepath.format(CPCT=cpct,
                                                                      omega=o,
                                                                      zeta=z)))
                     )
    for cpct, o, z in zip(CTCP, omega, zeta)]

# Create HTC files
wsp_steps = np.arange(CUTIN+1, CUTOUT+1, 1)
step_times = np.arange(1, len(wsp_steps)+1, 1)*DT + TSTART
t_end = step_times[-1] + DT
for i, ctrl_tuning_file in enumerate(ctrl_tuning_files):
    # load the master htc file
    htc = MyHTC(MASTER_FILE)

    # update controller block in htc file
    htc._update_ctrl_params(ctrl_tuning_file.params,
                            rated_rot_speed=omega_rtd_LSS)

    # generate a step-wind HAWC2 file
    # htc.make_step(save_dir=HAWC2_HTC_DIR,
    #               wsp=4, wsp_steps=wsp_steps, step_times=step_times,
    #               last_step_len=DT, start_record_time=TSTART,
    #               append=f"_hawc2s_flex_ctrleval_C{i+1}")
    htc.make_step(save_dir=HAWC2_HTC_DIR,
                  wsp=4, wsp_steps=wsp_steps, step_times=step_times,
                  last_step_len=DT, start_record_time=TSTART,
                  append=f"_hawc2s_flex_ctrleval_C{1}")
