from pathlib import Path
import json
import re
import random

from lacbox.htc import _clean_directory
from lacbox.io import load_ctrl_txt

from myteampack import MyHTC

# %% User inputs
DESIGN_NAME = "IEC_What_You_Did_There"
wsp = 24
nseeds = 10
minspeed = 0  # Minimum rotor speed [rad/s]

# %% Path preparations and file loading
ROOT = Path(__file__).parent
MASTER_FILE = ROOT / '_master' / f'{DESIGN_NAME}.htc'
CTRL_DIR = ROOT / 'res' / 'hawc2s' / 'ctrltune'
OPT_PATH_FLEX = ROOT / 'data' / f'{DESIGN_NAME}_flex.opt'
RES_DIR = ROOT / 'res' / 'hawc2' / 'ctrltune_eval'  # HTC save folder
HTC_DIR = Path('./htc/hawc2/ctrleval/')  # where to save the step-wind files

TI_REF = .14  # Turbulence intensities for IEC  B
TI = TI_REF*(0.75*wsp+5.6)/wsp
TIME_START = 100
TIME_STOP = 700
START_SEED = 42  # initialize the random-number generator for reproducability

# Load design parameters
with open(ROOT / "data" / f"{DESIGN_NAME}_params.json", 'r') as file:
    des_params = json.load(file)

# Read all control tuning files
ctrl_tune_files = []
for file in CTRL_DIR.iterdir():
    if file.name.endswith("ctrl_tuning.txt"):
        ctrl_tune_files.append(file)

if len(ctrl_tune_files) == 0:
    raise OSError("No control tuning files found.")

# clean the top-level htc directory if requested
_clean_directory(HTC_DIR, True)

# Create turbulent htc files
for ctrl_tune_file in ctrl_tune_files:
    ctrl_params = load_ctrl_txt(ctrl_tune_file)
    freq, damp = re.findall(r"f(0.\d+)_Z(0.\d+)", ctrl_tune_file.name)[0]

    for idx_seed in range(nseeds):
        sim_seed = random.randrange(int(2**16))
        htc = MyHTC(MASTER_FILE)

        # update controller block in htc file
        htc._update_ctrl_params(
            ctrl_params, min_rot_speed=minspeed,
            rated_rot_speed=des_params["omega_rot_rtd_rads"])

        # Update rotor diameter in controller dll
        r_hub = htc.new_htc_structure.main_body__4.c2_def.sec__2.values[-2]
        l_bld = htc.new_htc_structure.main_body__7.c2_def.sec__27.values[-2]
        r_rotor = r_hub + l_bld
        htc.dll.type2_dll__1.init.constant__47 = [47, 2*r_rotor]

        htc.make_turb(wsp=wsp, ti=TI, seed=sim_seed,
                      append=f"_f{freq}_Z{damp}_{sim_seed:d}",
                      save_dir=HTC_DIR, subfolder="",
                      opt_path=OPT_PATH_FLEX, resdir=RES_DIR,
                      dy=190, dz=190,
                      rigid=False, withdrag=True, tilt=None,
                      time_start=TIME_START, time_stop=TIME_STOP)
