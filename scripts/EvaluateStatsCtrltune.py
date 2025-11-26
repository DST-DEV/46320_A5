from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr

from lacbox.io import load_stats
import scivis

# %% User input
fname_stats = "IEC_What_You_Did_There_ctrleval_turb_stats"

eval_oper = True
eval_DELs = True

show_plots = True
save_plots = True
exp_fld = "plots/stats_eval"  # Figure export path( relative to project root)

# %% Data preparation

# File paths
ROOT = Path(__file__).parent.parent
STATS_PATH = ROOT / 'hawc_files' / 'stats' / (fname_stats + '.csv')
SAVE_PLOTS_PATH  = ROOT / exp_fld
SAVE_PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Load statistics
h2stats, _ = load_stats(STATS_PATH)

# Retrieve the control tuning parameters:
h2stats['f'] = h2stats.filename.str.extract(r"_f(0.\d+)")
h2stats['d'] = h2stats.filename.str.extract(r"_Z(0.\d+)")
h2stats['seed'] = h2stats.filename.str.extract(r"_(\d+).hdf5")

# Channels to evaluate
chan_ids_op_data = ['BldPit', 'RotSpd', 'Thrust', 'GenTrq', 'ElPow']
chan_id_loads = ['TbFA', 'TbSS','YbTilt', 'YbRoll', 'ShftTrs', 'OoPBRM',
                 'IPBRM']
sn_slopes_dict = {'TbFA': 4, 'TbSS': 4,'YbTilt': 4, 'YbRoll': 4, 'ShftTrs': 4,
                  'OoPBRM': 10, 'IPBRM': 10}
CHAN_DESCS = {'BldPit': 'pitch1 angle',
            'RotSpd': 'rotor speed',
            'Thrust': 'aero rotor thrust',
            'GenTrq': 'generator torque',
            'ElPow': 'pelec',
            'TbFA': 'momentmx mbdy:tower nodenr:   1',
            'TbSS': 'momentmy mbdy:tower nodenr:   1',
            'YbTilt': 'momentmx mbdy:tower nodenr:  11',
            'YbRoll': 'momentmy mbdy:tower nodenr:  11',
            'ShftTrs': 'momentmz mbdy:shaft nodenr:   4',
            'OoPBRM': 'momentmx mbdy:blade1 nodenr:   1 coo: hub1',
            'IPBRM': 'momentmy mbdy:blade1 nodenr:   1 coo: hub1',
            'FlpBRM': 'momentmx mbdy:blade1 nodenr:   1 coo: blade1',
            'EdgBRM': 'momentmy mbdy:blade1 nodenr:   1 coo: blade1',
            'OoPHub': 'momentmx mbdy:hub1 nodenr:   1 coo: hub1',
            'IPHub': 'momentmy mbdy:hub1 nodenr:   1 coo: hub1'
                }

# Prepare datasets for the statistics
base_stats = ["min", "mean", "max", "std"]
N_base_stats = len(base_stats)
sn_slopes = (4, 10)
freqs = np.sort(h2stats['f'].unique())
damps = np.sort(h2stats['d'].unique())

# Check number of wind bins (assuming they are the same for A5 & the DTU 10 MW)
N_sim = len(h2stats[(h2stats.desc=='pitch1 angle')
                     & (h2stats.f==h2stats.f[0])
                     & (h2stats.d==h2stats.d[0])].index)

# Prepare load channels
if eval_oper:
    if eval_DELs:
        ds_stats_keys = chan_ids_op_data + chan_id_loads
        SCAL_FACTOR = 1.35
        SAFE_FACTOR = 1.25
    else:
        ds_stats_keys = chan_ids_op_data
else:
    ds_stats_keys = chan_id_loads

ds_stats_shape = (len(freqs), len(damps), N_base_stats + 1, N_sim)

ds_stats_raw = xr.Dataset(
    {
        chan_id: (["f", "d", "stat", "sim"],
                  np.empty(ds_stats_shape))
        for chan_id in ds_stats_keys
     },
    coords={
        "f": freqs,
        "d": damps,
        "stat": base_stats + ["del10min"],
        "sim": np.arange(N_sim)
    },
)

ds_stats_eval = xr.Dataset(
    {
        chan_id: (["f", "d", "stat",],
                  np.empty(ds_stats_shape[:-1]))
        for chan_id in ds_stats_keys
     },
    coords={
        "f": freqs,
        "d": damps,
        "stat": base_stats + ["del1h"]
    },
)

# Retrieve statistics for control tuning combination
chan_units = {}
for ch in ds_stats_keys:
    if ch in chan_id_loads:
        load_ch = True
        m_chan = sn_slopes_dict[ch]
    else:
        load_ch = False

    stats_chan = h2stats.filter_channel(
        ch, CHAN_DESCS).sort_values(["f", "d"]).reset_index(drop=True)

    for f in freqs:
        stats_chan_f = stats_chan[stats_chan.f == f]

        chan_units[ch] = stats_chan_f["units"].values[0]
        chan_d = stats_chan_f["d"].unique()

        ds_stats_raw[ch].loc[{"f": f, "d": chan_d, "stat": base_stats}] \
            = np.swapaxes(stats_chan_f[base_stats].to_numpy().reshape(
                len(chan_d), N_sim, N_base_stats), 1, 2)

        if load_ch:
            dels_10min = stats_chan_f[f"del{m_chan}"].to_numpy().reshape(
                len(chan_d), N_sim)
            ds_stats_raw[ch].loc[{"f": f, "d": chan_d, "stat": "del10min"}
                                 ] = dels_10min

            dels_1h = np.sum(dels_10min**m_chan/N_sim, axis=-1)**(1/m_chan)
            ds_stats_eval[ch].loc[{"f": f, "d": chan_d, "stat": "del1h"}
                                  ] = dels_1h

if show_plots:

    fig, ax = mpl.pyplot.subplots(figsize=(6, 5))

    # Show heatmap
    im = ax.imshow(ds_stats_eval["TbFA"].sel(stat="del1h").values.T,
                   cmap="viridis", aspect="auto")

    # Show colorbar
    mpl.pyplot.colorbar(im, ax=ax, label="DEL 1h")

    # Set categorical tick labels
    ax.set_xticks(np.arange(len(freqs)))
    ax.set_yticks(np.arange(len(damps)))

    ax.set_xticklabels(freqs)
    ax.set_yticklabels(damps)

    ax.set_xlabel(r"$f\:[Hz]$")
    ax.set_xlabel(r"$d$")

    # Rotate x labels if needed
    mpl.pyplot.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add grid between cells (optional)
    ax.set_xticks(np.arange(len(freqs)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(damps)+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
