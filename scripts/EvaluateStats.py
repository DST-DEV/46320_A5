from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr

from lacbox.io import load_stats
import scivis

# %% User input

# fname_stats_A5 = "IEC_Ya_Later_tower_exclusion_v4_turb_stats"
fname_stats_A5 = "IEC_Ya_Later_new_seeds_turb_stats"
# fname_stats_A5 = "IEC_What_You_Did_There_turb_stats"
fname_stats_dtu = "dtu_10mw_new_seeds_turb_stats"
# fname_stats_dtu = "dtu_10mw_debug_file_turb_stats"

eval_oper = True
eval_DELs = True
eval_AEP = True
show_plots = True
save_plots = True
latex = False  # Not tested
exp_fld = "plots/stats_eval"  # Figure export path( relative to project root)

 # %% Data preparation

# File paths
ROOT = Path(__file__).parent.parent
STATS_PATH_ROOT = ROOT /"hawc_files" / "stats"
STATS_PATH_A5 = STATS_PATH_ROOT / (fname_stats_A5 + '.csv')
STATS_PATH_DTU = STATS_PATH_ROOT / (fname_stats_dtu + '.csv')
SAVE_PLOTS_PATH  = ROOT / exp_fld
SAVE_PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Load statistics
h2stats_A5, wsps_A5 = load_stats(STATS_PATH_A5, statstype='turb')
h2stats_dtu, wsps_dtu = load_stats(STATS_PATH_DTU, statstype='turb')

wsps_A5 = np.sort(wsps_A5)
h2stats_A5.fillna(0, inplace=True)
h2stats_dtu.fillna(0, inplace=True)

dtu_str, A5_str = "dtu", "redesign"
stats_dict = {dtu_str: h2stats_dtu, A5_str: h2stats_A5}

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
N_wsp = len(wsps_A5)
N_turbines = len(stats_dict)
sn_slopes = (4, 10)

# Retrieve and check turblence classes
tc_A5 = h2stats_A5["subfolder"].unique()  # Turbulence classes for A5
tc_dtu = h2stats_dtu["subfolder"].unique()  # Turbulence classes for DTU
assert len(tc_A5)>0, "No turbulance classes given for redesigned turbine"
assert len(tc_dtu)>0, "No turbulance classes given for DTU 10 MW"

tc_dict = {dtu_str: tc_dtu, A5_str: tc_A5}
tc = list(set(tc_A5).union(tc_dtu))
tc.sort()

# Check number of wind bins (assuming they are the same for A5 & the DTU 10 MW)
sim_count = h2stats_A5[(h2stats_A5.desc=='pitch1 angle')
                       & (h2stats_A5.subfolder==tc_A5[0])
                       ]["wsp"].value_counts()
assert sim_count.nunique() == 1, "Not all bins have same number of simulations!"
N_sim = sim_count.iloc[0]  # Number of simulations per wind bin

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

ds_stats_shape = (N_wsp, len(tc), N_base_stats + 1, N_turbines, N_sim)

ds_stats_raw = xr.Dataset(
    {
        chan_id: (["wsp", "tc", "stat", "turbine", "sim"],
                  np.empty(ds_stats_shape))
        for chan_id in ds_stats_keys
     },
    coords={
        "wsp": wsps_A5,
        "tc": tc,
        "stat": base_stats + ["del10min"],
        "turbine": ["dtu", "redesign"],
        "sim": np.arange(N_sim),
    },
)

ds_stats_eval = xr.Dataset(
    {
        chan_id: (["wsp", "tc", "stat", "turbine"],
                  np.empty(ds_stats_shape[:-1]))
        for chan_id in ds_stats_keys
     },
    coords={
        "wsp": wsps_A5,
        "tc": tc,
        "stat": base_stats + ["del1h"],
        "turbine": ["dtu", "redesign"]
    },
)

# Retrieve statistics for each wind bin
chan_units = {}
for turbine, stats in stats_dict.items():
    for tc_i in tc_dict[turbine]:
        # Filter stats for turbulence class
        stats_tc_i = stats[stats.subfolder == tc_i]

        for ch in ds_stats_keys:
            if ch in chan_id_loads:
                load_ch = True
                m_chan = sn_slopes_dict[ch]
            else:
                load_ch = False

            stats_chan = stats_tc_i.filter_channel(
                ch, CHAN_DESCS).sort_values(
                    ["wsp", "filename"]).reset_index(drop=True)

            chan_units[ch] = stats_chan["units"].values[0]
            wsp_i = stats_chan["wsp"].unique()

            ds_stats_raw[ch].loc[{"wsp": wsp_i, "stat": base_stats,
                              "turbine": turbine, "tc": tc_i}] \
                = np.swapaxes(stats_chan[base_stats].to_numpy().reshape(
                    len(wsp_i), N_sim, N_base_stats), 1, 2)

            if load_ch:
                dels_10min = stats_chan[f"del{m_chan}"].to_numpy().reshape(
                    len(wsp_i), N_sim)
                ds_stats_raw[ch].loc[{"wsp": wsp_i, "stat": "del10min",
                                  "turbine": turbine, "tc": tc_i}] = dels_10min

                dels_1h = np.sum(dels_10min**m_chan/N_sim, axis=-1)**(1/m_chan)
                ds_stats_eval[ch].loc[{"wsp": wsp_i, "stat": "del1h",
                                       "turbine": turbine, "tc": tc_i}
                                      ] = dels_1h

            ds_stats_eval[ch].loc[{"wsp": wsp_i, "stat": base_stats,
                                   "turbine": turbine, "tc": tc_i}] \
                = np.nanmean(ds_stats_raw[ch].sel(
                    stat=base_stats, turbine=turbine, tc=tc_i).values, axis=-1)

# %% Calculate fatigue and ultimate design loads
if eval_DELs:
    ultimate_loads = pd.DataFrame(columns=chan_id_loads,
                                  index=stats_dict.keys())
    dels_20a = pd.DataFrame(columns=chan_id_loads, index=stats_dict.keys())

    # Wind distribution (Rayleigh cumulative probability density)
    V_ave = np.array([10, 7.5])  # DTU: IEC I, Redesign: IEC III
    wsp_bins = np.append(wsps_A5-.5, wsps_A5[-1]+.5)
    cdf = 1 - np.exp(-np.pi/4 * (wsp_bins / V_ave.reshape((-1,1)))**2)
    cdf_bins = cdf[:, 1:] - cdf[:, :-1]

    # Number of cycles
    n_20 = 3600*8760*20  # number of cycles in 20 years assuming 1 Hz sampling
    n_eq = 1e7 # equivalent number of cycles based on the IEC standards

    for ch in chan_id_loads:
        # Calculate ultimate load
        ult_loads = lambda minmax: np.max(np.abs(minmax)) \
            * SCAL_FACTOR * SAFE_FACTOR

        ultimate_loads.loc[dtu_str, ch] = ult_loads(ds_stats_eval[ch].sel(
            stat=["min", "max"], turbine=dtu_str, tc="tca").values)
        ultimate_loads.loc[A5_str, ch] = ult_loads(ds_stats_eval[ch].sel(
            stat=["min", "max"], turbine=A5_str, tc="tcb").values)

        # Calculate lifetime fatigue load
        m_chan = sn_slopes_dict[ch]
        dels_1h = np.vstack(
            [ds_stats_eval[ch].sel(stat="del1h", turbine=dtu_str,
                                   tc="tca").values,
            ds_stats_eval[ch].sel(stat="del1h", turbine=A5_str,
                                  tc="tcb").values])
        dels_20a[ch] = (n_20/n_eq * np.sum(cdf_bins*dels_1h**m_chan, axis=1)) \
            **(1/m_chan)

    print("\n" + "-"*70)
    print("Fatigue loads")
    print("\t\t\t" + "\t".join( chan_id_loads))
    print("DTU:\t\t" + "\t".join([f"{dels_20a.loc[dtu_str, ch]*1e-3:5.2f}"
                               for ch in chan_id_loads]))
    print("Redesign:\t" + "\t".join([f"{dels_20a.loc[A5_str, ch]*1e-3:5.2f}"
                               for ch in chan_id_loads]))
    print("\nUltimate loads")
    print("DTU:\t\t"
          + "\t".join([f"{ultimate_loads.loc[dtu_str, ch]*1e-3:5.2f}"
                       for ch in chan_id_loads]))
    print("Redesign:\t"
          + "\t".join([f"{ultimate_loads.loc[A5_str, ch]*1e-3:5.2f}"
                       for ch in chan_id_loads]))
    print("-"*70)

# Plot operational data, loads and DELs
if show_plots:
    label_mapping = {"BldPit":r"\theta", "RotSpd":r"\omega", "Thrust":r"T",
                     "GenTrq":r"Q_g", "ElPow":r"P_{el}"}

    rc_profile = scivis.rcparams._prepare_rcparams(latex=latex)
    markers = {"min": "2", "mean": "x", "max": "1"}
    col_line_plots = {"dtu": ("#ff6361","#ffa600"),
                      "redesign": ("#003f5c","#58508d")}
    col_bars_dels = ["#A31C00", "#003f5c"]  # positive: red, negative: blue

    turbine_labels = {dtu_str: "DTU 10MW", A5_str: "Redesign"}

    tc_plot_dict = {dtu_str: "tca", A5_str: "tcb"}
    if not tc_plot_dict[dtu_str] in tc_dtu:
        raise KeyError("Missing turbulence class A for plots DTU 10 MW.")
    if not tc_plot_dict[A5_str] in tc_A5:
        raise KeyError("Missing turbulence class B for plots redesign.")


    with mpl.rc_context(rc_profile):
        for ch in ds_stats_keys:
            if ch == "ElPow":
                unit_scale = 1e-6
                y_unit = "MW"
            elif ch == "GenTrq":
                unit_scale = 1e-6
                y_unit = "MNm"
            elif chan_units[ch] == "kNm":
                unit_scale = 1e-3
                y_unit = "MNm"
            else:
                unit_scale=1
                y_unit = chan_units[ch]

            y_label = label_mapping[ch] if ch in label_mapping.keys() else ch

            # Plot Min/Mean/Max
            fig, ax = scivis.subplots(profile="partsize", scale=.7,
                                      latex=latex)

            for turbine in stats_dict.keys():
                for stat in ("min", "mean", "max"):
                    ax.scatter(wsps_A5,
                               ds_stats_eval[ch].sel(turbine=turbine,
                                                     tc=tc_plot_dict[turbine],
                                                     stat=stat) * unit_scale,
                               c=col_line_plots[turbine][0],
                               label=" - ".join([turbine_labels[turbine],
                                                  stat.title()]),
                               marker=markers[stat], zorder=3)

                # Shaded area between max & min curves
                ax.fill_between( wsps_A5,
                                *(ds_stats_eval[ch].sel(
                                    turbine=turbine,
                                    tc=tc_plot_dict[turbine],
                                    stat=["min", "max"]).values.T
                                    *unit_scale),
                                label="_", fc=col_line_plots[turbine][1],
                                alpha=0.2, zorder=2)

            ax.set_xlabel(r'$V\:[m/s]$')
            ax.set_ylabel(r"$" + y_label + r"\:[{" + y_unit + r"}]$")

            ax.set_xlim([4.5, 24.5])
            ax.set_xticks(np.arange(5,25,1))

            ax.minorticks_on()
            ax.grid(which='major', zorder=1)
            ax.grid(which='minor', visible=False)
            ax.tick_params(labelrotation=30)

            handles, labels = ax.get_legend_handles_labels()
            order = [0,3,1,4,2,5] # order of labels
            ax.legend([handles[i] for i in order], [labels[i] for i in order],
                      loc='lower center', fontsize=25,
                      bbox_to_anchor=(0.5, 1), ncol=3, frameon=False)

            if save_plots:
                fig.savefig(SAVE_PLOTS_PATH / (ch + "_scatterplot.svg"))

            # Plot DELs
            if ch in chan_id_loads:
                markerstyle = [{"marker": "x", "mec": col_line_plots[turb][0]}
                               for turb in [dtu_str, A5_str]]
                scivis.plot_line(wsps_A5,
                                 ds_stats_eval[ch].sel(stat="del1h",
                                                       tc=tc_plot_dict[turbine]
                                                       ).values.T * unit_scale,
                                 plt_labels=list(turbine_labels.values()),
                                 ax_labels=["V", y_label],
                                 ax_units = ["m/s", y_unit],
                                 colors=[col_line_plots[dtu_str][0],
                                         col_line_plots[A5_str][0]],
                                 linestyles="--", markers=markerstyle,
                                 ax_lims=[[4.5, 24.5], None],
                                 ax_ticks=[np.arange(5,25,1), None],
                                 profile="partsize", scale=.7, latex=latex,
                                 exp_fld=SAVE_PLOTS_PATH, fname=ch+"_DELS",
                                 savefig=save_plots
                                 )

        if eval_DELs:
            # Plot bar chart for design load comparison
            for load_name, load_df in zip(["FLS", "ULS"],
                                          [dels_20a, ultimate_loads]):
                fig, ax = scivis.subplots(profile="partsize", scale=.7,
                                          latex=latex)
                load_diff = (load_df.loc[A5_str, :] - load_df.loc[dtu_str, :])\
                    / load_df.loc[dtu_str, :] *100
                colors = [col_bars_dels[0] if diff >= 0 else col_bars_dels[1]
                          for diff in load_diff]

                x_pos = range(len(load_diff))
                ax.bar(x_pos, load_diff, fc=colors, ec="k", zorder=3)
                ax.axhline(0, color='black', linewidth=2)  # zero line

                ax.set_ylabel("Relative difference [%]")

                ax.tick_params(labelrotation=45)
                ax.set_xticks(x_pos, load_diff.index)

                ax.minorticks_on()
                ax.grid(which='major', zorder=1)
                ax.grid(which='minor', visible=False)

                if save_plots:
                    fig.savefig(SAVE_PLOTS_PATH
                                / (load_name + "_comparison.svg"))


# %% Calculate AEP
if eval_AEP:
    # Wind distribution (Rayleigh cumulative probability density)
    V_ave = 7.5  # IEC III for both turbines
    wsp_bins = np.append(wsps_A5-.5, wsps_A5[-1]+.5)
    cdf = 1 - np.exp(-np.pi/4 * (wsp_bins / V_ave)**2)
    cdf_bins = cdf[1:] - cdf[:-1]

    aep_func = lambda P_el: np.sum(P_el*cdf_bins)*8760/1e9
    AEP = {key: aep_func(ds_stats_eval["ElPow"].sel(stat="mean", turbine=key,
                                                    tc="tcb").values)
           for key in stats_dict.keys()}


    print("\n" + "-"*30)
    print("AEP")
    print("DTU:\t\t" + f"{AEP[dtu_str]:.2f}")
    print("Redesign:\t" + f"{AEP[A5_str]:.2f}")
    print("-"*30)

    if show_plots:
        rc_profile = scivis.rcparams._prepare_rcparams(profile="partsize",
                                                       scale=.7, latex=latex)

        cols_aep = {"bars": "#003f5c", "line": "#ffa600"}
        ax_r_col = '#C78100'
        with mpl.rc_context(rc_profile):
            fig, ax_l = scivis.subplots(profile="partsize", scale=.7,
                                        latex=latex)
            ax_r = ax_l.twinx()
            ax_r.zorder = 2

            ax_l.bar(wsps_A5, cdf_bins*100, fc=cols_aep["bars"], ec="k",
                     zorder=2)
            ax_r.plot(wsps_A5,
                      ds_stats_eval["ElPow"].sel(stat="mean", turbine=A5_str,
                                                 tc="tcb").values * 1e-6,
                      ls="--", c=cols_aep["line"], marker="o", zorder=2)

            ax_l.set_xlabel(r"$V_{ave}\:[m/s]$")
            ax_l.set_ylabel(r"Wind speed probabilty [%]")
            ax_r.set_ylabel(r"$P_{el}\:[MW]$")

            ax_l.set_xticks(wsps_A5)
            ax_l.set_xlim([wsps_A5[0]-.7, wsps_A5[-1]+.7])

            ax_l.minorticks_on()
            ax_r.minorticks_on()
            ax_l.tick_params(which="minor", axis="x", length=0)
            ax_l.tick_params(which="major", axis="x",
                             direction="out", length=5, top=False)
            ax_l.grid(which="major", zorder=1)
            ax_r.grid(which="both", visible=False)

            # Change color of right axis
            ax_r.tick_params(axis='y', which = "both", colors=ax_r_col)
            ax_r.yaxis.label.set_color(ax_r_col)
            ax_r.spines['right'].set_color(ax_r_col)

            if save_plots:
                    fig.savefig(SAVE_PLOTS_PATH / ("AEP.svg"))
