from pathlib import Path

from lacbox.io import load_stats
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd

#%% plot commands
#size
mpl.rcParams['figure.figsize'] = (16,10)

#font size of label, title, and legend
mpl.rcParams['font.size'] = 25
mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['axes.titlesize'] = 25
mpl.rcParams['legend.fontsize'] = 25

#Lines and markers
mpl.rcParams['lines.linewidth'] = 2.2
mpl.rcParams['lines.markersize'] = 15
mpl.rcParams['scatter.marker'] = "d"

#Latex font
plt.rcParams['font.family'] = 'serif'  # Simula il font di LaTeX
plt.rcParams['mathtext.fontset'] = 'cm'  # Usa Computer Modern per la matematica

#Export
mpl.rcParams['savefig.bbox'] = "tight"

col_plots = ['#003f5c','#58508d','#bc5090','#ff6361','#ffa600']
mark_plots = ['x','1','2']

#%% plot function
def plot_ch(stats,stats_dtu,ch,OP_DATA,SAVE_PLOTS_PATH):
    
    #check in which folder the plots need to be saved
    if OP_DATA:
        FOLDER = 'op_data'
    else:
        FOLDER = 'loads'
        
    #check if the folder where to save the plots exists
    output_fld = Path(SAVE_PLOTS_PATH, FOLDER)
    output_fld_DELs = Path(SAVE_PLOTS_PATH, "DELs")
    output_fld.mkdir(parents=True, exist_ok=True)
    output_fld_DELs.mkdir(parents=True, exist_ok=True)
    
    #save units and wind speed valies from the channel considered
    units = stats['units'].values
    wsp = stats['wsp']
    wsp_dtu = stats_dtu['wsp']
    
    # IEC Ya Later data
    mom_mean = stats['mean']
    mom_max = stats['max']
    mom_min = stats['min']
    mom_std = stats['std']
    # dtu data
    mom_mean_dtu = stats_dtu['mean']
    mom_max_dtu = stats_dtu['max']
    mom_min_dtu = stats_dtu['min']
    mom_std_dtu = stats_dtu['std']
    # create a Pandas dataframe to store data and then sort them
    df = pd.DataFrame({
    'wsp': wsp,
    'mom_mean': mom_mean,
    'mom_max': mom_max,
    'mom_min': mom_min,
    'mom_std': mom_std
    })
    df_dtu = pd.DataFrame({
    'wsp_dtu': wsp_dtu,
    'mom_mean_dtu': mom_mean_dtu,
    'mom_max_dtu': mom_max_dtu,
    'mom_min_dtu': mom_min_dtu,
    'mom_std_dtu': mom_std_dtu
    })
    if not OP_DATA:
        m_vals =(3,4,5,8,10,12)
        for m in m_vals:
            del_ = stats['del'+str(m)]
            del_dtu = stats_dtu['del'+str(m)]
            df['del'+str(m)] = del_
            df_dtu['del'+str(m)+'_dtu'] = del_dtu

    # sorting data based on the wind speed
    grouped = df.groupby('wsp', sort=True)
    grouped_dtu= df_dtu.groupby('wsp_dtu', sort=True)

    # calculate the mean for each channel for each group, since we have N seeds
    mean_by_ws = grouped.mean()
    mean_by_ws_dtu = grouped_dtu.mean()
    scale_plot = 1
    if ch == 'ElPow':
        units[0] = 'MW'
        scale_plot = 1e-6
    if ch == 'GenTrq':
        units[0] = 'MNm'
        scale_plot = 1e-6
    # mean, max, min figure for redesign and DTU 10MW in turbulent flow
    fig,ax = plt.subplots(1, 1)
    plot1 = ax.scatter(mean_by_ws['mom_mean'].index,mean_by_ws['mom_mean']*scale_plot,
                       color = col_plots[0],label='Redesign - Mean',marker=mark_plots[0],
                       zorder=2)
    plot2 = ax.scatter(mean_by_ws['mom_mean'].index,mean_by_ws['mom_max']*scale_plot,
                       color = col_plots[0],label='Redesign - Max',marker=mark_plots[1],
                       zorder=3)
    plot3 = ax.scatter(mean_by_ws['mom_mean'].index,mean_by_ws['mom_min']*scale_plot,
                       color = col_plots[0],label='Redesign - Min',marker=mark_plots[2],
                       zorder=4)
    plot4 = ax.scatter(mean_by_ws_dtu['mom_mean_dtu'].index,mean_by_ws_dtu['mom_mean_dtu']*scale_plot,
                       color = col_plots[-2],label='DTU 10MW - Mean',marker=mark_plots[0],
                       zorder=5)
    plot5 = ax.scatter(mean_by_ws_dtu['mom_mean_dtu'].index,mean_by_ws_dtu['mom_max_dtu']*scale_plot,
                       color = col_plots[-2],label='DTU 10MW - Max',marker=mark_plots[1],
                       zorder=6)
    plot6 = ax.scatter(mean_by_ws_dtu['mom_mean_dtu'].index,mean_by_ws_dtu['mom_min_dtu']*scale_plot,
                       color = col_plots[-2],label='DTU 10MW - Min',marker=mark_plots[2],
                       zorder=7)
    ax.fill_between(mean_by_ws['mom_mean'].index, mean_by_ws['mom_min']*scale_plot, mean_by_ws['mom_max']*scale_plot,
                    color=col_plots[0], alpha=0.2, zorder=8) #shaded area
    ax.fill_between(mean_by_ws_dtu['mom_mean_dtu'].index, mean_by_ws_dtu['mom_min_dtu']*scale_plot,
                    mean_by_ws_dtu['mom_max_dtu']*scale_plot, color=col_plots[-1], alpha=0.2,
                    zorder=8)
    ax.set_xlabel(r'V [m/s]')
    ax.set_ylabel(ch + f' [{units[0]}]')
    ax.set_xlim([4.5,24.5])
    ax.grid(which='major',zorder=1)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,3,1,4,2,5] # order of labels
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='lower center',
               bbox_to_anchor=(0.5, 1), ncol=3,frameon=False)
    ax.minorticks_on()
    ax.set_xticks(np.arange(5,25,1))
    ax.tick_params(direction='in',right=True,top =True)
    ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,
                      labelright=False)
    ax.tick_params(direction='in',which='minor',length=5,bottom=True,
                     top=True,left=True,right=True)
    ax.tick_params(direction='in',which='major',length=10,bottom=True,
                      top=True,right=True,left=True)
    plt.tight_layout()
    fig.savefig(f'{SAVE_PLOTS_PATH}/{FOLDER}/{ch}.pdf')
    #plt.close()
    
    # DELs plots
    if not OP_DATA:
        m_vals =(3,4,5,8,10,12)
        for m in m_vals:
            del_ = stats['del'+str(m)]
            del_dtu = stats_dtu['del'+str(m)]
            df['del'+str(m)] = del_
            df_dtu['del'+str(m)+'_dtu'] = del_dtu
        
        if 'BRM' in ch:
            m= 10
        else:
            m= 4

        # DEL plot 
        fig2,ax = plt.subplots(1, 1)
        ax.scatter(mean_by_ws['del'+str(m)].index,mean_by_ws['del'+str(m)],color =col_plots[0],
                   label='Redesign',marker=mark_plots[0],zorder=2)
        ax.scatter(mean_by_ws_dtu['del'+str(m)+'_dtu'].index,mean_by_ws_dtu['del'+str(m)+'_dtu'],
                   color =col_plots[-2],label='DTU 10MW',marker=mark_plots[0],zorder=3)
        ax.plot(mean_by_ws['del'+str(m)].index,mean_by_ws['del'+str(m)],color =col_plots[0],
                alpha=0.5,linestyle='--',zorder=4)
        ax.plot(mean_by_ws_dtu['del'+str(m)+'_dtu'].index,mean_by_ws_dtu['del'+str(m)+'_dtu']
                ,color =col_plots[-2],alpha=0.5,linestyle='--',zorder=5)
        ax.set_xlabel(r'V [m/s]')
        ax.set_ylabel(f'DEL{m} '+ ch + f' [{units[0]}]')
        ax.grid(which='major',zorder=1)
        ax.set_xlim([4.5,24.5])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2,frameon=False)
        ax.minorticks_on()
        ax.set_xticks(np.arange(5,25,1))
        ax.tick_params(direction='in',right=True,top =True)
        ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,
                        labelright=False)
        ax.tick_params(direction='in',which='minor',length=5,bottom=True,
                        top=True,left=True,right=True)
        ax.tick_params(direction='in',which='major',length=10,bottom=True,
                        top=True,right=True,left=True)
        plt.tight_layout()
        fig2.savefig(f'{SAVE_PLOTS_PATH}/DELs/{ch}.pdf')
        #plt.close()
        
        # std figure for redesign and DTU 10MW in turbulent flow
        fig1,ax = plt.subplots(1, 1)
        ax.scatter(mean_by_ws['mom_std'].index,mean_by_ws['mom_std'],color =col_plots[0],
                   label='Redesign - std',marker=mark_plots[0])
        ax.scatter(mean_by_ws_dtu['mom_std_dtu'].index,mean_by_ws_dtu['mom_std_dtu'],
                   color =col_plots[-2],label='DTU 10MW - std',marker=mark_plots[0])
        ax.plot(mean_by_ws['mom_std'].index,mean_by_ws['mom_std'],color =col_plots[0],
                alpha=0.5,linestyle='--')
        ax.plot(mean_by_ws_dtu['mom_std_dtu'].index,mean_by_ws_dtu['mom_std_dtu']
                ,color =col_plots[-2],alpha=0.5,linestyle='--')
        ax.set_xlabel(r'V [m/s]')
        ax.set_ylabel(ch + f' [{units[0]}]')
        ax.grid(which='major')
        ax.set_xlim([4.5,24.5])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2,frameon=False)
        ax.minorticks_on()
        ax.set_xticks(np.arange(5,25,1))
        ax.tick_params(direction='in',right=True,top =True)
        ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,
                        labelright=False)
        ax.tick_params(direction='in',which='minor',length=5,bottom=True,
                        top=True,left=True,right=True)
        ax.tick_params(direction='in',which='major',length=10,bottom=True,
                        top=True,right=True,left=True)
        plt.tight_layout()
        fig1.savefig(f'{SAVE_PLOTS_PATH}/{FOLDER}/{ch}_std.pdf')
        return fig, fig1, fig2
    else:
        return fig
#%% main script

# folder direction
ROOT = Path(__file__).parent
FOLDER_STATS = 'stats'
STATS_PATH = ROOT / FOLDER_STATS / 'IEC_Ya_Later_Jenni_seeds_turb_stats.csv'
STATS_PATH_DTU = ROOT / FOLDER_STATS /'dtu_10mw_turb_stats.csv'
SUBFOLDER = ['tca','tcb']  # which subfolder to plot: turb class A or B
SAVE_PLOTS_PATH  = ROOT/'plots_A4'

# which data we want to plot: operational or loads
OP_DATA = False

# loading data
h2stats, wsps = load_stats(STATS_PATH, statstype='turb')
h2stats_dtu,wsps_dtu = load_stats(STATS_PATH_DTU, statstype='turb')
h2stats = h2stats.fillna(0)
h2stats_dtu= h2stats_dtu.fillna(0)

# what channels we want to plot: operational and loads
chan_ids_op_data = ['BldPit', 'RotSpd', 'Thrust', 'GenTrq', 'ElPow']
chan_id_loads = ['TbFA', 'TbSS','YbTilt', 'YbRoll', 'ShftTrs', 'OoPBRM', 'IPBRM']
CHAN_DESCS_OP_DATA = {'BldPit': 'pitch1 angle',
            'RotSpd': 'rotor speed',
            'Thrust': 'aero rotor thrust',
            'GenTrq': 'generator torque',
            'ElPow': 'pelec'
            }
CHAN_DESCS_LOADS = {'TbFA': 'momentmx mbdy:tower nodenr:   1',
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
# basic statistics we want to plot
base_stats = ['mean', 'max', 'min']

if OP_DATA:
    data = chan_ids_op_data
    descr = CHAN_DESCS_OP_DATA
else:
    data = chan_id_loads
    descr = CHAN_DESCS_LOADS
    
    # for the loads I also create a dict to save ultimate loads and fatigue loads
    ultimate_loads = dict()
    fatigue_loads = dict()

    print('The design loads are being calculated for the following channels for DTU 10MW:')

# analyze each channel to plot the statistics and determine the design loads
jenni_ult_loads = np.array([363813.39, 124940.40, 58035.04, 24430.53,
                            20514.81, 72752.48, 40373.08])
jenni_fat_loads = np.array([126790.93, 57153.20, 32742.06, 4013.05,
                            2874.84, 31998.76, 30999.19])
results = {'chan_id':chan_id_loads,
           'ultimate loads redesign [kNm]': np.zeros_like(chan_id_loads),
           'ultimate loads DTU 10MW [kNm]': jenni_ult_loads,
           'ultimate loads percentage difference [%]': np.zeros_like(chan_id_loads),
           'fatigue loads redesign [kNm]': np.zeros_like(chan_id_loads),
           'fatigue loads DTU 10MW [kNm]': jenni_fat_loads,
           'fatigue loads percentage difference [%]': np.zeros_like(chan_id_loads)}
results = pd.DataFrame(results)
results.set_index("chan_id", inplace=True)
for ch in data:
    
    # use IA class for DTU
    df_sub_dtu = h2stats_dtu[h2stats_dtu.subfolder == 'tca']
    
    # use IIIB class for YA LATER
    df_sub = h2stats[h2stats.subfolder == 'tcb']
    stats_chan = df_sub.filter_channel(ch, descr)
    stats_chan_dtu = df_sub_dtu.filter_channel(ch, descr)

    if OP_DATA:
        fig1 = plot_ch(stats_chan,stats_chan_dtu,ch,OP_DATA,SAVE_PLOTS_PATH)
    else:
        fig1, fig2, fig3 = plot_ch(stats_chan,stats_chan_dtu,ch,OP_DATA,SAVE_PLOTS_PATH)

    # ultimate loads calculation and fatigue DELs
    if not OP_DATA:
        SCAL_FACTOR = 1.35
        SAFE_FACTOR = 1.25
        
        # group data and sort them based on wind speed
        wsp = stats_chan['wsp']
        max_min = pd.DataFrame({
            'wsp': wsp,
            'maximums': stats_chan['max'],
            'minimums': stats_chan['min']
            })
        grouped = max_min.groupby('wsp', sort=True)
        max_min_by_ws = grouped.mean()
        
        # calculate the maximum of both the absolute value of min and max curve 
        maxi = (np.abs(max_min_by_ws['maximums'])).max()
        mini = (np.abs(max_min_by_ws['minimums'])).max()
        
        # get the highest value for each channel
        ultimate_loads[ch] = np.max([maxi,mini])*SCAL_FACTOR*SAFE_FACTOR
        results.loc[ch,'ultimate loads redesign [kNm]'] = ultimate_loads[ch]
        #results['ultimate loads redesign [kNm]'][ch] = ultimate_loads[ch]
        results.loc[ch,'ultimate loads percentage difference [%]']  = (ultimate_loads[ch]-results.loc[ch,'ultimate loads DTU 10MW [kNm]'])/\
                                                results.loc[ch,'ultimate loads DTU 10MW [kNm]']*100
        print(f'\nChannel: {ch}      ----->      Ultimate Load: {ultimate_loads[ch]:.2f} {stats_chan_dtu["units"].values[0]}')

        # DELs
        V_avg = 10 # V_ave for class IA
        
        # for blades m in different than the other components
        if 'BRM' in ch:
            m = 10
        else:
            m = 4
        
        # save the 10-minute DELs and group them for each wind speed
        tenmin_dels = stats_chan['del'+str(m)]
        dels = pd.DataFrame({
            'wsp': wsp,
            'tenmin_dels': tenmin_dels
            })
        grouped_dels = dels.groupby('wsp', sort=True)
        
        # determine the 1-hour DEL
        hour_del = {}
        for wsp_val, group in grouped_dels:
            hour_del[wsp_val] = (1/len(group['tenmin_dels']) * np.sum(group['tenmin_dels']**m))**(1/m)
        hour_del = np.array(list(hour_del.values()))
        
        # the wind speed used for the Rayleigh probability are calculated as the 
        # edges of each 1 m/s bin around the centered value saved in wsp vector
        wsp = np.array(list(grouped_dels.groups.keys()))-0.5
        wsp = np.append(wsp, wsp[-1]+1)  # extend last wind speed bin
        
        # Reyliegh CDF
        cdf = 1-np.exp(-np.pi/4*(wsp/V_avg)**2)
        
        # fatigue loads
        n_20 = 3600*8760*20  # number of cycles in 20 years assuming 1 Hz sampling
        n_eq = 1e7 # equivalent n° of cycles based on the IEC standards
        
        # initializing the fatigue loads and probability at each wind speed
        fatigue_loads[ch] = 0
        probability = np.zeros_like(hour_del)
        for i in range(len(hour_del)):
            probability[i] = cdf[i+1]- cdf[i]
            fatigue_loads[ch] += (n_20/n_eq)*(hour_del[i]**m)*probability[i]
        fatigue_loads[ch] = fatigue_loads[ch] **(1/m)
        results.loc[ch,'fatigue loads redesign [kNm]'] = fatigue_loads[ch]
        #results['fatigue loads redesign [kNm]'][ch] = fatigue_loads[ch]
        results.loc[ch,'fatigue loads percentage difference [%]'] = (fatigue_loads[ch]-results.loc[ch,'fatigue loads DTU 10MW [kNm]'])/\
                                                results.loc[ch,'fatigue loads DTU 10MW [kNm]']*100
        print(f'\nChannel: {ch}      ----->      Fatigue Load: {fatigue_loads[ch]:.2f} {stats_chan["units"].values[0]}')
    
results = results.apply(pd.to_numeric, errors='coerce')    

#%% bar chart for percentage differences
if not OP_DATA:
    FOLDER = 'DESIGN_LOADS'
    output_fld = Path(SAVE_PLOTS_PATH, FOLDER)
    output_fld.mkdir(parents=True, exist_ok=True)
    
    y = results["ultimate loads percentage difference [%]"]
    x = results.index
    colors = np.where(y >= 0, col_plots[0], col_plots[-1])  # blue for positive, red for negative
    fig, ax = plt.subplots(1,1)
    bars = ax.bar(x, y, color=colors, edgecolor="black",zorder=3)
    ax.axhline(0, color='black', linewidth=2)  # zero line
    ax.set_ylabel("Ultimate load difference [%]")
    plt.xticks(rotation=45, ha='right')
    ax.grid(which='major',zorder=1)
    ax.minorticks_on()
    ax.tick_params(direction='in',right=True,top =False)
    ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,
                    labelright=False)
    ax.tick_params(direction='in',which='minor',length=5,bottom=False,
                    top=False,left=True,right=True)
    ax.tick_params(direction='in',which='major',length=10,bottom=True,
                    top=True,right=True,left=True)
    plt.tight_layout()
    fig.savefig(f'{SAVE_PLOTS_PATH}/{FOLDER}/ultimate_loads_barchart.pdf')
    
    y = results["fatigue loads percentage difference [%]"]
    x = results.index
    colors = np.where(y >= 0, col_plots[0], col_plots[-1])  # blue for positive, red for negative
    fig, ax = plt.subplots(1,1)
    bars = ax.bar(x, y, color=colors, edgecolor="black",zorder=3)
    ax.axhline(0, color='black', linewidth=2)  # zero line
    ax.set_ylabel("Fatigue load difference [%]")
    plt.xticks(rotation=45, ha='right')
    ax.grid(which='major',zorder=1)
    ax.minorticks_on()
    ax.tick_params(direction='in',right=True,top =False)
    ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,
                    labelright=False)
    ax.tick_params(direction='in',which='minor',length=5,bottom=False,
                    top=False,left=True,right=True)
    ax.tick_params(direction='in',which='major',length=10,bottom=True,
                    top=True,right=True,left=True)
    plt.tight_layout()
    fig.savefig(f'{SAVE_PLOTS_PATH}/{FOLDER}/fatigue_loads_barchart.pdf')
#%% AEP calculation
chan_id_power = 'Power'
chan_desc_power = {'Power':'pelec'}

# use for both turbines the Turbulent class B
df_sub = h2stats[h2stats.subfolder == 'tcb']
df_sub_dtu = h2stats_dtu[h2stats_dtu.subfolder == 'tcb']

# stats for the power channel
stats_chan = df_sub.filter_channel(chan_id_power, chan_desc_power)
stats_chan_dtu = df_sub_dtu.filter_channel(chan_id_power, chan_desc_power)
wsp = stats_chan['wsp']
wsp = np.sort(stats_chan['wsp'].unique())
wsp -= 0.5
wsp = np.append(wsp, wsp[-1]+1)
 
# sorting the values of power based on the wind speed
power_des  = pd.pivot_table(stats_chan, index='wsp', values='mean',aggfunc='mean')
power_dtu  = pd.pivot_table(stats_chan_dtu, index='wsp', values='mean',aggfunc='mean')

# calculate the Rayleigh CDF
V_avg = 7.5 # V_ave for class IIIB
cdf = 1-np.exp(-np.pi/4*(wsp/V_avg)**2)
probability = np.zeros_like(power_des)
for i in range(len(power_des)):
    probability[i] = cdf[i+1]- cdf[i]


AEP_des = np.sum(power_des['mean'].to_numpy().flatten()*probability.flatten())*8760/1e9  #GWh
AEP_dtu = np.sum(power_dtu['mean'].to_numpy().flatten()*probability.flatten())*8760/1e9  #GWh
print(f'\nRedesign AEP: {AEP_des:.4}  MWh')
print(f'\nDTU 10MW AEP: {AEP_dtu:.4}  MWh')

#%% AEP plots
FOLDER = 'AEP'
output_fld = Path(SAVE_PLOTS_PATH, FOLDER)
output_fld.mkdir(parents=True, exist_ok=True)

y = probability[:,0]
x = power_des.index
fig, ax = plt.subplots(1,1)
ax2 = ax.twinx()
bars = ax.bar(x, y, color=col_plots[0], edgecolor="black",zorder=3)
ax2.plot(power_des.index,power_des.values*1e-6,color=col_plots[-1],linestyle='--',
         marker='o',zorder=2)
ax.set_ylabel("Wind speed probabilty [-]")
ax.set_xlabel('V [m/s]')
ax2.set_ylabel('Power [MW]',color='#C78100')
ax.set_xlim([4.5,25])
ax.grid(which='major',zorder=1)
ax2.yaxis.label.set_color('#C78100')
ax2.spines['right'].set_color('#C78100')
ax.minorticks_on()
ax.set_xticks(np.arange(5,25,1))
ax.tick_params(direction='in',right=True,top =False)
ax2.tick_params(axis='y', which = "both", colors='#C78100')
ax.tick_params(labelbottom=True,labeltop=False,labelleft=True,
                labelright=False)
ax.tick_params(direction='in',which='minor',length=5,bottom=False,
                top=False,left=True,right=True)
ax.tick_params(direction='in',which='major',length=10,bottom=True,
                top=True,right=True,left=True)
plt.tight_layout()
fig.savefig(f'{SAVE_PLOTS_PATH}/{FOLDER}/ultimate_loads_barchart.pdf')

