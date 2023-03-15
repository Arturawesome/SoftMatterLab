 
import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import pygame
import math
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.cm as cm
from scipy import spatial
from scipy.optimize import curve_fit
from matplotlib.ticker import LinearLocator
from scipy import signal
from matplotlib.colors import LightSource, Normalize
plt.style.use('./article_style.mplstyle')




def read_er_m_p(f, N_relax, N_run, thermo):
    for i in range(int(N_relax / thermo)):
        f.readline()

    d=[[float(item) for item in f.readline().strip().split('\t')[2::]] for i in range(int(N_run/thermo))]
    return np.array(d)



def RegistPhaseChageSc2Mil(size, p_xy, ts_relax, ts_all):

    p_xy_medfilt = np.array(scipy.signal.medfilt(p_xy, 3))
    size_milling = np.mean(size[50 : ts_relax])
    change_flags = [0, 0, 0] #0- stay skooling, 1-change to milling, 2-skooling->milling->skoling,  3-change to turning (experimental)
    polar_gran_in_clear = 0.7
    flag_mill = 0
    flag_brak = 0
    # все хгначения меньше => либо мгновенный вылет, лыбо не хватает силы вылететь но skyling остается
    if sum(p_xy_medfilt[50 : ts_relax] < polar_gran_in_clear) >= len(p_xy_medfilt[50 : ts_relax])- 5:
        flag_mill = 1


    count_up_down = 0
    time_ts = np.arange(1, ts_all, 10)
    size_start = 0
    size_end = 0

    size_start = np.mean(size[ts_relax-25 : ts_relax])
    size_end = np.mean(size[ts_all-25 : ts_all])
    for i in range(len(time_ts) - 2):
        p_mean1 = np.mean(p_xy_medfilt[int(time_ts[i]) : int(time_ts[i + 1])] - 0.5)
        p_mean2 = np.mean(p_xy_medfilt[int(time_ts[i+1]) : int(time_ts[i + 2])] - 0.5)

        if p_mean1 * p_mean2 < 0:
            count_up_down += 1

    if count_up_down == 1:
        #1-change to milling
        change_flags[1] = 1

    elif count_up_down == 0:
        #0- stay skooling
        if size_end > 120:
            flag_mill = 1
            flag_brak = 1
        else:
            change_flags[0] = 1

    elif count_up_down > 1.5:
        #2-skooling->milling->skoling
        change_flags[2] = 1

    return change_flags, flag_mill, flag_brak






path = '/run/media/artur/Новый том/FISH_MD_data/Fish/Fish/sp_fish/SC2Mil_dump3/'
dump_folders = ['dump']
print('/run/media/artur/Новый том/FISH_MD_data/Fish/dump/log_0_100_5_Ip_In_0.1_20000_100000_100_0_0.6_0.99_0.0.txt')
name = 'log_0_100_5_' + 'Ip'+'_' + 'In'+'_0.1_20000_100000_100_0_0.6_0.99_0.0.txt'

log_data = np.arange(10,20, 1) #
In = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


num_osob = np.array([1, 2, 3, 4, 5, 7, 10, 12, 15])
F_k_osob = 0.01*np.arange(4, 64, 4)
N_relax = 2e4
N_run = 1e5

name_log = []
name_log_not_path = []
print(F_k_osob)

for lgd in log_data:
    for i_n in In:
        for n_osoba in num_osob:
            for f_k_os in F_k_osob:
                name = path +'log_'+ str(lgd) + '_100_5_8.0_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_'+ str(n_osoba) + '_'+ str(f_k_os) + '_1.0_0.0.txt'
                name_log.append(name)
                name_log_not_path.append('log_'+ str(lgd) + '_100_5_8.0_'+ str(round(i_n, 2)) + '_0.1_20000_100000_100_'+ str(n_osoba) + '_'+ str(f_k_os) + '_1.0_0.0.txt')









newcolors =[ '#eb2d2d','#0daf0d','#3e68e2','#ff8000','#840384','#85B4DC','#63CE9C','#CB63CE','#3A779F','#3A9F77','#BA8838', '#BA5438','#BA38A8', '#b5b5b5']

N_ts = int((N_relax + N_run) / 100)
ts_relax = 200
ts_all = 1200
time =  np.arange(0, ts_all)
ts0_m = [20]#, 0, 201]
ts1_m = [1200, 200, 1200]

fly_fish = np.zeros((len(In), len(F_k_osob), len(num_osob)))    # мгновенный вылет рыб, милинг остается
no_influens = np.zeros((len(In), len(F_k_osob), len(num_osob))) # рыбы не вылетели, идет незначительное искажение милинга
full_change = np.zeros((len(In), len(F_k_osob), len(num_osob))) # милинг поменялся на скулинг
mil_sch_MilSchol = np.zeros((len(In), len(F_k_osob), len(num_osob)))    # милинг - скулинг - милинг и вылет рыбок


fig32, axs32 = plt.subplots(nrows = 3, ncols=1, figsize = (1, 2*1.08679358417*163.044/161.338), sharex=True)
fig32.subplots_adjust(hspace=0)
count1 = 0
count2 = 0
count3 = 0

flag1 = 0
flag2 = 0
flag3 = 0

skips = 10
p_color = "#ff8000"
s_color = "#3e68e2"
major_s_xticks = [0, 1e4]
y_s_lim_max = 1.11e4
y_s_lim_mim = -900
y_p_lim_max = 1.1
y_p_lim_mim = -0.085
major_p_yticks = [0, 1]

ts_start = 20
for lgd in range(len(log_data)):
    if flag1 + flag2 + flag3 == 3:
        break
    for i_n in range(len(In)):
        if flag1 + flag2 + flag3 == 3:
            break
        for n_osoba in range(len(num_osob)):
            if flag1 + flag2 + flag3 == 3:
                break
            for f_k_os in range(len(F_k_osob)):
                if flag1 + flag2 + flag3 == 3:
                    break
                k = (len(In) *len(num_osob) * len(F_k_osob)) * lgd + len(num_osob)*len(F_k_osob)*i_n + n_osoba * len(F_k_osob) + f_k_os

                f = open(name_log[k], "r")
                st = read_er_m_p(f, 0, N_relax + N_run, 100)
                x_cm = st[:, 0]
                y_cm = st[:, 1]
                px = st[:, 2]
                py = st[:, 3]
                p_xy = np.sqrt(np.square(px) + np.square(py))
                erx = st[:, 4]
                ery = st[:, 5]
                er_xy = np.sqrt(np.square(erx) + np.square(ery))
                moment = abs(st[:, 6])
                size = st[:, 7]
                mean_v = np.mean(p_xy)
                c, flag_mill, flag_brak = RegistPhaseChageSc2Mil(size, p_xy, ts_relax, ts_all)
                f.close()
                if flag_mill == 0:
                    if c[2] > 0:
                        #fly_data.write(name_log[k] + "\n")
                        if count2 == 85:
                            flag1 = 1
                            #axs32[0].set_title("Fly")
                            axs32[0].plot(time[ts_start:ts_all:skips], p_xy[ts_start:ts_all:skips], "-", color=p_color)

                            axs32[0].set_xticklabels([])
                            axs32[0].minorticks_off()
                            axs32[0].set_xticks([])
                            axs32[0].set_ylim([y_p_lim_mim, y_p_lim_max])
                            axs32[0].set_ylabel('P', color=p_color)
                            axs32[0].set_yticks(major_p_yticks)
                            axs32[0].tick_params(axis="y", which="both", colors=p_color)
                            axs32[0].spines['left'].set_color(p_color)
                            axs32[0].spines['right'].set_color(s_color)
                            axs32[0].axvline(x=200, ymin=y_p_lim_mim, ymax=y_p_lim_max, ls="--", lw=0.3, color = "Black")
                            axs0_twin = axs32[0].twinx()



                            axs0_twin.plot(time[ts_start:ts_all:skips],size[ts_start:ts_all:skips], "-", color=s_color)
                            axs0_twin.set_ylabel('S', color=s_color)
                            axs0_twin.tick_params(axis="y", which="both",colors=s_color)
                            axs0_twin.set_yticks(major_s_xticks)
                            axs0_twin.set_xticklabels([])
                            axs0_twin.set_yticklabels(["0", "$S_{m}$"])
                            axs0_twin.set_xticks([])
                            axs0_twin.set_ylim([y_s_lim_mim, y_s_lim_max])
                            axs0_twin.minorticks_off()
                            axs0_twin.spines['left'].set_color(p_color)
                            axs0_twin.spines['right'].set_color(s_color)
                            axs0_twin.text(0, 0, f'$\\mathbf{{(a_1)}}$')



                        count2 +=1



                    if c[0] > 0:
                        #full_change[i_n, f_k_os, n_osoba] += c[1]
                        #full_change_data.write(name_log[k] + "\n")
                        if count1 == 0:
                            flag2 = 1
                            #axs32[1].set_title("Change")
                            axs32[2].spines['left'].set_color(p_color)
                            axs32[2].spines['right'].set_color(s_color)
                            axs32[2].plot(time[ts_start:ts_all:skips], p_xy[ts_start:ts_all:skips], "-", color=p_color)
                            axs32[2].set_xticklabels([])
                            axs32[2].minorticks_off()
                            axs32[2].set_ylim([y_p_lim_mim, y_p_lim_max])
                            axs32[2].set_ylabel('P', color=p_color)
                            axs32[2].set_yticks(major_p_yticks)
                            axs32[2].set_xticks([])
                            axs32[2].axvline(x=200, ymin=y_p_lim_mim, ymax=y_p_lim_max, ls="--", lw=0.3, color = "Black")
                            axs32[2].tick_params(axis="y", which="both", colors=p_color)


                            axs2_twin = axs32[2].twinx()

                            axs2_twin.plot(time[ts_start:ts_all:skips], size[ts_start:ts_all:skips], "-", color=s_color)
                            axs2_twin.set_ylabel('S', color=s_color)
                            axs2_twin.tick_params(axis="y", which="both",colors=s_color)
                            axs2_twin.set_xticks([])
                            axs2_twin.set_yticks(major_s_xticks)
                            axs2_twin.set_yticklabels(["0", "$S_{m}$"])
                            axs2_twin.minorticks_off()
                            axs2_twin.set_ylim([y_s_lim_mim, y_s_lim_max])
                            axs2_twin.spines['left'].set_color(p_color)
                            axs2_twin.spines['right'].set_color(s_color)


                        count1 +=1

                    if c[1] > 0:
                        #sch_Mil_Schol[i_n, f_k_os, n_osoba] += c[2]
                        #mil_sch_MilSchol_data.write(name_log[k] + "\n")
                        if count3 == 10:
                            flag3 = 1
                            #axs32[2].set_title("Bif")
                            axs32[1].plot(time[ts_start:ts_all:skips], p_xy[ts_start:ts_all:skips], "-", color=p_color)
                            axs32[1].set_xticklabels([])
                            axs32[1].set_ylim([y_p_lim_mim, y_p_lim_max])
                            axs32[1].set_xticks([])
                            axs32[1].minorticks_off()
                            axs32[1].set_yticks(major_p_yticks)
                            axs32[1].set_ylabel('P', color=p_color)
                            axs32[1].tick_params(axis="y", which="both", colors=p_color)
                            axs32[1].spines['left'].set_color(p_color)
                            axs32[1].spines['right'].set_color(s_color)
                            axs32[1].axvline(x=200, ymin=y_p_lim_mim, ymax=y_p_lim_max, ls="--", lw=0.3, color = "Black")
                            axs1_twin = axs32[1].twinx()

                            axs1_twin.plot(time[ts_start:ts_all:skips], size[ts_start:ts_all:skips], "-", color=s_color)
                            axs1_twin.set_ylabel('S', color=s_color)
                            axs1_twin.tick_params(axis="y", which="both",colors=s_color)
                            axs1_twin.set_xticks([])
                            axs1_twin.minorticks_off()
                            axs1_twin.set_yticklabels(["0", "$S_{m}$"])
                            axs1_twin.set_ylim([y_s_lim_mim, y_s_lim_max])
                            axs1_twin.set_yticks(major_s_xticks)
                            axs1_twin.spines['left'].set_color(p_color)
                            axs1_twin.spines['right'].set_color(s_color)


                        count3 +=1
                if flag_brak == 1:
                    continue

                #fly_fish[i_n, f_k_os, n_osoba] += c[0]
                #no_influens[i_n, f_k_os, n_osoba] += c[1]
                #full_change[i_n, f_k_os, n_osoba] += c[2]
                #mil_sch_MilSchol[i_n, f_k_os, n_osoba] += c[3]

#plt.tight_layout(pad = 0)
plt.savefig("/home/artur/Documents/Fish/res/color_maps/" +'Sc2MPhTrain.pdf', transparent=True, bbox_inches='tight', pad_inches = 0)










