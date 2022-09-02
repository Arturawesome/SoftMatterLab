import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import pygame
import math
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.cm as cm
from scipy import spatial
from scipy import signal
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits import mplot3d
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import os
import pathlib
def read_er_m_p(f, N_relax, N_run, thermo):
    for i in range(int(N_relax / thermo)):
        f.readline()

    d=[[float(item) for item in f.readline().strip().split('\t')[2::]] for i in range(int(N_run/thermo))]
    return np.array(d)

def get_disp(np_array):
    mean = sum(np_array) / len(np_array)
    D = sum(np.square(np_array - mean) ) / (len(np_array) -1)
    return D

def get_disp_sko(np_array):
    mean = sum(np_array) / len(np_array)
    D = np.sqrt(sum(np.square(np_array - mean) ) / (len(np_array) -1))
    return D

def autocorrelate(data, len_corr):
    lags = range(len_corr)

    acorr = len(lags) * [0]
    x = np.array(data)

    # Mean
    mean = np.mean(data)

    # Variance
    var = np.var(data)

    # Normalized data
    ndata = data - mean

    acorr = np.correlate(ndata, ndata, 'full')[len(ndata)-1:]
    acorr = acorr / var / len(ndata)
    return acorr


def get_angles(px_array, py_array):
    # https://lms2.sseu.ru/courses/eresmat/course1/razd9z1/par9_12z1.htm
    rot_angle = np.zeros(len(px_array))
    for i in range(len(px_array)):
        px = px_array[i]
        py = py_array[i]
        angle = pygame.math.Vector2(1, 0).angle_to((px, py))

        if py > 0:
            rot_angle[i] =  math.radians(angle)

        elif py < 0:
            rot_angle[i] =  math.radians(360 + angle)


    return  rot_angle

def display_colorbar(x, y, z, name):

    cmap = plt.cm.copper
    ls = LightSource(315, 45)
    rgb = ls.shade(z, cmap)

    fig, ax = plt.subplots()
    ax = plt.subplot(111)

    # Use a proxy artist for the colorbar...
    im = ax.pcolormesh(x, y, z, cmap=cmap)
    fig.colorbar(im)

   #
    #fig.colorbar(im)

    ax.set_title(name, size='x-large')
    plt.savefig(name + '.png',  bbox_inches='tight',pad_inches = 0.05)



path = '/run/media/artur/Новый том/FISH_MD_data/Fish/DATA_log/'
#path = "/run/media/artur/Новый том/FISH_MD_data/Fish/dump/"
dump_folders = ['dump']

name = 'log_0_100_5_' + 'Ip'+'_' + 'In'+'_0.1_20000_100000_100_0_0.6_0.99_0.0.txt'


log_data = np.arange(0, 100, 1) #0, 1, 2, 3, 4...
print(log_data[0], " - ", log_data[len(log_data)-1])
N_fish = [100]
size_box = [5]
Ip = np.arange(0, 10, 0.5)
In = np.arange(0, 1, 0.05)
d_t = np.array([0.1])
N_relax = np.array([20000])
N_run = np.array([100000])
thermo = np.array([100])
num_osob = np.array([0])
F_k_osob = np.array([0.6])
v_osob = np.array([0.99])
k_f_osob = np.array([0.0])
name_log = []
for i in log_data:
    for nfish in N_fish:
        for size in size_box:
            for i_n in In:
                for ip in Ip:
                    for dt in d_t:
                        for n_rel in N_relax:
                            for n_run in N_run:
                                for ther in thermo:
                                    for n_osoba in num_osob:
                                        for f_k_os in F_k_osob:
                                            for v_os in v_osob:
                                                for k_f_os in k_f_osob:
                                                    name = path +'log_'+ str(i) + '_'+ str(nfish) + '_'+ str(size) + '_'+ str(ip) + '_'+ str(round(i_n, 2)) + '_'+ str(dt) + '_'+ str(n_rel) + '_'+ str(n_run) + '_'+ str(ther) + '_'+ str(n_osoba) + '_'+ str(f_k_os) + '_'+ str(v_os) + '_'+ str(k_f_os) + '.txt'
                                                    #print(name)
                                                    try:
                                                        f = open(name, "r")
                                                        f.close()
                                                    except:
                                                        old_name = path +'log_'+ str(i) + '_'+ str(nfish) + '_'+ str(size) + '_'+ str(ip) + '_'+ str(round(i_n, 2)) + '_'+ str(dt) + '_'+ str(n_rel) + '_'+ str(n_run) +'_'+ str(ther) + '_'+ str(n_osoba) + '_'+ str(f_k_os) + '_1.0_'+ str(k_f_os) + '.txt'
                                                        os.rename(old_name, name)
                                                    name_log.append(name)




# 2              3              4              5                  6           7             8          9
# 0              1              2               3                  4          5             6
#<mean_x<<"\t"<<mean_y<<"\t"<<mean_px<<"\t"<<mean_py<<"\t"<<mean_erx<<"\t"<<mean_ery<<"\t"<<M<<"\t"<<size<<"\n"



save_path = "/run/media/artur/Новый том/FISH_MD_data/res/ris/"

av_p_in_run = np.zeros(len(log_data) * len(In) * len(Ip))
av_mil_in_run = np.zeros(len(log_data) * len(In) * len(Ip))

medfilt_dat_clear_mill = []# = np.zeros(int(N_run/thermo))

cear_milling_id_run = []
clear_milling_p = []
clear_milling_m = []

noise_p = []
noise_m = []

n_elem_in_gp = 400
min_val_polar_run = np.zeros(len(log_data) * len(In) * len(Ip))

n_elem_in_gm = 400
min_val_mill_run = np.zeros(len(log_data) * len(In) * len(Ip))
val_mill_run_full = np.zeros(len(log_data) * len(In) * len(Ip))
val_d_mill_run_full = np.zeros(len(log_data) * len(In) * len(Ip))
val_in_clear_milling_dp = []
val_in_clear_milling = []#np.zeros(len(log_data) * len(In) * len(Ip))

n_elem_in_g_dp = 1000
min_val_dp_run = np.zeros(len(log_data) * len(In) * len(Ip))



time = np.arange(N_relax, N_run + N_relax, thermo)
count_clear_milling = 0
polar_sko_in_milling = []
f_m = open("milling.txt", "w")
f_k = open("k_prove.txt", "w")
f_f = open("f_prove.txt", "w")
corr0_clear_mill_polar = []
corr0_clear_mill_angle = []
for lgd in range(len(log_data)):
    for i_n in range(len(In)): #
        for i_p in range(len(Ip)): # ip
            k = (len(In) *len(Ip)) * lgd + len(Ip)*i_n + i_p
            f_k.write(str(k) + " " + str(In[i_n]) + " " + str(Ip[i_p]) + "\n")
            f_f.write(name_log[k] + "\n")

            f = open(name_log[k], "r")
            st = read_er_m_p(f, N_relax, N_run, thermo)
            f.close()

            x_cm = st[:, 0]
            y_cm = st[:, 1]

            px = st[:, 2]
            py = st[:, 3]
            p_xy = np.sqrt(np.square(px) + np.square(py))

            erx = st[:, 4]
            ery = st[:, 5]

            #er_xy = np.sqrt(np.square(erx) + np.square(erx))

            milling = abs(st[:, 6])
            size = st[:, 7]

            av_p_in_run[k] =  sum(p_xy) / len(p_xy)
            av_mil_in_run[k] = sum(milling) / len(milling)

            min_val_min = sorted(milling)[:50]
            min_mil = sum(min_val_min) / 50
            min_val_mill_run[k] = min_mil

            val_mill_run_full[k] = sum(milling) / len(milling)
            val_d_mill_run_full[k] = get_disp(milling)

            # выделение всего милинга на всех симуляциях

            if np.log(min_mil) > 1.8:

                cear_milling_id_run.append(k)
                f_m.write(name_log[k] + "\n")
                clear_milling_p.append(sum(p_xy) / len(p_xy))
                clear_milling_m.append(sum(milling) / len(milling))
                polar_sko_in_milling.append(get_disp_sko(p_xy))
                val_in_clear_milling.append(sum(milling) / len(milling))
                val_in_clear_milling_dp.append(val_d_mill_run_full[k])
                angles_data = get_angles(px, py)
                angles_data_filt = np.array(scipy.signal.medfilt(angles_data, 3))

                count_clear_milling += 1

                p_xy_medfilt = np.array(scipy.signal.medfilt(p_xy, 5))

                medfilt_dat_clear_mill.append(sum(sorted(p_xy_medfilt)[:-10:-1]) / len(p_xy_medfilt[:-10:-1]))

                len_corr = 50
                acorr = autocorrelate(p_xy_medfilt, len_corr)
                for id_acorr in range(len(acorr)):
                    el_acor = acorr[id_acorr]
                    if el_acor < 0:
                        corr0_clear_mill_polar.append(id_acorr)
                        break
                acorr2 = autocorrelate(angles_data_filt, len_corr)
                for id_acorr in range(len(acorr2)):
                    el_acor = acorr2[id_acorr]
                    if el_acor < 0:
                        corr0_clear_mill_angle.append(id_acorr)
                        break


            if np.log(val_d_mill_run_full[k]) < -10:
                noise_m.append(sum(milling) / len(milling))
                noise_p.append(sum(p_xy) / len(p_xy))




f_m.close()
f_k.close()
f_f.close()
polar_sko_in_milling = sum(polar_sko_in_milling) / len(polar_sko_in_milling)
medfilt_dat_clear_mill = np.array(medfilt_dat_clear_mill)
len_ar = len(medfilt_dat_clear_mill)
polar_gran_in_clear = sum(sorted(medfilt_dat_clear_mill)[:-600:-1]) / len(sorted(medfilt_dat_clear_mill)[:-600:-1])
print("polar_gran_in_clear = ", polar_gran_in_clear)
#polar_gran_in_clear = sum(medfilt_dat_clear_mill) / len(medfilt_dat_clear_mill)

medfilt_dat_clear_mill = np.zeros(int(N_run/thermo))
medfilt_dat_clear_mill += polar_gran_in_clear


clear_scholong = []
polar_clear_scholong_runs = []
milling_clear_scholongn_runs = []

clear_swarming = []

polar_clear_swarming_runs = []
milling_clear_swarming_runs = []
clear_swarm_disp_polar = []

polar_bifurcation_runs = []
milling_bifurcation_runs = []
val_in_bifurcation = []

av_size_in_bifur_run = []
disp_size_in_bifur_run = []




count_down_global = []
count_up_global = []
relationship_time_bif = []

frequency = []
disp_polar_in_not_clean_phase = []
aver_polar_in_not_clean_phase = []
len_zeros_in_autocorr = []

polar_x_orientation_bif = []
polar_y_orientation_bif = []

corr0_clear_swarm_polar = []
corr0_clear_swarm_angle = []

corr0_clear_schol_polar = []
corr0_clear_schol_angle = []

corr0_clear_bulshit_polar = []
corr0_clear_bulshit_angle = []

medfilt_bulshit_mill = []
medfilt_bulshit_size = []

disp_size_bulshit = []

size_pick1 = []
size_pick2 = []
size_pick3 = []
size_pick4 = []
size_pick5 = []

milling_pick1 = []
milling_pick2 = []
milling_pick3 = []
milling_pick4 = []
milling_pick5 = []

polar_pick1 = []
polar_pick2 = []
polar_pick3 = []
polar_pick4 = []
polar_pick5 = []

f_swarm = open("swarm_info.txt", "w")
f_skul = open("skhul_info.txt", "w")
f_bulshit = open("bulshit_info.txt", "w")

f_pick1 = open("pick1.txt", "w")
f_pick2 = open("pick2.txt", "w")
f_pick3 = open("pick3.txt", "w")
f_pick4 = open("pick4.txt", "w")
f_pick5 = open("pick5.txt", "w")

for lgd in range(len(log_data)):
    for i_n in range(len(In)): #
        for i_p in range(len(Ip)): # ip
            k = (len(In) *len(Ip)) * lgd + len(Ip)*i_n + i_p

            if k in cear_milling_id_run:
                continue

            f = open(name_log[k], "r")
            st = read_er_m_p(f, N_relax, N_run, thermo)
            f.close()

            x_cm = st[:, 0]
            y_cm = st[:, 1]

            px = st[:, 2]
            py = st[:, 3]
            p_xy = np.sqrt(np.square(px) + np.square(py))

            erx = st[:, 4]
            ery = st[:, 5]
            er_xy = np.sqrt(np.square(erx) + np.square(ery))

            milling = abs(st[:, 6])
            size = st[:, 7]

            p_xy_medfilt = np.array(scipy.signal.medfilt(p_xy, 3))

            p_filt_minus_p_gran = p_xy_medfilt - polar_gran_in_clear

            # выделение 100% сварминга
            if sum(p_xy_medfilt > polar_gran_in_clear) < 1:
                clear_swarming.append(k)
                f_swarm.write(name_log[k] + "\n")
                polar_clear_swarming_runs.append(sum(p_xy) / len(p_xy))
                clear_swarm_disp_polar.append(get_disp_sko(p_xy))
                milling_clear_swarming_runs.append(sum(milling) / len(milling))

                len_corr = 50
                acorr = autocorrelate(p_xy_medfilt, len_corr)
                angles_data = get_angles(px, py)
                angles_data_filt = np.array(scipy.signal.medfilt(angles_data, 3))

                for id_acorr in range(len(acorr)):
                    el_acor = acorr[id_acorr]
                    if el_acor < 0:
                        corr0_clear_swarm_polar.append(id_acorr)
                        break
                acorr2 = autocorrelate(angles_data_filt, len_corr)
                for id_acorr in range(len(acorr2)):
                    el_acor = acorr2[id_acorr]
                    if el_acor < 0:
                        corr0_clear_swarm_angle.append(id_acorr)
                        break
            elif sum(p_xy_medfilt < polar_gran_in_clear) < 1:
                 # выделение 100% скулинга
                # schooling init

                clear_scholong.append(k)
                f_skul.write(name_log[k] + "\n")
                polar_clear_scholong_runs.append(sum(p_xy) / len(p_xy))
                milling_clear_scholongn_runs.append(sum(milling) / len(milling))

                len_corr = 50
                acorr = autocorrelate(p_xy_medfilt, len_corr)
                angles_data = get_angles(px, py)
                angles_data_filt = np.array(scipy.signal.medfilt(angles_data, 3))
                for id_acorr in range(len(acorr)):
                    el_acor = acorr[id_acorr]
                    if el_acor < 0:
                        corr0_clear_schol_polar.append(id_acorr)
                        break

                acorr2 = autocorrelate(angles_data_filt, len_corr)
                for id_acorr in range(len(acorr2)):
                    el_acor = acorr2[id_acorr]
                    if el_acor < 0:
                        corr0_clear_schol_angle.append(id_acorr)
                        break
            else:
                # выделение оставшихся фаз
                len_corr = 50
                acorr = autocorrelate(p_xy_medfilt, len_corr)
                angles_data = get_angles(px, py)
                angles_data_filt = np.array(scipy.signal.medfilt(angles_data, 3))

                milling_medfilt = np.array(scipy.signal.medfilt(milling, 7))

                # выделение 50 наименьших значений милинга
                exp_gran = sum(sorted(milling_medfilt)[:-50:-1]) / len(milling_medfilt[:-50:-1])
                medfilt_bulshit_mill.append(exp_gran)
                if np.log(exp_gran) < 2.1:
                    # выделение оставшегося скулинга по нименьшему милингу
                    clear_scholong.append(k)
                    polar_clear_scholong_runs.append(sum(p_xy) / len(p_xy))
                    f_skul.write(name_log[k] + "\n")
                    milling_clear_scholongn_runs.append(sum(milling) / len(milling))

                    len_corr = 50
                    acorr = autocorrelate(p_xy_medfilt, len_corr)
                    angles_data = get_angles(px, py)
                    angles_data_filt = np.array(scipy.signal.medfilt(angles_data, 3))
                    for id_acorr in range(len(acorr)):
                        el_acor = acorr[id_acorr]
                        if el_acor < 0:
                            corr0_clear_schol_polar.append(id_acorr)
                            break
                    acorr2 = autocorrelate(angles_data_filt, len_corr)
                    for id_acorr in range(len(acorr2)):
                        el_acor = acorr2[id_acorr]
                        if el_acor < 0:
                            corr0_clear_schol_angle.append(id_acorr)
                            break
                    continue



                sizefilt = np.array(scipy.signal.medfilt(size, 5))
                #exp_gran2 = sum(sorted(sizefilt)[:-50:-1]) / len(sizefilt[:-50:-1])
                # выделение оставшегося сварминга по среднему значению фаз
                exp_gran2 = sum(sorted(sizefilt)) / len(sizefilt)
                medfilt_bulshit_size.append(exp_gran2)
                disp_size_bulshit.append(get_disp(sizefilt))
                if exp_gran2 < 9.8:
                    size_pick1.append(k)
                    milling_pick1.append(sum(milling) / len(milling))
                    polar_pick1.append(sum(p_xy) / len(p_xy))
                    f_pick1.write(name_log[k] + "\n")
                if 9.8 < exp_gran2 < 12.8:
                    size_pick2.append(k)
                    milling_pick2.append(sum(milling) / len(milling))
                    polar_pick2.append(sum(p_xy) / len(p_xy))
                    f_pick2.write(name_log[k] + "\n")
                if 12.8 < exp_gran2 < 16.2:
                    size_pick3.append(k)
                    milling_pick3.append(sum(milling) / len(milling))
                    polar_pick3.append(sum(p_xy) / len(p_xy))
                    f_pick3.write(name_log[k] + "\n")
                if 16.2 < exp_gran2 < 19.4:
                    size_pick4.append(k)
                    milling_pick4.append(sum(milling) / len(milling))
                    polar_pick4.append(sum(p_xy) / len(p_xy))
                    f_pick4.write(name_log[k] + "\n")
                if exp_gran2 > 19.4:
                    size_pick5.append(k)
                    milling_pick5.append(sum(milling) / len(milling))
                    polar_pick5.append(sum(p_xy) / len(p_xy))
                    f_pick5.write(name_log[k] + "\n")

                if exp_gran2 > 16.2:
                    # выделение оставшегося сварминга
                    clear_swarming.append(k)
                    f_swarm.write(name_log[k] + "\n")
                    polar_clear_swarming_runs.append(sum(p_xy) / len(p_xy))
                    clear_swarm_disp_polar.append(get_disp_sko(p_xy))
                    milling_clear_swarming_runs.append(sum(milling) / len(milling))

                    len_corr = 50
                    acorr = autocorrelate(p_xy_medfilt, len_corr)
                    angles_data = get_angles(px, py)
                    angles_data_filt = np.array(scipy.signal.medfilt(angles_data, 3))

                    for id_acorr in range(len(acorr)):
                        el_acor = acorr[id_acorr]
                        if el_acor < 0:
                            corr0_clear_swarm_polar.append(id_acorr)
                            break
                    acorr2 = autocorrelate(angles_data_filt, len_corr)
                    for id_acorr in range(len(acorr2)):
                        el_acor = acorr2[id_acorr]
                        if el_acor < 0:
                            corr0_clear_swarm_angle.append(id_acorr)
                            break
                    continue




                #medfilt_bulshit_size.append(sum(sizefilt) / len(sizefilt))


                for id_acorr in range(len(acorr)):
                    el_acor = acorr[id_acorr]
                    if el_acor < 0:
                        corr0_clear_bulshit_polar.append(id_acorr)
                        break
                acorr2 = autocorrelate(angles_data_filt, len_corr)
                for id_acorr in range(len(acorr2)):
                    el_acor = acorr2[id_acorr]
                    if el_acor < 0:
                        corr0_clear_bulshit_angle.append(id_acorr)
                        break


                polar_x_orientation_bif.append(sum(px) / len(px))
                polar_y_orientation_bif.append(sum(py) / len(py))
                len_corr = 50
                acorr = autocorrelate(p_xy_medfilt, len_corr)
                av_size_in_bifur_run.append(sum(size) / len(size))
                disp_size_in_bifur_run.append(get_disp(size))
                for id_acorr in range(len(acorr)):
                    el_acor = acorr[id_acorr]
                    if el_acor < 0:
                        len_zeros_in_autocorr.append(id_acorr)
                        break

                #ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=True, facecolor = "white")
                #plt.plot(time, acorr,  '-')
                #plt.title(name_log[k])
                #plt.savefig(save_path + 'Acorr_P(ts)'+ str(k)+ "_"+str(In[i_n]) +'_Ip_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
                #plt.close(ff)
                aver_polar_in_not_clean_phase.append(sum(p_xy_medfilt) / len(p_xy_medfilt))
                disp_polar_in_not_clean_phase.append( get_disp(p_xy_medfilt))



                freq_count = 0
                count_down = -10
                count_up = -10

                flag_down = 0
                flag_up = 0
                flag_chenge = 0

                count_d = 0
                count_u = 0
                for id_el in range(len(p_filt_minus_p_gran) -1):
                    el_now = p_filt_minus_p_gran[id_el]
                    if id_el == 0:
                        el_past = p_filt_minus_p_gran[id_el]
                    else:
                        if (el_past > 0) and  (el_now > 0):
                            count_u += 1
                            if count_u > count_up:
                                count_up = count_u + 1
                        elif (el_past < 0) and (el_now < 0):
                            count_d += 1
                            if count_d > count_down:
                                count_down = count_d + 1
                                #count_d = 0
                                #flag_up = 1
                        elif el_past * el_now < 0:
                            freq_count += 1
                            flag_change = 1
                            if el_now > 0:
                                if count_d > count_down:
                                    count_down = count_d + 1
                                count_d = 0
                                flag_up = 1
                                count_u += 1
                            elif el_now < 0:
                                if count_u > count_up:
                                    count_up = count_u + 1
                                count_u = 0
                                count_d += 1
                                flag_down = 1

                if count_u > count_up:
                    count_up = count_u +1
                if count_d > count_down:
                    count_down = count_d +1

                count_down_global.append(count_down)
                count_up_global.append(count_up)
                frequency.append(freq_count)
                f_bulshit.write(name_log[k] + " " + str(k) + ";  " + str(freq_count) + "\n")



clear_swarm_disp_polar = sum(clear_swarm_disp_polar) / len(clear_swarm_disp_polar)
counter = 0

f_pick1.close()
f_pick2.close()
f_pick3.close()
f_pick4.close()
f_pick5.close()
f_swarm.close()
f_skul.close()
f_bulshit.close()


ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.hist(disp_size_bulshit, 100)
plt.title(f"down value")
plt.savefig(save_path + 'disp_size_bulshit.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.hist(np.log(min_val_mill_run), n_elem_in_gm)
plt.xlabel("Log(<M(50 min)>)")
plt.xlim([-6,6])
plt.savefig(save_path + 'Gist_M1_.pdf',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.hist(medfilt_bulshit_size, 100)
#plt.xlim([-6,6])
plt.xlabel("<S> mean filtr")
plt.savefig(save_path + 'Gist3.pdf',  bbox_inches='tight',pad_inches = 0.05)

plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)

plt.plot(av_p_in_run, np.log(av_mil_in_run),  '.')
plt.plot(clear_milling_p, np.log(clear_milling_m),  '.')
plt.plot(polar_clear_swarming_runs, np.log(milling_clear_swarming_runs),  '.', color = "blue")
plt.plot(polar_clear_scholong_runs, np.log(milling_clear_scholongn_runs),  '.', color = "red")
plt.ylim([-5, 12])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'P_M_depend'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)




ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)

plt.plot(av_p_in_run, np.log(av_mil_in_run),  '.')
plt.plot(clear_milling_p, np.log(clear_milling_m),  '.')
plt.plot(polar_clear_swarming_runs, np.log(milling_clear_swarming_runs),  '.', color = "blue")
plt.plot(polar_clear_scholong_runs, np.log(milling_clear_scholongn_runs),  '.', color = "red")

plt.plot(polar_pick1, np.log(milling_pick1),  '.', color = "orange")
plt.plot(polar_pick2, np.log(milling_pick2),  '.', color = "green")
plt.plot(polar_pick3, np.log(milling_pick3),  '.', color = "yellow")
plt.plot(polar_pick4, np.log(milling_pick4),  '.', color = "gray")
plt.plot(polar_pick5, np.log(milling_pick5),  '.', color = "black")
plt.ylim([-5, 12])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'P_M_depend_sized_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.hist(np.log(medfilt_bulshit_mill), 100)
#plt.xlim([-6,6])
plt.xlabel("Log(<M(50 min)>) mean filtr")
plt.savefig(save_path + 'Gist2.pdf',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)




#num_time_cut = 100
#t_max = max(count_down_global)
#t_min = min(count_down_global)
#d_t_cut = (t_max - t_min)/num_time_cut
#time_cut = np.arange(t_min, t_max + 3*d_t_cut,  d_t_cut)
#Disp_time_cut_polar = np.zeros(len(time_cut))
#Disp_time_cut_size = np.zeros(len(time_cut))
#Disp_time_cut_er = np.zeros(len(time_cut))
#Disp_time_cut_m = np.zeros(len(time_cut))

#num_frequency_cut = 100
#f_max = max(frequency)
#f_min = min(frequency)
#d_f_cut = (f_max - f_min) / num_frequency_cut
#frequency_cut = np.arange(f_min, f_max + d_f_cut,  d_f_cut)
#Disp_frequency_cut_polar = np.zeros(len(frequency_cut))
#Disp_frequency_cut_size = np.zeros(len(frequency_cut))
#Disp_frequency_cut_er = np.zeros(len(frequency_cut))
#Disp_frequency_cut_m = np.zeros(len(frequency_cut))

#for lgd in range(len(log_data)):
    #for i_n in range(len(In)): #
        #for i_p in range(len(Ip)): # ip
            #k = (len(In) *len(Ip)) * lgd + len(Ip)*i_n + i_p
            #if k in cear_milling_id_run or k in clear_swarming or k in clear_scholong:
                #continue
            #id_time = int(math.floor(count_down_global[counter] / d_t_cut))
            #id_freq = int(math.floor(frequency[counter] / d_f_cut))
            ##print(k, "; ", id_time)
            #f = open(name_log[k], "r")
            #st = read_er_m_p(f, N_relax, N_run, thermo)
            #x_cm = st[:, 0]
            #y_cm = st[:, 1]
            #px = st[:, 2]
            #py = st[:, 3]
            #p_xy = np.sqrt(np.square(px) + np.square(py))
            #erx = st[:, 4]
            #ery = st[:, 5]
            #er_xy = np.sqrt(np.square(erx) + np.square(ery))
            #milling = abs(st[:, 6])
            #size = st[:, 7]
            #p_xy_medfilt = np.array(scipy.signal.medfilt(p_xy, 3))

            #Disp_time_cut_polar[id_time] += get_disp(p_xy)
            #Disp_time_cut_size[id_time] += get_disp(size)
            #Disp_time_cut_er[id_time] += get_disp(er_xy)
            #Disp_time_cut_m[id_time] += get_disp(milling)

            #Disp_frequency_cut_polar[id_freq] += get_disp(p_xy)
            #Disp_frequency_cut_size[id_freq] += get_disp(size)
            #Disp_frequency_cut_er[id_freq] += get_disp(er_xy)
            #Disp_frequency_cut_m[id_freq] += get_disp(milling)

            #counter += 1

#fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (20,15))

#fig.suptitle("all_dispersion" + name_log[k])
#axs[0, 0].plot(time_cut, Disp_time_cut_polar,  '.')
#axs[0, 0].set_title("polar")
#axs[0, 0].set_ylabel("Disp_time_cut_polar")

#axs[1, 0].plot(time_cut, Disp_time_cut_size,  '.')
#axs[1, 0].set_title("size")
#axs[1, 0].set_ylabel("Disp_time_cut_size")

#axs[0, 1].plot(time_cut, Disp_time_cut_er,  '.')
#axs[0, 1].set_title("er")
#axs[0, 1].set_ylabel("Disp_time_cut_er")

#axs[1, 1].plot(time_cut, Disp_time_cut_m,  '.')
#axs[1, 1].set_ylabel("Miling")
#axs[1,1].set_title("Log(Disp_time_cut_m)")
#plt.tight_layout(pad = 1, w_pad=0.2, h_pad=.5)
#plt.savefig(save_path + 'bulshit/D(t_cut)'+str(k) + '.png',  bbox_inches='tight',pad_inches = 0.05)
#plt.close(fig)


#corr0_clear_bulshit_angle = np.array(corr0_clear_bulshit_angle)# * np.array(disp_polar_in_not_clean_phase)
#corr0_clear_bulshit_polar = np.array(corr0_clear_bulshit_polar)#* np.array(disp_polar_in_not_clean_phase)


fig, axs = plt.subplots(nrows=3, ncols=5, figsize = (50,15))

fig.suptitle("All zeros acorr" )
axs[0, 0].plot(corr0_clear_mill_angle, corr0_clear_mill_polar,  '.')
axs[0, 0].set_title("Milling")
axs[0, 0].set_xlabel("corr0_clear_mill_angle")
axs[0, 0].set_ylabel("corr0_clear_mill_polar")


axs[0, 1].plot(corr0_clear_swarm_angle, corr0_clear_swarm_polar,  '.')
axs[0, 1].set_title("CLEAR swarming")
axs[0, 1].set_xlabel("corr0_clear_swarm_angle")
axs[0, 1].set_ylabel("corr0_clear_swarm_polar")


axs[0, 2].plot(corr0_clear_schol_angle, corr0_clear_schol_polar,  '.')
axs[0, 2].set_title("CLEAR Scholing")
axs[0, 2].set_xlabel("corr0_clear_schol_angle")
axs[0, 2].set_ylabel("corr0_clear_schol_polar")



axs[0, 3].plot(corr0_clear_bulshit_angle, corr0_clear_bulshit_polar,  '.')
axs[0, 3].set_title("Bulshit")
axs[0, 3].set_xlabel("corr0_clear_bulshit_angle")
axs[0, 3].set_ylabel("corr0_clear_bulshit_polar")


axs[0, 4].plot(corr0_clear_bulshit_angle, corr0_clear_bulshit_polar,  '.')
axs[0, 4].plot(corr0_clear_schol_angle, corr0_clear_schol_polar,  '.')
axs[0, 4].plot(corr0_clear_swarm_angle, corr0_clear_swarm_polar,  '.')
axs[0, 4].plot(corr0_clear_mill_angle, corr0_clear_mill_polar,  '.')
axs[0, 4].set_title("All on one")
axs[0, 4].set_xlabel("corr0_angle")
axs[0, 4].set_ylabel("corr0_polar")

axs[1, 0].hist(corr0_clear_mill_angle, 100)
axs[1, 1].hist(corr0_clear_swarm_angle, 100)
axs[1, 2].hist(corr0_clear_schol_angle, 100)
axs[1, 3].hist(corr0_clear_bulshit_angle, 100)
#axs[1, 4].hist((corr0_clear_mill_angle+corr0_clear_swarm_angle+corr0_clear_schol_angle+corr0_clear_bulshit_angle), 100)

axs[2, 0].hist(corr0_clear_mill_polar, 100)
axs[2, 1].hist(corr0_clear_swarm_polar, 100)
axs[2, 2].hist(corr0_clear_schol_polar, 100)
axs[2, 3].hist(corr0_clear_bulshit_polar, 100)
#axs[2, 4].hist((corr0_clear_mill_polar+corr0_clear_swarm_polar+corr0_clear_schol_polar+corr0_clear_bulshit_polar), 100)


plt.tight_layout(pad = 1, w_pad=0.2, h_pad=.5)
plt.savefig(save_path + 'All_o_cor_data'+str(k) + '.pdf',  bbox_inches='tight',pad_inches = 0.05)
plt.close(fig)


count_down_global = np.array(count_down_global)
count_up_global = np.array(count_up_global)

phase_milling = np.zeros((len(Ip),len(In)))
phase_scholing = np.zeros((len(Ip),len(In)))
phase_swarming = np.zeros((len(Ip),len(In)))

phase_size1 = np.zeros((len(Ip),len(In)))
phase_size2 = np.zeros((len(Ip),len(In)))
phase_size3 = np.zeros((len(Ip),len(In)))
phase_size4 = np.zeros((len(Ip),len(In)))
phase_size5 = np.zeros((len(Ip),len(In)))



for lgd in range(len(log_data)):
    for i_n in range(len(In)): #
        for i_p in range(len(Ip)): # ip
            k = (len(In) *len(Ip)) * lgd + len(Ip)*i_n + i_p
            if k in size_pick1:
                phase_size1[i_p, i_n] += 1
            if k in size_pick2:
                phase_size2[i_p, i_n] += 1
            if k in size_pick3:
                phase_size3[i_p, i_n] += 1
            if k in size_pick4:
                phase_size4[i_p, i_n] += 1
            if k in size_pick5:
                phase_size5[i_p, i_n] += 1

            if k in cear_milling_id_run:
                phase_milling[i_p, i_n] += 1
            if k in clear_swarming:
                phase_swarming[i_p, i_n] += 1

            if k in clear_scholong:
                phase_scholing[i_p, i_n] += 1
phase_size1 /= np.amax(phase_size1)
phase_size2 /= np.amax(phase_size2)
phase_size3 /= np.amax(phase_size3)
phase_size4 /= np.amax(phase_size4)
phase_size5 /= np.amax(phase_size5)

phase_milling /= np.amax(phase_milling)
phase_swarming /= np.amax(phase_swarming)
phase_scholing /= np.amax(phase_scholing)
x = In
y = Ip
z = phase_milling
display_colorbar(x, y, z, 'milling')

z = phase_swarming
display_colorbar(x, y, z, 'phase_swarming')

z = phase_scholing
display_colorbar(x, y, z, 'phase_scholing')

z = phase_milling + phase_swarming + phase_scholing
display_colorbar(x, y, z, 'all_clear_phase')

z = phase_size1
display_colorbar(x, y, z, 'size_pick1')

z = phase_size2
display_colorbar(x, y, z, 'size_pick2')

z = phase_size3
display_colorbar(x, y, z, 'size_pick3')

z = phase_size4
display_colorbar(x, y, z, 'size_pick4')

z = phase_size5
display_colorbar(x, y, z, 'size_pick5')


#frequency = np.array(frequency) * np.array(disp_polar_in_not_clean_phase)
#aver_polar_in_not_clean_phase = np.array(aver_polar_in_not_clean_phase) * np.array(disp_polar_in_not_clean_phase)
#count_down_global = np.array(count_down_global) * np.array(disp_polar_in_not_clean_phase)
ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.hist(aver_polar_in_not_clean_phase, 100)
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'aver_polar_in_not_clean_phase'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(av_size_in_bifur_run, frequency, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'av_size_in_bifur_run_frequency'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(polar_x_orientation_bif, polar_y_orientation_bif, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'px_py'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8))

ax = plt.axes(projection='3d')
#plt.plot(polar_x_orientation_bif, polar_y_orientation_bif, ".")
ax.plot3D(polar_x_orientation_bif, polar_y_orientation_bif, av_size_in_bifur_run,  ".")
#plt.xlim([-6,6])
plt.savefig(save_path + 'px_py_size'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8))

ax = plt.axes(projection='3d')
#plt.plot(polar_x_orientation_bif, polar_y_orientation_bif, ".")
ax.plot3D(polar_x_orientation_bif, polar_y_orientation_bif, disp_size_in_bifur_run, '.')
#ax.view_init(60, 35)
#plt.xlim([-6,6])
plt.savefig(save_path + 'px_py_disp_size'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))

ax = plt.axes(projection='3d')
#plt.plot(polar_x_orientation_bif, polar_y_orientation_bif, ".")
ax.plot3D(aver_polar_in_not_clean_phase,frequency, np.log(len_zeros_in_autocorr), '.')
#ax.view_init(60, 35)
#plt.xlim([-6,6])
plt.savefig(save_path + 'av_p_freq_zeros'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)




ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(np.log(len_zeros_in_autocorr), av_size_in_bifur_run, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + '0_autocorr_av_size_not_clean_phase'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(np.log(len_zeros_in_autocorr), disp_size_in_bifur_run, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + '0_autocorr_disp_size_not_clean_phase'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(disp_size_in_bifur_run, frequency, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'disp_size_in_bifur_run_frequency'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(np.log(len_zeros_in_autocorr), aver_polar_in_not_clean_phase, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + '0_autocorr_aver_polar_not_clean_phase'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(np.log(len_zeros_in_autocorr), disp_polar_in_not_clean_phase, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + '0_autocorr_disp_polar_not_clean_phase'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(aver_polar_in_not_clean_phase,frequency, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'Polar_frequency'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(disp_polar_in_not_clean_phase,frequency, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'DispPolar_frequency'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(count_down_global,disp_polar_in_not_clean_phase, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'count_down_global_disp_polar'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(count_up_global,disp_polar_in_not_clean_phase, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'count_up_global_disp_polar'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(aver_polar_in_not_clean_phase,disp_polar_in_not_clean_phase, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'Polar_disp_polar'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.hist(disp_polar_in_not_clean_phase, 80)
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'disp_polar_in_not_clean_phase'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(np.log(count_down_global[count_down_global>-1]), 150)
##plt.xlim([-6,6])
#plt.title(f"down value")
#plt.savefig(save_path + 'logDown_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.hist(frequency, 100)
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'frequency'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.plot(np.log(len_zeros_in_autocorr), frequency, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + '0_autocorr_frequency'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))
plt.hist(np.log(len_zeros_in_autocorr), 200)
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'len_zeros_in_autocorr'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(np.log(count_up_global[count_up_global>-1]), 150)
##plt.xlim([-6,6])
#plt.title(f"up value")
#plt.savefig(save_path + 'logUp_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)

#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(count_down_global[count_down_global>-1], 150)
##plt.xlim([-6,6])
#plt.title(f"down value")
#plt.savefig(save_path + 'Down_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)

#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(count_up_global[count_up_global>-1], 150)
##plt.xlim([-6,6])
#plt.title(f"up value")
#plt.savefig(save_path + 'Up_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)

#relation = count_down_global / count_up_global
#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(relation, 100)
#plt.xlim([1,999])
#plt.title(f"up value")
#plt.savefig(save_path + 'relationship_time_bif_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)



ff=plt.figure(figsize=(15.0,15.0 * 0.8))

plt.plot(count_down_global, count_up_global,  '.')
plt.xlabel("time under meanfilt")
plt.ylabel("time over meanfilt")

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + '2D_time'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8))

plt.plot(np.log(count_down_global), np.log(count_up_global),  '.')
plt.xlabel("time under meanfilt")
plt.ylabel("time over meanfilt")

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'log_2D_time'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.hist(np.log(val_mill_run_full), n_elem_in_gm)
plt.hist(np.log(val_in_clear_milling), int(n_elem_in_gm/6))
plt.xlim([-6,12])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'Gist_Mful_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.hist(np.log(val_d_mill_run_full), n_elem_in_gm)
plt.hist(np.log(val_in_clear_milling_dp), int(n_elem_in_gm/6))
#plt.xlim([-6,6])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'Gist_M_disp_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

'''

#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(min_val_polar_run, n_elem_in_gp)
##plt.xlim([-6,6])
#plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
#plt.savefig(save_path + 'Gist_P1_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)

#plt.close(ff)

#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(min_val_dp_run, n_elem_in_g_dp)
#plt.xlim([0,0.1])
#plt.ylim([0, 300])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'Gist_PDISP_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)





phase_milling = np.zeros((len(Ip),len(In)))
phase_scholing = np.zeros((len(Ip),len(In)))
phase_swarming = np.zeros((len(Ip),len(In)))




for lgd in range(len(log_data)):
    for i_n in range(len(In)): #
        for i_p in range(len(Ip)): # ip
            k = (len(In) *len(Ip)) * lgd + len(Ip)*i_n + i_p

            if k in cear_milling_id_run:
                phase_milling[i_p, i_n] += 1
            if k in clear_swarming:
                phase_swarming[i_p, i_n] += 1

            if k in clear_scholong:
                phase_scholing[i_p, i_n] += 1

phase_milling /= np.amax(phase_milling)
phase_swarming /= np.amax(phase_swarming)
phase_scholing /= np.amax(phase_scholing)
x = In
y = Ip
z = phase_milling
display_colorbar(x, y, z, 'milling')

z = phase_swarming
display_colorbar(x, y, z, 'phase_swarming')

z = phase_scholing
display_colorbar(x, y, z, 'phase_scholing')

z = phase_milling + phase_swarming + phase_scholing
display_colorbar(x, y, z, 'all_clear_phase')


                #fig, axs = plt.subplots(nrows=3, ncols=2, figsize = (20,15), frameon=True, facecolor = "white")
                #inset_corr = axs[0, 0].twinx()
                #fig.suptitle("Milling" + name_log[k])
                #axs[0, 0].plot(time, p_xy,  '.')
                #axs[0, 0].set_title("P")
                #axs[0, 0].plot(time, p_xy_medfilt,  '--', color = "black")
                #axs[0, 0].plot(time, medfilt_dat_clear_mill,  '-', color = "gray")
                #axs[0, 0].set_ylabel("Polar$")
                #inset_corr.plot(time, acorr, "-", color = "red")
                #axs[1, 0].plot(time, angles_data,  '.')
                #axs[1, 0].plot(time, angles_data_filt,  '--', color = "black")
                #inset_corr_angle = axs[1, 0].twinx()

                #axs[1, 0].set_title("Angle")
                #inset_corr_angle.plot(time, acorr2, "-", color = "red")
                #axs[1, 0].set_ylabel("angles_data")
                #axs[0, 1].plot(time, size,  '.')
                #axs[0, 1].set_title("Size")
                #axs[0, 1].set_ylabel("Size")
                #axs[1, 1].plot(time, np.log(milling),  '.')
                #axs[1, 1].set_ylabel("Log(M)")
                #axs[1,1].set_title("Log(M)")
                #axs[2, 0].plot(time, er_xy,  '.')
                #axs[2, 0].set_ylabel("er_xy")
                #axs[2, 0].set_title("er_xy")
                #plt.tight_layout(pad = 1, w_pad=0.2, h_pad=.5)
                #plt.savefig(save_path + 'bulshit/BulshitPhase'+str(k) + '.png',  bbox_inches='tight',pad_inches = 0.05)
                #plt.close(fig)



'''





























