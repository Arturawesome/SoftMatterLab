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
def read_er_m_p(f, N_relax, N_run, thermo):
    for i in range(int(N_relax / thermo)):
        f.readline()

    d=[[float(item) for item in f.readline().strip().split('\t')[2::]] for i in range(int(N_run/thermo))]
    return np.array(d)

def get_disp(np_array):
    mean = sum(np_array) / len(np_array)
    D = sum(np.square(np_array - mean) ) / len(np_array)
    return D

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
    plt.savefig(name + '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)



path = '/run/media/artur/Новый том/FISH_MD_data/Fish/dump/'
dump_folders = ['dump']
print('/run/media/artur/Новый том/FISH_MD_data/Fish/dump/log_0_100_5_Ip_In_0.1_20000_100000_100_0_0.6_0.99_0.0.txt')
name = 'log_0_100_5_' + 'Ip'+'_' + 'In'+'_0.1_20000_100000_100_0_0.6_0.99_0.0.txt'


log_data = np.arange(0,10, 1) #0, 1, 2, 3, 4...
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
for i in range(len(log_data)):
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
f_m = open("milling.txt", "w")
for lgd in range(len(log_data)):
    for i_n in range(len(In)): #
        for i_p in range(len(Ip)): # ip
            k = (len(In) *len(Ip)) * lgd + len(Ip)*i_n + i_p

            f = open(name_log[k], "r")
            st = read_er_m_p(f, N_relax, N_run, thermo)

            x_cm = st[:, 0]
            y_cm = st[:, 1]


            px = st[:, 2]
            py = st[:, 3]
            p_xy = np.sqrt(np.square(px) + np.square(py))

            erx = st[:, 4]
            ery = st[:, 5]


            milling = abs(st[:, 6])
            size = st[:, 7]

            av_p_in_run[k] =  sum(p_xy) / len(p_xy)
            av_mil_in_run[k] = sum(milling) / len(milling)

            min_val_min = sorted(milling)[:50]
            min_mil = sum(min_val_min) / 50
            min_val_mill_run[k] = min_mil

            val_mill_run_full[k] = sum(milling) / len(milling)
            val_d_mill_run_full[k] = get_disp(milling)

            if np.log(min_mil) > 2.2:
                cear_milling_id_run.append(k)
                f_m.write(name_log[k] + "\n")
                clear_milling_p.append(sum(p_xy) / len(p_xy))
                clear_milling_m.append(sum(milling) / len(milling))
                val_in_clear_milling.append(sum(milling) / len(milling))
                val_in_clear_milling_dp.append(val_d_mill_run_full[k])

                count_clear_milling += 1

                p_xy_medfilt = np.array(scipy.signal.medfilt(p_xy, 5))

                medfilt_dat_clear_mill.append(sum(sorted(p_xy_medfilt)[:-10:-1]) / len(p_xy_medfilt[:-10:-1]))
                #medfilt_dat_clear_mill.append(sum(sorted(p_xy_medfilt)[:-10:-1]) / len(p_xy_medfilt[:-5:-1]))
                #ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
                #plt.plot(time, p_xy,  '.')
                #plt.plot(time, p_xy_medfilt,  '-')
                #plt.savefig(save_path + 'P(ts)_medfilt'+str(Iread_er_m_pp[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
                #plt.close(ff)


            if np.log(val_d_mill_run_full[k]) < -10:
                noise_m.append(sum(milling) / len(milling))
                noise_p.append(sum(p_xy) / len(p_xy))


            #min_val_pol = sorted(p_xy)[:50]
            #min_pol = sum(min_val_pol)/50

            #min_val_polar_run[k] = min_pol
            #min_val_dp_run[k] = get_disp(min_val_pol)





f_m.close()
medfilt_dat_clear_mill = np.array(medfilt_dat_clear_mill)
len_ar = len(medfilt_dat_clear_mill)
polar_gran_in_clear = sum(sorted(medfilt_dat_clear_mill)[:-int(len_ar/2):-1]) / len(sorted(medfilt_dat_clear_mill)[:-int(len_ar/2):-1])
#polar_gran_in_clear = sum(medfilt_dat_clear_mill) / len(medfilt_dat_clear_mill)
medfilt_dat_clear_mill = np.zeros(int(N_run/thermo))
medfilt_dat_clear_mill += polar_gran_in_clear

bufurcation_id_runs = []
clear_scholong = []
polar_clear_scholong_runs = []
milling_clear_scholongn_runs = []

clear_swarming = []

polar_clear_swarming_runs = []
milling_clear_swarming_runs = []

polar_bifurcation_runs = []
milling_bifurcation_runs = []
val_in_bifurcation = []
f_swarm = open("swarm_info.txt", "w")
f_skul = open("skhul_info.txt", "w")

count_down_global = []
count_up_global = []
relationship_time_bif = []

frequency = []
disp_polar_in_not_clean_phase = []
aver_polar_in_not_clean_phase = []
for lgd in range(len(log_data)):
    for i_n in range(len(In)): #
        for i_p in range(len(Ip)): # ip
            k = (len(In) *len(Ip)) * lgd + len(Ip)*i_n + i_p

            if k in cear_milling_id_run:
                continue

            f = open(name_log[k], "r")
            st = read_er_m_p(f, N_relax, N_run, thermo)

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


            if sum(p_xy_medfilt > polar_gran_in_clear) < 1:
                clear_swarming.append(k)
                f_swarm.write(name_log[k] + "\n")
                polar_clear_swarming_runs.append(sum(p_xy) / len(p_xy))
                milling_clear_swarming_runs.append(sum(milling) / len(milling))

            elif sum(p_xy_medfilt < polar_gran_in_clear) < 1:
                # schooling init

                clear_scholong.append(k)
                f_skul.write(name_log[k] + "\n")
                polar_clear_scholong_runs.append(sum(p_xy) / len(p_xy))
                milling_clear_scholongn_runs.append(sum(milling) / len(milling))


            else:
                #ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=True, facecolor = "white")
                #plt.plot(time, p_xy,  '.')
                #plt.plot(time, p_xy_medfilt,  '-', color = "black")
                #plt.plot(time, medfilt_dat_clear_mill,  '-', color ="red")


                #plt.title(name_log[k])
                #plt.savefig(save_path + 'p3(ts)_medfilt'+ str(k)+ "_"+str(In[i_n]) +'_Ip_'+str(Ip[i_p])+ '.png',  bbox_inches='tight',pad_inches = 0.05)
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



















count_down_global = np.array(count_down_global)
count_up_global = np.array(count_up_global)


ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.hist(aver_polar_in_not_clean_phase, 100)
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'aver_polar_in_not_clean_phase'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.plot(aver_polar_in_not_clean_phase,frequency, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'Polar_frequency'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.plot(disp_polar_in_not_clean_phase,frequency, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'DispPolar_frequency'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.plot(count_down_global,disp_polar_in_not_clean_phase, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'count_down_global_disp_polar'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.plot(count_up_global,disp_polar_in_not_clean_phase, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'count_up_global_disp_polar'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.plot(aver_polar_in_not_clean_phase,disp_polar_in_not_clean_phase, ".")
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'Polar_disp_polar'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.hist(disp_polar_in_not_clean_phase, 100)
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'disp_polar_in_not_clean_phase'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)



#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(np.log(count_down_global[count_down_global>-1]), 150)
##plt.xlim([-6,6])
#plt.title(f"down value")
#plt.savefig(save_path + 'logDown_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.hist(frequency, 250)
#plt.xlim([-6,6])
plt.title(f"down value")
plt.savefig(save_path + 'frequency'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(np.log(count_up_global[count_up_global>-1]), 150)
##plt.xlim([-6,6])
#plt.title(f"up value")
#plt.savefig(save_path + 'logUp_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)

#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(count_down_global[count_down_global>-1], 150)
##plt.xlim([-6,6])
#plt.title(f"down value")
#plt.savefig(save_path + 'Down_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)

#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(count_up_global[count_up_global>-1], 150)
##plt.xlim([-6,6])
#plt.title(f"up value")
#plt.savefig(save_path + 'Up_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)

#relation = count_down_global / count_up_global
#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(relation, 100)
#plt.xlim([1,999])
#plt.title(f"up value")
#plt.savefig(save_path + 'relationship_time_bif_Gist_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
#plt.close(ff)



ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)

plt.plot(count_down_global, count_up_global,  '.')
plt.xlabel("time under meanfilt")
plt.ylabel("time over meanfilt")

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + '2D_time'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)

plt.plot(np.log(count_down_global), np.log(count_up_global),  '.')
plt.xlabel("time under meanfilt")
plt.ylabel("time over meanfilt")

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'log_2D_time'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

'''
ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.hist(np.log(min_val_mill_run), n_elem_in_gm)
plt.xlim([-6,6])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'Gist_M1_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.hist(np.log(val_mill_run_full), n_elem_in_gm)
plt.hist(np.log(val_in_clear_milling), int(n_elem_in_gm/6))
plt.xlim([-6,12])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'Gist_Mful_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)

ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
plt.hist(np.log(val_d_mill_run_full), n_elem_in_gm)
plt.hist(np.log(val_in_clear_milling_dp), int(n_elem_in_gm/6))
#plt.xlim([-6,6])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'Gist_M_disp_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)


#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(min_val_polar_run, n_elem_in_gp)
##plt.xlim([-6,6])
#plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
#plt.savefig(save_path + 'Gist_P1_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)

#plt.close(ff)

#ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)
#plt.hist(min_val_dp_run, n_elem_in_g_dp)
#plt.xlim([0,0.1])
#plt.ylim([0, 300])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'Gist_PDISP_'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
plt.close(ff)




ff=plt.figure(figsize=(15.0,15.0 * 0.8), frameon=False)

plt.plot(av_p_in_run, np.log(av_mil_in_run),  '.')
plt.plot(clear_milling_p, np.log(clear_milling_m),  '.')
plt.plot(polar_clear_swarming_runs, np.log(milling_clear_swarming_runs),  '.', color = "blue")
plt.plot(polar_clear_scholong_runs, np.log(milling_clear_scholongn_runs),  '.', color = "red")
plt.ylim([-5, 12])

plt.title(f"Ip = {Ip[i_p]}, In = {In[i_n]}")
plt.savefig(save_path + 'P_M_depend'+str(Ip[i_n]) +'_In_'+str(Ip[i_p])+ '.pdf', transparent=True, bbox_inches='tight',pad_inches = 0.05)
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






'''





























