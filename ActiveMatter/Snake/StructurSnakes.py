import numpy as np

import scipy as scipy
from scipy import spatial

import matplotlib.pyplot as plt
import pygame
import math
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.colors import LightSource, Normalize
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import scipy.io

from itertools import chain
from statistics import pvariance
import logging
import psutil
import os
import decimal
# reneame all uncorrecty dump files
# read_state
# plus point on border
# make voronoi
# find neghbot particle
# cals all parameter




def GetPsiK(x0y0, xy_neighs, k):
    # if region_mask[region_id]:
    #     psi_6[jj] = ret_psi_k(vor.points[jj], vor.points[neigh_id[jj]], 6)
    psi6 = 0
    for j in range(len(xy_neighs)):
        r_ij = xy_neighs[j, :] - x0y0
        theta_ij = angle_of_vector_Ox(r_ij[0], r_ij[1])
        psi6 += np.exp(complex(0, k * theta_ij))
    psi6 /= len(xy_neighs)
    psi6 = abs(psi6)
    return psi6

def PolyArea(x,y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def read_state(f):
    f.readline()                                                                        # ITEM: TIMESTEP
    f.readline()                                                                        # 0
    f.readline()                                                                        # ITEM: NUMBER OF ATOMS
    N = int(f.readline())
    f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
    xsize = [float(item) for item in f.readline().split(' ')]
    ysize = [float(item) for item in f.readline().split(' ')]
    zsize = [float(item) for item in f.readline().split(' ')]
    f.readline()                                                                        # ITEM: ATOMS id ...
    #print(f.readline().strip().split(' ')[2:-1])
    d=[[float(item) for item in f.readline().strip().split(' ')[0:4]] for i in range(N)]
    return [N, xsize, ysize, d]

# dipole35_0.01 : 35 - box length; 0.01 - temperature of the system
#start_time = time.time()

def AddPoints(st_mas, percent):
    # ========================== INFORMATION ===============================
    # прога для добавления доплноительных частиц в массив данных
    # ============= INPUT ============
    # st_mas - исходный массив координат и скоростей,  x_1 y_1 vx_1 vy_1
    # ============= OUTPUT ===========
    # dat_xyz - массив дополненных частиц x_i y_i vx_i vy_i
    # x_left, x_right, y_bottom, y_up - левая правая нижняя и верхняя граница области моделирования
    # x_box_delta, y_box_delta - размер дельта области для добавления частиц

    dat_xyz = np.array(st_mas[-1])
    id_particle    = dat_xyz[:, 0]
    x_particle     = dat_xyz[:, 1]
    y_particle     = dat_xyz[:, 2]
    z_particle     = dat_xyz[:, 3]

    #vor = scipy.spatial.Voronoi(dat[:,0:2])

    # ======== define box size and area of cut line ===================
    x_left      = st_mas[1][0]
    x_right     = st_mas[1][1]
    y_up        = st_mas[2][1]
    y_bottom    = st_mas[2][0]
    x_box       = x_right - x_left
    y_box       = y_up - y_bottom
    x_box_delta = percent * x_box
    y_box_delta = percent * y_box
    # ==================================================================
    xy_part_dop = np.array(1)
    #counter1 = 0

    for i in range(st_mas[0]):

        if x_particle[i] < x_left + x_box_delta:

            dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i] + x_box, y_particle[i], z_particle[i]]], axis = 0)

            if  y_particle[i] < y_bottom + y_box_delta:

                dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i] + x_box , y_particle[i] + y_box, z_particle[i] ]], axis = 0)
                dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] + y_box, z_particle[i] ]], axis = 0)

            elif  y_particle[i] > y_up - y_box_delta:

                dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i] + x_box, y_particle[i] - y_box, z_particle[i] ]], axis = 0)
                dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] - y_box, z_particle[i] ]], axis = 0)

        elif x_particle[i] > x_right - x_box_delta:

            dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i] - x_box, y_particle[i], z_particle[i] ]], axis = 0)

            if y_particle[i] < y_bottom + y_box_delta:

                dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i] - x_box, y_particle[i] + y_box, z_particle[i] ]], axis = 0)
                dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] + y_box, z_particle[i] ]], axis = 0)

            elif y_particle[i] > y_up - y_box_delta:

                dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i] - x_box, y_particle[i] - y_box, z_particle[i] ]], axis = 0)
                dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] - y_box, z_particle[i] ]], axis = 0)

        elif y_particle[i] > y_up - y_box_delta:

            dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] - y_box, z_particle[i] ]], axis = 0)

        elif y_particle[i] < y_bottom + y_box_delta:

            dat_xyz = np.append(dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] + y_box, z_particle[i] ]], axis = 0)

    return dat_xyz, x_left, x_right, y_bottom, y_up, x_box_delta, y_box_delta

def pairCorrelationFunction_2D(x, y, Sx, Sy, rMax, dr):
    # Find particles which are close enough to the box center that a circle of radius
    # rMax will not cross any edge of the box
    bools1 = x > rMax
    bools2 = x < (Sx - rMax)
    bools3 = y > rMax
    bools4 = y < (Sy - rMax)
    interior_indices, = np.where(bools1 * bools2 * bools3 * bools4)
    num_interior_particles = len(interior_indices)

    if num_interior_particles < 1:
        raise  RuntimeError ("No particles found for which a circle of radius rMax\
                will lie entirely within a square of side length S.  Decrease rMax\
                or increase the size of the square.")

    edges = np.arange(0., rMax + 1.1 * dr, dr)
    num_increments = len(edges) - 1
    g = np.zeros([num_interior_particles, num_increments])
    radii = np.zeros(num_increments)
    numberDensity = len(x) / Sx / Sy

    # Compute pairwise correlation for each interior particle
    for p in range(num_interior_particles):
        index = interior_indices[p]
        d = np.sqrt((x[index] - x)**2 + (y[index] - y)**2)
        d[index] = 2 * rMax

        (result, bins) = np.histogram(d, bins=edges)#, normed=False)
        g[p, :] = result/numberDensity

    # Average g(r) for all interior particles and compute radii
    g_average = np.zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = np.mean(g[:, i]) / (np.pi * (rOuter**2 - rInner**2))

    return (g_average, radii, interior_indices)

def GetVoronoiNeighbors(dat_xyz1, xy, dx, dy, n_parts_init, xmin_init, xmax_init, ymin_init, ymax_init):
    vor = scipy.spatial.Voronoi(xy)
    npoints = len(vor.points)
    ver_index = np.full(len(vor.vertices),False)
    region_mask = np.full(len(vor.regions), False)
    neigh_id=[[] for i in range(n_parts_init)]

    #----------------------------------
    # otsechenie pogranichnux regionov i poisk min max ploshadei
    #----------------------------------
    for i in range(npoints):
        ind = vor.point_region[i]
        certies = vor.regions[ind]
        if (vor.points[i][0] > xmin_init  - 0.5 * dx  and vor.points[i][0] < xmax_init + 0.5*dx and vor.points[i][1] > ymin_init - 0.5*dy  and vor.points[i][1] < ymax_init + 0.5*dy):
            ver_index[certies] = True

    #----------------------------------
    # otsechenie pogranichnux regionov i poisk min max ploshadei
    #----------------------------------
    for jj in range(len(vor.regions)):
        region = vor.regions[jj]
        if not -1 in region and not False in ver_index[region]:
            region_mask[jj] = True
        else:
            region_mask[jj] = False

    #----------------------------------
    # sostavlenie spiska blishaishix sosedei (neigh_id)  neigh_id[i] = [id_1, id_2, ... , id_n];   i - index of particle, id_1,2...n - индексы ссоседних частиц
    #----------------------------------
    for rid in vor.ridge_points:

        rid0 = int(dat_xyz1[rid[0], 0])
        rid1 = int(dat_xyz1[rid[1], 0])
        if rid[0] == 998 or rid[1] == 998:
            print(str(rid1) + "; " + str(rid0))

        if region_mask[vor.point_region[rid[0]]] + region_mask[vor.point_region[rid[1]]] == True:   # at least one particle in the region
                                    # two adjacent particles: one inside the other outside the region
            if rid1 not in neigh_id[rid0]:  # condition to avoid repetition
                neigh_id[rid0].append(rid1)
            if rid0 not in neigh_id[rid1]:
                neigh_id[rid1].append(rid0)
    return neigh_id, vor, region_mask




def IdentShake(t_i, xy, r_snake, xmin, xmax, ymin, ymax, partDiam, size, posNeighbor):
    snakes_id = []
    len_snakes = []
    snakes_info = []



    # get neighbors from Lammps file Voronoi
    neigh_id, posNeighbor = getNeighbors(data_path+'dump'+str(size)+'_' + str(t_i) + '.neighbors', N, posNeighbor)
    #print(neigh_id[1])


    jj_using_glob = np.zeros(N+1)
    for jj in range(1,N+1):
        if jj_using_glob[jj] > 0: #check if jj particle is in snake, skip
            continue


        #j_help = np.zeros(N+1) #j_help = []      #array of considered particles
        jj_using = [] #array of particles used in snake
        jj_using.append(jj) #jj - first particle in snake
        jj_using_glob[jj] = 1 #jj particle is in snake
        length = 0

        jj_el_snake = jj
        while True: #search end of chain
            x0 = xy[jj_el_snake, 0]
            y0 = xy[jj_el_snake, 1]

            # neighbors coordinates
            x1 = xy[neigh_id[jj_el_snake], 0]
            y1 = xy[neigh_id[jj_el_snake], 1]
            r=[]

            # distances between central partilce and her neighbors
            for i in range(len(neigh_id[jj_el_snake])):

                if(abs(x1[i] - x0) > xmax/2 and x1[i] < x0):
                    xi = x1[i] + xmax
                elif(abs(x1[i] - x0) > xmax/2 and x1[i] > x0):
                    xi = x1[i] - xmax
                else:
                    xi = x1[i]

                if(abs(y1[i] - y0) > ymax/2 and y1[i] < y0):
                    yi = y1[i] + ymax
                elif(abs(y1[i] - y0) > ymax/2 and y1[i] > y0):
                    yi = y1[i] - ymax
                else:
                    yi = y1[i]

                r_i = np.sqrt((xi - x0)*(xi - x0) + (yi - y0)*(yi - y0))
                if r_i < r_snake:
                    r.append([i, r_i])


            # condition for snake existing
            if len(r) > 1:
                if len(r) == 2:
                    if neigh_id[jj_el_snake][r[0][0]] in jj_using and neigh_id[jj_el_snake][r[1][0]] in jj_using:
                        break
                for item in r:
                    if neigh_id[jj_el_snake][item[0]] not in jj_using:
                        length += item[1] #count number of particles in snake
                        jj_using.insert(0, neigh_id[jj_el_snake][item[0]]) #add particle to start of array of particles used in snake
                        jj_using_glob[neigh_id[jj_el_snake][item[0]]] = 1 #jj particle is in snake
                        jj_el_snake = neigh_id[jj_el_snake][item[0]]
                        break
            else:
                break #find the end


        del(jj_el_snake)
        for jj_el_snake in jj_using: #loop by particles in current snake


            if jj_el_snake != jj_using[-1]: #skip considered particles, search for end of array
                continue


            x0 = xy[jj_el_snake, 0]
            y0 = xy[jj_el_snake, 1]

            # neighbors coordinates
            x1 = xy[neigh_id[jj_el_snake], 0]
            y1 = xy[neigh_id[jj_el_snake], 1]
            r=[]

            # distances between central partilce and her neighbors
            for i in range(len(neigh_id[jj_el_snake])):
                if(abs(x1[i] - x0) > xmax/2 and x1[i] < x0):
                    xi = x1[i] + xmax
                elif(abs(x1[i] - x0) > xmax/2 and x1[i] > x0):
                    xi = x1[i] - xmax
                else:
                    xi = x1[i]

                if(abs(y1[i] - y0) > ymax/2 and y1[i] < y0):
                    yi = y1[i] + ymax
                elif(abs(y1[i] - y0) > ymax/2 and y1[i] > y0):
                    yi = y1[i] - ymax
                else:
                    yi = y1[i]

                r_i = np.sqrt((xi - x0) * (xi - x0) + (yi - y0) * (yi - y0))
                if r_i < r_snake:
                    r.append([i, r_i])

            # condition for snake existing
            if r:
                for item in r: #(index of neighbor, r)

                    #plus neighbors partilce id in list
                    if neigh_id[jj_el_snake][item[0]] not in jj_using:
                        length += item[1] #count number of particles in snake
                        jj_using.append(neigh_id[jj_el_snake][item[0]]) #add particle to array of particles used in snake
                        jj_using_glob[neigh_id[jj_el_snake][item[0]]] = 1 #jj particle is in snake



        snakes_id.append(jj_using) #add current snake to array of snakes' ID
        len_snakes.append(length) #add length of current snake to array of snakes' lengths


    #checkOrder(snakes_id)

    return  snakes_id, len_snakes, posNeighbor

def plot_snakes(size, frame, xy, xmin, xmax, ymin, ymax, snakes_id, img_path):
    #plot identified snakes

    ff=plt.figure(figsize=(22,22 * (ymax - ymin) / (xmax - xmin)))#frameon=False
    ax = plt.subplot(111)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_aspect('equal', 'datalim')


    list_i = 0
    for list_el in snakes_id:    #mh1
        ax.plot(xy[list_el, 0], xy[list_el, 1], 'o', markersize=12)    #mh1

        #for id_part in list_el:
            #ax.text(xy[id_part, 0], xy[id_part, 1], str(id_part), fontsize=15)

    # for i in range(len(xy)):
    #     ax.text(xy[i, 0], xy[i, 1], str(xy[i,2]), fontsize=15)
    # ax.plot(xy[:, 0], xy[:, 1], 'o', markersize=12, color = 'Black')


    plt.savefig(img_path + 'v2w' + str(size) + '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
    ff.clear()
    plt.close(ff)





def PlotSystem(frame, x, y, xmin, xmax, ymin, ymax, img_path):
    ff=plt.figure(figsize=(22,22 * (ymax - ymin) / (xmax - xmin)))#frameon=False
    ax = plt.subplot(111)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax.set_aspect('equal', 'datalim')

    ax.plot(x, y, '.')    #mh1
    plt.savefig(img_path + 'snapshot' +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
    ff.clear()
    plt.close(ff)



def PlotVoroSystem(frame, n_parts_init, neigh_id,vor, region_mask, x, y, xmin, xmax, ymin, ymax, xmin_init, xmax_init, ymin_init, ymax_init, img_path):
    ff=plt.figure(figsize=(30,30 * (ymax - ymin) / (xmax - xmin)))#frameon=False
    ax = plt.subplot(111)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax.set_aspect('equal', 'datalim')

    for jj in range(len(vor.points)):
        region_id = vor.point_region[jj]

        if region_mask[region_id]:
            polygon = [vor.vertices[i] for i in vor.regions[region_id]]
            ax.fill(*zip(*polygon), facecolor = "white", edgecolor = 'black', linewidth = 0.5)
            #ax.plot(vor.points[jj][0], vor.points[jj][1], "o")
            #if jj > n_parts_init - 1:
            ax.text(vor.points[jj][0], vor.points[jj][1], str(int(dat_xyz1[jj, 0])) )
            #else:
            #    ax.text(vor.points[jj][0], vor.points[jj][1], str(jj))

    rect = patches.Rectangle((xmin_init, ymin_init), xmax_init - xmin_init, ymax_init - ymin_init, linewidth = 4, edgecolor='Black', facecolor='none')
    ax.add_patch(rect)
    ax.grid(color='r', linestyle='-', linewidth=2)

    plt.savefig(img_path + 'Neigh' +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
    ff.clear()
    plt.close(ff)



def ReneameDump(name, path, prefix, end_name):
    #path = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/"
    #end_name = "_0.lammpstrj"
    new_prefix = prefix
    #name = path + new_prefix + end_name
    try:
        f = open(name, "r")
        f.close()
        print("yes;   " + name)
    except:
        print("NON;   " + name)
        for filename in os.listdir(path):
            if filename.startswith(new_prefix):
                #print(filename)
                new_filename = new_prefix + end_name
                #print(new_filename)
                os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
                print(new_filename)

# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================
# =============== MAIN LOOP ==========================================================

path_md = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/MD/Nikita/Data/"
img_path = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/result/ris/"
prefix = "dump_100_100_"
density = [0.7]#[0.7, 0.8]
num_h = 1#20
heights = np.arange(1., 1. + 0.025 * num_h, 0.025)
num_mu = 1#50
# 1 - 10.8 for rho = 0.7
mu_values = np.arange(5.0, 5. +0.2 * num_mu, 0.2)
Temperature = 1.0
istart = 2
all_cadr = 15
num_cadr = 2

for id_rho in range(len(density)):
    rho = density[id_rho]
    for id_h in range(num_h):
        # dump_100_100_0.7_1.1_1.0_7.2_0.lammpstrj
        # dump_100_100_0.7_1.1_1.0_6.6_0.lammpstrj
        #h = 1.+0.025*id_h
        h = 1.1
        for id_mu in range(num_mu):
            #mu = round(mu_values[id_mu], 1)
            mu = 6.6
            new_prefix = prefix + str(rho) + "_" + str(h) + "_" + str(Temperature) + "_" + str(mu)
            name = path_md + new_prefix + "_0.lammpstrj"
            name = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/MD/dipole35_0.5.lammpstrj"
            f = open(name, "r")
            for skip in range(istart):
                st = read_state(f)
            counter_for_skip = 0
            for index in range(istart, all_cadr - 10, int((all_cadr - 1 ) / num_cadr)):
                if counter_for_skip > 0:
                    print('index__1 = ', index__1, ";   index = ", index)
                    for skip in range(index__1, index - 1, 1):
                        st=read_state(f)
                index__1 = index
                counter_for_skip += 1
                print('counter_for_skip = ', counter_for_skip)
                st  = read_state(f)
                nparts_init = st[0]
                dat_xyz = np.array(st[-1])
                id_particle    = dat_xyz[:, 0]
                x_particle     = dat_xyz[:, 1]
                y_particle     = dat_xyz[:, 2]
                z_particle     = dat_xyz[:, 3]
                xmin_init, xmax_init, ymin_init, ymax_init = min(x_particle), max(x_particle), min(y_particle), max(y_particle)
                PlotSystem(index__1, x_particle, y_particle,  xmin_init, xmax_init, ymin_init, ymax_init, img_path)

                dat_xyz1, x_left, x_right, y_bottom, y_up, dx, dy = AddPoints(st, 0.1)
                dat_xyz1[:, 0] -= 1 # onlu for MD when id particles start from 1 not from 0
                x_particle     = dat_xyz1[:, 1]
                y_particle     = dat_xyz1[:, 2]
                z_particle     = dat_xyz1[:, 3]
                xmin, xmax, ymin, ymax = min(x_particle), max(x_particle), min(y_particle), max(y_particle)
                PlotSystem(index__1, x_particle, y_particle,  xmin, xmax, ymin, ymax, img_path + "Add_")

                neigh_id, vor, region_mask = GetVoronoiNeighbors(dat_xyz1, dat_xyz1[:, 1:3], dx, dy, nparts_init, xmin_init, xmax_init, ymin_init, ymax_init)
                PlotVoroSystem(index__1, nparts_init, neigh_id, vor, region_mask, x_particle, y_particle, xmin, xmax, ymin, ymax, xmin_init, xmax_init, ymin_init, ymax_init, img_path)
                print(neigh_id[531])
                print(neigh_id[559])
                print(neigh_id[46])
                print(neigh_id[33])
                print(neigh_id[1498])



            #ReneameDump(name, path_md, new_prefix,  "_0.lammpstrj")





'''
for t_i in tArr:

    for size in sizeArr:   #35, 36 doesn't work for some t

        #print(size)
        r_snake = 0.8
        dr = 0.1
        rMax = 10
        partDiam = 1.0        #particle's diametr
        r_mas =  np.arange(0., rMax + 1.1 * dr, dr)
        Gr = np.zeros(len(r_mas) - 1)

        pos = 0
        posNeighbor = 0

        for _ in range(10):
            xy, pos = getData(data_path+'dipole'+str(size)+'_' + str(t_i) + '.lammpstrj', N, pos)
            neigh_id, posNeighbor=getNeighbors(data_path+'dump'+str(size)+'_' + str(t_i) + '.neighbors', N, posNeighbor)

        count = 0
        mean_len_snakes = 0
        mean_var = 0

        bins_num = 50
        len_snakes_glob = []
        for frame in range(50):
            count += 1
            xy, pos = getData(data_path+'dipole'+str(size)+'_' + str(t_i) + '.lammpstrj', N, pos)
            xmin = min(xy[:, 0])
            xmax = max(xy[:, 0])
            ymin = min(xy[:, 1])
            ymax = max(xy[:, 1])


            xy[:, 0] += abs(xmin)
            xy[:, 1] += abs(ymin)

            xbox = (xmax - xmin)
            ybox = (ymax - ymin)

            xmin = (min(xy[:, 0]))
            xmax = (max(xy[:, 0]))
            ymin = (min(xy[:, 1]))
            ymax = (max(xy[:, 1]))

            # snakes_id=[]
            # plot_snakes(size, frame, xy, xmin, xmax, ymin, ymax, snakes_id, img_path+'/temp/')
            snakes_id, len_snakes, posNeighbor = IdentShake(t_i, xy, r_snake, xmin, xmax, ymin, ymax, partDiam, size, posNeighbor)


            mean_len_snakes += sum(len_snakes) / len(len_snakes)
            mean_var += pvariance(len_snakes)
            len_snakes_glob += len_snakes


region_mask = np.full(len(vor.regions), 0)
for jj in range(len(vor.points)):
region_id = vor.point_region[jj]
if xy[jj, 0] > xmin_init  and xy[jj, 0] < xmax_init  and xy[jj, 1] > ymin_init and xy[jj, 1] < ymax_init:
    region_mask[region_id] = 1










'''

































