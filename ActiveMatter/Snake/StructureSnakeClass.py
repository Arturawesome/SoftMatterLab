import numpy as np
import DataSetWorks1
from DataSetWorks1 import LammpsCoords

import StructuralAnalysis
from StructuralAnalysis import SyatemStructur
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



class Snake(SyatemStructur):
    def __init__(self,
                 heights: list,
                 mu_values: list,
                 Temperature: float,
                 percent: float,
                 density: list,
                 list_psi: list,
                 istart: int,
                 all_cadr: int,
                 num_cadr: int,

                 path_md: str,
                 img_path: str,
                 prefix:str
                 ):

        # all path and txt variables
        self.img_path = img_path
        self.path_md = path_md
        self.prefix = prefix
        self.list_psi = list_psi

        # all nympy variables
        self.heights = np.array(heights, dtype=float)
        self.mu_values = np.array(mu_values,  dtype=float)
        self.density = np.array(density, dtype = float)
        self.z_distance = []
        self.r_mas = []
        self.st = None
        self.dat_xyz = None     # np.array after Add_Points() dat_xyz[:, 0] - MD id particle, dat_xyz[:, 1:4] - xyz position
        self.dat_xyz_init = None # np.array dat_xyz[:, 0] - MD id particle, dat_xyz[:, 1:4] - xyz position
        #self.dat_id_part_inside_init = None
        self.r_ij = None

        # all float variables
        self.Temperature = Temperature
        self.percent = percent  #
        self.x_box = None
        self.y_box = None
        self.dx = None
        self.dy = None
        self.xmin_init = None
        self.xmax_init = None
        self.ymin_init = None
        self.ymax_init = None

        # all integer variables
        self.num_h = len(self.heights)
        self.num_mu  = len(self.mu_values)
        self.istart = istart
        self.all_cadr = all_cadr
        self.num_cadr = num_cadr
        self.nparts_init = None

        # VORONOI VARIABLES
        self.vor = None
        self.region_mask = None
        self.neigh_id = None

        # All snake's variables
        self.snakes_id = []
        self.len_snakes = []
        self.r_snake =  0.8

        # Structure variables
        self.psi = None
        self.psi3 = None
        self.psi4 = None
        self.psi5 = None
        self.psi6 = None
        self.psi7 = None


    def Get_Voronoi_Neighbors(self):
        # function for
        self.vor = scipy.spatial.Voronoi(self.dat_xyz[:, 1:3])
        self.region_mask = np.full(len(self.vor.regions), False)
        npoints = len(self.vor.points)
        ver_index = np.full(len(self.vor.vertices),False)
        self.neigh_id = [[] for i in range(self.nparts_init)]
        self.r_ij = [[] for _ in range(self.nparts_init)]
        #self.dat_id_part_inside_init = np.zeros(self.nparts_init)

        #----------------------------------
        # otsechenie pogranichnux regionov i poisk min max ploshadei
        #----------------------------------
        for i in range(npoints):
            ind = self.vor.point_region[i]
            certies = self.vor.regions[ind]
            if (self.vor.points[i][0] > self.xmin_init  - 0.5 * self.dx  and self.vor.points[i][0] < self.xmax_init + 0.5*self.dx and self.vor.points[i][1] > self.ymin_init - 0.5*self.dy  and self.vor.points[i][1] < self.ymax_init + 0.5*self.dy):
                ver_index[certies] = True
            #
            # if i < self.nparts_init:
            #     if (self.dat_xyz[i, 1] > self.xmin_init  + 0.5 * self.dx  and self.dat_xyz[i, 1] < self.xmax_init - 0.5*self.dx and self.dat_xyz[i, 2] > self.ymin_init + 0.5*self.dy  and self.dat_xyz[i, 1] < self.ymax_init - 0.5*self.dy):
            #         id_part = self.dat_xyz_init[i]
            #         self.dat_id_part_inside_init[id_part] = 1

        #----------------------------------
        # otsechenie pogranichnux regionov i poisk min max ploshadei
        #----------------------------------
        for jj in range(len(self.vor.regions)):
            region = self.vor.regions[jj]
            if not -1 in region and not False in ver_index[region]:
                self.region_mask[jj] = True
            else:
                self.region_mask[jj] = False

        #----------------------------------
        # sostavlenie spiska blishaishix sosedei (self.neigh_id)  self.neigh_id[i] = [id_1, id_2, ... , id_n];   i - index of particle, id_1,2...n - индексы ссоседних частиц
        #----------------------------------
        for rid in self.vor.ridge_points:   # get adjasent particles - > [id1, id2] -> id1 and id2 - index of particle in array
                                            # but trully id of particle from MD consist in dat_xyz[:, 0] ()
            rid0 = int(self.dat_xyz[rid[0], 0]) # index of particle
            rid1 = int(self.dat_xyz[rid[1], 0]) # index of adjasent particle

            # at least one particle in the region two adjacent particles: one inside the other outside the region
            # region_id = self.vor.point_region[rid[0]]
            if self.region_mask[self.vor.point_region[rid[0]]] + self.region_mask[self.vor.point_region[rid[1]]] == True:
                if rid1 not in self.neigh_id[rid0]:  # condition to avoid repetition
                    self.neigh_id[rid0].append(rid1)

                    x0 = self.dat_xyz_init[rid0, 1]
                    y0 = self.dat_xyz_init[rid0, 2]
                    z0 = self.dat_xyz_init[rid0, 3]
                    x1 = self.dat_xyz_init[rid1, 1]
                    y1 = self.dat_xyz_init[rid1, 2]
                    z1 = self.dat_xyz_init[rid1, 3]

                    x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1)
                    self.r_ij[rid0].append([(x1 - x0), (y1 - y0), abs(z1 - z0)])

                if rid0 not in self.neigh_id[rid1]:
                    self.neigh_id[rid1].append(rid0)

                    x0 = self.dat_xyz_init[rid1, 1]
                    y0 = self.dat_xyz_init[rid1, 2]
                    z0 = self.dat_xyz_init[rid1, 3]
                    x1 = self.dat_xyz_init[rid0, 1]
                    y1 = self.dat_xyz_init[rid0, 2]
                    z1 = self.dat_xyz_init[rid0, 3]
                    x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1)
                    self.r_ij[rid1].append([(x1 - x0), (y1 - y0), abs(z1 - z0)])

        return self.neigh_id, self.vor, self.region_mask

    def Consideration_Periodic_Boundary_Distance(self, x0, y0, x1, y1):

        if(abs(x1 - x0) > self.xmax_init/2.2 and x1 < x0):
            xi = x1 + self.xmax_init
        elif(abs(x1 - x0) > self.xmax_init/2.2 and x1 > x0):
            xi = x1 - self.xmax_init
        else:
            xi = x1


        if(abs(y1 - y0) > self.ymax_init/2.2 and y1 < y0):
            yi = y1 + self.ymax_init
        elif(abs(y1 - y0) > self.ymax_init/2.2 and y1 > y0):
            yi = y1 - self.ymax_init
        else:
            yi = y1
        return xi, yi

    def Add_Points(self):
        # ========================== INFORMATION ===============================
        # function of adding additional points to the array
        # of particles taking into account the periodic boundary

        self.dat_xyz = np.array(self.st[-1])
        id_part    = self.dat_xyz[:, 0]
        x_part     = self.dat_xyz[:, 1]
        y_part     = self.dat_xyz[:, 2]
        z_part     = self.dat_xyz[:, 3]


        # ======== define box size and area of cut line ===================
        box_data = self.st[1]
        x_left      = box_data[0][0]
        x_right     = box_data[0][1]
        y_up        = box_data[1][1]
        y_bottom    = box_data[1][0]
        self.x_box       = x_right - x_left
        self.y_box       = y_up - y_bottom
        self.dx = self.percent * self.x_box
        self.dy = self.percent * self.y_box
        print("x_r = %s;   x_l = %s;   y_u = %s;   y_b = %s" %(x_left, x_right, y_up, y_bottom))

        for i in range(self.st[0]):

            if x_part[i] < x_left + self.dx:

                self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] + self.x_box, y_part[i], z_part[i]]], axis = 0)

                if  y_part[i] < y_bottom + self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] + self.x_box , y_part[i] + self.y_box, z_part[i] ]], axis = 0)
                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] + self.y_box, z_part[i] ]], axis = 0)

                elif  y_part[i] > y_up - self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] + self.x_box, y_part[i] - self.y_box, z_part[i] ]], axis = 0)
                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] - self.y_box, z_part[i] ]], axis = 0)

            elif x_part[i] > x_right - self.dx:

                self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] - self.x_box, y_part[i], z_part[i] ]], axis = 0)

                if y_part[i] < y_bottom + self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] - self.x_box, y_part[i] + self.y_box, z_part[i] ]], axis = 0)
                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] + self.y_box, z_part[i] ]], axis = 0)

                elif y_part[i] > y_up - self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i] - self.x_box, y_part[i] - self.y_box, z_part[i] ]], axis = 0)
                    self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] - self.y_box, z_part[i] ]], axis = 0)

            elif y_part[i] > y_up - self.dy:

                self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] - self.y_box, z_part[i] ]], axis = 0)

            elif y_part[i] < y_bottom + self.dy:

                self.dat_xyz = np.append(self.dat_xyz, [[id_part[i], x_part[i], y_part[i] + self.y_box, z_part[i] ]], axis = 0)

        #return self.dat_xyz, x_left, x_right, y_bottom, y_up, self.dx, self.dy

    def Plot_Voro_System(self, xy, plot_type, frame):
        ff=plt.figure(figsize=(30, 30 * (max(xy[:, 1]) - min(xy[:, 1])) / (max(xy[:, 0]) - min(xy[:, 0]))))
        ax = plt.subplot(111)
        ax.set_xlim([min(xy[:, 0]), max(xy[:, 0])])
        ax.set_ylim([min(xy[:, 1]), max(xy[:, 1])])

        ax.set_aspect('equal', 'datalim')
        ax.grid(color='r', linestyle='-', linewidth=2)

        for jj in range(len(self.vor.points)):
            region_id = self.vor.point_region[jj]
            if self.region_mask[region_id]:
                polygon = [self.vor.vertices[i] for i in self.vor.regions[region_id]]
                ax.fill(*zip(*polygon), facecolor = "white", edgecolor = 'black', linewidth = 0.7)
                ax.text(self.vor.points[jj][0], self.vor.points[jj][1], str(int(self.dat_xyz[jj, 0])))
                if jj<self.nparts_init:
                    ax.text(self.vor.points[jj][0], self.vor.points[jj][1], str(int(self.dat_xyz_init[jj, 0])))
                    ax.text(self.vor.points[jj][0], self.vor.points[jj][1], str(jj))

        rect = patches.Rectangle((self.xmin_init, self.ymin_init), self.xmax_init - self.xmin_init, self.ymax_init - self.ymin_init, linewidth = 5, edgecolor='Black', facecolor='none')
        ax.add_patch(rect)

        plt.savefig(self.img_path + plot_type +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
        ff.clear()
        plt.close(ff)

    def Init_Main_Variable(self):
        return 0

    def Plot_System(self, xy, plot_type, frame):
        ff = plt.figure(figsize=(22,22 * (max(xy[:, 1]) - min(xy[:, 1])) / (max(xy[:, 0]) - min(xy[:, 0]))))
        ax = plt.subplot(111)
        ax.set_xlim([min(xy[:, 0]), max(xy[:, 0])])
        ax.set_ylim([min(xy[:, 1]), max(xy[:, 1])])
        ax.set_aspect('equal', 'datalim')
        ax.plot(xy[:, 0], xy[:, 1], '.')    #mh1
        plt.savefig(self.img_path + plot_type +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
        ff.clear()
        plt.close(ff)

    def Plot_Snakes(self, xy, plot_type, frame):
        #plot identified snakes

        ff=plt.figure(figsize=(22,22 * (max(xy[:, 1]) - min(xy[:, 1])) / (max(xy[:, 0]) - min(xy[:, 0]))))
        ax = plt.subplot(111)
        ax.set_xlim([min(xy[:, 0]), max(xy[:, 0])])
        ax.set_ylim([min(xy[:, 1]), max(xy[:, 1])])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_aspect('equal', 'datalim')



        for list_el in self.snakes_id:    #mh1
            ax.plot(xy[list_el, 0], xy[list_el, 1], 'o', markersize=12)    #mh1

        for jj in range(self.nparts_init):
            ax.text(xy[jj, 0], xy[jj, 1], str(int(self.dat_xyz_init[jj, 0])))
        ax.plot(xy[:, 0], xy[:, 1], '.')    #mh1
            #for id_part in list_el:
                #ax.text(xy[id_part, 0], xy[id_part, 1], str(id_part), fontsize=15)

        # for i in range(len(xy)):
        #     ax.text(xy[i, 0], xy[i, 1], str(xy[i,2]), fontsize=15)
        # ax.plot(xy[:, 0], xy[:, 1], 'o', markersize=12, color = 'Black')


        plt.savefig(self.img_path + plot_type +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
        ff.clear()
        plt.close(ff)

    def IdentShake(self):
        self.snakes_id = []
        self.len_snakes = []
        snakes_info = []

        jj_using_glob = np.zeros(self.nparts_init)
        for jj in range(self.nparts_init ):
            if jj_using_glob[jj] > 0: # skip the particle if it's in a snake
                continue
            jj_using = [] #array of particles used in snake
            jj_using.append(jj) #jj - first particle in snake
            jj_using_glob[jj] = 1 #jj particle is in snake
            length = 0

            jj_el_snake = jj

            while True: #search end of chain
                x0 = self.dat_xyz_init[jj_el_snake, 1]
                y0 = self.dat_xyz_init[jj_el_snake, 2]

                # neighbors coordinates
                x1 = self.dat_xyz_init[self.neigh_id[jj_el_snake], 1]
                y1 = self.dat_xyz_init[self.neigh_id[jj_el_snake], 2]
                r = []

                # distances between central partilce and her neighbors
                for i in range(len(self.neigh_id[jj_el_snake])):

                    if(abs(x1[i] - x0) > self.xmax_init/2 and x1[i] < x0):
                        xi = x1[i] + self.xmax_init
                    elif(abs(x1[i] - x0) > self.xmax_init/2 and x1[i] > x0):
                        xi = x1[i] - self.xmax_init
                    else:
                        xi = x1[i]


                    if(abs(y1[i] - y0) > self.ymax_init/2 and y1[i] < y0):
                        yi = y1[i] + self.ymax_init
                    elif(abs(y1[i] - y0) > self.ymax_init/2 and y1[i] > y0):
                        yi = y1[i] - self.ymax_init
                    else:
                        yi = y1[i]

                    r_i = np.sqrt((xi - x0)*(xi - x0) + (yi - y0)*(yi - y0))

                    if r_i < self.r_snake:
                        r.append([i, r_i])


                # condition for snake existing
                if len(r) > 1:
                    if len(r) == 2:
                        if self.neigh_id[jj_el_snake][r[0][0]] in jj_using and self.neigh_id[jj_el_snake][r[1][0]] in jj_using:
                            break

                    for item in r:
                        if self.neigh_id[jj_el_snake][item[0]] not in jj_using:
                            length += item[1] #count number of particles in snake
                            jj_using.insert(0, self.neigh_id[jj_el_snake][item[0]]) #add particle to start of array of particles used in snake
                            jj_using_glob[self.neigh_id[jj_el_snake][item[0]]] = 1 #jj particle is in snake
                            if self.neigh_id[jj_el_snake][item[0]] == 506 or jj == 506:
                                print("kjslfdhgkldsfjahglkads:    ", end="")
                                print(len(self.snakes_id))
                            jj_el_snake = self.neigh_id[jj_el_snake][item[0]]
                            break
                else:
                    break #find the end


            del(jj_el_snake)
            for jj_el_snake in jj_using: #loop by particles in current snake
                if jj_el_snake != jj_using[-1]: #skip considered particles, search for end of array
                    continue

                x0 = self.dat_xyz_init[jj_el_snake, 1]
                y0 = self.dat_xyz_init[jj_el_snake, 2]

                # neighbors coordinates
                x1 = self.dat_xyz_init[self.neigh_id[jj_el_snake], 1]
                y1 = self.dat_xyz_init[self.neigh_id[jj_el_snake], 2]
                r = [] # for devination of snake r must be 3 or more EXAMPLE:  (---+----) [r1 r2 r3]     (---------) [r1 r2]
                        #                                                          |
                # distances between central partilce and her neighbors
                for i in range(len(self.neigh_id[jj_el_snake])):
                    if(abs(x1[i] - x0) > self.xmax_init/2 and x1[i] < x0):
                        xi = x1[i] + self.xmax_init
                    elif(abs(x1[i] - x0) > self.xmax_init/2 and x1[i] > x0):
                        xi = x1[i] - self.xmax_init
                    else:
                        xi = x1[i]

                    if(abs(y1[i] - y0) > self.ymax_init/2 and y1[i] < y0):
                        yi = y1[i] + self.ymax_init
                    elif(abs(y1[i] - y0) > self.ymax_init/2 and y1[i] > y0):
                        yi = y1[i] - self.ymax_init
                    else:
                        yi = y1[i]

                    r_i = np.sqrt((xi - x0) * (xi - x0) + (yi - y0) * (yi - y0))
                    if r_i <  self.r_snake:
                        r.append([i, r_i])

                # condition for snake existing
                if r:
                    for item in r: #(index of neighbor, r)
                        #plus neighbors partilce id in list
                        if self.neigh_id[jj_el_snake][item[0]] not in jj_using:
                            length += item[1] #count number of particles in snake
                            jj_using.append(self.neigh_id[jj_el_snake][item[0]]) #add particle to array of particles used in snake
                            jj_using_glob[self.neigh_id[jj_el_snake][item[0]]] = 1 #jj particle is in snake
            self.snakes_id.append(jj_using) #add current snake to array of snakes' ID
            self.len_snakes.append(length) #add length of current snake to array of snakes' lengths

        del jj_using
        del jj_using_glob
        del r

        return  self.snakes_id, self.len_snakes

    def Init_Parameter(self):
        self.r_mas = []
        self.psi = np.zeros((self.nparts_init, len(self.list_psi)))
        self.z_distance = []#np.zeros(self.nparts_init)
        self.dx_diff = [[] for _ in range(self.nparts_init)]
        self.dy_diff = [[] for _ in range(self.nparts_init)]


    def Calc_Parameter(self):
        psi_mas = [[] for _ in range(len(self.list_psi))]
        rr_mas = []
        z_h = []
        for jj in [557]:#range(#self.nparts_init):

            for rij in self.r_ij[jj]:
                rr_mas.append((rij[0] ** 2 + rij[1] ** 2) ** 0.5)
                z_h.append(rij[2])


            for psi_i in range(len(self.list_psi)):
                psi_k = self.list_psi[psi_i]
                psi_mas[psi_i].append(self.Get_Psik_Single(np.array(self.r_ij[jj]), psi_k))

                self.psi[jj, psi_i] += self.Get_Psik_Single(np.array(self.r_ij[jj]), psi_k)


        print(self.list_psi)
        print("PSI = %s"%self.psi[557, :])





    def Particle_Dispacment_1_Timestep(self, dat_xyz_previusly):
        # func to calc gist for diffusion
        # INPUT:
        # solid_part - массив твердых частиц (в этом коде он не соответвует индексам частиц)
        # это массив номера строчек частиц

        # массивы координат частиц общиме массивы без разделения на фазы
        #          mas_part_dif0,              mas_part_dif_ts_1,                 mas_part_dif_ts
        # начальный (нулевой) таймстеп        предыдущий таймстеп                текущий таймстеп
        # x_left, x_right, y_up, y_bottom - размеры области моделирования


        for i in range(self.nparts_init):
            x0 = dat_xyz_previusly[i, 1]
            y0 = dat_xyz_previusly[i, 2]

            x1 = self.dat_xyz_init[i, 1]
            y1 = self.dat_xyz_init[i, 2]

            x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1)

            self.dx_diff[i].append(x1 - x0)
            self.dy_diff[i].append(y1 - y0)



        x_box       = (x_right - x_left) / 2
        y_box       = (y_up - y_bottom) / 2
        y_up1       = 2 * y_box
        x_r         = 2 * x_box

        # pos1 - частица в твердой фазе (расчет твердой фазы происхолдит относительно начачльного массива частиц, где учитывалась только нумерация строчек)
        # для вычислкения индекса частиц надо использовать вспомогательный массив
        for pos1 in solid_part:                               # ind_part - номер частицы в исхожном массиве (строчка)
            #print('pos1 = ', pos1, 'dat1[pos1] = ', dat1[pos1])
            id1 = int(mas_part_dif_ts[pos1, 0])                    # нахожим индекс частицы (массив на данном тс аналогичен входному массиву в функции вороного, добавлена только столбец с ID частицами)
            #print('id1 = ', id1,'mas_part_dif_ts[pos1==id1] = ', mas_part_dif_ts[pos1])
            p2 = numpy.where(mas_part_dif_ts_1[:, 0] == id1)
            pos2 = p2[0][0]
            #pos2 = mas_part_dif_ts_1[:, 0].index(id1)         # номер строчки частицы с предыдущего таймстепа
            #pos0 =  mas_part_dif0[:, 0].index(id1)            # позиция в исходном массиве (номер строчки)
            # проверка на фазу частицы на предыдущем таймстепе:
            if pos1 not in solid_part_ts_1:
                rx[id1].append(brak)
                ry[id1].append(brak)
                continue
            #print('pos1 = ', pos1, 'pos2 = ', pos2)
            #x0, y0 =  mas_part_dif0[pos0, 1], mas_part_dif0[pos0, 2]
            x, y, x_1, y_1 = mas_part_dif_ts[pos1, 1], mas_part_dif_ts[pos1, 2], mas_part_dif_ts_1[pos2, 1], mas_part_dif_ts_1[pos2, 2]
            #print('x = ', mas_part_dif_ts[1, 1], '; x_1 = ',mas_part_dif_ts_1[1, 1])
            if x - x_box < x_1 < x + x_box and y - y_box < y_1 < y + y_box:
                rx[id1].append(x - x_1)
                ry[id1].append(y - y_1)
            elif y - y_box > y_1:
                ry[id1].append(y - (y_1 + y_up1))
                if x - x_box < x_1 < x + x_box:
                    rx[id1].append(x - x_1)
                elif x - x_box > x_1 :
                    rx[id1].append(x - (x_1 + x_r))
                elif x + x_box < x_1:
                    rx[id1].append(x - (x_1 - x_r))
            elif y + y_box < y_1:
                ry[id1].append(y - (y_1 - y_up1))
                if x - x_box < x_1 < x + x_box:
                    rx[id1].append(x - x_1)
                elif x - x_box > x_1 :
                    rx[id1].append(x - (x_1 + x_r))
                elif x + x_box < x_1:
                    rx[id1].append(x - (x_1 - x_r))
            elif y - y_box < y_1 < y + y_box:
                ry[id1].append(y - y_1)
                if x - x_box > x_1 :
                    rx[id1].append(x - (x_1 + x_r))
                elif x + x_box < x_1:
                    rx[id1].append(x - (x_1 - x_r))

            len_r = len(rx[id1]) - 1

            rr = ((rx[id1][len_r])**2 + (ry[id1][len_r])**2 )**0.5
            ind_gist = int(rr / dr_dif)
            #print(ind_gist)
            gist[ind_gist] += 1
        for pos in range(npoints):
            if pos in solid_part:
                continue
            else:
                id1 = int(mas_part_dif_ts[pos, 0])
                rx[id1].append(brak)
                ry[id1].append(brak)



    def Calc_Loop(self):

        for id_rho in range(len(self.density)):
            rho = density[id_rho]
            for id_h in range(self.num_h):
                # dump_100_100_0.7_1.1_1.0_7.2_0.lammpstrj
                # dump_100_100_0.7_1.1_1.0_6.6_0.lammpstrj
                #h = 1.+0.025*id_h
                h = 1.1
                for id_mu in range(self.num_mu):
                    #mu = round(mu_values[id_mu], 1)
                    mu = 6.6
                    new_prefix = self.prefix + str(rho) + "_" + str(h) + "_" + str(Temperature) + "_" + str(mu)
                    name = self.path_md + new_prefix + "_0.lammpstrj"
                    name = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/MD/dipole35_0.5.lammpstrj"

                    dump = LammpsCoords(string_content = 'id x y z ',
                                        required_data = 'id x y z ',
                                        path = name)
                    dump.display_info()
                    dump.OpenTrjGetInfo(1)
                    self.st = dump.ReadState()
                    self.nparts_init = self.st[0]
                    self.Init_Parameter()

                    #structur_anal = Structur()

                    counter_for_skip = 0
                    for index in range(istart, all_cadr - 10, int((all_cadr - 1 ) / num_cadr)):
                        if counter_for_skip > 0:
                            print('index__1 = ', index__1, ";   index = ", index)
                            for skip in range(index__1, index - 1, 1):
                                self.st = dump.ReadState()
                        index__1 = index
                        counter_for_skip += 1
                        print('counter_for_skip = ', counter_for_skip)
                        self.st = dump.ReadState()

                        self.dat_xyz_init = np.array(self.st[-1])

                        self.xmin_init, self.xmax_init = min(self.dat_xyz_init[:, 1]), max(self.dat_xyz_init[:, 1])
                        self.ymin_init, self.ymax_init = min(self.dat_xyz_init[:, 2]), max(self.dat_xyz_init[:, 2])
                        print("%f,    %f" %( self.xmin_init, self.xmax_init))

                        self.Add_Points()

                        # onlu for MD when id particles start from 1 not from 0
                        if min(self.dat_xyz[:, 0]) == 1:
                            self.dat_xyz[:, 0] -= 1

                        if min(self.dat_xyz_init[:, 0]) == 1:
                            self.dat_xyz_init[:, 0] -= 1


                        #xmin, xmax, ymin, ymax = min(x_part), max(x_part), min(y_part), max(y_part)
                        # Plot_System(self, xy, plot_type, frame)
                        #self.Plot_System(self.dat_xyz_init[:, 1:3], "Init", index__1)
                        #self.Plot_System(self.dat_xyz[:, 1:3], "add", index__1)

                        self.Get_Voronoi_Neighbors()
                        #self.Plot_Voro_System(self.dat_xyz[:, 1:3], "VOR_Add", index__1)
                        self.IdentShake()
                        self.Plot_Snakes(self.dat_xyz_init[:, 1:3], "IdentWor", index__1)
                        self.Calc_Parameter()





path_md = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/MD/Nikita/Data/"
img_path = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/result/ris/"
prefix = "dump_100_100_"
density = [0.7]#[0.7, 0.8]

num_h = 1#20
num_mu = 1#50

heights = np.arange(1., 1. + 0.025 * num_h, 0.025)

mu_values = np.arange(5.0, 5. +0.2 * num_mu, 0.2)
Temperature = 1.0
istart = 2
all_cadr = 15
num_cadr = 2

KingCobra = Snake(heights = [1],
                 mu_values = [1],
                 Temperature = 1.,
                 percent = 0.1,
                 density = density,
                 list_psi = [3, 4, 5, 6],
                 istart = istart,
                 all_cadr = all_cadr,
                 num_cadr = num_cadr,
                 path_md = path_md,
                 img_path = img_path,
                 prefix = prefix)

KingCobra.Calc_Loop()




















