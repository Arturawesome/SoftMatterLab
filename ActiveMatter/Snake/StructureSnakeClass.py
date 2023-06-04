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
                 list_triplets:list,
                 istart: int,
                 all_cadr: int,
                 num_cadr: int,
                 path_md: str,
                 txt_path:str,
                 img_path: str,
                 prefix:str
                 ):

        # all path and txt variables
        self.img_path = img_path
        self.txt_path = txt_path
        self.path_md = path_md
        self.prefix = prefix


        # list cariables
        self.list_psi = list_psi
        self.list_triplets = list_triplets



        # all nympy variables
        self.heights = np.array(heights, dtype=float)
        self.mu_values = np.array(mu_values,  dtype=float)
        self.density = np.array(density, dtype = float)
        self.z_distance = []
        self.st = None
        self.dat_xyz = None     # np.array after Add_Points() dat_xyz[:, 0] - MD id particle, dat_xyz[:, 1:4] - xyz position
        self.dat_xyz_init = None # np.array dat_xyz[:, 0] - MD id particle, dat_xyz[:, 1:4] - xyz position
        self.dat_xyz_previusly = None
        self.sigma2t = None
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
        self.n_triplets_points = None

        # VORONOI VARIABLES
        self.vor = None
        self.region_mask = None
        self.neigh_id = None

        # All snake's variables
        self.snakes_id = []
        self.angles_in_snakes = []
        self.len_snakes = []
        self.agnles_stat = None

        self.r_snake =  1.1#0.8

        # Structure variables
        self.psi_array_gist = None
        self.psi = None
        self.psi3 = None
        self.psi4 = None
        self.psi5 = None
        self.psi6 = None
        self.psi7 = None

        self.dy_diff = None
        self.dx_diff = None
        self.dr_diff = None



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

                    # calc the distance the betwen central parrticle and its adjacent particle
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

                    # calc the distance the betwen central parrticle and its adjacent particle
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

        self.dat_xyz = np.array(self.st[-1]).copy()
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
    def Plot_Gist(self, data, n_elem_gist, plot_type):
        ff=plt.figure(figsize=(15.0,15.0 * 0.8))
        plt.hist(data, n_elem_gist)
        plt.xlabel(plot_type)
        #plt.xlim([1, ])
        plt.savefig(self.img_path + 'Gist' + plot_type + '_.pdf',  bbox_inches='tight',pad_inches = 0.05)
        plt.close(ff)


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

    def Plot_Sigma_2t(self, plot_type):
        ff = plt.figure(figsize=(22,22 ))
        ax = plt.subplot(111)
        ax.plot(np.arange(0, len(self.sigma2t)), self.sigma2t, 'o-')    #mh1
        plt.savefig(self.img_path + plot_type +  '_.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
        ff.clear()
        plt.close(ff)


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
            ax.plot(xy[list_el, 0], xy[list_el, 1], '.', markersize=12)    #mh1
        for id_snake in range(len(self.snakes_id)):
            list_el = self.snakes_id[id_snake]
            for el in list_el:
                ax.text(xy[el, 0], xy[el, 1], str(id_snake), fontsize = 5)


        #ax.plot(xy[:, 0], xy[:, 1], '.')    #mh1
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
        self.angles_in_snakes = []

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
                len_r = len(r)
                if len_r > 1:
                    # identification of triplets piont
                    if len_r > 2:
                        for id_el in range(len(self.list_triplets)):
                            num_triplet = self.list_triplets[id_el] # 3 4 5
                            if len_r == num_triplet:
                                self.n_triplets_points.append(num_triplet)
                                continue

                    if len_r == 2:
                        if self.neigh_id[jj_el_snake][r[0][0]] in jj_using and self.neigh_id[jj_el_snake][r[1][0]] in jj_using:
                            break

                    for item in r:
                        if self.neigh_id[jj_el_snake][item[0]] not in jj_using:
                            length += item[1] #count number of particles in snake
                            jj_using.insert(0, self.neigh_id[jj_el_snake][item[0]]) #add particle to start of array of particles used in snake
                            jj_using_glob[self.neigh_id[jj_el_snake][item[0]]] = 1 #jj particle is in snake
                            #if self.neigh_id[jj_el_snake][item[0]] == 506 or jj == 506:
                                #print("kjslfdhgkldsfjahglkads:    ", end="")
                                #print(len(self.snakes_id))
                            jj_el_snake = self.neigh_id[jj_el_snake][item[0]]

                            break
                # ===============================================
                # ===============================================
                #elif len(r) >2 : break (for)
                # ===============================================
                # ===============================================
                else:
                    break #find the end


            # ===============================================
            # ===============================================
            # Particle sorting in snakes??????????
            # ===============================================
            # ===============================================
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
            self.angles_in_snakes.append([])
            if length > 2:
                self.len_snakes.append(length) #add length of current snake to array of snakes' lengths


        for id_snake in range(len(self.snakes_id)):
            angles = []
            len_snake = len(self.snakes_id[id_snake])
            if len_snake > 2:
                for index in range(len_snake - 2):
                    if index + 2 < len_snake:

                        id_init = self.snakes_id[id_snake][index]
                        id_center = self.snakes_id[id_snake][index+1]
                        id_end = self.snakes_id[id_snake][index+2]

                        x0, y0, z0 = self.dat_xyz_init[id_init, 1], self.dat_xyz_init[id_init, 2], self.dat_xyz_init[id_init, 3]
                        x1, y1, z1 = self.dat_xyz_init[id_center, 1], self.dat_xyz_init[id_center, 2], self.dat_xyz_init[id_center, 3]
                        x2, y2, z2 = self.dat_xyz_init[id_end, 1], self.dat_xyz_init[id_end, 2], self.dat_xyz_init[id_end, 3]

                        self.z_distance.append(abs(z1-z0))

                        x1t, y1t = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1)
                        x2t, y2t = self.Consideration_Periodic_Boundary_Distance(x1, y1, x2, y2)

                        x_init_center = x1t - x0
                        y_init_center = y1t - y0

                        x_center_end = x2t - x1
                        y_center_end = y2t - y1

                        radians = self.Angle_Of_Vectors([x_center_end, y_center_end], [x_init_center, y_init_center])
                        self.agnles_stat.append(radians)

                        angles.append(radians)
                        if id_snake == 32:
                            print("id %s,  %s,  %s " %(id_init, id_center, id_end) )
                            print("v1 %s,  %s"%(x_init_center, y_init_center))
                            print("v2 %s,  %s"%(x_center_end, y_center_end))
                            print("angle %s"%radians, end="\n\n")

                self.angles_in_snakes[id_snake] = angles



        del jj_using
        del jj_using_glob
        del r

        return  self.snakes_id, self.len_snakes

    def WriteDataTxt(self):
        return 0



    def Init_Parameter(self):
        #self.neigh_id,   self.r_ij init in Get_Voronoi_Neighbors
        #self.snakes_id, self.angles_in_snakes, self.len_snakes  init in Snake Ident
        self.len_snakes = []
        self.agnles_stat = []

        self.z_distance = []    # in IdentShake
        self.n_triplets_points = []# = np.zeros(len(self.list_triplets)) # in IdentShake
        # n_triplets_points = [ [] , [] , [] atc];
        # n_triplets_points[0] list for triplets,  n_triplets_points[1] for quadro point etc

        self.dx_diff = np.zeros((self.nparts_init, self.num_cadr+1)) # Particle_Dispacment_1_Timestep and Get_Sigma2t
        self.dy_diff = np.zeros((self.nparts_init, self.num_cadr+1))  # Particle_Dispacment_1_Timestep and Get_Sigma2t
        self.dr_diff = [] # Particle_Dispacment_1_Timestep
        self.sigma2t = np.zeros(self.num_cadr) #Get_Sigma2t

        self.psi = np.zeros((self.nparts_init, len(self.list_psi))) # in Calc_Parameter
        self.psi_array_gist = [[] for _ in range(len(self.list_psi))] # in Calc_Parameter


    def Calc_Parameter(self, ts):
        for jj in range(self.nparts_init):

            for psi_i in range(len(self.list_psi)):
                psi_k = self.list_psi[psi_i]

                self.psi_array_gist[psi_i].append(self.Get_Psik_Single(np.array(self.r_ij[jj]), psi_k)) # StructureAnalysis
                self.psi[jj, psi_i] += self.Get_Psik_Single(np.array(self.r_ij[jj]), psi_k)


            self.Particle_Dispacment_1_Timestep(jj, ts)
            if ts == self.num_cadr - 1:
                self.Get_Sigma2t(jj)



    def Particle_Dispacment_1_Timestep(self, id_part, ts):
        # func to calc gist for diffusion
        # INPUT:
        # solid_part - массив твердых частиц (в этом коде он не соответвует индексам частиц)
        # это массив номера строчек частиц

        # массивы координат частиц общиме массивы без разделения на фазы
        #          mas_part_dif0,              mas_part_dif_ts_1,                 mas_part_dif_ts
        # начальный (нулевой) таймстеп        предыдущий таймстеп                текущий таймстеп
        # x_left, x_right, y_up, y_bottom - размеры области моделирования

        #for i in range(self.nparts_init):
        x0 = self.dat_xyz_previusly[id_part, 1]
        y0 = self.dat_xyz_previusly[id_part, 2]

        x1 = self.dat_xyz_init[id_part, 1]
        y1 = self.dat_xyz_init[id_part, 2]

        x1, y1 = self.Consideration_Periodic_Boundary_Distance(x0, y0, x1, y1)

        self.dx_diff[id_part, ts] = x1 - x0
        self.dy_diff[id_part, ts] = y1 - y0
        self.dr_diff.append(((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0))**0.5)

    def Get_Sigma2t(self, id_part):
        # calc the sigma ^2 values:
        for ts_ in range(self.num_cadr - 1):
            dx = sum(self.dx_diff[id_part, 0: ts_+1])
            dy = sum(self.dy_diff[id_part, 0: ts_+1])
            self.sigma2t[ts_] += (dx*dx + dy*dy)  / self.nparts_init

    def Save_Txt_File(self, data, data_name, dimension):
        full_name = self.txt_path + data_name
        f = open(full_name, "w")
        if dimension == 1:
            a = len(data)
            for i in range(a):
                f.write(str(data[i]) + "\n")

        elif dimension == 2:
            row, col = data.shape
            for i in range(row):
                for j in range(col):
                    f.write(str(data[i, j]) + " ")
            f.write("\n")
        f.close()


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
                    #name = "dipole35_0.5.lammpstrj"

                    dump = LammpsCoords(string_content = 'id x y z ',
                                        required_data = 'id x y z ',
                                        path = name)
                    dump.display_info()
                    dump.OpenTrjGetInfo(1)
                    self.st = dump.ReadState()
                    self.nparts_init = self.st[0]
                    self.Init_Parameter()

                    #structur_anal = Structur()

                    ts_index = 0
                    dr_pcf = 0.1
                    r_pcf = 5
                    radii_pcf =  np.arange(0., r_pcf + 1.1 * dr_pcf, dr_pcf)
                    gr = np.zeros(len(radii_pcf) - 1)
                    count_cadrs = 0

                    for index in range(istart, all_cadr - 1, int((all_cadr - 1 ) / num_cadr)):
                        if ts_index > 0:
                            print('index__1 = ', index__1, ";   index = ", index)
                            for skip in range(index__1, index - 1, 1):
                                self.st = dump.ReadState()
                        index__1 = index
                        count_cadrs += 1

                        self.st = dump.ReadState()

                        self.dat_xyz_init = np.array(self.st[-1]).copy()

                        self.xmin_init, self.xmax_init = min(self.dat_xyz_init[:, 1]), max(self.dat_xyz_init[:, 1])
                        self.ymin_init, self.ymax_init = min(self.dat_xyz_init[:, 2]), max(self.dat_xyz_init[:, 2])

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
                        self.IdentShake()
                        #self.Plot_Snakes(self.dat_xyz_init[:, 1:3], "IdentWor", index__1)

                        # ============== PCF ==============================
                        #pcf, radii_pcf = self.GetPCF(self.dat_xyz_init[:, 1:4], 2, r_pcf, dr_pcf)
                        #gr += pcf
                        # ============== PCF ==============================

                        #self.Plot_Voro_System(self.dat_xyz[:, 1:3], "VOR_Add", index__1)
                        ts_index += 1
                        print('ts_index = ', ts_index)

                        if ts_index >= 2:

                            self.Calc_Parameter(ts_index-1)


                        self.dat_xyz_previusly = self.dat_xyz_init.copy()

                        if ts_index == 1:
                            print("%s", self.snakes_id[32])
                            print(len(self.snakes_id[32]))
                            print(len(self.angles_in_snakes[32]))
                            print("%s"%self.angles_in_snakes[32])
                            print(sum(self.angles_in_snakes[32]))
                    #self.Plot_PCF(gr, radii_pcf, self.img_path)
                    self.Plot_Gist(self.len_snakes, 50, "GistLenSnake")
                    self.Plot_Gist(self.agnles_stat, 50, "AgnlesStat")
                    self.Plot_Gist(self.n_triplets_points, len(self.n_triplets_points), "Triplet")
                    self.Plot_Gist(self.z_distance, 50, "Zcoord")
                    self.Plot_Gist(self.dr_diff, 50, "dr_diff")
                    self.Plot_Sigma_2t("sigma2t")
                    for id_psi in range(len(self.list_psi)):
                        psik = self.list_psi[id_psi]
                        self.Plot_Gist(self.psi_array_gist[id_psi], 50, "Psi" + str(psik))

                        self.Save_Txt_File(self.psi_array_gist[id_psi], "GistPsi"+ str(psik), 1)
                        self.Save_Txt_File([sum(self.psi_array_gist[id_psi])/len(self.psi_array_gist[id_psi])], "AvPsi"+ str(psik), 1)


                    self.Save_Txt_File(self.sigma2t, "sigma2t", 1)

                    self.Save_Txt_File(self.z_distance, "Z_Coord", 1)
                    self.Save_Txt_File([sum(self.z_distance)/len(self.z_distance)], "AvLen", 1)

                    self.Save_Txt_File(self.len_snakes, "GistLenSnake", 1)
                    self.Save_Txt_File([sum(self.len_snakes)/len(self.len_snakes)], "AvLen", 1)

                    self.Save_Txt_File(self.agnles_stat, "GistAngles", 1)
                    self.Save_Txt_File([sum(self.agnles_stat)/len(self.agnles_stat)], "AvAngle", 1)

                    self.Save_Txt_File(self.n_triplets_points, "GistTripletspoint", 1)
                    self.Save_Txt_File([sum(self.n_triplets_points)/len(self.n_triplets_points)], "AvTripletspoint", 1)




                """
                гистограмма длины змеек+
                    накопление статистики по кадрам

                средняя длина змеек+
                    слодить все и делить на количство змеек и количество кадров

                гистограмма углов в змейке +
                    накопление статистики по кадрам +

                средний угол в змейке+
                    слодить все и делить на количство змеек и количество кадров+

                гистограмма триплетов+
                    накопление статистики по кадрам

                срежнее количество триплетов+
                    слодить все и делить на количство змеек и количество кадров

                диффузия+
                    накопление статистики по кадрам

                гистограмма параметров пси_к+
                    накопление статистики по кадрам

                среднее значение пси_к+
                    слодить все и делить на количство змеек и количество кадров

                гистограмма з координат
                    накопление статистики по кадрам

                """







path_md = ""
txt_path = "results/txt/"
img_path = "results/ris/"
prefix = "dump_100_100_"
density = [0.7]#[0.7, 0.8]

num_h = 1#20
num_mu = 1#50

heights = np.arange(1., 1. + 0.025 * num_h, 0.025)

mu_values = np.arange(5.0, 5. +0.2 * num_mu, 0.2)
Temperature = 1.0
istart = 2
all_cadr = 20
num_cadr = 4
list_triplets = [3, 4, 5, 6]
KingCobra = Snake(heights = [1],
                 mu_values = [1],
                 Temperature = 1.,
                 percent = 0.07,
                 density = density,
                 list_psi = [3, 4, 5, 6],
                 list_triplets = list_triplets,
                 istart = istart,
                 all_cadr = all_cadr,
                 num_cadr = num_cadr,
                 path_md = path_md,
                 txt_path = txt_path,
                 img_path = img_path,
                 prefix = prefix)

KingCobra.Calc_Loop()




















