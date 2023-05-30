import numpy as np
import DataSetWorks1
from DataSetWorks1 import LammpsCoords
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



class Snake:
    def __init__(self,
                 heights: list,
                 mu_values: list,
                 Temperature: float,
                 percent: float,
                 density: list,
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

        # all nympy variables
        self.heights = np.array(heights, dtype=float)
        self.mu_values = np.array(mu_values,  dtype=float)
        self.density = np.array(density, dtype = float)
        self.st = None
        self.dat_xyz = None
        self.dat_xyz_init = None

        # all float variables
        self.Temperature = Temperature
        self.percent = percent
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
        self.self.region_mask = None
        self.self.neigh_id = None

    def AddPoints(self):
        # ========================== INFORMATION ===============================
        # function of adding additional points to the array
        # of particles taking into account the periodic boundary

        self.dat_xyz = np.array(self.st[-1])
        id_particle    = self.dat_xyz[:, 0]
        x_particle     = self.dat_xyz[:, 1]
        y_particle     = self.dat_xyz[:, 2]
        z_particle     = self.dat_xyz[:, 3]


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

            if x_particle[i] < x_left + self.dx:

                self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i] + self.x_box, y_particle[i], z_particle[i]]], axis = 0)

                if  y_particle[i] < y_bottom + self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i] + self.x_box , y_particle[i] + self.y_box, z_particle[i] ]], axis = 0)
                    self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] + self.y_box, z_particle[i] ]], axis = 0)

                elif  y_particle[i] > y_up - self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i] + self.x_box, y_particle[i] - self.y_box, z_particle[i] ]], axis = 0)
                    self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] - self.y_box, z_particle[i] ]], axis = 0)

            elif x_particle[i] > x_right - self.dx:

                self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i] - self.x_box, y_particle[i], z_particle[i] ]], axis = 0)

                if y_particle[i] < y_bottom + self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i] - self.x_box, y_particle[i] + self.y_box, z_particle[i] ]], axis = 0)
                    self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] + self.y_box, z_particle[i] ]], axis = 0)

                elif y_particle[i] > y_up - self.dy:

                    self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i] - self.x_box, y_particle[i] - self.y_box, z_particle[i] ]], axis = 0)
                    self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] - self.y_box, z_particle[i] ]], axis = 0)

            elif y_particle[i] > y_up - self.dy:

                self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] - self.y_box, z_particle[i] ]], axis = 0)

            elif y_particle[i] < y_bottom + self.dy:

                self.dat_xyz = np.append(self.dat_xyz, [[id_particle[i], x_particle[i], y_particle[i] + self.y_box, z_particle[i] ]], axis = 0)

        #return self.dat_xyz, x_left, x_right, y_bottom, y_up, self.dx, self.dy




    def Get_Voronoi_Neighbors(self.dat_xyz, xy, dx, dy,  self.nparts_init, xmin_init, xmax_init, ymin_init, ymax_init):
        self.vor = scipy.spatial.Voronoi(xy)
        self.self.region_mask = np.full(len(self.vor.regions), False)
        npoints = len(self.vor.points)
        ver_index = np.full(len(self.vor.vertices),False)
        self.neigh_id=[[] for i in range(self.nparts_init)]

        #----------------------------------
        # otsechenie pogranichnux regionov i poisk min max ploshadei
        #----------------------------------
        for i in range(npoints):
            ind = self.vor.point_region[i]
            certies = self.vor.regions[ind]
            if (self.vor.points[i][0] > xmin_init  - 0.5 * self.dx  and self.vor.points[i][0] < xmax_init + 0.5*self.dx and self.vor.points[i][1] > ymin_init - 0.5*self.dy  and self.vor.points[i][1] < ymax_init + 0.5*self.dy):
                ver_index[certies] = True

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
        for rid in self.vor.ridge_points:

            rid0 = int(self.dat_xyz[rid[0], 0])
            rid1 = int(self.dat_xyz[rid[1], 0])

            # at least one particle in the region two adjacent particles: one inside the other outside the region
            if self.region_mask[self.vor.point_region[rid[0]]] + self.region_mask[self.vor.point_region[rid[1]]] == True:
                if rid1 not in self.neigh_id[rid0]:  # condition to avoid repetition
                    self.neigh_id[rid0].append(rid1)
                if rid0 not in self.neigh_id[rid1]:
                    self.neigh_id[rid1].append(rid0)
        return self.neigh_id, self.vor, self.region_mask


    def PlotVoroSystem(self, frame,  self.neigh_id, self.vor, self.region_mask, x, y, xmin, xmax, ymin, ymax, xmin_init, xmax_init, ymin_init, ymax_init, img_path):
        ff=plt.figure(figsize=(30, 30 * (max(xy[:, 1]) - min(xy[:, 1])) / (max(xy[:, 0]) - min(xy[:, 0]))))
        ax = plt.subplot(111)
        ax.set_xlim([min(xy[:, 0]), max(xy[:, 0])])
        ax.set_ylim([min(xy[:, 1]), max(xy[:, 1])])

        ax.set_aspect('equal', 'datalim')

        for jj in range(len(self.vor.points)):
            region_id = self.vor.point_region[jj]

            if self.region_mask[region_id]:
                polygon = [self.vor.vertices[i] for i in self.vor.regions[region_id]]
                ax.fill(*zip(*polygon), facecolor = "white", edgecolor = 'black', linewidth = 0.5)
                #ax.plot(self.vor.points[jj][0], self.vor.points[jj][1], "o")
                #if jj > n_parts_init - 1:
                ax.text(self.vor.points[jj][0], self.vor.points[jj][1], str(int(self.dat_xyz[jj, 0])) )
                #else:
                #    ax.text(self.vor.points[jj][0], self.vor.points[jj][1], str(jj))

        rect = patches.Rectangle((xmin_init, ymin_init), xmax_init - xmin_init, ymax_init - ymin_init, linewidth = 4, edgecolor='Black', facecolor='none')
        ax.add_patch(rect)
        ax.grid(color='r', linestyle='-', linewidth=2)

        plt.savefig(img_path + 'Neigh' +  '_' + str(frame) + '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
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
                    # f = open(name, "r")
                    # for skip in range(istart):
                    #     self.st = read_state(f)
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
                        nparts_init = self.st[0]
                        self.dat_xyz_init = np.array(self.st[-1])

                        xmin_init, xmax_init = min(self.dat_xyz_init[:, 1]), max(self.dat_xyz_init[:, 1])
                        ymin_init, ymax_init = min(self.dat_xyz_init[:, 2]), max(self.dat_xyz_init[:, 2])

                        #self.PlotSystem(index__1, x_particle, y_particle,  xmin_init, xmax_init, ymin_init, ymax_init, img_path)

                        self.AddPoints()

                        # onlu for MD when id particles start from 1 not from 0
                        if self.dat_xyz[0, 0] == 1:
                            self.dat_xyz[:, 0] -= 1

                        if self.dat_xyz_init[0, 0] == 1:
                            self.dat_xyz_init[:, 0] -= 1


                        #xmin, xmax, ymin, ymax = min(x_particle), max(x_particle), min(y_particle), max(y_particle)
                        # Plot_System(self, xy, plot_type, frame)
                        self.Plot_System(self.dat_xyz_init[:, 1:3], "Init", index__1)
                        self.Plot_System(self.dat_xyz[:, 1:3], "add", index__1)

                        #self.neigh_id, self.vor, self.region_mask = GetVoronoiNeighbors(self.self.dat_xyz, self.self.dat_xyz[:, 1:3], dx, dy, nparts_init, xmin_init, xmax_init, ymin_init, ymax_init)
                        #PlotVoroSystem(index__1, nparts_init, self.neigh_id, self.vor, self.region_mask, x_particle, y_particle, xmin, xmax, ymin, ymax, xmin_init, xmax_init, ymin_init, ymax_init, img_path)



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
                 istart = istart,
                 all_cadr = all_cadr,
                 num_cadr = num_cadr,
                 path_md = path_md,
                 img_path = img_path,
                 prefix = prefix)

KingCobra.Calc_Loop()




















