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

# VARIABLES DESCRIPTION
# dat_xyz - coordiante massive dat_xyz[N, 3] where dat_xyz[N, 0] - x position, [N, 1] - y positions, [N, 2] - z positions
#
class SyatemStructur:
    def __init__(self
                 ):
        self.dat_xyz = None,
        self.id_parts = None,
        self.g_r = None
        self.radii = None
        self.box_size_xyz = None
        self.nparts = None


    def Coord_Transform(self):
        # function to transform to the coordinate to the first quarter
        xmin = min(self.dat_xyz[:, 0])
        ymin = min(self.dat_xyz[:, 1])
        zmin = min(self.dat_xyz[:, 2])

        if xmin < 0:
            self.dat_xyz[:, 0] += abs(xmin)
        if ymin < 0:
            self.dat_xyz[:, 1] += abs(ymin)
        if zmin < 0:
            self.dat_xyz[:, 2] += abs(zmin)

        self.box_size_xyz = np.zeros(3)
        self.box_size_xyz[0] = max(self.dat_xyz[:, 0])
        self.box_size_xyz[1] = max(self.dat_xyz[:, 1])
        self.box_size_xyz[2] = max(self.dat_xyz[:, 2])

    def GetPCF(self, dat_xyz, dimension, rMax, dr):
        self.dat_xyz = dat_xyz
        self.Coord_Transform()
        self.g_r = np.zeros(len(np.arange(0., rMax + 1.1 * dr, dr)))
        if dimension == 2:
            return self.PCF_2D(rMax, dr)
        if dimension == 3:
            return 0
    def Angle_Of_Vector_AxesX(self, x, y):
        angle = pygame.math.Vector2(1, 0).angle_to((x, y))

        return math.radians(angle)

    def Angle_Of_Vectors(self, vector1, vector2):
        unit_vector_1 = vector1 / np.linalg.norm(vector1)
        unit_vector_2 = vector2 / np.linalg.norm(vector2)

        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        return math.degrees(angle)

    # def Angle_Of_Vectors(self, x0, y0, x1, y1):
    #     angle = pygame.math.Vector2(x0, y0).angle_to((x1, y1))
    #     return angle #math.radians(angle)


    def Get_Psik_Single(self, r_ij: np.ndarray, k:int):#x0y0: np.ndarray, xy_neighs: np.ndarray, k:int):
        # x0y0 - [x0, y0]; xy_neighs = [[x1, y1], [x2, y2] ...]
        #     psi_6[jj] = ret_psi_k(vor.points[jj], vor.points[neigh_id[jj]], 6)
        psik = 0
        # print("in Get_Psik_Single")
        # print("len r_ij = %s"%len(r_ij))
        # print(r_ij)
        for i in range(len(r_ij)):
            #r_ij = xy_neighs[j, :] - x0y0
            theta_ij = self.Angle_Of_Vector_AxesX(r_ij[i, 0], r_ij[i, 1])
            psik += np.exp(complex(0, k * theta_ij))
        psik /= len(r_ij)
        psik = abs(psik)
        return psik
    def Plot_PCF(self, Gr, r_gr, img_path):
        ff=plt.figure(figsize=(10, 10))
        ax = plt.subplot(111)
        ax.plot(r_gr, Gr, "-")
        plt.savefig(img_path + 'PCF_2D.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
        ff.clear()
        plt.close(ff)


    def PCF_2D(self, rMax, dr):
        self.Coord_Transform()

        # Find particles which are close enough to the box center that a circle of radius
        # rMax will not cross any edge of the box
        x = self.dat_xyz[:, 0]
        y = self.dat_xyz[:, 1]
        bools1 = x > rMax
        bools2 = x < (self.box_size_xyz[0] - rMax)
        bools3 = y > rMax
        bools4 = y < (self.box_size_xyz[1] - rMax)
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
        numberDensity = len(x) / self.box_size_xyz[0] / self.box_size_xyz[1]

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

        return g_average, radii



            # вставить остальные гран условия
    def Simga2T(rx, ry, dr_dif, brak, num_ts, npoints):
        #  вычисление MSD^2
        # rx - массив смещения частиц относительно соседних таймстепов
        # dr_diff шаг смещения
        # num_ts - число обрабатываенмых таймстепов
        # определеить максимальное r
        r_r0 = [[] for i in range(num_ts + 1)]
        sigma2 = []
        r_r0[0] = [0]

        for i in range(npoints):
            ind = [ii for ii, d in enumerate(rx[i]) if d == brak]             # поиск индексов бракованных таймстепов для частиц
            if ind == []:
                for jj in range(len(rx[i])):
                    dx = sum(rx[i][0 : jj + 1])
                    dy = sum(ry[i][0 : jj + 1])

                    r_r0[jj + 1].append(((dx*dx + dy*dy) ** 0.5) **2 )
            for j in range(len(ind)):
                if j == 0:                                                  # до первого вхождения брака
                    for jj in range(ind[j]):
                        dx = sum(rx[i][0 : jj + 1])
                        dy = sum(ry[i][0 : jj + 1])
                        r_r0[jj + 1].append(((dx*dx + dy*dy) ** 0.5) **2 )
                elif int(ind[j - 1] - ind[j]) == 1:                         # выброс последовательных таймстепов где частица в газовой фазе
                    continue
                elif int(ind[j - 1] - ind[j]) <  3:                         # выброс последовательных таймстепов где частица в газовой фазе
                    continue
                elif j == len(ind) - 1:                                     # последнее вхождение брака
                    for jj in range(ind[j], len(rx[i])):
                        dx = sum(rx[i][ind[j] : jj + 1])
                        dy = sum(ry[i][ind[j] : jj + 1])
                        ts_ind = jj - ind[j]
                    # print('ts_ind (len(ind) - 1) = ', ts_ind + 1)
                        r_r0[ts_ind + 1].append(((dx*dx + dy*dy) ** 0.5) **2 )
                else:                                                       #
                    for jj in range(ind[j - 1] + 1, ind[j]):
                        #if
                        dx = sum(rx[i][ind[j] : jj + 1])
                        dy = sum(ry[i][ind[j] : jj + 1])
                        ts_ind = jj - ind[j -1]
                    #  print('ts_ind ind[j - 1] + 1, ind[j]) = ', ts_ind + 1)
                        r_r0[ts_ind + 1].append(((dx*dx + dy*dy) ** 0.5) **2 )

        for ii in range(int(num_ts)):
            sigma2.append(sum(r_r0[ii]) / len(r_r0[ii]))

        return sigma2



























