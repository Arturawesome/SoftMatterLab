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


    def PCF_2D(self,  rMax, dr):
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


    def Particle_Dispacment_1_Timestep(rx, ry, gist, dr_dif, mas_part_dif_ts_1, mas_part_dif_ts, x_left, x_right, y_up, y_bottom, solid_part, solid_part_ts_1, brak, dat1, npoints):
        # func to calc gist for diffusion
        # INPUT:
        # solid_part - массив твердых частиц (в этом коде он не соответвует индексам частиц)
        # это массив номера строчек частиц

        # массивы координат частиц общиме массивы без разделения на фазы
        #          mas_part_dif0,              mas_part_dif_ts_1,                 mas_part_dif_ts
        # начальный (нулевой) таймстеп        предыдущий таймстеп                текущий таймстеп
        # x_left, x_right, y_up, y_bottom - размеры области моделирования

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



























