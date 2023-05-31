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
class Structur:
    def __init__(self,
                 dat_xyz: np.ndarray,
                 id_parts: np.ndarray,
                 ):
        self.dat_xyz = dat_xyz,
        self.id_parts = id_parts,
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
        self.Coord_Transform
        self.g_r = np.zeros(len(np.arange(0., rMax + 1.1 * dr, dr)))
        if dimension == 2:
            return self.PCF_2D(rMax, dr)
        if dimension == 3:
            return 0


    def Get_Psik_Single(self, r_ij: np.ndarray, k:int):#x0y0: np.ndarray, xy_neighs: np.ndarray, k:int):
        # x0y0 - [x0, y0]; xy_neighs = [[x1, y1], [x2, y2] ...]
        #     psi_6[jj] = ret_psi_k(vor.points[jj], vor.points[neigh_id[jj]], 6)
        psik = 0
        for j in range(len(xy_neighs)):
            #r_ij = xy_neighs[j, :] - x0y0
            theta_ij = angle_of_vector_Ox(r_ij[0], r_ij[1])
            psik += np.exp(complex(0, k * theta_ij))
        psik /= len(xy_neighs)
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

























