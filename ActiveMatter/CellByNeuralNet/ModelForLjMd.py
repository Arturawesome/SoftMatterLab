'''
Name       : ModelForLjMd.py

Author informaion: Artur D. Nasyrov
                   PhD student in BMSTU, Moscow, Russia
                   engineer, Research and Development Center
                   "Photonic and IK-technic"; "Soft Matter & Phys of Fluids Centre"

Description: My first independent neural network for prediction
             the trajectory of classical particles in the LJ liquid



Technical description:

NnLennardJonesIneraction: the class that define the force acting on particle

Technology : ANN, OOP


'''

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.spatial import distance
from scipy import spatial

import torch
from torch import nn
import torch.nn.functional as F

mod = SourceModule("""
    #include <cmath>
    #include <math.h>

    extern "C" __global__ void HellowWorld(int r_cut)
    {
        printf("HellowWorld");
    }
    extern "C" __global__ void ReturnNeghbors(float *x_p, float *y_p, int *neigh_ar, float r_cut, int n_part, int num_max_neighbors)
    {
        //calc relative of time
        int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y+blockIdx.y*blockDim.y;
        int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс времени
        int id_part_in_array = id_part * num_max_neighbors;

        if (id_part < n_part)
        {
            float x0, y0, x1, y1;
            int count = 0;
            if (id_part == 5) printf("id = %i; ", id_part);

            x0 = x_p[id_part];
            y0 = y_p[id_part];

            for(int i = 0; i < n_part; i++)
            {
                if (id_part == i) continue;
                x1 = x_p[i];
                y1 = y_p[i];


                if ((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0) < r_cut*r_cut)
                {
                    neigh_ar[id_part_in_array + count] = i;
                    count++;
                    if (id_part == 5) printf("%i; ", i);
                }
            }
            if (id_part == 5) printf("\\n");
        }
    }

    extern "C" __global__ void LjPF(float *x_p, float *y_p, int *neigh_ar, float *f_x, float *f_y, float *energy,  float *virial, int n_part, int num_max_neighbors)
    {
        int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y+blockIdx.y*blockDim.y;
        int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс времени
        int id_part_in_array = id_part * num_max_neighbors;
        if (id_part < n_part)
        {
            double delx, dely, r2inv, r6inv, force;
            int id_neigh;

            for(int i = 0; neigh_ar[id_part_in_array + i] >=1; i++)
            {
                id_neigh = neigh_ar[id_part_in_array + i];
                delx = x_p[id_part] - x_p[id_neigh];
                dely = y_p[id_part] - y_p[id_neigh];

                r2inv = delx * delx + dely * dely;
                r6inv = r2inv * r2inv * r2inv;

                r2inv = 1 / r2inv;
                r6inv = 1 / r6inv;

                force = 48 * r2inv * r6inv * (r6inv - 0.5);
                f_x[id_part] += delx * force;
                f_y[id_part] += dely * force;

                energy[id_part] += 4 * (r6inv * r6inv + r6inv);
                virial[id_part] += force * (delx * delx + dely * dely);




            }
        }

    }


    """)



class NnLennardJonesIneraction(nn.Module):
    # the input: the torch tenzor with 2 cols and number_of_neighbors rows.
    def __init__(self, number_of_neighbors: int, latent: int):
        super().__init__(LennardJonesIneraction)

        self.fc1 = nn.Linear(number_of_neighbors, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2) # 2 -> fx and fy coordinates of force exert in particles


    def forward (self, x):

        pred = nn.relu(self.fc1(x))
        pred = nn.relu(self.fc2(x))
        pred = nn.relu(self.fc3(x))

        return pred



class MD:

    def __init__(self,
                 num_tp: int,
                 rho: float,
                 r_cut: float,
                 d_t: float,
                 num_time_steps: int,
                 xy_box: list,
                 periodic_bond: list,
                 lmp_path_file: str,
                 ):

        self.num_tp = num_tp
        self.n_part =  None
        self.r_cut = np.float32(r_cut)
        self.d_t = np.float32(d_t)
        self.xy_box = np.array(xy_box, dtype=float)

        self.num_time_steps = int(num_time_steps)
        self.lmp_path_file = lmp_path_file
        self.periodic_bond = periodic_bond
        self.particles = None
        self.h_neigh_r = None
        self.V = None
        self.rho = rho
        self.d_neigh_r = None
        self.d_xp = None
        self.d_yp = None
        self.d_vx = None
        self.d_vy = None


    def init_state(self):
        """
        Initialize the particle position by cubic lattice
        """
        # random position and type of particle


        self.n_part =  np.int32(self.xy_box[0] * self.xy_box[1] * self.rho)
        n_2= np.int32(math.floor(self.n_part**(1/2)+0.5))
        self.n_part = n_2 * n_2

        self.particles = np.zeros((self.n_part, 3)) # x_pos, y_pos, type

        ix, iy = 0, 0
        bx = self.xy_box[0]
        by = self.xy_box[1]

        for i in range(self.n_part):
            self.particles[i, 0]= (ix + 0.5) * bx / n_2
            self.particles[i, 1]= (iy + 0.5) * by / n_2
            ix += 1

            if ix == n_2:
                ix = 0
                iy += 1
            if iy == n_2:
                iy = 0

        self.particles = self.particles.astype(np.float32)
        #self.particles = np.c_[self.xy_box * np.random.random((self.n_part, 2)), np.ones(self.n_part)]



    def get_neighbors(self):
        num_max_neighbors = np.int32(10)
        self.h_neigh_r = (np.zeros(self.n_part * num_max_neighbors) - 1).astype(np.int32)

        self.d_neigh_r = cuda.mem_alloc(self.h_neigh_r.nbytes)
        cuda.memcpy_htod(self.d_neigh_r, self.h_neigh_r)

        a = self.particles[:, 0].astype(np.float32)
        self.d_xp = cuda.mem_alloc(a.nbytes)
        cuda.memcpy_htod(self.d_xp, a)

        a = self.particles[:, 1].astype(np.float32)
        self.d_yp = cuda.mem_alloc(a.nbytes)
        cuda.memcpy_htod(self.d_yp, a)

        func_neagh = mod.get_function("ReturnNeghbors")

        func_neagh(self.d_xp, self.d_yp, self.d_neigh_r, self.r_cut, self.n_part, num_max_neighbors , block=(1,1,1), grid=(10, 1))
        cuda.memcpy_dtoh(self.h_neigh_r, self.d_neigh_r)
        #id_part_in_array = 5 * num_max_neighbors
        #for i in range(num_max_neighbors):
        #    print(self.h_neigh_r[id_part_in_array + i])
        #    print(" ")



    def run_nvt(self):
        for ts in range(self.num_tp):
            print("ts = " + str(ts) + "; ")
            #ke= nhchain(Q, dt, dt_2, dt_4, dt_8, nparticles, vxi, xi, ke, vel)

    def update_position(self, vx, vy, d_t, periodic_bond):
        if sum(periodic_bond) > 1:
            self.position[0] += vx * d_t
            self.position[1] += vy * d_t
            #self.position[2] += vz * d_t
        else:
            self.position[0] += vx * d_t
            self.position[1] += vy * d_t


    def plot_system(self, path_save):
        print(self.particles)
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.plot(self.particles[:, 0], self.particles[:, 1], "o")
        for i in range(self.n_part):
            ax.text(self.particles[i, 0], self.particles[i, 1], str(i) +"; \n" + str(round(self.particles[i, 0], 2)) + "; " +   str(round(self.particles[i, 1], 2)), fontsize = 5)

        plt.savefig(path_save + "Sys.pdf", transparent=True, bbox_inches='tight',pad_inches = 0)
        plt.close(fig)

    def calc_force(self):
        return 0




sim = MD(num_tp = 1,
                 rho = 0.8,
                 r_cut = 2.,
                 d_t = 0.01,
                 num_time_steps = 10,
                 xy_box = [10, 10],
                 periodic_bond = [1, 1],
                 lmp_path_file = "/run/media/artur/Новый том/CellDynamic/LjLiquidDataSet/",
)
sim.init_state()
sim.plot_system("/home/artur/Documents/CellularDynamic/NeuralNetForCell/NnExperimentalVersion/BlmRis/")
sim.get_neighbors()
































