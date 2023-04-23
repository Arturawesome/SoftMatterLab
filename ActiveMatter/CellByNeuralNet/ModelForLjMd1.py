'''
Name       : ModelForLjMd.py

Author informaion: Artur D. Nasyrov
                   PhD student in BMSTU, Moscow, Russia
                   engineer, Research and Deself.velopment Center
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
    extern "C" __global__ void ReturnNeghbors(float *x_p, float *y_p, int *neigh_ar, float r_cut, int n_part, int num_max1bors)
    {
        //calc relative of time
        int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y+blockIdx.y*blockDim.y;
        int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс времени
        int id_part_in_array = id_part * num_max1bors;

        if (id_part < n_part)
        {

            float x0, y0, x1, y1;
            int count = 0;

            if (id_part == 35) printf("id = %i; ", id_part);

            x0 = x_p[id_part];
            y0 = y_p[id_part];
            // delete all neighbors from the previousl step
            for(int i = 0; i < num_max1bors; i++)
            {
                neigh_ar[id_part_in_array + i] = -1;
            }

            for(int i = 0; i < n_part; i++)
            {
                if (id_part == i) continue;
                x1 = x_p[i];
                y1 = y_p[i];


                if ((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0) < r_cut*r_cut)
                {
                    neigh_ar[id_part_in_array + count] = i;
                    count++;
                     if (id_part == 35) printf(" %i; ", i);
                }
            }
            if (id_part == 35) printf("\\n");
        }
    }
extern "C" __global__ void ReturnNeghbors2(float *x_p, float *y_p, int *neigh_ar, float r_cut, int n_part, int num_max1bors, float *delx, float *dely, float x_max, float y_max)
    {
        //calc relative of time
        int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y+blockIdx.y*blockDim.y;
        int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс времени
        int id_part_in_array = id_part * num_max1bors;

        if (id_part < n_part)
        {
            float x0, y0, x1, y1,rsquare, x_r, x_l, y_up, y_b;
            int count = 0;
            //if (id_part == 5) printf("id = %i; ", id_part);

            x0 = x_p[id_part];
            y0 = y_p[id_part];


            for(int i = 0; i < n_part; i++)
            {
                if (id_part == i) continue;
                x1 = x_p[i];
                y1 = y_p[i];
                x_r = x1 + x_max / 2;
                x_l = x1 - x_max / 2;
                y_up = y1 + y_max / 2;
                y_b = y1 - y_max / 2;


                if (((x1>x_l)&&(x1<x_r))&&((y1>y_b)&&(y1<y_up)))
                {
                    rsquare = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1);
                    if (rsquare < r_cut*r_cut)
                    {
                        neigh_ar[id_part_in_array + count] = i;
                        delx[id_part_in_array + count] = (x0 - x1) * (x0 - x1);
                        dely[id_part_in_array + count] = (y0 - y1) * (y0 - y1);
                        count++;
                        // if (id_part == 5) printf("%i; ", i);
                    }


                }
                else if ((x1 < r_cut) || (x1 > x_max - r_cut) || (y1 < r_cut) || (y1 > y_max - r_cut))
                {
                    if  ((x1>x_l)&&(x1<x_r)&&(y1>y_up))
                    {
                        rsquare = (x0 - x1) * (x0 - x1) + (y0 + (y_max - y1)) * (y0 + (y_max - y1));
                        if (rsquare < r_cut*r_cut)
                        {
                            neigh_ar[id_part_in_array + count] = i;
                            delx[id_part_in_array + count] = (x0 - x1) * (x0 - x1);
                            dely[id_part_in_array + count] = (y0 + (y_max - y1)) * (y0 + (y_max - y1));
                            count++;
                            // if (id_part == 5) printf("%i; ", i);
                        }
                    }
                    else if ((x1>x_r)&&(y1>y_up))
                    {
                        rsquare  = (x0+(x_max-x1))*(x0+(x_max-x1)) + (y0+(y_max-y1))*(y0+(y_max-y1));
                        if (rsquare < r_cut*r_cut)
                        {
                            neigh_ar[id_part_in_array + count] = i;
                            delx[id_part_in_array + count] = (x0+(x_max-x1))*(x0+(x_max-x1));
                            dely[id_part_in_array + count] = (y0+(y_max-y1))*(y0+(y_max-y1));
                            count++;
                            // if (id_part == 5) printf("%i; ", i);
                        }
                    }
                    else if ((y1>y_b)&&(y1<y_up)&&(x1>x_r))
                    {
                        rsquare = (x_max-x1+x0)*(x_max-x1+x0)+(y1-y0)*(y1-y0);
                        if (rsquare < r_cut*r_cut)
                        {
                            neigh_ar[id_part_in_array + count] = i;
                            delx[id_part_in_array + count] = (x_max-x1+x0)*(x_max-x1+x0);
                            dely[id_part_in_array + count] = (y1-y0)*(y1-y0);
                            count++;
                            // if (id_part == 5) printf("%i; ", i);
                        }

                    }
                    else if ((x1>x_r)&&(y1<y_b))
                    {
                        rsquare = (x_max-x1+x0)*(x_max-x1+x0)+(y1+y_max-y0)*(y1+y_max-y0);
                        if (rsquare < r_cut*r_cut)
                        {
                            neigh_ar[id_part_in_array + count] = i;
                            delx[id_part_in_array + count] = (x_max-x1+x0)*(x_max-x1+x0);
                            dely[id_part_in_array + count] = (y1+y_max-y0)*(y1+y_max-y0);
                            count++;
                            // if (id_part == 5) printf("%i; ", i);
                        }

                    }
                    else if  ((x1>x_l)&&(x1<x_r)&&(y1<y_b))
                    {
                        rsquare = (x0-x1)*(x0-x1)+(y_max-y0+y1)*(y_max-y0+y1);
                        if (rsquare < r_cut*r_cut)
                        {
                            neigh_ar[id_part_in_array + count] = i;
                            delx[id_part_in_array + count] = (x0-x1)*(x0-x1);
                            dely[id_part_in_array + count] = (y_max-y0+y1)*(y_max-y0+y1);
                            count++;
                            // if (id_part == 5) printf("%i; ", i);
                        }

                    }
                    else if ((x1<x_l)&&(y1<y_b))
                    {
                        rsquare = (x_max-x0+x1)*(x_max-x0+x1) + (y_max-y0+y1)*(y_max-y0+y1);
                        if (rsquare < r_cut*r_cut)
                        {
                            neigh_ar[id_part_in_array + count] = i;
                            delx[id_part_in_array + count] = (x_max-x0+x1)*(x_max-x0+x1);
                            dely[id_part_in_array + count] = (y_max-y0+y1)*(y_max-y0+y1);
                            count++;
                            // if (id_part == 5) printf("%i; ", i);
                        }

                    }
                    else if ((y1>y_b)&&(y1<y_up)&&(x1<x_l))
                    {
                        rsquare = (x_max-x0+x1)*(x_max-x0+x1)+(y1-y0)*(y1-y0);
                        if (rsquare < r_cut*r_cut)
                        {
                            neigh_ar[id_part_in_array + count] = i;
                            delx[id_part_in_array + count] = (x_max-x0+x1)*(x_max-x0+x1);
                            dely[id_part_in_array + count] = (y1-y0)*(y1-y0);
                            count++;
                            // if (id_part == 5) printf("%i; ", i);
                        }

                    }
                    else if ((x1<x_l)&&(y1>y_up))
                    {
                        rsquare = (x_max-x0+x1)*(x_max-x0+x1)+(y0+(y_max-y1))*(y0+(y_max-y1));
                        if (rsquare < r_cut*r_cut)
                        {
                            neigh_ar[id_part_in_array + count] = i;
                            delx[id_part_in_array + count] = (x_max-x0+x1)*(x_max-x0+x1);
                            dely[id_part_in_array + count] = (y0+(y_max-y1))*(y0+(y_max-y1));
                            count++;
                            // if (id_part == 5) printf("%i; ", i);
                        }
                    }
                }
            }
            // if (id_part == 5) printf("\\n");
        }
    }


    extern "C" __global__ void LjPFVelUpdate(float *x_p, float *y_p, int *neigh_ar, float *f_x, float *f_y, float *energy,  float *virial, int n_part, int num_max1bors)
    {
        int ix = threadIdx.x + blockIdx.x * blockDim.x;
        int iy = threadIdx.y+blockIdx.y*blockDim.y;
        int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс времени
        int id_part_in_array = id_part * num_max1bors;
        if (id_part < n_part)
        {
            f_x[id_part] = 0;
            f_y[id_part] = 0;
            energy[id_part] = 0;
            virial[id_part] = 0;
            double delx, dely, r2inv, r6inv, force;
            int id_neigh;

            for(int i = 0; i < num_max1bors; i++)
            {
                if (neigh_ar[id_part_in_array + i] < 0)  break;
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
            if (id_part == 5) printf("cuda, fx = %f; fy = %f", f_x[id_part], f_y[id_part]);
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
                 temperature: float,
                 line_flag: bool
                 ):
        self.fig_path = "/home/artur/Документы/Cell/CellByNeuralNet/BLMRis/"
        self.lmp = None
        self.num_tp = num_tp
        self.n_part =  None
        self.r_cut = np.float32(r_cut)
        self.d_t = np.float32(d_t)
        self.xy_box = np.array(xy_box, dtype=float)
        self.num_time_steps = int(num_time_steps)
        # wtf is that
        self.mass = np.array([1., 1.])
        self.vxi = np.array([0.0, 0.0])
        self.xi = np.array([0.0, 0.0])
        self.ke = 0     #kinetci energy
        self.pe = 0     #potential energy
        self.F_tot = 0  # total force in system
        self.num_max1bors = np.int32(20)
        self.temperature = temperature
        self.lmp_path_file = lmp_path_file
        self.id_part_print = 0
        self.line_flag = line_flag

        self.periodic_bond = periodic_bond

        self.particles = None
        self.vel = None
        self.h_vy = None
        self.h_vx = None
        self.V = None
        self.rho = rho



        self.h_neigh_r = None
        self.h_virial  = None
        self.h_xpos = None  # x particle positions (host)
        self.h_ypos = None  # y particle positions (host)
        self.h_fx = None
        self.h_fy = None
        self.h_pe = None

        self.x_vx = None
        self.x_vy = None


        self.d_neigh_r = None
        self.d_virial = None
        self.d_fx = None
        self.d_fy = None
        self.d_pe = None
        self.d_xp = None
        self.d_yp = None
        self.d_vx = None
        self.d_vy = None


    def init_state(self):
        """
        Initialize the particle position by cubic lattice
        """
        # random position and type of particle

        if self.line_flag == True:
            self.n_part =  np.int32(self.xy_box[0] * self.xy_box[1] * self.rho)
            print("self.n_part = " + str(self.n_part))
            n_2_x = np.int32(math.floor(self.n_part**(1/2)+0.5)) + 1
            print("n_2_x = " + str(n_2_x))
            n_2_y = 1
        else:
            self.n_part =  np.int32(self.xy_box[0] * self.xy_box[1] * self.rho)
            n_2_x = np.int32(math.floor(self.n_part**(1/2)+0.5))
            n_2_y = np.int32(math.floor(self.n_part**(1/2)+0.5))
            self.n_part = n_2_x * n_2_y

        #self.particles = np.zeros((self.n_part, 3)) # x_pos, y_pos, type

        self.h_xpos = np.zeros(self.n_part)
        self.h_ypos = np.zeros(self.n_part)


        ix, iy = 0, 0
        bx = self.xy_box[0]
        by = self.xy_box[1]

        for i in range(self.n_part):
            self.h_xpos[i] = (ix + 0.5) * bx / n_2_x
            self.h_ypos[i] = (iy + 0.5) * by / n_2_y
            #self.particles[i, 0] = (ix + 0.5) * bx / n_2
            #self.particles[i, 1]= (iy + 0.5) * by / n_2
            ix += 1
            if ix == n_2_x:
                ix = 0
                iy += 1
            if iy == n_2_y:
                iy = 0

        #self.particles = self.particles.astype(np.float32)
        self.h_xpos = self.h_xpos.astype(np.float32)
        self.h_ypos = self.h_ypos.astype(np.float32)

        print("Particle positions completed (square latiice)")

        # initialize self.velocity by uniform distribution
        self.vel = np.random.uniform(-0.5, 0.5, self.n_part * 2).reshape(self.n_part, 2)

        # define center of mass
        cm_x = self.vel[:, 0].sum() / self.n_part
        cm_y = self.vel[:, 1].sum() / self.n_part

        # initialize kinetic energy
        # elminate center of mass drift (make zero momentum)

        self.h_vx = self.vel[:, 0] - cm_x
        self.h_vy = self.vel[:, 1] - cm_y
        self.vel[:, 0] -= cm_x
        self.vel[:, 1] -= cm_y

        # obtain kinetic energy from self.velocity
        self.ke = 0.5 * np.square(self.vel).sum()

        # define 'scale self.velocity'
        T_temp = self.ke * 2 / (3 * self.n_part)
        print("T_temp = " + str(T_temp))
        scale = math.sqrt(self.temperature / T_temp)
        print("self.temperature = " + str(self.temperature))
        print("scale = " + str(scale))

        self.h_vx = np.multiply(scale, self.h_vx)
        self.h_vy = np.multiply(scale, self.h_vy)

        self.h_vx = self.h_vx.astype(np.float32)
        self.h_vy = self.h_vx.astype(np.float32)

        self.vel = np.multiply(scale, self.vel)

        self.ke = 0.5 * np.square(self.vel).sum()

        print("Velocity Initialization Completed")

        #self.num_max1bors = np.int32(20)
        self.h_neigh_r = (np.zeros(self.n_part * self.num_max1bors) - 1).astype(np.int32)

        print("Neighbor array Initialization completed (zeros) ")
        self.h_fx = np.zeros(self.n_part).astype(np.float32)
        self.h_fy = np.zeros(self.n_part).astype(np.float32)
        self.h_pe = np.zeros(self.n_part).astype(np.float32)
        self.h_virial = np.zeros(self.n_part).astype(np.float32)
        print("Force and Potential Enregy arrays Initialization completed (zeros) ")

        #self.particles = np.c_[self.xy_box * np.random.random((self.n_part, 2)), np.ones(self.n_part)]

    def init_device(self):
        self.grid = (32, 32, 1)
        self.block = (2, 2, 1)
        self.d_neigh_r = cuda.mem_alloc(self.h_neigh_r.nbytes)
        cuda.memcpy_htod(self.d_neigh_r, self.h_neigh_r)

        self.d_xp = cuda.mem_alloc(self.h_xpos.nbytes)
        self.d_yp = cuda.mem_alloc(self.h_ypos.nbytes)
        self.d_vx = cuda.mem_alloc(self.h_vx.nbytes)
        self.d_vy = cuda.mem_alloc(self.h_vy.nbytes)
        self.d_fx = cuda.mem_alloc(self.h_fx.nbytes)
        self.d_fy = cuda.mem_alloc(self.h_fy.nbytes)
        self.d_pe = cuda.mem_alloc(self.h_pe.nbytes)
        self.d_virial = cuda.mem_alloc(self.h_virial.nbytes)



    def h_update_force_potential(self):
        self.h_fx = np.zeros(self.n_part).astype(np.float32)
        self.h_fy = np.zeros(self.n_part).astype(np.float32)
        for id_part in range(self.n_part):
            id_part_in_array = id_part * self.num_max1bors
            if id_part == self.id_part_print:
                print(str(self.id_part_print) + ": " + str(self.h_xpos[id_part]) + "; " + str(self.h_ypos[id_part]) )
            for i in range(self.num_max1bors):
                if self.h_neigh_r[id_part_in_array + i] < 0:
                    break
                id_neigh = self.h_neigh_r[id_part_in_array + i]
                delx = self.h_xpos[id_part] - self.h_xpos[id_neigh]
                dely = self.h_ypos[id_part] - self.h_ypos[id_neigh]



                r2inv = delx * delx + dely * dely
                r6inv = r2inv * r2inv * r2inv

                r2inv = 1 / r2inv
                r6inv = 1 / r6inv

                force = 48 * r2inv * r6inv * (r6inv - 0.5)
                if id_part == self.id_part_print:
                    print(str(id_part) + "-" + str(id_neigh) +"; dx, dy" + str(delx) + "; " +str(dely) +"; f = "+ str(force))
                self.h_fx[id_part] += delx * force
                self.h_fy[id_part] += dely * force
                #self.h_fx[id_neigh] -= delx * force
                #self.h_fy[id_neigh] -= dely * force
                self.h_pe[id_part] += 4 * (r6inv * r6inv + r6inv)
                self.h_virial[id_part] += force * (delx * delx + dely * dely)

    def d_update_force_potential(self):

        #LjPF(float *x_p, float *y_p, int *neigh_ar, float *f_x, float *f_y, float *energy,  float *virial, int n_part, int num_max1bors)

        func_ljFP = mod.get_function("LjPFVelUpdate")
        cuda.memcpy_htod(self.d_xp, self.h_xpos)
        cuda.memcpy_htod(self.d_yp, self.h_ypos)

        func_ljFP(self.d_xp, self.d_yp, self.d_neigh_r, self.d_fx, self.d_fy, self.d_pe, self.d_virial, self.n_part, self.num_max1bors , block=(1,1,1), grid=(10, 1))

        cuda.memcpy_dtoh(self.h_fx, self.d_fx)
        cuda.memcpy_dtoh(self.h_fy, self.d_fy)
        cuda.memcpy_dtoh(self.h_pe, self.d_pe)
        cuda.memcpy_dtoh(self.h_virial, self.d_virial)
        print("\nhost: " + str(self.h_fx[self.id_part_print]) + "; " + str(self.h_fy[self.id_part_print]))
        return 0

    def d_get_neighbors(self):

        cuda.memcpy_htod(self.d_xp, self.h_xpos)
        cuda.memcpy_htod(self.d_yp, self.h_ypos)
        print("in d_get_neighbors, befor cuda")
        func_neagh = mod.get_function("ReturnNeghbors")

        #(       float *x_p, float *y_p, int *neigh_ar, float r_cut, int n_part,   int num_max1bors)

        func_neagh(self.d_xp, self.d_yp, self.d_neigh_r, self.r_cut, self.n_part, self.num_max1bors , block=self.block, grid=self.grid)
        print("in d_get_neighbors, after cuda \n" )
        cuda.memcpy_dtoh(self.h_neigh_r, self.d_neigh_r)
        id_part_in_array = self.id_part_print * self.num_max1bors
        print("neighbors:")
        print(self.h_neigh_r[id_part_in_array : id_part_in_array + self.num_max1bors])



    # Nose-hoover Chain Algorithm 31 (page 474-475) from Frenkel & Smit's book:
    def h_nhchain(self):
        dt_2 = self.d_t / 2
        dt_4 = self.d_t / 4
        dt_8 = self.d_t / 8
        # mass Q and

        G2 = (self.mass[0] * self.vxi[0] * self.vxi[0] - self.temperature)
        self.vxi[1] += G2 * dt_4   # eq 2.6
        self.vxi[0] *= math.exp(-self.vxi[1] * dt_8)    # eq 2.7

        G1 = (2 * self.ke - 3 * self.n_part * self.temperature) / self.mass[0]

        self.vxi[0] += G1 * dt_4   # eq 2.8
        self.vxi[0] *= math.exp(-self.vxi[1] * dt_8)    # eq 2.7

        self.xi[0]+= self.vxi[0] * dt_2  # eq 2.9
        self.xi[1]+= self.vxi[1] * dt_2  # eq 2.10

        s = math.exp(-self.vxi[0] * dt_2)    # eq 2.11
        self.vel = np.multiply(s, self.vel)  # eq 2.11
        self.ke *= (s * s)   # kinetik enegy

        self.vxi[0] *= np.exp(-self.vxi[1] * dt_8)  # eq 2.7

        G1=(2 * self.ke - 3 * self.n_part * self.temperature) / self.mass[0]

        self.vxi[0] += G1 * dt_4 # eq 2.8
        self.vxi[0] *= math.exp(-self.vxi[1] * dt_8)  # eq 2.7

        G2 = (self.mass[0] * self.vxi[0] * self.vxi[0] - self.temperature) / self.mass[1]

        self.vxi[1] += G2 * dt_4 # eq 2.6



    def run_nvt(self, dump):

        # init state for all
        self.init_state()
        self.init_device()
        self.lmp = open(self.lmp_path_file + str(round(self.temperature, 2)) +"_den_" + str(self.rho) + '.lammpstrj', 'w')
        self.get_lammps_file(0)
        for ts in range(self.num_tp): # self.num_time_steps
            print(str(ts + dump) + "\n")

            # update kinetic energy by Nose-Hoover Chain
            self.h_nhchain()
            # first-half of Nose-Hoover Chain Algorithm.
            self.h_update_position()

            # update potential energy, force, virial pressure
            print("in loop")
            self.d_get_neighbors()
            print("in loop after d_get_neighbors \n in loop before h_update_force_potential")
            #self.d_update_force_potential()
            self.h_update_force_potential()
            print("in after h_update_force_potential")
            # Second-half of Nose-Hoover Chain Algorithm.
            self.h_update_velocity()
            self.h_update_position()

            # calculate kinetic energy from velocity of particles
            self.ke = 0.5 * np.square(self.vel).sum()
            # calculate kinetic energy again from Nose-Hoover Chain
            self.h_nhchain()

            # add if
            if ts % dump == 0:
                print(ts)
                self.plot_system(ts + dump)
                self.get_lammps_file(ts + dump)

        self.lmp.close()


        #for ts in range(self.num_tp):

            #print("ts = " + str(ts) + "; ")
            #h_nhchain()
            #h_update_position()
            #d_get_neighbors()
            #d_update_force_potential()



    def h_update_velocity(self):
        print("vx, vy, = " + str(self.vel[self.id_part_print, 0]) + "; " + str(self.vel[self.id_part_print, 1]) )
        self.vel[:, 0] -= self.d_t * self.h_fx
        self.vel[:, 1] -= self.d_t * self.h_fy
        print("fx, fy = " + str(self.h_fx[self.id_part_print])+ str(self.h_fy[self.id_part_print]))
        print("vx1, vx2 = " + str(self.vel[self.id_part_print, 0])+ str(self.vel[self.id_part_print, 1]))

    def h_update_position(self):


        self.h_xpos += self.vel[:, 0] * self.d_t / 2
        self.h_ypos += self.vel[:, 1] * self.d_t / 2

        if 1 < 0:
            for i in range(self.n_part):
                if self.h_xpos[i] > self.xy_box[0]:
                    self.h_xpos[i] -= self.xy_box[0]
                if self.h_xpos[i] < 0:
                    self.h_xpos[i] += self.xy_box[0]
                if self.h_ypos[i] > self.xy_box[1]:
                    self.h_ypos[i] -= self.xy_box[1]
                if self.h_ypos[i] <0:
                    self.h_ypos[i] += self.xy_box[1]


    def get_lammps_file(self, ts):
         self.lmp.write("ITEM: TIMESTEP" + "\n")
         self.lmp.write(str(ts) + "\n")
         self.lmp.write("ITEM: NUMBER OF ATOMS\n")
         self.lmp.write(str(self.n_part) + "\n")
         self.lmp.write("ITEM: BOX BOUNDS pp pp pp\n")
         self.lmp.write(str(0) + " " + str(self.xy_box[0]) + "\n" )
         self.lmp.write(str(0) + " " + str(self.xy_box[1]) + "\n" )
         self.lmp.write(str(0.1) + " " + str(-0.1) + "\n" )
         self.lmp.write("ITEM: ATOMS id x y\n")
         for i in range(self.n_part):
             self.lmp.write(str(i) + " " + str(self.h_xpos[i])+ " " + str(self.h_ypos[i]) + "\n")


    def plot_system(self, ts):

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.plot(self.h_xpos, self.h_ypos, ".")

        for i in range(self.n_part):
            ax.text(self.h_xpos[i], self.h_ypos[i], str(i) +"; \n" + str(round(self.h_xpos[i], 2)) + "; " +   str(round(self.h_ypos[i], 2)), fontsize = 5)
            v = np.sqrt(self.vel[i, 0] * self.vel[i, 0] +  self.vel[i, 1] * self.vel[i, 1])
            ax.arrow(self.h_xpos[i], self.h_ypos[i], self.vel[i, 0], self.vel[i, 1], ec = 'black', fc = 'black', head_width = v * 0.14, head_length = v * 0.15, width = v * 0.25 * 0.0125, length_includes_head = True) # рисовка стрелки

        plt.savefig(self.fig_path + str(ts) +"Sys.png", transparent=True, bbox_inches='tight',pad_inches = 0)
        plt.close(fig)






sim = MD(num_tp = 1000000,
                 rho = 1,
                 r_cut = 2.,
                 d_t = 0.001,
                 num_time_steps = 10,
                 xy_box = [2, 1],
                 periodic_bond = [1, 1],
                 lmp_path_file = "/home/artur/Документы/Cell/CellByNeuralNet/BLMtxt/",
                 temperature = 0.1,
                 line_flag = True
)
sim.run_nvt(200)
#sim.plot_system("/home/artur/Документы/Cell/CellByNeuralNet/BLMRis/")
sim.d_get_neighbors()































