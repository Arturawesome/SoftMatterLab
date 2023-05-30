import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import pygame
import math
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.cm as cm
from scipy import spatial
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

# dipole35_0.01 : 35 - box length; 0.01 - temperature of the system
#start_time = time.time()

data_path = "/home/softmatter/snakes/0.3/"
log_file = "/home/softmatter/snakes/snakeIdent.log"

N = 1600
frames = 101

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)


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


def IdentShake(t_i, xy, r_shake, xmin, xmax, ymin, ymax, partDiam, size, posNeighbor):
    snakes_id = []
    len_snakes = []
    snakes_info = []
    #vor = scipy.spatial.Voronoi(xy)
    #npoints = len(vor.points)
    #ver_index=np.full(len(vor.vertices),True)


    # opredelenie pogranichnux regionov
    #----------------------------------
    #for i in range(len(vor.vertices)):
        #ver=vor.vertices[i]
        #if ver[0]<xmin or ver[0]>xmax or ver[1]<ymin or ver[1]>ymax:
            #ver_index[i]=False

    #----------------------------------
    # sostavlenie spiska blishaishix sosedei (neigh_id)  neigh_id[i] = [id_1, id_2, ... , id_n];   i - index of particle, id_1,2...n - индексы ссоседних частиц
    #----------------------------------

    #for rid in vor.ridge_points:
        #neigh_id[rid[0]].append(rid[1])
        #neigh_id[rid[1]].append(rid[0])




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
                if r_i < r_shake:
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
                if r_i < r_shake:
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


#bullshit functions
def checkOrder(snakes_id):
    for snake in snakes_id:
        x0 = xy[snake[0], 0]
        y0 = xy[snake[0], 1]

        # neighbors coordinates
        x1 = xy[neigh_id[snake[0]], 0]
        y1 = xy[neigh_id[snake[0]], 1]
        r0=[]

        # distances between central partilce and her neighbors
        for i in range(len(neigh_id[snake[0]])):
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
            if r_i < r_shake:
                r0.append([i, r_i])


        x0 = xy[snake[-1], 0]
        y0 = xy[snake[-1], 1]

        # neighbors coordinates
        x1 = xy[neigh_id[snake[-1]], 0]
        y1 = xy[neigh_id[snake[-1]], 1]
        rl=[]

        # distances between central partilce and her neighbors
        for i in range(len(neigh_id[snake[-1]])):
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
            if r_i < r_shake:
                rl.append([i, r_i])

        print(len(r0), len(rl))
        if len(r0) > 1 or len(rl) > 1:
            print(snake, snake[0], r0, neigh_id[snake[0]], snake[-1], rl, neigh_id[snake[-1]])

#bullshit functions
def getDistance(part0, part1, xy, xmin, xmax):
    x0 = xy[part0, 0]
    y0 = xy[part0, 1]
    x1 = xy[part1, 0]
    y1 = xy[part1, 1]
    if(abs(x1 - x0) > xmax/2 and x1 < x0):
        xi = x1 + xmax
    elif(abs(x1 - x0) > xmax/2 and x1 > x0):
        xi = x1 - xmax
    else:
        xi = x1

    if(abs(y1 - y0) > ymax/2 and y1 < y0):
        yi = y1 + ymax
    elif(abs(y1 - y0) > ymax/2 and y1 > y0):
        yi = y1 - ymax
    else:
        yi = y1

    return np.sqrt((xi - x0)*(xi - x0) + (yi - y0)*(yi - y0))


def getNeighbors(path, N, pos):
    data = [[] for i in range(N+1)]
    f = open(path, 'r')
    f.seek(pos)
    s = f.readline()
    while (s != 'ITEM: ENTRIES index c_6[1] c_6[2] c_6[3] \n' and s != 'ITEM: ENTRIES index c_6[1] c_6[2] c_6[3]\n'):
        s = f.readline()
    while True:
        s = f.readline()
        if (s == 'ITEM: TIMESTEP\n' or s ==''):
            break

        if (int(s.split()[1]) != 0 and int(s.split()[2]) != 0):
            data[int(s.split()[1])].append(int(s.split()[2]))
            #print(int(s.split()[0]), int(s.split()[1]), int(s.split()[2]))
    pos = f.tell()
    f.close()
    return data, pos


def getData(path, N, pos):

    data = np.zeros((N+1, 2))
    f = open(path, 'r')
    f.seek(pos)
    s = f.readline()
    while (s != 'ITEM: ATOMS id x y z\n'):
        s = f.readline()
    for i in range(1,N+1):
        s = f.readline()
        data[i, 0] = s.split()[1]
        data[i, 1] = s.split()[2]
        # data[i, 2] = s.split()[0]
    pos = f.tell()
    f.close()
    return data, pos


def save_txt_data(Gr, r, length, name):
    f = open(name, "w")
    for i in range(length):
        f.write(str(r[i]) + " " + str(Gr[i]) + "\n")
    f.close()


def save_txt_data2D(mas, row, col, name):
    f = open(name, "w")
    for i in range(row):
        for j in range(col):
            f.write(str(mas[i, j]) + " ")
        f.write("\n")
    f.close()


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



with open(log_file, "w") as log:
    log.write("time\tt_i\tsize\tcpu\tmem\n")


p1=psutil.Process(os.getpid())
logging.basicConfig(level=logging.INFO, filename=log_file, filemode="a", format="%(asctime)s %(levelname)s %(message)s")
logging.info(f"STARTED")

sizeArr=[35, 35.5, 36, 36.5, 37, 37.5, 38, 38.5, 39, 39.5, 40, 40.1, 40.2, 40.3, 40.4, 40.5, 40.6, 40.7, 40.8, 40.9, 41, 41.1, 41.2, 41.3, 41.4, 41.5, 41.6, 41.7, 41.8, 41.9, 42,42.5, 43, 44, 45, 46, 47, 48]
#[36,37,38,39,40, 40.05, 40.1, 40.15, 40.2, 40.25, 40.3, 40.35, 40.4, 40.45, 40.5, 40.55, 40.6, 40.65, 40.7, 40.75, 40.8, 40.85, 40.9, 40.95, 41, 41.05, 41.1, 41.15, 41.2, 41.25, 41.3, 41.35, 41.4, 41.45, 41.5, 41.55, 41.6, 41.65, 41.7, 41.75, 41.8, 41.85,41.9,41.95,42,42.3,42.7,43,44,45,45.5,46,47,48,50,55]
#[36,37,38,39,40, 40.2, 40.25, 40.45, 40.6, 40.8, 41, 41.2, 41.4, 41.6, 41.8,42,42.3,42.7,43,44,45,45.5,46,47,48,50,55]#list(float_range(40,42,'0.05'))#[40,40.4,40.7,41,41.1,41.2,41.3,41.4,41.5,41.6,41.8,42,42.3,42.7,43,44,45,45.5,46,47,48,50,55]

tArr=[0.01, 1,2.5,3]#[1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]#[0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.8,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]#[0.001,0.005,0.01,0.05,0.1,0.5,1.0]


for t_i in tArr:
    print(str(t_i))

    img_path = "/home/softmatter/snakes/processing/" + str(t_i) + "/img/"
    txt_path = "/home/softmatter/snakes/processing/" + str(t_i) + "/txt/"
    hist_path = "/home/softmatter/snakes/processing/" + str(t_i) + "/img/histograms/"
    dens_len = "/home/softmatter/snakes/processing/" + str(t_i) + "/dens_len_new.txt"
    dens_var = "/home/softmatter/snakes/processing/" + str(t_i) + "/dens_var_new.txt"
    dens_len_count = "/home/softmatter/snakes/processing/" + str(t_i) + "/dens_len_count_new.txt"

    open(dens_len, 'w').close()
    open(dens_var, 'w').close()
    open(dens_len_count, 'w').close()

    for size in sizeArr:   #35, 36 doesn't work for some t

        #print(size)
        r_shake = 0.8
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
            snakes_id, len_snakes, posNeighbor = IdentShake(t_i, xy, r_shake, xmin, xmax, ymin, ymax, partDiam, size, posNeighbor)


            ##histograms
            # for n, bins in enumerate([100, 50, 25, 10]):
            #
            #     # Set up the plot
            #     h = plt.subplot(2, 2, n + 1)
            #
            #     # Draw the plot
            #     h.hist(len_snakes, bins, color = 'blue', edgecolor = 'black')
            #
            #
            # plt.title('size = %.2f' % size)
            # plt.tight_layout()
            # plt.savefig(hist_path + 'hist' + str(size) + '.png', bbox_inches='tight')
            # plt.close()


            mean_len_snakes += sum(len_snakes) / len(len_snakes)
            mean_var += pvariance(len_snakes)
            len_snakes_glob += len_snakes






        with open(dens_len, "a") as f1:
            f1.write("%s %s\n" % (N / (size * size), mean_len_snakes / count))


        with open(dens_var, "a") as f2:
            f2.write("%s %s\n" % (N / (size * size), mean_var / count))




        counts, bins = np.histogram(len_snakes_glob, bins_num + 1)   #hist: length, count
        bins_mid = np.zeros(bins_num)
        for i in range(bins_num):
            bins_mid[i] = (bins[i] + bins[i+1]) / 2

        with open(dens_len_count, "a") as f3:
            for i in range(bins_num):
                f3.write("%s %s %s\n" % (N / (size * size), bins_mid[i], counts[i] / count))



        logging.info(f"%.1f\t%.1f\t%.2f%%" % (t_i, size, p1.memory_percent()))
    logging.info(f"\n\n")



