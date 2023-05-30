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
wolt = [10, 11, 12, 13, 14,15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 59]
save_path = "/run/media/artur/B620539220535907/snake/result/ris/"

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

        (result, bins) = np.histogram(d, bins=edges, normed=False)
        g[p, :] = result/numberDensity

    # Average g(r) for all interior particles and compute radii
    g_average = np.zeros(num_increments)
    for i in range(num_increments):
        radii[i] = (edges[i] + edges[i+1]) / 2.
        rOuter = edges[i + 1]
        rInner = edges[i]
        g_average[i] = np.mean(g[:, i]) / (np.pi * (rOuter**2 - rInner**2))

    return (g_average, radii, interior_indices)


def IdentShake(xy, r_shake, xmin, xmax, ymin, ymax, partDiam, nx_grid, ny_grid, save_path, w, cadr):

    #nx_grid = 20
    #ny_grid = 15

    grid_slovent = 1e6 + np.zeros((nx_grid, ny_grid))
    num_solvent = np.zeros((nx_grid, ny_grid))

    dr_grid_x = (xmax - xmin) / (nx_grid)
    dr_grid_y = (ymax - ymin) / (ny_grid)



    rminForSolvent = 1e6
    snakes_id = []
    len_snakes = []
    snakes_info = []
    vor = scipy.spatial.Voronoi(xy)
    npoints = len(vor.points)
    ver_index=np.full(len(vor.vertices),True)


    # opredelenie pogranichnux regionov
    #----------------------------------
    for i in range(len(vor.vertices)):
        ver=vor.vertices[i]
        if ver[0]<xmin or ver[0]>xmax or ver[1]<ymin or ver[1]>ymax:
            ver_index[i]=False

    # otsechenie pogranichnux regionov
    #----------------------------------
    region_mask=np.full(len(vor.regions),True)
    for jj in range(len(vor.regions)):
        region=vor.regions[jj]
        if not -1 in region and not False in ver_index[region]:
            region_mask[jj]=True
        else:
            region_mask[jj]=False

    #----------------------------------
    # sostavlenie spiska blishaishix sosedei (neigh_id)  neigh_id[i] = [id_1, id_2, ... , id_n];   i - index of particle, id_1,2...n - индексы ссоседних частиц
    #----------------------------------
    neigh_id=[[] for i in range(npoints)]
    for rid in vor.ridge_points:
        neigh_id[rid[0]].append(rid[1])
        neigh_id[rid[1]].append(rid[0])



    jj_using_glob = np.zeros(len(vor.points))
    for jj in range(len(vor.points)):

        j_help = np.zeros(len(vor.points)) #j_help = []
        jj_using = []

        if jj_using_glob[jj] > 0:
            continue
        jj_using.append(jj)
        jj_using_glob[jj] = 1
        count = 0

        for jj_el_snake in jj_using:
            if j_help[jj_el_snake] == 1:#jj_el_snake in j_help:
                continue
            region_id = vor.point_region[jj_el_snake]

            if region_mask[region_id]:
                x0 = xy[jj_el_snake, 0]
                y0 = xy[jj_el_snake, 1]

                # neighbors coordinates
                x1 = xy[neigh_id[jj_el_snake], 0]
                y1 = xy[neigh_id[jj_el_snake], 1]

                # distances between central partilce and her neighbors
                r = np.sqrt((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0))

                # condition for snake existing
                if sum(r < r_shake) >= 1:
                    if j_help[jj_el_snake] == 0: #jj_el_snake not in j_help:
                        j_help[jj_el_snake] == 1 #j_help.append(jj_el_snake)
                    for item in enumerate(r):
                        if item[1] < r_shake:
                            if item[1] < partDiam:
                                #calc catet in triangle
                                grid_x_id = math.floor(x0 / dr_grid_x)
                                grid_y_id = math.floor(y0 / dr_grid_y)

                                if item[1] < grid_slovent[grid_x_id, grid_y_id]:
                                    hSolvent = (partDiam ** 2 - item[1] ** 2) ** 0.5 + partDiam
                                    grid_slovent[grid_x_id, grid_y_id] = hSolvent
                                    num_solvent[grid_x_id, grid_y_id] = 1


                            #plus neighbors partilce id in list
                            if neigh_id[jj_el_snake][item[0]] not in jj_using:
                                count += 1
                                jj_using.append(neigh_id[jj_el_snake][item[0]])
                                jj_using_glob[neigh_id[jj_el_snake][item[0]]] = 1

        snakes_id.append(jj_using)
        if count > 2:
            len_snakes.append(count + 1)





    #print(grid_slovent)



    #ff=plt.figure(figsize=(22,22 * (ymax - ymin) / (xmax - xmin)),frameon=False)
    #ax = plt.subplot(111)
    #ax.set_xlim([xmin, xmax])
    #ax.set_ylim([ymin, ymax])
    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])
    #ax.set_aspect('equal', 'datalim')
    #ax.plot(xy[:, 0], xy[:, 1], '.', color = "Black")
    ##ax.plot(xy[snakes_info, 0], xy[snakes_info, 1], 'o', color = "Red")
    #for list_el in snakes_id:
        #if len(list_el) > 2:
            #ax.plot(xy[list_el, 0], xy[list_el, 1], '.')
    #plt.axis('off')
    #plt.savefig(save_path + 'v2w' + str(w) + "_" +str(cadr) + '.png', transparent=True, bbox_inches='tight',pad_inches = 0)
    grid_slovent *= num_solvent
    return grid_slovent, num_solvent, snakes_id, len_snakes


    #plt.close(ff)






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


dencity = np.zeros(len(wolt))

save_path_txt = "/run/media/artur/B620539220535907/snake/result/txt/"
for w_id in range(len(wolt)):
    nx_grid = 15
    ny_grid = 10

    grid_slovent_glob = np.zeros((nx_grid, ny_grid))
    num_solvent_glob =  np.zeros((nx_grid, ny_grid))

    sovlent_list = []
    r_shake = 22
    num_parts = 0
    dr = 0.1
    rMax = 100
    partDiam = 1.31
    r_mas =  np.arange(0., rMax + 1.1 * dr, dr)
    Gr = np.zeros(len(r_mas) - 1)
    gist_r = np.zeros(len(r_mas))
    w = wolt[w_id]
    data = scipy.io.loadmat('/run/media/artur/B620539220535907/snake/Parts/w' + str(w) + '.mat')
    row, col = data["parts"].shape
    print(row, col)
    xy = np.array(data["parts"][0, 0])
    num_cadr = 100#col-10
    conc = 0
    snakes_id = []
    len_snakes_global = []



    for cadr in range(num_cadr):

        xy = np.array(data["parts"][0, cadr])
        num_parts = len(xy[:, 0])
        xmin = min(xy[:, 0])
        xmax = max(xy[:, 0])
        ymin = min(xy[:, 1])
        ymax = max(xy[:, 1])
        #print(xmin, xmax)
        pxTomk_x = 293.5 / (xmax - xmin)
        pxTomk_y = 165.1 / (ymax - ymin)
        pxTomk = (pxTomk_y + pxTomk_x) / 2
        #print(pxTomk)

        if cadr == 0:
            dr *= pxTomk
            rMax *= pxTomk
            r_mas *= pxTomk
            r_shake *= pxTomk


        xy[:, 0] += abs(xmin)
        xy[:, 1] += abs(ymin)
        xy[:, 0] *= pxTomk
        xy[:, 1] *= pxTomk


        xbox = (xmax - xmin) * pxTomk
        ybox = (ymax - ymin) * pxTomk

        conc += (num_parts / xbox / ybox)

        xmin = (min(xy[:, 0]))
        xmax = (max(xy[:, 0]))
        ymin = (min(xy[:, 1]))
        ymax = (max(xy[:, 1]))


        grid_slovent, num_solvent, snakes_id, len_snakes = IdentShake(xy, r_shake, xmin, xmax, ymin, ymax, partDiam, nx_grid, ny_grid, save_path, w, cadr)
        len_snakes_global.append(len_snakes[::])
        #print(len_snakes_global)
        grid_slovent_glob += grid_slovent
        num_solvent_glob += num_solvent

        grid_slovent.tolist()
        grid_slovent = list(chain.from_iterable(grid_slovent))
        sovlent_list.append(grid_slovent)








        #g_r, r, reference_indices = pairCorrelationFunction_2D(xy[:, 0], xy[:, 1], xbox, ybox, rMax, dr)
        #Gr += g_r

    #ff=plt.figure(figsize=(20,20 * (ymax - ymin) / (xmax - xmin)),frameon=False)
    #ax = plt.subplot(111)
    #ax.set_xlim([xmin, xmax])
    #ax.set_ylim([ymin, ymax])
    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])
    #ax.set_aspect('equal', 'datalim')
    #ax.plot(xy[:, 0], xy[:, 1], '.')
    #plt.axis('off')
    #plt.savefig(save_path + 'w' + str(w)+"_" +str(cadr) + '.png', transparent=True, bbox_inches='tight',pad_inches = 0)
    #plt.close(f)

    dencity[w_id] = conc / num_cadr
    for i in range(nx_grid):
        for j in range(ny_grid):
            if num_solvent_glob[i, j] != 0:
                grid_slovent_glob[i, j] /= num_solvent_glob[i, j]

    save_txt_data2D(grid_slovent_glob, nx_grid, ny_grid, save_path_txt +  'Konfimend2_w' + str(w)+'.txt')

    ff=plt.figure(figsize=(10,10* (ymax - ymin) / (xmax - xmin)),frameon=False)

    axs = plt.subplot(111)
    #cmap = plt.cm.Blues
    cmap = plt.cm.YlOrRd
    #cmap = plt.cm.hot_r

    norm_max = 2.5
    z = grid_slovent_glob

    x = np.arange(0, nx_grid)
    y = np.arange(0, ny_grid)
    axs.pcolormesh(x, y, z.T, cmap=cmap, vmin=np.min(z), vmax=norm_max)
    axs.set_ylabel("Y coordinate, mkm")
    axs.set_xlabel("X coordinate, mkm")

    plt.savefig(save_path + str(w)+'_MapConf.png', transparent=True, bbox_inches='tight',pad_inches = 0)

    #save_txt_data(Gr, r, len(Gr), save_path_txt +  'Gr_w' + str(w)+'.txt')


    sovlent_list = list(chain.from_iterable(sovlent_list))
    print(sovlent_list)
    ff=plt.figure(figsize=(20,10),frameon=False)
    ax = plt.subplot(111)
    #ax.set_xlim([0, 10])
    #ax.get_xaxis().set_ticks([])
    #ax.get_yaxis().set_ticks([])
    #ax.set_aspect('equal', 'datalim')
    ax.hist(sovlent_list, 50)
    #plt.axis('off')
    plt.savefig(save_path + 'gist_size/w' + str(w)+'_gist_solvent.png', transparent=True, bbox_inches='tight',pad_inches = 0)
    plt.close(ff)


#f = open("/run/media/artur/B620539220535907/snake/result/txt/dencity.txt", "w")
#for n_id in range(len(dencity)):
    #n = dencity[n_id]
    #w = wolt[n_id]
    #f.write(str(w) + " " + str(n) + "\n")
#f.close()



