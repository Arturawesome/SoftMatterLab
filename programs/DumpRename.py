import numpy as np
import os


def ReneameDump(name, path, prefix, end_name):
    new_prefix = prefix
    #name = path + new_prefix + end_name
    try:
        f = open(name, "r")
        f.close()
        print("yes;   " + name)
    except:
        print("NON;   " + name)
        for filename in os.listdir(path):
            if filename.startswith(new_prefix) and filename.endswith(".txt"):
                #print(filename)
                new_filename = new_prefix + end_name
                #print(new_filename)
                os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

#path_md = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/MD/MIPS_prove_rdf/"
path_exp = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/SnakeData/Experiments/7Output/"
folders_date = ["2023.07.03", "2023.07.04"]
series = [["series1", "series2", "series3", "series4"], ["series1", "series2", "series3", "series4"]]

subseries = [[8, 7, 12, 12], [8, 4, 14, 5]]

for id_fold in range(len(folders_date)):
    date = folders_date[id_fold]
    for id_series in range(len(series[id_fold])):
        sersie  = series[id_fold][id_series]
        print(sersie)
        for id_sub in range(subseries[id_fold][id_series]):
            prefix = str(id_sub +1)
            full_path = path_exp + date + "/" + sersie + "/" + str(id_sub +1) + "/"
            ReneameDump(full_path + prefix + ".txt", full_path, prefix, ".txt")
            print(full_path)

    print("\n\n\n")




#
#
#
# for id_rho in range(len(density)):
#     rho = density[id_rho]
#     for id_h in range(num_h):
#         # dump_100_100_0.7_1.1_1.0_7.2_0.lammpstrj
#         # dump_100_100_0.7_1.1_1.0_6.6_0.lammpstrj
#         h = 1 + round(0.025, 3) * int(id_h)
#         #h = 1.1
#         for id_mu in range(num_mu):
#             mu = round(mu_values[id_mu], 1)
#             #mu = 6.6
#             new_prefix = prefix + str(rho) + "_" + str(h) + "_" + str(Temperature) + "_" + str(mu)
            #ReneameDump(name, path_md, new_prefix,  "_0.lammpstrj")






'''
for t_i in tArr:

    for size in sizeArr:   #35, 36 doesn't work for some t

        #print(size)
        r_snake = 0.8
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
            snakes_id, len_snakes, posNeighbor = IdentShake(t_i, xy, r_snake, xmin, xmax, ymin, ymax, partDiam, size, posNeighbor)


            mean_len_snakes += sum(len_snakes) / len(len_snakes)
            mean_var += pvariance(len_snakes)
            len_snakes_glob += len_snakes


region_mask = np.full(len(vor.regions), 0)
for jj in range(len(vor.points)):
region_id = vor.point_region[jj]
if xy[jj, 0] > xmin_init  and xy[jj, 0] < xmax_init  and xy[jj, 1] > ymin_init and xy[jj, 1] < ymax_init:
    region_mask[region_id] = 1










'''

































