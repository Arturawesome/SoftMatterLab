/*
 ============================================================================
Name       : flow_anal.cu
Author     : Nasyrov A.D.

Copyright  : Your copyright notice
Description: analysis of active sysytem with periodic flow force




Еechnology : CUDA, Object Oriented Programming

The Titan X parametres:
Maximum number of threads per block:           1024
Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
 ============================================================================
*/
#include <string.h>
#include <math.h>
#include <chrono>

#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <stdio.h>
#include<cuda_runtime.h>

extern "C" __global__ void CalcImpulsMomentGPU(double *d_xcord_now, double *d_ycord_now, double *d_vx_now, double *d_vy_now, double *d_imp_of_part)
{
    // calac made relative each particle
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y+blockIdx.y*blockDim.y;
	int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс уцентральной частицы (-1)
	if (id_part < nparts)
    {
        d_imp_of_part[id_part] += abs(d_xcord_now[id_part] * d_vy_now[id_part] - d_ycord_now[id_part] * d_vx_now[id_part]);
    }


}

extern "C" __global__ void ReturnSigma2tDependences(double *d_sigma2t, double *d_sigma2t_itog, int len_ts_for_sigm2t, int nparts)
{
    //calc relative of time
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y+blockIdx.y*blockDim.y;
	int id_time = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс времени

	for(int id_part = 0; id_part < nparts; id_part++)
    {
        d_sigma2t_itog[id_time] += (d_sigma2t[id_part * len_ts_for_sigm2t + id_time] / nparts);
    }
    //printf("( %f )", d_sigma2t_itog[id_time]);
}

extern "C" __global__ void CalcMsd(double *d_xcord_now, double *d_ycord_now, double *d_xcord_past, double *d_ycord_past,  double *d_sigma2tx, double *d_sigma2ty, double *d_sigma2t, double x_box, double *d_sigma_gist, int nparts, double dr_g, int len_ts_for_sigm2t, int ts)
{
    // calc relative of particles
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y+blockIdx.y*blockDim.y;
	int id_part = iy * (gridDim.x * blockDim.x) + ix;  					 //индекс уцентральной частицы (-1)


	if (id_part < nparts)
    {

        double drx, dry, r_right, r_left, x_now, x_past, y_now, y_past;
        x_now = d_xcord_now[id_part];
        y_now = d_ycord_now[id_part];
        x_past = d_xcord_past[id_part];
        y_past = d_ycord_past[id_part];
        r_right = x_now+x_box/2.1;
        r_left = x_now-x_box/2.1;
        dry = y_now - y_past;

        if(x_past > r_left && x_past < r_right)
        {
            drx = x_now - x_past;
        }
        else if(x_past < r_left)
        {
            drx = x_box - x_now + x_past;
        }
        else
        {
            drx = x_box - x_past + x_now;
        }

        //== displacment each particles in present timepstep =======================
        d_sigma2tx[id_part * len_ts_for_sigm2t + ts] = drx;
        d_sigma2ty[id_part * len_ts_for_sigm2t + ts] = dry;


        //== id displametnt in gistogram =======================
        d_sigma_gist[id_part * len_ts_for_sigm2t + ts] = int(floor(sqrt(drx * drx + dry * dry) / dr_g));
        //printf("(%i; %i; %f) ", id_part, int(floor(sqrt(drx * drx + dry * dry) / dr_g)), sqrt(drx * drx + dry * dry) );

        if(ts % len_ts_for_sigm2t == (len_ts_for_sigm2t - 1))
        {
            drx = d_sigma2tx[id_part * len_ts_for_sigm2t + 0];
            dry = d_sigma2ty[id_part * len_ts_for_sigm2t + 0];
            d_sigma2t[id_part * len_ts_for_sigm2t + 0] = sqrt(drx * drx + dry * dry);

            for(int id_ts = 1; id_ts < len_ts_for_sigm2t; id_ts++)
            {
                drx += d_sigma2tx[id_part * len_ts_for_sigm2t + id_ts];
                dry += d_sigma2ty[id_part * len_ts_for_sigm2t + id_ts];
                d_sigma2t[id_part * len_ts_for_sigm2t + id_ts] = sqrt(drx * drx + dry * dry);
            }
        }
    }
}


class Flow{
    std::fstream 	f_flow;					// filе with flow grid data
	std::fstream 	f_trj;					// filе with lammps f_trj
	std::string 	s;					// vrenennam0 peremenna9 dl9 chteni9
    int     nparts;	            			// number of particles in system (lammps trj file)
    int     x_grid;
    int     y_grid;
    int     n_el_gist;
    int     len_ts_for_sigm2t;

    double  *h_vx_grid;                          // vx grid on host
    double  *h_vy_grid;                          // vy grid on host
    double  *h_vabs_grid;                        // abs v grid on host
    double  *h_lx_grid;                          // arrray for impuls moment x
    double  *h_ly_grid;                          // arrray for impuls moment y


    double  *h_sigma_gist2;
    double  *h_sigma_gist;                      // mean square displacment of praticle for calc diffusion (GISTOGRAM OF displacment)
    double  *h_xcord;
    double  *h_ycord;
    double  *h_vx;
    double  *h_vy;
    double  *h_imp_of_part;                     // impuls moment of each paticles
    double  *h_sigma2t_itog;



    double  dr_g;                              // step in r for diffusion gistogram
    std::vector<std::vector<double>> sigma2tx;
    std::vector<std::vector<double>> sigma2ty;
    std::vector<std::vector<double>> sigma2t;

    double x_min, x_max, y_min, y_max;
    double  act;


    // ======GPU======GPU======GPU======GPU======GPU======GPU======GPU======GPU======GPU
    dim3    dimGrid_map;        			//
    dim3    dimBlock_map;
    double  *d_vx_grid;                          // vx grid on device
    double  *d_vy_grid;                          // vy grid on device
    double  *d_vabs_grid;                        // abs v grid on device
    double  *d_lx_grid;                          // arrray for impuls moment x
    double  *d_ly_grid;                          // arrray for impuls moment y

    double  *d_imp_of_part;                     // impuls moment of each paticles

    double  *d_xcord_now;
    double  *d_ycord_now;
    double  *d_xcord_past;
    double  *d_ycord_past;

    double  *d_sigma_gist;                       // mean square displacment of praticle for calc diffusion gistogram
    double  *d_sigma2tx;                        // смещение чатиц на таймтепах размер nparts * len_ts_for_sigm2t
    double  *d_sigma2ty;
    double  *d_sigma2t;

    double  *d_sigma2t_itog;                    // итог по вычислению сигма, размер len_ts_for_sigm2t


public:
    Flow(std::string &full_name_file, std::string &full_name_file_flow, double r_c_dif_gist, int n_in_gist, int len_ts_sigma2t);
    ~Flow();
    int Read_Header();
    void Read_State();
    void Read_flow_info();
    void Calc_msd_gist(int ts);
    void ReturnGist(int ts);
    void print_array_2D(int num_str, int num_col, double *array);
    void print_array_1D(int num_str,double *array);
    void Read_flow_header();
    void WriteGist(std::ofstream& full_name_file, int counter);
    void WriteSigma2t(std::ofstream& full_name_file, int counter);
    void CalcImpulsMomet(int ts);
};

void Flow::CalcImpulsMomet(int ts)
{
    // func too calc impuls moment in system
    double x0 = 0, y0 = 0, z0 = 0;      // point relative to which the calculation is made



}

void Flow::Calc_msd_gist(int ts)
{
    // func to calc msd gist
    if (ts > 0)
    {
        cudaMemcpy(d_xcord_now, h_xcord, nparts * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ycord_now, h_ycord, nparts * sizeof(double), cudaMemcpyHostToDevice);

        dimBlock_map.x = static_cast<int>(32);
        dimBlock_map.y = static_cast<int>(1);
        dimBlock_map.z = static_cast<int>(1);

        dimGrid_map.x = static_cast<int>(32);
        dimGrid_map.y = static_cast<int>(1);
        dimGrid_map.z = static_cast<int>(1);

        CalcMsd<<<dimGrid_map, dimBlock_map>>>(d_xcord_now, d_ycord_now, d_xcord_past, d_ycord_past, d_sigma2tx, d_sigma2ty, d_sigma2t, (x_max - x_min), d_sigma_gist, nparts, dr_g, len_ts_for_sigm2t, ts-1);
        cudaDeviceSynchronize();

         if(ts % len_ts_for_sigm2t == 0)
        {
            dimBlock_map.x = static_cast<int>(len_ts_for_sigm2t);
            dimBlock_map.y = static_cast<int>(1);
            dimBlock_map.z = static_cast<int>(1);

            dimGrid_map.x = static_cast<int>(1);
            dimGrid_map.y = static_cast<int>(1);
            dimGrid_map.z = static_cast<int>(1);
            cudaMemcpy(h_sigma_gist2, d_sigma_gist, nparts * len_ts_for_sigm2t * sizeof(double), cudaMemcpyDeviceToHost);

            ReturnSigma2tDependences<<<dimGrid_map, dimBlock_map >>>(d_sigma2t, d_sigma2t_itog, len_ts_for_sigm2t, nparts);
            ReturnGist(ts);
        }
    }
    cudaMemcpy(d_xcord_past, h_xcord, nparts * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ycord_past, h_ycord, nparts * sizeof(double), cudaMemcpyHostToDevice);
}

void Flow::ReturnGist(int ts)
{
    int id_gist;
    for(int id_part = 0; id_part < nparts; id_part++)
    {
        for(int id_ts = 0; id_ts < len_ts_for_sigm2t; id_ts ++)
        {
            id_gist = h_sigma_gist2[id_part * len_ts_for_sigm2t + id_ts];

            h_sigma_gist[id_gist] += 1;
        }
    }
}
Flow::Flow(std::string &full_name_file, std::string &full_name_file_flow, double r_c_dif_gist, int n_in_gist, int len_ts_sigma2t){		//konstructor
    n_el_gist = n_in_gist;

    len_ts_for_sigm2t = len_ts_sigma2t -1;
	f_trj.open(full_name_file.c_str());					// open the lammps file trajectory
    nparts = Read_Header();					        // number of particles in system

    h_xcord = new double[nparts];					// allocate memory on host for coord
    h_ycord = new double[nparts];					// allocate memory on host for coord

    h_vx = new double[nparts];					// allocate memory on host for velocity
    h_vy = new double[nparts];					// allocate memory on host for velocity
    h_imp_of_part = new double[nparts];
    memset(h_imp_of_part, 0, sizeof(h_imp_of_part));


    h_sigma2t_itog = new double[len_ts_for_sigm2t];

    h_sigma_gist2 = new double[nparts * len_ts_for_sigm2t];
    h_sigma_gist = new double[n_in_gist];

    memset(h_sigma_gist, 0, sizeof(h_sigma_gist));
    dr_g = r_c_dif_gist / n_in_gist;

    memset(h_sigma2t_itog, 0, sizeof(h_sigma2t_itog));

//     std::vector<std::vector<double>> sigma2tx(nparts);
//     std::vector<std::vector<double>> sigma2ty(nparts);
//     std::vector<std::vector<double>> sigma2t(len_ts_for_sigm2t);

    std::cout<<"nparts = "<<nparts<<"\n";
	int id_part;
    for(int i = 0; i<nparts; i++)
    {				     // read psticle data
        f_trj>>id_part;							     // read id particle
        f_trj>>h_xcord[id_part];						// read x-koordinatu
        f_trj>>h_ycord[id_part];						// read y-koordinatu
        //f_trj>>h_vx[id_part];						// read vx-velocity
        //f_trj>>h_vx[id_part];						// read vy-velocity
    }


    // allocate memort on host
    f_flow.open(full_name_file_flow.c_str());
    Read_flow_header();
    std::cout<<"x_grid, y_grid = "<<x_grid<<" "<<y_grid<<"\n";

    h_vx_grid = new double[x_grid * y_grid];
    h_vy_grid = new double[x_grid * y_grid];

    h_lx_grid = new double[x_grid * y_grid];
    h_ly_grid = new double[x_grid * y_grid];

    Read_flow_info();

    // allocate memort on device

    cudaMalloc((void**)&d_vx_grid, x_grid * y_grid*sizeof(double));
    cudaMalloc((void**)&d_vy_grid, x_grid * y_grid*sizeof(double));

    cudaMalloc((void**)&d_lx_grid, x_grid * y_grid*sizeof(double));
    cudaMalloc((void**)&d_ly_grid, x_grid * y_grid*sizeof(double));

    cudaMalloc((void**)&d_xcord_now, nparts * sizeof(double));
    cudaMalloc((void**)&d_ycord_now, nparts * sizeof(double));

    cudaMalloc((void**)&d_xcord_past, nparts * sizeof(double));
    cudaMalloc((void**)&d_ycord_past, nparts * sizeof(double));

    cudaMalloc((void**)&d_imp_of_part, nparts * sizeof(double));

    cudaMalloc((void**)&d_sigma2tx, nparts * len_ts_for_sigm2t * sizeof(double));
    cudaMalloc((void**)&d_sigma2ty, nparts * len_ts_for_sigm2t * sizeof(double));
    cudaMalloc((void**)&d_sigma2t, nparts * len_ts_for_sigm2t * sizeof(double));
    cudaMalloc((void**)&d_sigma2t_itog, len_ts_for_sigm2t * sizeof(double));
    cudaMalloc((void**)&d_sigma_gist, nparts * len_ts_for_sigm2t * sizeof(double));

    cudaMemcpy(d_sigma2t_itog, h_sigma2t_itog, len_ts_for_sigm2t * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imp_of_part, h_imp_of_part, nparts * sizeof(double), cudaMemcpyHostToDevice);

	//printf("Flow : Device Variable:\t%s\n", cudaGetErrorString(cudaGetLastError()));
   //printf("Flow 2 : cudaGetErrorName:\t%s\n", cudaGetErrorName(cudaGetLastError()));
}
Flow::~Flow()
{
   f_trj.close();
    f_flow.close();
    delete [] h_vx_grid;
    delete [] h_vy_grid;

    delete [] h_lx_grid;
    delete [] h_ly_grid;

    delete [] h_sigma_gist;
    delete [] h_xcord;
    delete [] h_ycord;
    delete [] h_vx;
    delete [] h_vy;

    cudaFree(d_vx_grid);
    cudaFree(d_vy_grid);
    //cudaFree(d_vabs_grid);
    cudaFree(d_lx_grid);
    cudaFree(d_ly_grid);

    cudaFree(d_sigma2tx);
    cudaFree(d_sigma2ty);
    cudaFree(d_sigma2t);
    cudaFree(d_sigma_gist);
    cudaFree(d_sigma2t_itog);

    cudaFree(d_xcord_now);
    cudaFree(d_ycord_now);
    cudaFree(d_xcord_past);
    cudaFree(d_ycord_past);

}

int Flow::Read_Header(){
    int     	n;
    f_trj>>s;f_trj>>s; 							// ITEM: TIMESTEP
    f_trj>>s;								// 0
    f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;				// ITEM:NUMBER OF ATOMS
    f_trj>>n;								// 10000
    f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;		// ITEM: BOX BOUNDS pp pp pp
    f_trj>>x_min;							// chtenie razmerov
    f_trj>>x_max;							// chtenie razmerov
    f_trj>>y_min;							// chtenie razmerov
    f_trj>>y_max;							// chtenie razmerov
    f_trj>>s;//z_min;								// propusk z-coordinat
    f_trj>>s;//z_max;								// propusk z-coordinat
    f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;f_trj>>s;//f_trj>>s;f_trj>>s;	// chtenie musora
    //ITEM ATOMS id    x    y   vx  vy
    return n;
}

void Flow::Read_State(){
    Read_flow_header();
    Read_flow_info();
    Read_Header();
    int id_part;
    for(int i = 0; i < nparts; i++){
    	f_trj>>id_part;
		//read data coordinat
    	f_trj>>h_xcord[id_part];
    	f_trj>>h_ycord[id_part];
		// read velocity data

		//f_trj>>h_vx[id_part - 1];
		//f_trj>>h_vy[id_part - 1];
    }
}
void Flow::Read_flow_header()
{

    f_flow>>s;f_flow>>s; 							//ITEM: TIMESTEP
    f_flow>>s;								// 1000
    f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;				// ITEM: NUMBER OF ATOMS
    f_flow>>s;								// 924

    //std::vector<std::vector<int>> neigh_list(nparts);
    f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;		//ITEM: BOX BOUNDS pp pp grid_size_xy
    f_flow>>s;							// chtenie razmerov x_min
    f_flow>>s;							// chtenie razmerov x_mox
    f_flow>>s;							// chtenie razmerov
    f_flow>>s;							// chtenie razmerov
    f_flow>>x_grid;//z_min;								// propusk z-coordinat
    f_flow>>y_grid;//z_max;								// propusk z-coordinat
    f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;f_flow>>s;//f_trj>>s;f_trj>>s;	// chtenie musora
     //ITEM    ATOMS       id         x         y        vx        vy
}

void Flow::Read_flow_info(){

    double value;
    for(int cols = 0; cols < x_grid; cols++)
    {
        for(int row = 0; row < y_grid; row ++)
        {
            f_flow>>value;
            h_vx_grid[x_grid * row + cols] += value;
            f_flow>>value;
            h_vy_grid[x_grid * row + cols] += value;
        }
    }
}

void Flow::print_array_1D(int num_str,double *array)
{
    std::cout<<"in prind1d \n";
    for(int i = 0; i < num_str; i++)
    {
        std::cout<<array[i]<<" ";
    }
}
void Flow::print_array_2D(int num_str,int num_col,double *array)
{
    std::cout<<"in print 2d \n";
//     id_part * len_ts_for_sigm2t + id_ts
    for(int i = 0; i < num_str; i++)
    {
        for(int j = 0; j < num_col; j++)
        {
            std::cout<<"( "<<i<<" "<<array[num_col * i + j]<<") ";
        }
        std::cout<<"\n";
    }
}
void Flow::WriteGist(std::ofstream& full_name_file, int counter)
{
    double count = 0;
    for(int order = 0; order < n_el_gist; order++)
	{
		full_name_file<<h_sigma_gist[order] / ((counter - 1)  * len_ts_for_sigm2t)<<"\n";
        count += h_sigma_gist[order] / ((counter - 1) * (len_ts_for_sigm2t));
	}
	full_name_file<<count;
}
void Flow::WriteSigma2t(std::ofstream& full_name_file, int counter)
{
    cudaMemcpy(h_sigma2t_itog, d_sigma2t_itog, len_ts_for_sigm2t * sizeof(double), cudaMemcpyDeviceToHost);
    for(int order = 0; order < len_ts_for_sigm2t; order++)
	{
		full_name_file<<h_sigma2t_itog[order] / (counter - 1) <<"\n";
	}

}

int main()
{

    std::string full_path, full_path2,full_save,full_save2;
    std::string path = "/run/media/artur/Новый том/Mansevitch_quantum_dots/trj/MD_trj/";
	std::string save_path = "//run/media/artur/Новый том/Mansevitch_quantum_dots/res/txt/";
    std::string D_r[] = {"0.5"};//, "1", "1.5", "2"};
    std::string A[] = {"1"};//, "2"};
    double r_dif = 5;
    int n_in_gist = 5000;
    int len_t_sigma = 32;   // только степени двойки
    int ts_in = 0;
    int counter = 0;
    for(int ia = 0; ia < 1; ia ++)
    {
        for(int idr = 0; idr < 1; idr++)
        {
            full_path = path + "F_WCA_0.5_per_150_A_" + A[ia] + "_Dr_" +  D_r[idr] +"_rho_1.4.lammpstrj";
            full_path2 = path + "F_WCA_0.5_per_150_A_" + A[ia] + "_Dr_" +  D_r[idr] +"_rho_1.4.lammpstrj_setka_";
            full_save = save_path + "GIST_" + A[ia] + "_Dr_" +  D_r[idr] +"_CUDA.txt";
            full_save2 = save_path + "Sigma2t_" + A[ia] + "_Dr_" +  D_r[idr] +"_CUDA.txt";
            std::cout<<full_path<<"\n";
            Flow Set(full_path, full_path2, r_dif, n_in_gist, len_t_sigma);
            for(int ts  = 0; ts < 100; ts++)
            {
                if(ts % len_t_sigma == 0)
                {
                    ts_in = 0;
                    counter++;
                }
                //std::cout<<"ts = "<<ts<<"\n";
                Set.Read_State();
                Set.Calc_msd_gist(ts_in);
                ts_in++;
            }
            std::ofstream fout_Gist(full_save);      // fail dl9 vuvoda dannux
            std::ofstream fout_Sigma2t(full_save2);      // fail dl9 vuvoda dannux
            Set.WriteGist(fout_Gist, counter);
            Set.WriteSigma2t(fout_Sigma2t, counter);

        }
    }
    return (0);
}
