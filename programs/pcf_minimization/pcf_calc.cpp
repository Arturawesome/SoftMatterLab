/*
 ============================================================================
Name       : theor_calk_pcf.cu
Author     : Nasyrov A.D.

Description: Calculating the pair correlation fucntion in liquid
            using inegral equation

            In this C++/CUDA program
                                    1) we calc integrals of p_n(r)
                                    2) realization of minimization method

Technology : Object Oriented Programming
 ============================================================================
*/
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cmath>
#include <iomanip>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multiroots.h>
#include "cubature.h"
double koefTT=1.0;
int f_avr(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval);
int f_msd(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval);
int f_gamma1(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval);
int f_gamma2(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval);

int f_dat(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval);
int f_e(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval);
int f_r(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval);
int f_sigm0(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval);

double Ferr(const gsl_vector* v, void* params);
int Ferr_n(const gsl_vector *x, void *param, gsl_vector *ret);

class Scor{
    double 	*Dat;		// masssfi so srezom funkcii
    double *domain;
    double 	rho;		// plotnost; sistemu
    double 	T;			// temperatura sistemu
    double 	scale;		// parametr mashtabirovani9
    double 	h;			// shag setki postroeni9 massiva Dat
    double 	Asm;		// pokazatel' assimetrii
    double 	sigma0;		// teoreticheskii parametr dl9 rascheta stepeni delokalizacii blishaishix uzlov
    int 	N;				// chislo uzlov
    int n_alpha;
    double 	R;			// radius postroeni9
    int 	n_2d;
    int 	l_type;
    int 	u_type;
    int 	dim;
    double 	Fixed_tm4x;
    double 	Fixed_tm4y;
    double 	Fixed_tm3x;
    // -------
    double 	tEx;
    double 	tS0;
    double 	tAsm;
    double 	tM3;
    double 	tM3x;
    double 	tM4x;
    double 	tM4y;
    double 	ttm4x;
    double 	ttm4y;
    double 	ttm3x;
    double 	tmxy2;
    double 	Tmxy2;
    double 	kM3;

    double 	km1;
    double 	km2;
    double 	km3;
    double 	km4;

    double 	t_km1;
    double 	t_km2;
    double 	t_km3;
    double 	t_km4;


    double avr;
    double sigma_na;
    double gamma_1;
    double gamma_2;
    double normirova_pna;


    double t_avr;
    double t_sigma_na;
    double t_gamma_1;
    double t_gamma_2;
    double t_normirova_pna;




    double 	a;


public:
    double  F_avr(double x, double y);
    double  F_msd(double x, double y);
    double  F_gamma1(double x, double y);
    double  F_gamma2(double x, double y);
    double 	F(double x,double y);
    double 	F_e(double x,double y);
    double 	F_r(double x,double y);
    double 	F_sigm0(double x,double y);

    double 	sr;
    double 	st;
    double 	tr;
    double  skoef;
    double 	En;


    Scor(double A, double Rho, double T_in, double sigma0_in, double Asm_in,
         double Gamma2, double scale_in, int n, int L_Type, int U_type, int Dim);
    ~Scor();

    void    Compute_avr();
    void    Compute_msd();
    void    Compute_gamma1();
    void    Compute_gamma2();


    void    Compute_En();
    void 	Compute_Dat();		// vuchislenie znachenii massiva Dat
    void 	Compute_E();		// vuchislenie Ex
    void 	Compute_Asm();		// vuchislenie koefficienta assimetrii
    void 	Compute_sigma0();	// vuchislenie delokalizacii
    void 	Dump_To_File();

    double 	Err_n(double sKM_1, double sKM_2, double sKM_3, double sKM_4, double &out1, double &out2, double &out3, double &out4);
    //double Err(double Tr, double Tsr, double Tst,double Tm4x,double Tm4y,double Tm3x);
    double  Err(double KM_1, double KM_2, double KM_3,double KM_4);


    void 	Find_Min();

    void 	Compute_M3();
    void 	Compute_M4x();
    void 	Compute_M4y();
    void 	Compute_Tmxy2();
};

Scor::Scor(double A, double Rho, double T_in, double sigma0_in, double Asm_in,
           double Gamma2,  double scale_in, int n,int L_Type,int U_Type, int n_alp){

    // A - average distance in correlation sphere
    // Rho is density, T_in - tmeperatures of system (LJ units)

    // sigma0_in is MSD, Asm_in - assymetri koefficient.
    // ========================
    // plus my parameters MSD, gamma1 gamma2 etc
    // ========================

    // scale_in is area of plotting of the pna(r).
    // pna(r) must be completely contained in scale_in
    // n is the number of node in the scale_in area


    // inicializaci9 znacheni9
    a = A;
    rho = Rho;
    T = T_in;			// inicializaci9 znacheni9
    sigma0 = sigma0_in;		// inicializaci9 znacheni9
    Asm = Asm_in;			// inicializaci9 znacheni9
    scale = scale_in;		// inicializaci9 znacheni9
    N = n;			// inicializaci9 chisla uzlov
    n_alpha = n_alp;


    // the inetgral thrue values
    avr  = A;
    sigma_na = sigma0_in;
    gamma_1 = Asm_in; //asymmetry;
    gamma_2 = Gamma2;


    l_type = L_Type;
    u_type = U_Type;


    tmxy2 = 0.0;
    n_2d = N * N;			// vuchislenie ramerov massivov

    domain = new double[2];

    if (n_alpha == 1)
    {
        domain[0] = 0;
        domain[1] = 4;
    }
    else if (n_alpha == 2)
    {
        // avr = 2; [0; 5]
        domain[0] = 0 ;
        domain[1] = 5;
    }
    else
    {
        domain[0] = n_alpha - 3;
        domain[1] = n_alpha + 4;
    }


    R = scale * pow(rho, -1./dim);
    // R = n_alpha - scale * pow(rho, -1./dim);
    std::cout<<"R="<<R<<"\n";
    h = 2.0 * R / (n - 1);	// shag postoreni9 funkcii Dat
    Dat = new double[N * N];	// vudelenie pam9ti
    //sigma0=0.0268774;
    //Asm=1.65874;
    for(int i = 0; i < n_2d; i++)
    {
        Dat[i] = 0;
    }
    //sr=sigma0_in/(1+2*Asm);
    //st=0.5*(sigma0_in-sr);

    // koefficients in pna exponent
    km1 = 0;
    km2 = 0;
    km3 = 0;
    km4 = 0;

    sr = sigma0_in / (1 + (dim - 1.0) * Asm);
    st = (sigma0_in - sr) / (dim - 1.0);
    tr = 0.0;
    std::cout<<"rho="<<rho<<"\n";
    //std::cout<<"pow(rho,-1.0/dim)="<<pow(rho,-1.0)<<"\n";
    //a=Lattice_a_koef(l_type)*pow(rho,-1.0/dim);
    std::cout<<"a="<<a<<"\n";

}
Scor::~Scor(){
    delete [] Dat;
}
void Scor::Compute_Dat(){
	int 			i=0;
	double 		x=-R;
	double 		y=0;
	double 		S=0;

	if(dim==2)
    {
        // in this block the inegral relations are calculated

		double 	xmin[2] = {-R,-R};    //set area
		double 	xmax[2] = {R,R};
		double 	val_dat;  // enter the values
		double 	val_e;    //
		double 	val_r;
		double 	val_sigm0;
		double 	err;  // error

        double 	norm_pna;  // enter the values
		double 	avr_pna;    //
		double 	msd_pna;
		double 	gamma1_pna;
        double 	gamma2_pna;

		int 		maxiter=10000;

        //double norm_pna, avr_pna, msd_pna, gamma1_pna, gamma2_pna;

        // f_dat - pna(r)
        // f_e inegral pna(r) * r
        // replace by Compute_Asm Compute_E etc. See bellow
        // f_dat(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval)

		// hcubature(1,f_dat, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_dat, &err);
		//hcubature(1,f_e, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_e, &err);
		//hcubature(1,f_r, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_r, &err);
		//hcubature(1,f_sigm0, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_sigm0, &err);


        //===================================================
        //===================================================
        hcubature(1,f_dat, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &norm_pna, &err);
		hcubature(1,f_avr, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &avr_pna, &err);
		hcubature(1,f_msd, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &msd_pna, &err);
		hcubature(1,f_gamma1, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &gamma1_pna, &err);
        hcubature(1,f_gamma2, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &gamma2_pna, &err);

		//tEx=val_e/val_dat;
		//tS0=val_sigm0/val_dat;
		//tAsm=(val_sigm0-val_r)/val_r;

        t_normirova_pna = norm_pna;
        t_avr = avr_pna / norm_pna;
        t_sigma_na = msd_pna / norm_pna;
        t_gamma_1 = gamma1_pna / norm_pna;
        t_gamma_2 = gamma2_pna / norm_pna;



		//if(tS0<10e-5) tS0=1000;
		if(isnan(t_normirova_pna)==1) 	tEx=1000;
		if(isnan(t_avr)==1) 	tS0=1000;
		if(isnan(t_sigma_na)==1) 	tAsm=1000;
		if(isinf(t_gamma_1)==1) 	tEx=1000;
		if(isinf(t_gamma_2)==1) 	tS0=1000;

	}
}
void Scor::Compute_Asm(){
    int 	i=0;
    double 	x=-R;
    double 	y=0;
    double 	S1=0;
    double 	S2=0;
    if(dim==3){
	for(int iy=0;iy<=N/2;iy++){
	    for(int ix=0;ix<N;ix++){
		S1+=(x*x)*2*M_PI*Dat[i]*y;
		S2+=(y*y)*2*M_PI*Dat[i]*y;
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tAsm=0.5*S2/S1;
    }
    else{
	for(int iy=0;iy<=N/2;iy++){
	    for(int ix=0;ix<N;ix++){
		S1+=(x*x)*Dat[i];
		S2+=(y*y)*Dat[i];
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tAsm=S2/S1;
    }
    //std::cout<<"tAsm="<<tAsm<<"\n";
}
void Scor::Compute_E(){
    int 	i = 0;
    double 	x = -R;
    double 	y = 0;
    double 	S = 0;
    if(dim == 3)
    {
        for(int iy = 0;iy <= N / 2; iy++)
        {
            for(int ix = 0; ix < N; ix++)
            {
                S += x * 2 * M_PI * Dat[i] * y;
                i++;
                x += h;
            }
            x = -R;
            y += h;
        }
        tEx =S * h * h;
    }
    else
    {
        for(int iy = 0; iy <= N / 2; iy++)
        {
            for(int ix = 0; ix < N; ix++)
            {
                S += x * Dat[i];
                i++;
                x +=h;
            }
            x = -R;
            y += h;
        }
        tEx = S * h * h;
    }
    //std::cout<<"Ex="<<tEx<<"\n";
}
void Scor::Compute_avr(){
    int 	i = 0;
    double 	x = domain[0];
    double 	y = 0;
    double 	integral = 0;
    for(int i = 0; i < N; i++)
    {
        integral += x*Dat[i];
        x += h;
    }
    t_avr = integral * h;

}

void Scor::Compute_msd(){
    int 	i = 0;
    double 	x = domain[0];
    double 	y = 0;
    double 	integral = 0;
    for(int i = 0; i < N; i++)
    {
        integral += (x - avr) * (x - avr) * Dat[i];
        x += h;
    }
    t_sigma_na = integral * h;

}

void Scor::Compute_gamma1(){
    int 	i = 0;
    double 	x = domain[0];
    double 	y = 0;
    double 	integral = 0;
    for(int i = 0; i < N; i++)
    {
        integral += (x - avr) * (x - avr) * (x - avr) * Dat[i] / pow(sigma_na, 3/2);
        x += h;
    }
    t_gamma_1 = integral * h;

}

void Scor::Compute_gamma2(){
    int 	i = 0;
    double 	x = domain[0];
    double 	y = 0;
    double 	integral = 0;
    for(int i = 0; i < N; i++)
    {
        integral += (x - avr) * (x - avr) * (x - avr) * (x - avr) * Dat[i] / pow(sigma_na, 2);
        x += h;
    }
    t_gamma_2 = integral * h;

}

void Scor::Compute_En(){
    int 	i=0;
    double 	x=-R;
    double 	y=0;
    double 	S=0;
    for(int iy=0;iy<=N/2;iy++){
	for(int ix=0;ix<N;ix++){
	    S+=pow(sqrt(pow(x+a,2)+pow(y,2)),-12)*2*M_PI*Dat[i]*y;
	    i++;
	    x+=h;
	}
	x=-R;
	y+=h;
    }
    En=S*h*h;
}


void Scor::Compute_M3(){
    int 	i=0;
    double 	x=-R;
    double 	y=0;
    double 	S=0;
    if(dim==3){
	for(int iy=0;iy<=N/2;iy++){
	    for(int ix=0;ix<N;ix++){
		S+=x*x*x*2*M_PI*Dat[i]*y;
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tM3=S*h*h;
	std::cout<<"M3="<<tM3<<"\n";
    }
    else{
	for(int iy=0;iy<=N/2;iy++){
	    for( int ix=0;ix<N;ix++){
		S+=x*x*x*Dat[i];
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tM3x=S*h*h;
    }
    //std::cout<<"Ex="<<tEx<<"\n";
}
void Scor::Compute_M4x(){
    int 	i=0;
    double 	x=-R;
    double 	y=0;
    double 	S=0;
    if(dim==3){
	for(int iy=0;iy<=N/2;iy++){
	    for(int ix=0;ix<N;ix++){
		S+=x*x*x*x*2*M_PI*Dat[i]*y;
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tM4x=S*h*h;
    }
    else{
	for(int iy=0;iy<=N/2;iy++){
	    for( int ix=0;ix<N;ix++){
		S+=x*x*x*x*Dat[i];
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tM4x=S*h*h;
    }
    //std::cout<<"Ex="<<tEx<<"\n";
}
void Scor::Compute_M4y(){
    int 	i=0;
    double 	x=-R;
    double 	y=0;
    double 	S=0;
    if(dim==3){
	for(int iy=0;iy<=N/2;iy++){
	    for(int ix=0;ix<N;ix++){
		S+=y*y*y*y*2*M_PI*Dat[i]*y;
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tM4y=S*h*h;
    }
    else{
	for(int iy=0;iy<=N/2;iy++){
	    for( int ix=0;ix<N;ix++){
		S+=y*y*y*y*Dat[i];
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tM4y=S*h*h;
    }
    //std::cout<<"Ex="<<tEx<<"\n";
}
void Scor::Compute_Tmxy2(){
    int 	i=0;
    double 	x=-R;
    double 	y=0;
    double 	S=0;
    if(dim==3){
	for(int iy=0;iy<=N/2;iy++){
	    for(int ix=0;ix<N;ix++){
		S+=x*y*y*2*M_PI*Dat[i]*y;
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	Tmxy2=S*h*h;
    }
    else{
	for(int iy=0;iy<=N/2;iy++){
	    for( int ix=0;ix<N;ix++){
		S+=x*y*y*Dat[i];
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	Tmxy2=S*h*h;
    }
    //std::cout<<"Ex="<<tEx<<"\n";
}
void Scor::Compute_sigma0(){
    int 	i=0;
    double 	x=-R;
    double 	y=0;
    double 	S=0;
    if(dim==3){
	for(int iy=0;iy<=N/2;iy++)
    {
	    for(int ix=0;ix<N;ix++)
        {
            S += (x*x+y*y) * 2 * M_PI * Dat[i] * y;
            i++;
            x += h;
	    }
	    x=-R;
	    y+=h;
	}
	tS0=S*h*h;
    }
    else{
	for(int iy=0;iy<=N/2;iy++){
	    for(int ix=0;ix<N;ix++){
		S+=(x*x+y*y)*Dat[i];
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tS0=S*h*h;
    }
    //std::cout<<"S0="<<tS0<<"\n";
}




void Scor::Dump_To_File(){
    std::ofstream f("Rez2.txt");
    Compute_Dat();
    for(int iy=0;iy<=N;iy++){
	for(int ix=0;ix<N;ix++){

            f<<ix*h-R<<" "<<(iy*h)-R<<" "<<F(ix*h-R,iy*h-R)<<"\n";
	}
    }
    f.close();
}






double Scor::F(double x, double y){



    double r = sqrt(pow(x+a,2)+pow(y,2));
    //return exp(-0.5*(x+tr)*(x+tr)/sr+0.5*tr*tr/sr-0.5*y*y/st-koefTT*((exp(-r)/r)-(exp(-a)/a))/T);
    return exp(-x*km1 - x*x*km2  -  x*x*x*km3 -  x*x*x*x*km4 - 4*(pow(x,-12.0)-pow(x,-6.0))/T );





}


double Scor::F_avr(double x, double y){
    return x*F(x,y);
}
int f_avr(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	Scor *SC=(Scor*)params;
	fval[0]=SC->F_avr(x[0],x[1]);
	return 0;
}


double Scor::F_msd(double x, double y){
    // t_avr - avr in iteration,
    // avr - the truth avr
    return (x - t_avr) * (x - t_avr) * F(x, y);
}
int f_msd(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	Scor *SC=(Scor*)params;
	fval[0]=SC->F_msd(x[0],x[1]);
	return 0;
}


double Scor::F_gamma1(double x, double y){
    return  (x - t_avr) * (x - t_avr) * (x - t_avr) * F(x, y) / pow(sigma_na, 3/2);
}
int f_gamma1(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	Scor *SC=(Scor*)params;
	fval[0]=SC->F_gamma1(x[0],x[1]);
	return 0;
}


double Scor::F_gamma2(double x, double y){
    return  (x - t_avr) * (x - t_avr) * (x - t_avr) * (x - t_avr) * F(x, y) / pow(sigma_na, 2);
}
int f_gamma2(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	Scor *SC=(Scor*)params;
	fval[0]=SC->F_gamma2(x[0],x[1]);
	return 0;
}





double Scor::F_e(double x, double y){
    return x*F(x,y);
}
int f_e(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	Scor *SC=(Scor*)params;
	fval[0]=SC->F_e(x[0],x[1]);
	return 0;
}


double Scor::F_r(double x, double y){
    return x*x*F(x,y);
}
int f_r(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	Scor *SC=(Scor*)params;
	fval[0]=SC->F_r(x[0],x[1]);
	return 0;
}

double Scor::F_sigm0(double x, double y){
    return (x*x+y*y)*F(x,y);
}




int f_dat(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	Scor *SC=(Scor*)params;
	fval[0]=SC->F(x[0],x[1]);
	return 0;
}



int f_sigm0(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	Scor *SC=(Scor*)params;
	fval[0]=SC->F_sigm0(x[0],x[1]);
	return 0;
}


//double Scor::Err(double Tr, double Tsr, double Tst,double Tm4x,double Tm4y,double Tm3x){
double Scor::Err(double KM_1, double KM_2, double KM_3,double KM_4){
    // this function calculet errors

    // me need to replace the values of tM4x et all to my koefficient
    //tr=Tr;
    //sr=Tsr;
    //st=Tst;
    //ttm4x=Tm4x;
    //ttm4y=Tm4y;
    //ttm3x=Tm3x;


    //===================================================
    //===================================================
    km1 = KM_1;
    km2 = KM_2;
    km3 = KM_3;
    km4 = KM_4;


    Compute_Dat();
    Compute_avr();
    Compute_msd();
    Compute_gamma1();
    Compute_gamma2();

    //Compute_M4x();
    //Compute_M4y();
    //Compute_M3();
    //std::cout<<tS0<<"\t"<<tEx<<"\t"<<tAsm<<"\n";
    std::cout<<"s="<<tS0<<" Asm="<<tAsm<<" tm4x="<<tM4x<<" tm4y="<<tM4y<<" tm3x="<<tM3x<<" tx="<<tEx<<"\n";
     if(fabs(st)<1e-4||fabs(sr)<1e-4) return 100;


    return pow(avr - t_avr,2)+pow(sigma_na-t_sigma_na, 2)+ pow(gamma_1-t_gamma_1,2)+ pow(gamma_2-t_gamma_2, 2);

    //return pow(sigma0-tS0,2)+pow(tEx,2)+pow(Asm-tAsm,2)+pow(tM3,2);
    //return fabs(sigma0-tS0)+fabs(tEx)+fabs(Asm-tAsm);
}


double Ferr(const gsl_vector* v, void* params){
    Scor *SC=(Scor*)params;

    return SC->Err(gsl_vector_get(v,0), gsl_vector_get(v,1), gsl_vector_get(v,2), gsl_vector_get(v,3));
    //return SC->Err(gsl_vector_get(v,0),gsl_vector_get(v,1),gsl_vector_get(v,2),gsl_vector_get(v,3));
    //return SC->Err(gsl_vector_get(v,0),gsl_vector_get(v,1),gsl_vector_get(v,2),gsl_vector_get(v,3));
}


//double Scor::Err_n(double Tr,double Tsr,double Tst,double &out1,double &out2,double &out3){
double Scor::Err_n(double sKM_1, double sKM_2, double sKM_3, double sKM_4, double &out1, double &out2, double &out3, double &out4){
    // plus one input in function


    //===================================================
    //===================================================
    km1 = sKM_1;
    km2 = sKM_2;
    km3 = sKM_3;
    km4 = sKM_4;
    /*
     * t_normirova_pna = norm_pna;
        t_avr = avr_pna / norm_pna;
        t_sigma_na = msd_pna / norm_pna;
        t_gamma_1 = gamma1_pna / norm_pna;
        t_gamma_2 = gamma2_pna / norm_pna;

     */
    //tr=Tr;
    //sr=Tsr;
    //st=Tst;
    // calc koef in pna -> tEx tS0 etc
    Compute_Dat();

    //out1=tEx;
    //out2=sigma0-tS0;
    //out3=Asm-tAsm;

    out1=t_avr;
    out2=t_sigma_na;
    out3=t_gamma_1;
    out4=t_gamma_2;

    if(fabs(st)<1e-4||fabs(sr)<1e-4){
	    out1=1000;
	    out2=1000;
	    out3=1000;
    }
    // return nevyzka
    return pow(sigma_na-t_sigma_na, 2) + pow(gamma_1 - t_gamma_1, 2) + pow(gamma_2 - t_gamma_2, 2) +  pow(avr-t_avr, 2);
    //return fabs(sigma0-tS0)+fabs(tEx)+fabs(Asm-tAsm);
}

int Ferr_n(const gsl_vector *x, void *param, gsl_vector *ret){
    // constructor of our solver: calc errors and get values
    // Providing the function to solve see site

	//double x1, x2, x3;
    double x1, x2, x3, x4;

	Scor *SC=(Scor*)param;

    // calc nevyzka
	// SC->Err_n(gsl_vector_get(x, 0),gsl_vector_get(x, 1),gsl_vector_get(x, 2),x1,x2,x3);
    //
    SC->Err_n(gsl_vector_get(x, 0),gsl_vector_get(x, 1),gsl_vector_get(x, 2), gsl_vector_get(x, 3), x1, x2, x3, x4);


    // This fucntion sets the value of the i-th element of a vector vec to x
    // gsl_vector_set(vec,i, x);
	gsl_vector_set(ret, 0, x1);
	gsl_vector_set(ret, 1, x2);
	gsl_vector_set(ret, 2, x3);
    gsl_vector_set(ret, 3, x4);
    //
    //gsl_vector_set(ret,3,x4);
	return GSL_SUCCESS;
}


void Scor::Find_Min(){
    // https://www.gnu.org/software/gsl/doc/html/multiroots.html

    /*
     * ==============================
     * guesstimation of parameters
     * ==============================
    */

    // initialization the solver function
	gsl_multiroot_function 		f;

    // Iteration and assemble all data
	gsl_multiroot_fsolver 		*workspace_f;

    // set vector from gsl lib
	gsl_vector 			*x;
    int n = 4;     //the number of parapeters of our function
	//int n = 3;     //the number of parapeters of our function
                // n = 3: f(x1, x2, x3) where x123 are variables
	int status;
	x = gsl_vector_alloc(n);

    // This function returns a pointer to a newly allocated instance of a solver of type "gsl_multiroot_fsolver_hybrid"
    // or a system of n=3 dimensions. (In my version must be 4)
    workspace_f = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrid, n);
	//workspace_f = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrid, 3);
	std::printf("F solver: %s\n", gsl_multiroot_fsolver_name(workspace_f));

    // set solver:
	f.f = &Ferr_n;
	f.n = n;

    // link to object
	f.params = this;


    // set the start koefficient values
    // This fucntion sets the value of the i-th element of a vector vec to x
    // gsl_vector_set(vec,i, x);
	//gsl_vector_set(x,0,tr);
	//gsl_vector_set(x,1,sr);
	//gsl_vector_set(x,2,st);


    //===================================================
    //===================================================
    gsl_vector_set(x, 0, km1);
	gsl_vector_set(x, 1, km2);
	gsl_vector_set(x, 2, km3);
    gsl_vector_set(x, 3, km4);


    //These functions set, or reset, an existing solver s to use the function f
    //or function and derivative fdf, and the initial guess
	gsl_multiroot_fsolver_set(workspace_f, &f, x);

    for(int times = 0; times < 2500; times++){
		status = gsl_multiroot_fsolver_iterate(workspace_f);
		std::printf("%d times: ", times);
		for(int i = 0; i < n; i++) std::printf("%10.3e ", gsl_vector_get(workspace_f->x, i));
		std::printf("\n");
		if((status == GSL_EBADFUNC) || (status == GSL_ENOPROG)){
			std::printf("Status: %s\n", gsl_strerror(status));
			break;
		}
	}
	for(int i = 0; i < n; i++) std::printf("%3d %25.17f\n", i, gsl_vector_get(workspace_f->x, i));

    // This function returns the i-th element of a vector vec
    // gsl_vector_get(vec,i);
	//tr = gsl_vector_get(workspace_f->x,0);
	//sr = gsl_vector_get(workspace_f->x,1);
	//st = gsl_vector_get(workspace_f->x,2);


    //===================================================
    //===================================================
    km1 = gsl_vector_get(workspace_f->x,0);
	km2 = gsl_vector_get(workspace_f->x,1);
	km3 = gsl_vector_get(workspace_f->x,2);
    km4 = gsl_vector_get(workspace_f->x,3);

	gsl_multiroot_fsolver_free(workspace_f);






    /*
     * ==============================
     * accurate estimate of parameters
     * the parameters value from previosly section is used as the starting points
     * ==============================
    */

	//koefTT=0;
	//int n=3;
	//gsl_vector 	*x=gsl_vector_alloc(n);

	gsl_vector 	*ss=gsl_vector_alloc(n);
    // method https://www.gnu.org/software/gsl/doc/html/multimin.html#c.gsl_multimin_fdfminimizer_set
	const 	gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2rand;


    // This is a workspace for minimizing functions without derivatives
	gsl_multimin_fminimizer 	*s = NULL;


    // This data type defines a general function of n variables with parameters
    // (set solver)
	gsl_multimin_function 	minex_func;

	size_t iter=0;
	//int status;
	double size;

    // set the start koefficient values
    // This fucntion sets the value of the i-th element of a vector vec to x
    // gsl_vector_set(vec,i, x);
	//gsl_vector_set(x, 0, tr);
	//gsl_vector_set(x, 1, sr);
	//gsl_vector_set(x, 2, st);


    //===================================================
    //===================================================
    gsl_vector_set(x, 0, km1);
	gsl_vector_set(x, 1, km2);
	gsl_vector_set(x, 2, km3);
    gsl_vector_set(x, 3, km4);

	// ??????? ?????????? ????
	gsl_vector_set_all(ss,0.1);
	// ????????????? ???????? ???????????
	minex_func.n = n;
	minex_func.f = Ferr;
	minex_func.params = this;
	s = gsl_multimin_fminimizer_alloc(T,n);

    // The function initializes the minimizer s to minimize
    // the function minex_func, starting from the initial point x.
	gsl_multimin_fminimizer_set(s, &minex_func, x, ss);
	//std::cout<<F(x,this)<<"\n";
	do{
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);
		if (status) break;
		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, 1e-15);
		if (status == GSL_SUCCESS)	std::cout<<"converged to minimum at\n";
		Compute_En();
		std::cout<<iter<<" tr="<<gsl_vector_get(s->x,0)<<" sr="<<gsl_vector_get(s->x,1)<<" st="<<gsl_vector_get(s->x,2)<<" f="<<s->fval<<" size="<<size<<"\tkoefTT="<<koefTT<<"\n";
		//std::cout<<iter<<" tr="<<gsl_vector_get(s->x,0)<<" sr="<<gsl_vector_get(s->x,1)<<" st="<<gsl_vector_get(s->x,2)<<" tm4x="<<gsl_vector_get(s->x,3)<<" tm4y="<<gsl_vector_get(s->x,4)<<" tm3x="<<gsl_vector_get(s->x,5)<<" f="<<s->fval<<" size="<<size<<"\tEn="<<En<<"\n";
	}while ((status == GSL_CONTINUE) && iter < 1000);

	//tr = gsl_vector_get(s->x, 0);
	//sr = gsl_vector_get(s->x, 1);
	//st = gsl_vector_get(s->x, 2);


    //===================================================
    //===================================================
    km1 = gsl_vector_get(workspace_f->x,0);
	km2 = gsl_vector_get(workspace_f->x,1);
	km3 = gsl_vector_get(workspace_f->x,2);
    km4 = gsl_vector_get(workspace_f->x,3);

	//tmxy2=gsl_vector_get(s->x,3);
	//ttm4x=gsl_vector_get(s->x,3);
	//ttm4y=gsl_vector_get(s->x,4);

	gsl_vector_free(x);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free(s);

}






int main()
{
    //Scor ProbaRDF(double A, double Rho, double T_in, double sigma0_in, double Asm_in, double scale_in,int n,int L_Type,int U_Type,int Dim);
    Scor ProbaRDF(1, 1, 4, 0.5, 0.2, 2, 20, 1, 1, 2, 1);



return 0;
}






















