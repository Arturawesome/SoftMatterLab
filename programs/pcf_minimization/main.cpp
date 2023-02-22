#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include "spectrum.h"
#include <cmath>
#include <lattice.h>
#include <rdf.h>
#include <iomanip>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multiroots.h>
#include "cubature.h"
/*  	????????? ???????????? ?????? ?????????????? ??????? ????????? ?? ?????? ?????? ?????????? ?????? 
 * 	?????? ??? ?? ???? ???????? ?????
 *????? ????? ?????? ?????: 
 * 	1)	ssf_fna -- ??? ??????????? ?????????? ?????? ?????????????? ??????? ????????? ?? ?????? ?????????? sigma_0 ? \phi(A) ? ????????????? ?????????? 
 * 	2)	Spectrum2 -- ??? ??????????? ???????????? ?????????? ??????????? ??? ?????????? ???
 *
 *	? ???? ?????????? ???????? ??????? ????????? ????????? ????? ??????????? ????????? ???? (2)  
 */
double koefTT=1.0;

void S_Line(){
    std::cout<<"----------------------------------------------------\n";
}
double Lattice_a_koef(int Type){  	// koefficient sv9zi meshdu rho^-1/d i xarakternum rassto9niem meshdu chasticami kristalla
    switch(Type){
	case 1: return 1.122462048309373;
	case 2: return 1.091123636;
	case 3: return 1.122462048309373;
	case 4: return 1.0;
	case 5: return 1.0;
	case 6: return 1.074569931823542;
    }
    return 1;
}

class Vect_Asm_Dat{
    int 	type;
    double 	rho;
    void 	Init_fcc();
    void 	Init_bcc();
public:
  Vect 	*r;
  int 	N;
  Vect_Asm_Dat(int Type,double Rho);
  ~Vect_Asm_Dat();
  //void Re_Inti(int Type,double Rho);
};
Vect_Asm_Dat::Vect_Asm_Dat(int Type, double Rho){
    rho=Rho;
    N=-1;
    switch(Type){
      case 1: Init_fcc(); break;
      case 2: Init_bcc(); break;
    }
}
Vect_Asm_Dat::~Vect_Asm_Dat(){
   if(N !=-1)  delete [] r;
}
void Vect_Asm_Dat::Init_fcc(){
    double a=1.122462048309373*pow(rho,-1./3.)/sqrt(2.0); 
    N=4;
    r=new Vect[N];
    r[0].Init(a,a,0);
    r[1].Init(2*a,0,2*a);
    r[2].Init(a,a,2*a);
    r[3].Init(0,0*a,2*a);
}
void Vect_Asm_Dat::Init_bcc(){
    N=4;
    double a=1.091123636*pow(rho,-1./3.)/sqrt(3.0);
    r=new Vect[N];
    r[0].Init(a,-a,a);
    r[1].Init(2.0*a,-2.0*a,2.0*a);
    r[2].Init(-2.0*a,-2.0*a,0.0*a);
    r[3].Init(0.0*a,0.0*a,-2.0*a);
    //r[3].Init(0.0*a,2.0*a,0.0*a);
}




class s_cor{
    double 	*Dat;		// masssfi so srezom funkcii 
    double 	rho;		// plotnost; sistemu
    double 	T;			// temperatura sistemu	
    double 	scale;		// parametr mashtabirovani9
    double 	h;			// shag setki postroeni9 massiva Dat 
    double 	Asm;		// pokazatel' assimetrii 
    double 	sigma0;		// teoreticheskii parametr dl9 rascheta stepeni delokalizacii blishaishix uzlov 
    int 	N;				// chislo uzlov 
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

    double 	a;
    double 	Sphere_SF_Integrate_el(double r,double Rk,double R_max,int N_Integrate);
    double 	Circle_SF_Integrate_el(double r,double Rk,double R_max,int N_Integrate);
public:
    double 	F(double x,double y);
    double 	F_e(double x,double y);
    double 	F_r(double x,double y);
    double 	F_sigm0(double x,double y);
    double 	F_3d(double x,double y);
    double 	F_e_3d(double x,double y);
    double 	F_r_3d(double x,double y);
    double 	F_sigm0_3d(double x,double y);
    double 	sr;
    double 	st;
    double 	tr;
    double      skoef;
    double 	En;
    s_cor(double A,double Rho,double T_in,double sigma0_in,double Asm_in,double scale_in,int n,int L_Type,int U_type,int Dim);
    ~s_cor();
    void 	Compute_Dat();		// vuchislenie znachenii massiva Dat
    void 	Compute_E();		// vuchislenie Ex
    void 	Compute_Asm();		// vuchislenie koefficienta assimetrii
    void 	Compute_sigma0();	// vuchislenie delokalizacii 
    void 	Dump_To_File();
    void 	Compute_g(double *A1,double *A2,int N_RDF, double mass,int N_Integrate);
    double 	Err_n(double Tr,double Tsr,double Tst,double &out1,double &out2,double &out3);
    double 	Err(double Tr,double Tsr,double Tst);
    double 	Err(double Tr,double Tsr,double Tst,double km3);
    double 	Err(double Tr, double Tsr, double Tst,double Tm4x,double Tm4y);
    double      Err(double Tr, double Tsr, double Tst,double Tm4x,double Tm4y,double Tm3x);
    double      Err_Var_Koef(double Tr, double Tsr, double Tst,double Tvar_koef);
    double 	Err_2D(double Tsr,double Tst);
    void 	Find_Min();
    void 	Find_Min_2d();
    void        Find_Min_Koef_Var();
    void 	Compute_En();
    void 	Compute_M3();
    void 	Compute_M4x();
    void 	Compute_M4y();
    void 	Compute_Tmxy2();
};
int f_dat(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	s_cor *SC=(s_cor*)params;
	fval[0]=SC->F(x[0],x[1]);
	return 0;
}
int f_e(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	s_cor *SC=(s_cor*)params;
	fval[0]=SC->F_e(x[0],x[1]);
	return 0;
}
int f_r(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	s_cor *SC=(s_cor*)params;
	fval[0]=SC->F_r(x[0],x[1]);
	return 0;
}
int f_sigm0(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	s_cor *SC=(s_cor*)params;
	fval[0]=SC->F_sigm0(x[0],x[1]);
	return 0;
}
int f_dat_3d(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	s_cor *SC=(s_cor*)params;
	fval[0]=SC->F_3d(x[0],x[1]);
	return 0;
}
int f_e_3d(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	s_cor *SC=(s_cor*)params;
	fval[0]=SC->F_e_3d(x[0],x[1]);
	return 0;
}
int f_r_3d(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	s_cor *SC=(s_cor*)params;
	fval[0]=SC->F_r_3d(x[0],x[1]);
	return 0;
}
int f_sigm0_3d(unsigned ndim, const double *x, void *params, unsigned fdim, double *fval) {
	s_cor *SC=(s_cor*)params;
	fval[0]=SC->F_sigm0_3d(x[0],x[1]);
	return 0;
}

s_cor::s_cor(double A,double Rho, double T_in, double sigma0_in, double Asm_in, double scale_in,int n,int L_Type,int U_Type,int Dim){
    //Fixed_tm4x=0.000379377;
    //Fixed_tm4y=0.000865806;
    //Fixed_tm3x=0.000773754;
    tmxy2=0.0;
    N=n;			// inicializaci9 chisla uzlov 
    l_type=L_Type;
    u_type=U_Type;
    n_2d=N*N;			// vuchislenie ramerov massivov
    scale=scale_in;		// inicializaci9 znacheni9 
    sigma0=sigma0_in;		// inicializaci9 znacheni9
    rho=Rho;			// inicializaci9 znacheni9
    T=T_in;			// inicializaci9 znacheni9
    Asm=Asm_in;			// inicializaci9 znacheni9
    dim=Dim;
    R=scale*pow(rho,-1./dim);
    std::cout<<"R="<<R<<"\n";
    h=2.0*R/(n-1);	// shag postoreni9 funkcii Dat
    Dat=new double[N*N];	// vudelenie pam9ti 
    //sigma0=0.0268774;
    //Asm=1.65874;
    for(int i=0;i<n_2d;i++){
	Dat[i]=0;
    }
    //sr=sigma0_in/(1+2*Asm);
    //st=0.5*(sigma0_in-sr);
    sr=sigma0_in/(1+(dim-1.0)*Asm);
    st=(sigma0_in-sr)/(dim-1.0);
    tr=0.0;
    std::cout<<"rho="<<rho<<"\n";
    //std::cout<<"pow(rho,-1.0/dim)="<<pow(rho,-1.0)<<"\n";
    //a=Lattice_a_koef(l_type)*pow(rho,-1.0/dim);
    a=A;
    std::cout<<"a="<<a<<"\n";

}
s_cor::~s_cor(){
    delete [] Dat;
}
void s_cor::Compute_Dat(){
	int 			i=0;
	double 		x=-R;
	double 		y=0;
	double 		S=0;
	if(dim==3){
		/*for(int iy=0;iy<=N/2;iy++){					// N/2 ??? ?????????? ??????? ?????????? 
			for(int ix=0;ix<N;ix++){
				Dat[i]=F(x,y);
				S+=2*M_PI*Dat[i]*y;
				i++;
				x+=h;
			}
			x=-R;
			y+=h;
		}
		S*=(h*h);
		i=0;
		for(int iy=0;iy<=N/2;iy++){
			for(int ix=0;ix<N;ix++){
				Dat[i]/=S;
				i++;
			}
		}*/
		double 	xmin[2] = {-R,-R};
		double 	xmax[2] = {R,R};
		double 	val_dat;
		double 	val_e;
		double 	val_r;
		double 	val_sigm0;
		double 	err;
		int 		maxiter=10000;
		hcubature(1,f_dat_3d, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_dat, &err);
		hcubature(1,f_e_3d, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_e, &err);
		hcubature(1,f_r_3d, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_r, &err);
		hcubature(1,f_sigm0_3d, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_sigm0, &err);
		tEx=val_e/val_dat;
		tS0=val_sigm0/val_dat;
		tAsm=0.5*(val_sigm0-val_r)/val_r;
		//if(tS0<10e-5) tS0=1000;
		if(isnan(tEx)==1) 	tEx=1000;
		if(isnan(tS0)==1) 	tS0=1000;
		if(isnan(tAsm)==1) 	tAsm=1000;
		if(isinf(tEx)==1) 	tEx=1000;
		if(isinf(tS0)==1) 	tS0=1000;
		if(isinf(tAsm)==1) 	tAsm=1000;
	}
	else{
		/*for(int iy=0;iy<=N/2;iy++){					// N/2 ??? ?????????? ??????? ?????????? 
			for(int ix=0;ix<N;ix++){
				Dat[i]=F(x,y);
				S+=Dat[i];
				i++;
				x+=h;
			}
			x=-R;
			y+=h;
		}
		S*=(h*h);*/
		//if(S<1e-20) S=1;
		//if(S>1e20) S=1;
		
		double 	xmin[2] = {-R,-R};
		double 	xmax[2] = {R,R};
		double 	val_dat;
		double 	val_e;
		double 	val_r;
		double 	val_sigm0;
		double 	err;
		int 		maxiter=10000;
		hcubature(1,f_dat, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_dat, &err);
		hcubature(1,f_e, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_e, &err);
		hcubature(1,f_r, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_r, &err);
		hcubature(1,f_sigm0, this, 2, xmin, xmax, maxiter, 0, 1e-6, ERROR_INDIVIDUAL, &val_sigm0, &err);
		//std::cout<<"hcubature S="<<val_dat<<"\ttEx="<<val_e/val_dat<<"\tsigm0="<<val_sigm0/val_dat<<"\tAsm="<<(val_sigm0-val_r)/val_r<<"\n";
		/*i=0;
		for(int iy=0;iy<=N/2;iy++){
			for(int ix=0;ix<N;ix++){
				Dat[i]/=S;
				i++;
			}
		}*/
		tEx=val_e/val_dat;
		tS0=val_sigm0/val_dat;
		tAsm=(val_sigm0-val_r)/val_r;
		//if(tS0<10e-5) tS0=1000;
		if(isnan(tEx)==1) 	tEx=1000;
		if(isnan(tS0)==1) 	tS0=1000;
		if(isnan(tAsm)==1) 	tAsm=1000;
		if(isinf(tEx)==1) 	tEx=1000;
		if(isinf(tS0)==1) 	tS0=1000;
		if(isinf(tAsm)==1) 	tAsm=1000;
	}
}
void s_cor::Compute_Asm(){
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
void s_cor::Compute_E(){
    int 	i=0;
    double 	x=-R;
    double 	y=0;
    double 	S=0;
    if(dim==3){
	for(int iy=0;iy<=N/2;iy++){
	    for(int ix=0;ix<N;ix++){
		S+=x*2*M_PI*Dat[i]*y;
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}
	tEx=S*h*h;
    }
    else{
	for(int iy=0;iy<=N/2;iy++){
	    for( int ix=0;ix<N;ix++){
		S+=x*Dat[i];
		i++;
		x+=h;
	    }
	    x=-R;
	    y+=h;
	}   
	tEx=S*h*h;
    }
    //std::cout<<"Ex="<<tEx<<"\n";
}
void s_cor::Compute_M3(){
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
void s_cor::Compute_M4x(){
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
void s_cor::Compute_M4y(){
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
void s_cor::Compute_Tmxy2(){
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
void s_cor::Compute_sigma0(){
    int 	i=0;
    double 	x=-R;
    double 	y=0;
    double 	S=0;
    if(dim==3){
	for(int iy=0;iy<=N/2;iy++){
	    for(int ix=0;ix<N;ix++){
		S+=(x*x+y*y)*2*M_PI*Dat[i]*y;
		i++;
		x+=h;
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
void s_cor::Dump_To_File(){
    std::ofstream f("Rez2.txt");
    Compute_Dat();
    for(int iy=0;iy<=N;iy++){
	for(int ix=0;ix<N;ix++){
	    //f<<ix*h-R<<" "<<(iy)*h<<" "<<Dat[ix+iy*N]<<"\n";
            f<<ix*h-R<<" "<<(iy*h)-R<<" "<<F(ix*h-R,iy*h-R)<<"\n";
	}
    }
    f.close();
}
double s_cor::Err(double Tr, double Tsr, double Tst){
    //SC->Err(5.0,2*2.*Logs[i].sigm2_0*T_k/(1.+Logs[i].Asm[ias]),Logs[i].Asm[ias]*2*2.*Logs[i].sigm2_0*T_k/(1.+Logs[i].Asm[ias]));
    tr=Tr;
    sr=Tsr;
    st=Tst;
    //tr=Tr;
    //sr=Tsr/(1+Tst);
    //st=Tst*Tsr/(1+Tst);
    //std::cout<<"tr="<<tr<<"\t"<<"sr="<<sr<<"\t"<<"st="<<st<<"\n";
    Compute_Dat();
    //Compute_Asm();
    //Compute_E();
    //Compute_sigma0();
    //Compute_Tmxy2();
    
    //Compute_M3();
    //std::cout<<Tmxy2<<"\n";
    //if(fabs(st/(a*a))<1e-4||fabs(sr/(a*a))<1e-4) return 100;
    //std::cout<<"E1="<<pow(sigma0-tS0,2)<<"\tE2="<<pow(tEx,2)<<"\tE3="<<pow(Asm-tAsm,2)<<"\n";
    std::cout<<"s0="<<tS0<<"\tAsm="<<tAsm<<"\ttEx="<<tEx<<"\n";
    double rezErr= pow(sigma0-tS0,2)+pow(tEx,2)+pow(Asm-tAsm,2);
    //if(isnan(rezErr)==1) rezErr=100;
    //std::cout<<"rezErr="<<rezErr<<"\t"<<isnan(rezErr)<<"\n";

    return rezErr;
    //return fabs(sigma0-tS0)+fabs(tEx)+fabs(Asm-tAsm);
}
double s_cor::Err_Var_Koef(double Tr, double Tsr, double Tst,double Tvar_koef){
    //SC->Err(5.0,2*2.*Logs[i].sigm2_0*T_k/(1.+Logs[i].Asm[ias]),Logs[i].Asm[ias]*2*2.*Logs[i].sigm2_0*T_k/(1.+Logs[i].Asm[ias]));
    tr=Tr;
    sr=Tsr;
    st=Tst;
    //tr=Tr;
    //sr=Tsr/(1+Tst);
    //st=Tst*Tsr/(1+Tst);
    //std::cout<<"tr="<<tr<<"\t"<<"sr="<<sr<<"\t"<<"st="<<st<<"\n";
    Compute_Dat();
    //Compute_Asm();
    //Compute_E();
    //Compute_sigma0();
    //Compute_Tmxy2();
    
    //Compute_M3();
    //std::cout<<Tmxy2<<"\n";
    //if(fabs(st/(a*a))<1e-4||fabs(sr/(a*a))<1e-4) return 100;
    //std::cout<<"E1="<<pow(sigma0-tS0,2)<<"\tE2="<<pow(tEx,2)<<"\tE3="<<pow(Asm-tAsm,2)<<"\n";
    std::cout<<"s0="<<tS0<<"\tAsm="<<tAsm<<"\ttEx="<<tEx<<"\n";
    double rezErr= pow(sigma0-tS0,2)+pow(tEx,2)+pow(Asm-tAsm,2);
    //if(isnan(rezErr)==1) rezErr=100;
    //std::cout<<"rezErr="<<rezErr<<"\t"<<isnan(rezErr)<<"\n";

    return rezErr;
    //return fabs(sigma0-tS0)+fabs(tEx)+fabs(Asm-tAsm);
}
double s_cor::Err_n(double Tr,double Tsr,double Tst,double &out1,double &out2,double &out3){
    tr=Tr;
    sr=Tsr;
    st=Tst;
    Compute_Dat();
    //Compute_Asm();
    //Compute_E();
    //Compute_sigma0();
    //Compute_Tmxy2();
    //Compute_M3();
    //std::cout<<Tmxy2<<"\n";
    //if(fabs(st)<1e-4||fabs(sr)<1e-4) //return 100;
    out1=tEx;
    out2=sigma0-tS0;
    out3=Asm-tAsm;
    if(fabs(st)<1e-4||fabs(sr)<1e-4){
	    out1=1000;
	    out2=1000;
	    out3=1000;
    }
    //std::cout<<"E1="<<pow(sigma0-tS0,2)<<"\tE2="<<pow(tEx,2)<<"\tE3="<<pow(Asm-tAsm,2)<<"\n";
    //std::cout<<"s0="<<tS0<<"\tAsm="<<tAsm<<"\ttEx="<<tEx<<"\n";
    return pow(sigma0-tS0,2)+pow(tEx,2)+pow(Asm-tAsm,2);
    //return fabs(sigma0-tS0)+fabs(tEx)+fabs(Asm-tAsm);
}
double s_cor::Err(double Tr, double Tsr, double Tst,double _tmxy2){
    tmxy2=_tmxy2;
    tr=Tr;
    sr=Tsr;
    st=Tst;
    Compute_Dat();
    Compute_Asm();
    Compute_E();
    Compute_sigma0();
    Compute_Tmxy2();
    //Compute_M3();
    //std::cout<<"s="<<tS0<<" Asm="<<tAsm<<" tm4x="<<tM4x<<" tm4y="<<tM4y<<" tmxy2="<<tmxy2<<" tx="<<tEx<<"\n";
    //std::cout<<"Tmxy2="<<Tmxy2<<"\n";
     if(fabs(st)<1e-4||fabs(sr)<1e-4) return 100;
    return pow(sigma0-tS0,2)+pow(tEx,2)+pow(Asm-tAsm,2)+pow(Tmxy2,2);
    //return pow(sigma0-tS0,2)+pow(tEx,2)+pow(Asm-tAsm,2)+pow(tM3,2);
    //return fabs(sigma0-tS0)+fabs(tEx)+fabs(Asm-tAsm);
}
double s_cor::Err(double Tr, double Tsr, double Tst,double Tm4x,double Tm4y,double Tm3x){
    tr=Tr;
    sr=Tsr;
    st=Tst;
    ttm4x=Tm4x;
    ttm4y=Tm4y;
    ttm3x=Tm3x;
    Compute_Dat();
    Compute_Asm();
    Compute_E();
    Compute_sigma0();
    Compute_M4x();
    Compute_M4y();
    Compute_M3();
    //std::cout<<tS0<<"\t"<<tEx<<"\t"<<tAsm<<"\n";
    std::cout<<"s="<<tS0<<" Asm="<<tAsm<<" tm4x="<<tM4x<<" tm4y="<<tM4y<<" tm3x="<<tM3x<<" tx="<<tEx<<"\n";
     if(fabs(st)<1e-4||fabs(sr)<1e-4) return 100;
    return pow(sigma0-tS0,2)+pow(tEx,2)+pow(Asm-tAsm,2)+1000000.*pow(tM4x-Fixed_tm4x,2)+1000000.*pow(tM4y-Fixed_tm4y,2)+1000.*pow(tM3x-Fixed_tm3x,2);
    
    //return pow(sigma0-tS0,2)+pow(tEx,2)+pow(Asm-tAsm,2)+pow(tM3,2);
    //return fabs(sigma0-tS0)+fabs(tEx)+fabs(Asm-tAsm);
}
double s_cor::Err_2D( double Tsr, double Tst){
    sr=Tsr;
    st=Tst;
    Compute_Dat();
    Compute_Asm();
    Compute_E();
    Compute_sigma0();
    //std::cout<<tS0<<"\t"<<tEx<<"\t"<<tAsm<<"\n";
     if(fabs(st)<1e-4||fabs(sr)<1e-4) return 100;
    return pow(sigma0-tS0,4)+pow(tEx,4)+pow(Asm-tAsm,4);
}
double s_cor::F(double x, double y){
    //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-1.0*pow(sqrt(pow(x+a,2)+pow(y,2)),-4)/T);
    double r=sqrt(pow(x+a,2)+pow(y,2));
    //double koefTeilor=-exp(-a)/(a*a)-exp(-a)/a;
    //double f=1.0*(134.553*(pow(r,-50)-pow(r,-49))+1);
    //if(r>1.0204081632653061) f=0.0;
    //double alpha=10*(r-1.35);
    //double f=pow(r,-14)+0.5*(1-(exp(2*alpha)-1)/(exp(2*alpha)+1));
    //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-koefTT*(pow(r,-3))/T);
    //if(r>1.122462048309373) return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st);
    //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st);
   // return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-koefTT*(1+4*(pow(r,-12.0)-pow(r,-6.0)))/T);
    //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-f/T);
     //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st);
    //if (r>50./49) return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st);
    //else return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-koefTT*(134.553*(pow(r,-50)-pow(r,-49))+1.0)/T);
    //return exp(-0.5*(x+tr)*(x+tr)/sr+0.5*tr*tr/sr-0.5*y*y/st-koefTT*(pow(r,-24.0)-pow(a,-24.0))/T);
    //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-0.0*pow(r,-12.0)/T-0*(x+tr)*y*y/tmxy2);
    //return exp(-0.5*(x)*(x)/sr-tr*x-0.5*y*y/st-koefTT*(exp(-r)/r)/T);
    //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-koefTT*(koefTeilor*r));
    return exp(-0.5*(x+tr)*(x+tr)/sr+0.5*tr*tr/sr-0.5*y*y/st-koefTT*((exp(-r)/r)-(exp(-a)/a))/T);
    //return exp(-0.5*(x)*(x)/sr-tr*x-0.5*y*y/st-koefTT*100*(exp(-4.*r)/r)/T);
    
    //if(r>5./4) return exp(-0.5*(x+fabs(tr)/sqrt(T))*(x+fabs(tr)/sqrt(T))/sr-0.5*y*y/st);
    //return exp(-0.5*(x+fabs(tr)/sqrt(T))*(x+fabs(tr)/sqrt(T))/sr-0.5*y*y/st-koefTT*(12.20703125*(pow(r,-5)-pow(r,-4))+1.0)/T);
    
    //if(r>5./4) return exp(-0.5*x*x/sr-0.5*y*y/st+tr*x);
    //return exp(-0.5*x*x/sr-0.5*y*y/st+tr*x-koefTT*(12.20703125*(pow(r,-5)-pow(r,-4))+1.0)/T);
    
    //return exp(-0.5*(x+fabs(tr)/sqrt(T))*(x+fabs(tr)/sqrt(T))/sr-0.5*y*y/st-koefTT*pow(r,-12.0)/T);
    //tr=0;
    //return 10000.*exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-koefTT*0.95*(exp(-15.*(r-1))/r)/T+1);
    //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-koefTT*pow(r,-15.0)/T);
    //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-1.0*(exp(-r)/r)/T+0*tmxy2*(x)*y*y);
    //return exp(-0.5*(x+tr)*(x+tr)/sr-0.5*y*y/st-1.0*(exp(-r)/r)/T-1.0*kM3*x*x*x+0.018*x*x*x*x);
}
double s_cor::F_e(double x, double y){
    return x*F(x,y);
}
double s_cor::F_r(double x, double y){
    return x*x*F(x,y);
}
double s_cor::F_sigm0(double x, double y){
    return (x*x+y*y)*F(x,y);
}
double s_cor::F_3d(double x, double y){
	return M_PI*F(x,y)*fabs(y);
}
double s_cor::F_r_3d(double x, double y){
	return M_PI*x*x*F(x,y)*fabs(y);
}
double s_cor::F_e_3d(double x, double y){
    return M_PI*x*F(x,y)*fabs(y);
}
double s_cor::F_sigm0_3d(double x, double y){
    return M_PI*(x*x+y*y)*F(x,y)*fabs(y);
}

double s_cor::Sphere_SF_Integrate_el(double r,double Rk,double R_max,int N_Integrate){
    double S=0;
    double r2=r*r;
    double k1=Rk*Rk+r2;
    double k2=2.*Rk*r;
    //std::cout<<"r="<<r<<" Rk="<<Rk<<"\n";
    double h_theta=asin(R_max/Rk)/(N_Integrate-1);
    double theta=0;
    double arg=0;
    for(int i=0;i<N_Integrate-1;i+=2){
	//f(x_{k-1}) 
	arg=sqrt(k1-k2*cos(theta));
	if(arg<R_max) 	S+=F(-Rk+r*cos(theta),r*sin(theta))*sin(theta);
	theta+=h_theta;
	//f(x_{k})
	arg=sqrt(k1-k2*cos(theta));
	if(arg<R_max) 	S+=4*F(-Rk+r*cos(theta),r*sin(theta))*sin(theta);
	theta+=h_theta;
	//f(x_{k+1})
	arg=sqrt(k1-k2*cos(theta));
	if(arg<R_max) 	S+=F(-Rk+r*cos(theta),r*sin(theta))*sin(theta);
    }
    return 2*M_PI*r2*S*h_theta/3.;
}
double s_cor::Circle_SF_Integrate_el(double r,double Rk,double R_max,int N_Integrate){
    double S=0;
    double r2=r*r;
    double k1=Rk*Rk+r2;
    double k2=2.*Rk*r;
    double h_theta=asin(R_max/Rk)/(N_Integrate-1);
    double theta=0;
    double arg=0;
    for(int i=0;i<N_Integrate-1;i+=2){
	//f(x_{k-1}) 
	arg=sqrt(k1-k2*cos(theta));
	if(arg<R_max) 	S+=F(-Rk+r*cos(theta),r*sin(theta));
	theta+=h_theta;
	//f(x_{k})
	arg=sqrt(k1-k2*cos(theta));
	if(arg<R_max) 	S+=4*F(-Rk+r*cos(theta),r*sin(theta));
	theta+=h_theta;
	//f(x_{k+1})
	arg=sqrt(k1-k2*cos(theta));
	if(arg<R_max) 	S+=F(-Rk+r*cos(theta),r*sin(theta));
    }
    return PI*sqrt(r2)*S*h_theta/3.;
}
void s_cor::Compute_g(double *Steca,double *Dat_RDF,int N_RDF, double mass,int N_Integrate){
    //double *Dat_RDF=new double[N_RDF];
    double *Dat_RDF_Temp=new double[N_RDF];
    for(int i=0;i<N_RDF;i++){
	//Dat_RDF[i]=0;
	Dat_RDF_Temp[i]=0;
    }
    double h_rdf=pow(rho,-1.0/dim)*Steca[0];
    //double h_rdf=pow(1.0,-1.0/3.0)*Steca[0];
    // ???????????? ????;
    double 	r_uzl=R; 
    std::cout<<"r_uzl="<<r_uzl<<"\n";
    int 	k_min=static_cast<int>((a-r_uzl)/h_rdf)-1;
    int 	k_max=static_cast<int>((a+r_uzl)/h_rdf)+1;
    int 	lk=static_cast<int>(2.*r_uzl/h_rdf)+2;
    double S=0;
    if(k_min<0) k_min=0;
    if(k_max>N_RDF) k_max=N_RDF;
    if(lk>=N_RDF) lk=N_RDF;
    //std::cout<<"k_min="<<k_min<<"\n";
    //std::cout<<"k_max="<<k_max<<"\n";
    //std::cout<<"lk="<<lk<<"\n";
    if(dim==3){
	for(int k=0;k<N_RDF;k++){
	    Dat_RDF_Temp[k]=Sphere_SF_Integrate_el(h_rdf*(k+k_min+1),a,r_uzl,N_Integrate);
	    S+=Dat_RDF_Temp[k];
	}
    }
    if(dim==2){
      	for(int k=0;k<lk;k++){
	    Dat_RDF_Temp[k]=Circle_SF_Integrate_el(h_rdf*(k+k_min+1),a,r_uzl,N_Integrate);
	    S+=Dat_RDF_Temp[k];
	}
    }

    S*=h_rdf;
    std::cout<<"S="<<S<<"\n";
    for(int k=k_min;k<k_max;k++){
	//std::cout<<"Dat_0="<<Dat_RDF[k]<<"\t";
	Dat_RDF[k]+=pow(rho,-1.0/dim)*Dat_RDF_Temp[k-k_min]*mass/(S);
	//std::cout<<"Dat_k="<<Dat_RDF[k]<<"\n";
	Dat_RDF_Temp[k-k_min]=0;
    } 
    std::cout<<"mass="<<mass<<"\n";
    std::cout<<"R="<<a<<"\n";
    for(int i=0;i<N_RDF;i++){
	Dat_RDF_Temp[i]=0;
    }
    std::ofstream frdf("RDF.txt");
    double r;
    if(dim==3){
	for(int i=0;i<N_RDF;i++){
	    r=(i+1)*h_rdf;
	    frdf<<r<<"\t"<<pow(rho,-1.0/dim)*Dat_RDF[i]/(4.0*M_PI*pow(r,2))<<"\n";
	}
    }
    if(dim==2){
      	for(int i=0;i<N_RDF;i++){
	    r=(i+1)*h_rdf;
	    frdf<<r<<"\t"<<pow(rho,-1.0/dim)*Dat_RDF[i]/(2.0*M_PI*r)<<"\n";
	}
    }
    frdf.close();
    // ???????????? 
    //delete [] Dat_RDF;
    delete [] Dat_RDF_Temp;
}
double Ferr(const gsl_vector* v, void* params){
    s_cor *SC=(s_cor*)params;
    return SC->Err(gsl_vector_get(v,0),gsl_vector_get(v,1),gsl_vector_get(v,2));
    //return SC->Err(gsl_vector_get(v,0),gsl_vector_get(v,1),gsl_vector_get(v,2),gsl_vector_get(v,3));
    //return SC->Err(gsl_vector_get(v,0),gsl_vector_get(v,1),gsl_vector_get(v,2),gsl_vector_get(v,3));
}
double Ferr_Var_Koef(const gsl_vector* v, void* params){
    s_cor *SC=(s_cor*)params;
    return SC->Err(gsl_vector_get(v,0),gsl_vector_get(v,1),gsl_vector_get(v,2));
    //return SC->Err(gsl_vector_get(v,0),gsl_vector_get(v,1),gsl_vector_get(v,2),gsl_vector_get(v,3));
    //return SC->Err(gsl_vector_get(v,0),gsl_vector_get(v,1),gsl_vector_get(v,2),gsl_vector_get(v,3));
}
double Ferr_2D(const gsl_vector* v, void* params){
    s_cor *SC=(s_cor*)params;
    return SC->Err_2D(gsl_vector_get(v,0),gsl_vector_get(v,1));
}
int Ferr_n(const gsl_vector *x, void *param, gsl_vector *ret){
	double x1, x2, x3;
	s_cor *SC=(s_cor*)param;
	SC->Err_n(gsl_vector_get(x, 0),gsl_vector_get(x, 1),gsl_vector_get(x, 2),x1,x2,x3);
	gsl_vector_set(ret,0,x1);
	gsl_vector_set(ret,1,x2);
	gsl_vector_set(ret,2,x3);
	return GSL_SUCCESS;
}


void s_cor::Find_Min(){
	gsl_multiroot_function 		f;
	gsl_multiroot_fsolver 		*workspace_f;
	//gsl_multiroot_function_fdf 	fdf;
	//gsl_multiroot_fdfsolver 	*workspace;
	gsl_vector 			*x;
	int n=3;
	int status;
	x = gsl_vector_alloc(n);
	workspace_f = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrid, 3);
	std::printf("F solver: %s\n", gsl_multiroot_fsolver_name(workspace_f));
	f.f = &Ferr_n;
	f.n = n;
	f.params = this;
	gsl_vector_set(x,0,tr);
	gsl_vector_set(x,1,sr);
	gsl_vector_set(x,2,st);	
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
	tr=gsl_vector_get(workspace_f->x,0);
	sr=gsl_vector_get(workspace_f->x,1);
	st=gsl_vector_get(workspace_f->x,2);
	gsl_multiroot_fsolver_free(workspace_f);
	
	
	
	//koefTT=0;
	//int n=3;
	//gsl_vector 	*x=gsl_vector_alloc(n);
	gsl_vector 	*ss=gsl_vector_alloc(n);
	const 	gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2rand;
	gsl_multimin_fminimizer 	*s = NULL;
	gsl_multimin_function 	minex_func;
	size_t iter=0;
	//int status;
	double size;
	// ??????? ?????????? ??????????? 
	gsl_vector_set(x,0,tr);
	gsl_vector_set(x,1,sr);
	gsl_vector_set(x,2,st);
	//gsl_vector_set(x,3,tmxy2);
	//gsl_vector_set(x,3,ttm4x);
	//gsl_vector_set(x,4,ttm4y);
	//gsl_vector_set(x,5,ttm3x);
	//gsl_vector_set(x,3,0);
	// ??????? ?????????? ???? 
	gsl_vector_set_all(ss,0.1);
	// ????????????? ???????? ???????????
	minex_func.n=n;
	minex_func.f=Ferr;
	minex_func.params=this;
	s = gsl_multimin_fminimizer_alloc (T,n);
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
	tr=gsl_vector_get(s->x,0);
	sr=gsl_vector_get(s->x,1);
	st=gsl_vector_get(s->x,2);
	//tmxy2=gsl_vector_get(s->x,3);
	//ttm4x=gsl_vector_get(s->x,3);
	//ttm4y=gsl_vector_get(s->x,4);
	//ttm3x=gsl_vector_get(s->x,5);
	//kM3=st=gsl_vector_get(s->x,3);
	gsl_vector_free(x);	
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free(s);  
	/*int n=3;
	gsl_vector 	*x=gsl_vector_alloc(n);
	gsl_vector 	*ss=gsl_vector_alloc(n);
	const 	gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2rand;
	gsl_multimin_fminimizer 	*s = NULL;
	gsl_multimin_function 	minex_func;
	size_t iter=0;
	int status;
	double size;
	// ??????? ?????????? ??????????? 
	gsl_vector_set(x,0,tr);
	gsl_vector_set(x,1,sr);
	gsl_vector_set(x,2,st);
	//gsl_vector_set(x,3,0);
	// ??????? ?????????? ???? 
	gsl_vector_set_all(ss,0.0001);
	// ????????????? ???????? ???????????
	minex_func.n=n;
	minex_func.f=Ferr;
	minex_func.params=this;
	s = gsl_multimin_fminimizer_alloc (T,n);
	gsl_multimin_fminimizer_set(s, &minex_func, x, ss);
	//std::cout<<F(x,this)<<"\n";
	do{   
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);
		if (status) break;
		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, 1e-12);
		if (status == GSL_SUCCESS)	std::cout<<"converged to minimum at\n";
		Compute_En();
		std::cout<<iter<<" tr="<<gsl_vector_get(s->x,0)<<" sr="<<gsl_vector_get(s->x,1)<<" st="<<gsl_vector_get(s->x,2)<<" f="<<s->fval<<" size="<<size<<"\tEn="<<En<<"\n";        
	}while (status == GSL_CONTINUE && iter < 2500);
	tr=gsl_vector_get(s->x,0);
	sr=gsl_vector_get(s->x,1);
	st=gsl_vector_get(s->x,2);
	//kM3=st=gsl_vector_get(s->x,3);
	gsl_vector_free(x);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free(s);*/
} 
void s_cor::Find_Min_Koef_Var(){
	gsl_multiroot_function 		f;
	gsl_multiroot_fsolver 		*workspace_f;
	//gsl_multiroot_function_fdf 	fdf;
	//gsl_multiroot_fdfsolver 	*workspace;
	gsl_vector 			*x;
	int n=4;
	int status;
	x = gsl_vector_alloc(n);
	workspace_f = gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_hybrid, 3);
	std::printf("F solver: %s\n", gsl_multiroot_fsolver_name(workspace_f));
	f.f = &Ferr_n;
	f.n = n;
	f.params = this;
	gsl_vector_set(x,0,tr);
	gsl_vector_set(x,1,sr);
	gsl_vector_set(x,2,st);	
        gsl_vector_set(x,3,skoef);	
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
	tr=gsl_vector_get(workspace_f->x,0);
	sr=gsl_vector_get(workspace_f->x,1);
	st=gsl_vector_get(workspace_f->x,2);
        skoef=gsl_vector_get(workspace_f->x,3);
	gsl_multiroot_fsolver_free(workspace_f);
	
	
	
	//koefTT=0;
	//int n=3;
	//gsl_vector 	*x=gsl_vector_alloc(n);
	gsl_vector 	*ss=gsl_vector_alloc(n);
	const 	gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2rand;
	gsl_multimin_fminimizer 	*s = NULL;
	gsl_multimin_function 	minex_func;
	size_t iter=0;
	//int status;
	double size;
	// ??????? ?????????? ??????????? 
	gsl_vector_set(x,0,tr);
	gsl_vector_set(x,1,sr);
	gsl_vector_set(x,2,st);
        gsl_vector_set(x,3,skoef);
	//gsl_vector_set(x,3,tmxy2);
	//gsl_vector_set(x,3,ttm4x);
	//gsl_vector_set(x,4,ttm4y);
	//gsl_vector_set(x,5,ttm3x);
	//gsl_vector_set(x,3,0);
	// ??????? ?????????? ???? 
	gsl_vector_set_all(ss,0.1);
	// ????????????? ???????? ???????????
	minex_func.n=n;
	minex_func.f=Ferr_Var_Koef;
	minex_func.params=this;
	s = gsl_multimin_fminimizer_alloc (T,n);
	gsl_multimin_fminimizer_set(s, &minex_func, x, ss);
	//std::cout<<F(x,this)<<"\n";
	do{   
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);
		if (status) break;
		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, 1e-12);
		if (status == GSL_SUCCESS)	std::cout<<"converged to minimum at\n";
		Compute_En();
		std::cout<<iter<<" tr="<<gsl_vector_get(s->x,0)<<" sr="<<gsl_vector_get(s->x,1)<<" st="<<gsl_vector_get(s->x,2)<<" f="<<s->fval<<" size="<<size<<"\tkoefTT-1="<<koefTT-1.0<<"\n";     
		//std::cout<<iter<<" tr="<<gsl_vector_get(s->x,0)<<" sr="<<gsl_vector_get(s->x,1)<<" st="<<gsl_vector_get(s->x,2)<<" tm4x="<<gsl_vector_get(s->x,3)<<" tm4y="<<gsl_vector_get(s->x,4)<<" tm3x="<<gsl_vector_get(s->x,5)<<" f="<<s->fval<<" size="<<size<<"\tEn="<<En<<"\n";        
	}while ((status == GSL_CONTINUE) && iter < 2500);
	tr=gsl_vector_get(s->x,0);
	sr=gsl_vector_get(s->x,1);
	st=gsl_vector_get(s->x,2);
        skoef=gsl_vector_get(s->x,3);
	//tmxy2=gsl_vector_get(s->x,3);
	//ttm4x=gsl_vector_get(s->x,3);
	//ttm4y=gsl_vector_get(s->x,4);
	//ttm3x=gsl_vector_get(s->x,5);
	//kM3=st=gsl_vector_get(s->x,3);
	gsl_vector_free(x);	
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free(s);  
	/*int n=3;
	gsl_vector 	*x=gsl_vector_alloc(n);
	gsl_vector 	*ss=gsl_vector_alloc(n);
	const 	gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2rand;
	gsl_multimin_fminimizer 	*s = NULL;
	gsl_multimin_function 	minex_func;
	size_t iter=0;
	int status;
	double size;
	// ??????? ?????????? ??????????? 
	gsl_vector_set(x,0,tr);
	gsl_vector_set(x,1,sr);
	gsl_vector_set(x,2,st);
	//gsl_vector_set(x,3,0);
	// ??????? ?????????? ???? 
	gsl_vector_set_all(ss,0.0001);
	// ????????????? ???????? ???????????
	minex_func.n=n;
	minex_func.f=Ferr;
	minex_func.params=this;
	s = gsl_multimin_fminimizer_alloc (T,n);
	gsl_multimin_fminimizer_set(s, &minex_func, x, ss);
	//std::cout<<F(x,this)<<"\n";
	do{   
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);
		if (status) break;
		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, 1e-12);
		if (status == GSL_SUCCESS)	std::cout<<"converged to minimum at\n";
		Compute_En();
		std::cout<<iter<<" tr="<<gsl_vector_get(s->x,0)<<" sr="<<gsl_vector_get(s->x,1)<<" st="<<gsl_vector_get(s->x,2)<<" f="<<s->fval<<" size="<<size<<"\tEn="<<En<<"\n";        
	}while (status == GSL_CONTINUE && iter < 2500);
	tr=gsl_vector_get(s->x,0);
	sr=gsl_vector_get(s->x,1);
	st=gsl_vector_get(s->x,2);
	//kM3=st=gsl_vector_get(s->x,3);
	gsl_vector_free(x);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free(s);*/
} 
void s_cor::Find_Min_2d(){
    gsl_vector 	*x=gsl_vector_alloc(2);
    gsl_vector 	*ss=gsl_vector_alloc(2);
    const 	gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2rand;
    gsl_multimin_fminimizer 	*s = NULL;
    gsl_multimin_function 	minex_func;
    size_t iter=0;
    int status;
    double size;
    gsl_vector_set(x,0,sr);
    gsl_vector_set(x,1,st);
    // ??????? ?????????? ???? 
    gsl_vector_set_all(ss,0.2);
    // ????????????? ???????? ???????????
    minex_func.n=2;
    minex_func.f=Ferr_2D;
    minex_func.params=this;
    s = gsl_multimin_fminimizer_alloc (T,2);
    gsl_multimin_fminimizer_set (s, &minex_func, x, ss);
    //std::cout<<F(x,this)<<"\n";
    do{   
	iter++;
	status = gsl_multimin_fminimizer_iterate(s);
	if (status) break;
	size = gsl_multimin_fminimizer_size (s);
	status = gsl_multimin_test_size (size, 1e-12);
	if (status == GSL_SUCCESS)	std::cout<<"converged to minimum at\n";
	Compute_En();
	std::cout<<iter<<" sr="<<gsl_vector_get(s->x,0)<<" st="<<gsl_vector_get(s->x,1)<<" f="<<s->fval<<" size="<<size<<"\tEn="<<En<<"\n";        
    }while (status == GSL_CONTINUE && iter < 2500);
    sr=gsl_vector_get(s->x,0);
    st=gsl_vector_get(s->x,1);
    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);
}
void s_cor::Compute_En(){
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


class Log{					// class dl9 edinichnoi zapisi v istorii
public:	
    // vxodnue parametru 
    int 	dim;				// razmernost' reshetki
    int 	l_type;				// tip reshetki
    int 	u_type;				// tip potenciala
    double 	u_param;			// parametr potenciala
    int 	n_rdf;				// chislo uzlov RDF
    int 	n_asm;				// chislo assimetrichnux uzlov 
    double 	rc;				// radius otsechki 
    int 	n_MC;				// parametr integrirovani9 metodo MC
    double 	rho;				// plotnost' sistemu 
    double 	T;				// temperatura sistemu 
    double 	Tk;				// koncehna9 temperatura
    int 	nT;				// chislo shagov po temperaturam 
    int 	n_rdf_mesh;			// chislo uzlov setki rdf 
    int 	n_uzl_mesh;			// chislo uzlov setki odnogo uzla kristallicheskoi reshetki
    int 	n_sph_integr;			// chislo iteracii dl9 postroeni9 setki na sphere 
    // dopolnitel'nue parametru 
    int 	dop_param;			// esli dopolnitel'nue parametru proinicializirovanu to 1, inache 0 
    double 	sigm2_0;			// parametr sv9zannue s blishaishum uzlom 
    double 	A;				// parametr A dl9 dvuxmernux reshetok
    double 	phi;				// parametr phi dl9 trexmernux reshetok 
    double 	*Asm;				// massiv koefficientov assimetrii 
    double 	*sigma_r;
    double 	*sigma_t;
    double 	entropy_const;			// entropiyna9 konstatnta na osnove garmonicheskogo priblisheni9
    double 	e_mod;				// energi9 modelunga 
    // methods
    Log();
    ~Log();
    void 	Read(std::fstream &f);		// formirovanie loga na osnove chteni9 strochki iz faila f 
    void 	Read_Old(std::fstream &f);	// funkci9 dl9 chteni9 starux logov iz istorii (pomimo osnovnux chitauts9 takshe dopolnitel'nue parametru)
    int 	Comp(Log &log);			// esli logi sovpadaut po bazovum znacheni9m (n_asm-mogut nesovpadat') vozvrashaets9 1, v protivnom sluchae 0
    void 	Copy_Dop_Param(Log &log);	// kopiruet v dannui log dopolnitel'nue parametru iz &log
    void 	Copy_Bas_Param(Log &log);	// kopirovanie v dannui vxodnux parametrov iz loga &log
    void 	Dump_To_File(std::ofstream &f); 
    void 	Init_Asm(int length);		// funci9 vupoln9usha9 vudelenie pam9ti dl9 masssiva Asm  
    void 	Print_To_Screen();	// vuvdo basovux parametrov loga na ekran 
};
Log::Log(){
	dop_param=0;				// poumolchaniu dopolnitel'nue parametru neproinicializirovanu 
}
Log::~Log(){
	if(dop_param==1){
		delete [] Asm;				// osvoboshdenie pam9ti
		delete [] sigma_r;
		delete [] sigma_t;
	}
}
void Log::Read(std::fstream& f){
	// ??????????? ???_??????? ???_?????????? ????????_?????????? ?????_????? ????_?????????????_????? ??????_??????? ????????_MC ????????? ??????????? 
	f>>dim;					// chtenie razmernosti sistemu 
	f>>l_type;					// chtenie tipa reshetki 
	f>>u_type;					// chtenie tipa potenciala 
	f>>u_param;					// chtenie parametrov potenciala 
	f>>n_rdf;					// chtenie chisla uzlov 
	f>>n_asm;					// chtenie chisla assimetrichnux uzlov 
	f>>rc;					// chtenie radiusa otsechki 
	f>>n_MC;					// chtenie parametra MC 
	f>>rho;					// chtenie plotnosti 
	f>>T;					// chtenie temperaturu 
	f>>Tk;					// chtenie konechnoi temperaturu 
	f>>nT;					// chtenie chisla shagov po temperature 
	f>>n_rdf_mesh;				// chtenie chisla uzlov setki rdf
	f>>n_uzl_mesh;				// chtenie chisla uzlov setkki na odin uzel kristallichsekoi reshetki 
	f>>n_sph_integr;				// chtenie chisla iteracii dk9 postroeni9 steki na sphere 
}
void Log::Read_Old(std::fstream& f){
	// ??????????? ???_??????? ???_?????????? ????????_?????????? ?????_????? ????_?????????????_????? ??????_??????? ????????_MC ????????? ??????????? 
	f>>dim;					// chtenie razmernosti sistemu 
	f>>l_type;					// chtenie tipa reshetki 	
	f>>u_type;					// chtenie tipa potenciala 
	f>>u_param;					// chtenie parametrov potenciala 
	f>>n_rdf;					// chtenie chisla uzlov 
	f>>n_asm;					// chtenie chisla assimetrichnux uzlov 
	f>>rc;					// chtenie radiusa otsechki 
	f>>n_MC;					// chtenie parametra MC 
	f>>rho;					// chtenie plotnosti 
	f>>T;					// chtenie temperaturu 
	f>>Tk;					// chtenie konechnoi temperaturu 
	f>>nT;					// chtenie chisla shagov po temperature 
	f>>n_rdf_mesh;				// chtenie chisla uzlov setki rdf
	f>>n_uzl_mesh;				// chtenie chisla uzlov setkki na odin uzel kristallichsekoi reshetki 
	f>>n_sph_integr;				// chtenie chisla iteracii dk9 postroeni9 steki na sphere 
	// chtenie dopolnitel'nux parametrov 
	dop_param=1;
	f>>sigm2_0;
	f>>A;
	f>>phi;
	f>>entropy_const;
	f>>e_mod;
	Asm=new double[100];
	sigma_r=new double[100];
	sigma_t=new double[100];
	for(int i=0;i<n_asm;i++){
		f>>Asm[i];
		f>>sigma_r[i];
		f>>sigma_t[i];
	}
}
int Log::Comp(Log& log){
    // bazovumi znacheni9mi schitausts9 te, kotorux dostatochno dl9 vuchisleni9 prametrov RDF 
    if(dim!=log.dim) return 0;			// razmernosti ne sovpadaut 
    if(l_type!=log.l_type) return 0;		// tipu reshetok ne sovpadaut 
    if(u_type!=log.u_type) return 0;		// tipu potncialov ne sovpadaut 
    if(u_param!=log.u_param) return 0;		// parametru potencialov ne sovpadaut  
    if(rc!=log.rc) return 0;			// nesovpadaut radiusu otsechki  
    if(n_MC!=log.n_MC) return 0; 		// nesovpadaut parametru MC modelirovani9 
    if(rho!=log.rho) return 0;			// plotnosti sistem ne sovpadaut 
    return 1;					// sovpadenie nabludaets9 po vsem parametram 
}
void Log::Copy_Dop_Param(Log& log){
    sigm2_0=log.sigm2_0;			// kopirovanie paramtra blishaishego uzla
    A=log.A;					// kopirovanie parametra A dl9 dvuxmeri9
    phi=log.phi;				// kopbrvanie parametra phi dl9 trexmeri9 
    entropy_const=log.entropy_const;
    e_mod=log.e_mod;
    for(int i=0;i<n_asm;i++){
	Asm[i]=log.Asm[i];			// kopirovanie koefficientov assimetrii 
	sigma_r[i]=log.sigma_r[i];
	sigma_t[i]=log.sigma_t[i];
    }
}
void Log::Copy_Bas_Param(Log& log){
    dim=log.dim;
    l_type=log.l_type;
    u_type=log.u_type;
    u_param=log.u_param;
    n_rdf=log.n_rdf;
    n_asm=log.n_asm;
    rc=log.rc;
    n_MC=log.n_MC;
    rho=log.rho;
    T=log.T;
    Tk=log.Tk;
    nT=log.nT;
    n_rdf_mesh=log.n_rdf_mesh;
    n_uzl_mesh=log.n_uzl_mesh;
    n_sph_integr=log.n_sph_integr;
}
void Log::Dump_To_File(std::ofstream& f){
    // vuvod bazovux parametrov 
    f<<dim<<"\t";
    f<<l_type<<"\t";
    f<<u_type<<"\t";
    f<<u_param<<"\t";
    f<<n_rdf<<"\t";
    f<<n_asm<<"\t";
    f<<rc<<"\t";
    f<<n_MC<<"\t";
    f<<rho<<"\t";
    f<<T<<"\t";
    f<<Tk<<"\t";
    f<<nT<<"\t";
    f<<n_rdf_mesh<<"\t";			// chtenie chisla uzlov setki rdf
    f<<n_uzl_mesh<<"\t";			// chtenie chisla uzlov setkki na odin uzel kristallichsekoi reshetki 
    f<<n_sph_integr<<"\t";			// chtenie chisla iteracii dk9 postroeni9 steki na sphere 
    // vuvod dopolnitel'nux parametrov 
    f<<sigm2_0<<"\t";
    f<<A<<"\t";
    f<<phi<<"\t";
    f<<entropy_const<<"\t";
    f<<e_mod<<"\t";
    for(int i=0;i<n_asm;i++){
	f<<Asm[i]<<"\t";
	f<<sigma_r[i]<<"\t";
	f<<sigma_t[i]<<"\t";
	//f<<0<<"\t";
    }
    f<<"\n";
}
void Log::Init_Asm(int length){
    Asm=new double[length];
    sigma_r=new double[length];
    sigma_t=new double[length];
    dop_param=1;
}
void Log::Print_To_Screen(){
    std::cout<<"Dim="<<dim<<"\nl_type="<<l_type<<"\nu_type="<<u_type<<"\nu_param="<<u_param<<"\n";
    std::cout<<"n_rdf="<<n_rdf<<"\nn_asm="<<n_asm<<"\nrc="<<rc<<"\nn_MC="<<n_MC<<"\n";
    std::cout<<"rho="<<rho<<"\nT="<<T<<"\nn_rdf_mesh="<<n_rdf_mesh<<"\nn_uzl_mesh="<<n_uzl_mesh<<"\nn_sph_integr="<<n_sph_integr<<"\n";
    std::cout<<"sigm2_0="<<sigm2_0;
    if(dim==3) std::cout<<"\nphi="<<phi;
    else std::cout<<"\nA="<<A;
    std::cout<<"\nEntropy_const="<<entropy_const;
    std::cout<<"\nE_Modelung="<<e_mod;
    if(dop_param==1){
	std::cout<<"\nASM:\n";
	for(int i=0;i<n_asm;i++){
	    std::cout<<Asm[i]<<"\t";
	    std::cout<<sigma_r[i]<<"\t";
	    std::cout<<sigma_t[i]<<"\n";
	}
    }
    std::cout<<"\n";
} 

 

class History{								// ????? ??? ???????? ??????? ????? ?????? ????????? 
    int 	N_Log;				// chislo zapisei v istorii 
    int 	N_New_Max;			// maksimal'noe chislo logov kotoroe moshet but' sformirovano v xode dannogo rascheta 
    int 	N_New_Logs;			// chislo novux sformirovannux logov 
    Log 	*Logs;				// massiv logov istorii 
    Log 	*New_Logs;			// massiv dl9 vremennogo xraneni9 logov kotorue mogut but'; sformirovanu v xode dannogo rascheta 
    int 	p_update;			// priznak neobxodimosti obovleni9 faila istorii
public: 
    History(int i);
    ~History();
    int Test(Log &log,Lattice_Param *LP,int N_ASM_MAX);	// vupolnenie testirovani9 na nalichie bolee rannix raschetov 
};
History::History(int i){
    p_update=0;
    N_New_Max=i;				// inicializaci9 maksimal'nogo chisla novux logov 
    N_New_Logs=0;				// inicializaci9 chisla novux sformirovannux logov 
    std::fstream f_dat("history_dat.txt");	// otkrutie faila s informaciei o sushestvuushix logax
    std::fstream f_log("history_log.txt");	// otkrutie faila s logami 
    f_dat>>N_Log;				// chtenie chisla zapisi v istorii;
    Logs=new Log[N_Log];			// vudelenie pam9ti pod logi
    New_Logs=new Log[N_New_Max];
    for(int i=0;i<N_Log;i++){			// ciklu dl9 chteni9 logov iz faila 
	Logs[i].Read_Old(f_log);			// forirovanie loga na osnove strochki faila 
    }  
    f_dat.close();				// zakrutie failov 
    f_log.close();				// zakrutie failov 
}
History::~History(){
    std::cout<<"~History\n";
    if(N_New_Logs!=0||p_update==1){
	int n=N_Log+N_New_Logs;			// novoe chislo logov 
	std::ofstream f_dat("history_dat.txt");	// otkrutie faila na zapis'
	f_dat<<n;					// zapis' novogo chisla logov 
	f_dat.close();				// zakrutie faila 
	std::ofstream f_log("history_log.txt");	// otkrutie faila so spiskom logov na zapis'
	f_log<<std::setprecision(16);
	for(int i=0;i<N_Log;i++){
		Logs[i].Dump_To_File(f_log);		// soxranenie starux logov v fail 
	}
	for(int i=0;i<N_New_Logs;i++){
		New_Logs[i].Dump_To_File(f_log);	// soxranenie novux logov v fail 
	}
	f_log.close();				// zakrutie faila
    }
    delete [] Logs; 				// osvoboshdenie failov 
}
int History::Test(Log &log,Lattice_Param *LP,int N_ASM_MAX){
	std::cout<<"Start Test\n";
	double t1;
	double t2;
	Vect_Asm_Dat 	VAD(log.l_type,log.rho);
	int p=0;									// parametr 
	Log *sim_log;							// vspomogatel'na9 peremenna9 
	for(int i=0;i<N_Log;i++){					// sravnenie log-a so starumi 
		if(Logs[i].Comp(log)==1){
			p=1;							// naiden podobnui log
			sim_log=&Logs[i];				// soxranenie sovpadaushego starogo logda
			break;							// vuxod iz cikla 
		}
	}
	if(p==0){								// poisk provodits9 v novom spiske 
		for(int i=0;i<N_New_Logs;i++){
			if(New_Logs[i].Comp(log)==1){
				p=1;						// naiden podxod9shii log 
				sim_log=&New_Logs[i];		// soxranenie sovpadaushego novogo logda 
				break;						// vuxod iz cikla 
			}
		}
	}
	std::cout<<"N_Asm="<<log.n_asm<<"\n";
	// ------------------------------
	// ------------------------------
	// ------------------------------
	if(p==0){					// podxod9shiil log ne bul naiden 	
		std::cout<<"log.dim="<<log.dim<<"\n";
		std::cout<<"log.l_type="<<log.l_type<<"\n";
		std::cout<<"log.u_type="<<log.u_type<<"\n";
		std::cout<<"log.u_param="<<log.u_param<<"\n";
		std::cout<<"log.rho="<<log.rho<<"\n";
		std::cout<<"log.rc="<<log.rc<<"\n";
		std::cout<<"log.n_MC="<<log.n_MC<<"\n";
		LP->Init_Lattice(log.dim,log.l_type,log.u_type,log.u_param,log.rho,log.rc,log.n_MC);	// inicializaci9 kristallicheskoi reshetki v raschetchike 
		// vuchislenie parametrov 
		New_Logs[N_New_Logs].Copy_Bas_Param(log);			// kopirovanie vxodnux parametrov v novui log 
		New_Logs[N_New_Logs].Init_Asm(log.n_asm);   			// vudelenie pam9ti dl9 koefficientov novogo loga 
		// vuchislenie dopolnitel'nux parametrov dl9 novogo log-a 
		New_Logs[N_New_Logs].sigm2_0=LP->Compute_sigma2_1(log.n_MC);	// vuchislenie paraemtra sigma2_1
		
		New_Logs[N_New_Logs].entropy_const=LP->Compute_Entropy_Const(log.rho,log.n_MC);
		std::cout<<"New_Logs[N_New_Logs].entropy_const="<<New_Logs[N_New_Logs].entropy_const<<"\n";
		
		std::cout<<"New_Logs[N_New_Logs].sigm2_0=LP->Compute_sigma2_1(log.n_MC);="<<LP->Compute_sigma2_1(log.n_MC)<<"\n";
		if(log.dim==2){							// neobxodimo vuchislit' parametr A
			New_Logs[N_New_Logs].A=LP->Compute_cs()/(4.*M_PI*log.rho*New_Logs[N_New_Logs].sigm2_0);			// vuchislenie 
			//std::cout<<" New_Logs[N_New_Logs].A=LP->Compute_cs() ="<<LP->Compute_cs()<<"\n";
			//std::cout<<" New_Logs[N_New_Logs].A=LP->Compute_cs()/(4.*M_PI*log.rho*New_Logs[N_New_Logs].sigm2_0); ="<<LP->Compute_cs()/(4.*M_PI*log.rho*New_Logs[N_New_Logs].sigm2_0)<<"\n";
		}
		else{								// neobxodimo vuchislit' parametr phi
			double temp;						// vremenna9 peremenna9 dl9 vuchisleni9 parametra phi
			temp=LP->Compute_sigm_inf(log.n_MC);			// vuchislenie parametra udalennoi tochki 
			New_Logs[N_New_Logs].phi=0.5*pow(temp/New_Logs[N_New_Logs].sigm2_0,-0.5);				// vuchislenie parametra phi
		}
		std::cout<<"Lattice* L=new Lattice(log.l_type,N_ASM_MAX\n";
		Lattice* L=new Lattice(log.l_type,N_ASM_MAX,pow(log.rho,-1./log.dim)*Lattice_a_koef(log.l_type));	// inicializaci9 reshetki dl9 rascheta napravlenii dl9 uzlov dl9 kororux vupoln9ets9 raschet koefficientov assimetrii 
		L->Compute_Uzl_Mod();
		for(int i=0;i<log.n_asm;i++){							// proverka na nalichie takogo uzla 
			if(i<L->N_Uzl_Mod){
				if(i<VAD.N){
					New_Logs[N_New_Logs].Asm[i]=LP->Compute_Asm(VAD.r[i],log.n_MC,t1,t2);
					//New_Logs[N_New_Logs].Asm[i]=LP->Compute_Asm(L->U_Mod[i].r,log.n_MC,t1,t2);
					New_Logs[N_New_Logs].sigma_r[i]=t1;
					New_Logs[N_New_Logs].sigma_t[i]=t2;
				}
				else{
					New_Logs[N_New_Logs].Asm[i]=LP->Compute_Asm(L->U_Mod[i].r,log.n_MC,t1,t2);	// raschet koefficienta assimetrii 
					New_Logs[N_New_Logs].sigma_r[i]=t1;
					New_Logs[N_New_Logs].sigma_t[i]=t2;
				}
			}
			else{
				New_Logs[N_New_Logs].Asm[i]=1.0;					// iniciaalizai9 uzla kak simmetrichnogo 
			}
		}
		New_Logs[N_New_Logs].entropy_const=LP->Compute_Entropy_Const(log.rho,log.n_MC);
		log.Init_Asm(log.n_asm);
		log.Copy_Dop_Param(New_Logs[N_New_Logs]);
		N_New_Logs++;
		log.Print_To_Screen();
		std::cout<<"log.Print_To_Screen();\n";
		delete L;
		return 0;
	}
	std::cout<<" if(sim_log->n_asm<log.n_asm)\n";
	if(sim_log->n_asm<log.n_asm){		// v sushestvuushem loge men'she koefficientov assimetrii chem trebuets9 
		LP->Init_Lattice(log.dim,log.l_type,log.u_type,log.u_param,log.rho,log.rc,log.n_MC);
		Lattice* L=new Lattice(log.l_type,N_ASM_MAX,pow(log.rho,-1./log.dim)*Lattice_a_koef(log.l_type));	// inicializaci9 reshetki dl9 rascheta napravlenii dl9 uzlov dl9 kororux vupoln9ets9 raschet koefficientov assimetrii 
		L->Compute_Uzl_Mod();
		sim_log->Print_To_Screen();
		p_update=1;
		// vuchislenie dopolnitel'nux koefficientov 
		log.Init_Asm(log.n_asm);		// vudelenie pam9ti pod polnui spisok koefficientov assimetrii 
		log.Copy_Dop_Param(*sim_log);
		for(int i=sim_log->n_asm;i<log.n_asm;i++){
			if(i<L->N_Uzl_Mod){
				if(i<VAD.N){
					New_Logs[N_New_Logs].Asm[i]=LP->Compute_Asm(VAD.r[i],log.n_MC,t1,t2);
					New_Logs[N_New_Logs].sigma_r[i]=t1;
					New_Logs[N_New_Logs].sigma_t[i]=t2;
					//New_Logs[N_New_Logs].Asm[i]=LP->Compute_Asm(L->U_Mod[i].r,log.n_MC);
				}
				else{
					New_Logs[N_New_Logs].Asm[i]=LP->Compute_Asm(L->U_Mod[i].r,log.n_MC,t1,t2);	// raschet koefficienta assimetrii 
					New_Logs[N_New_Logs].sigma_r[i]=t1;
					New_Logs[N_New_Logs].sigma_t[i]=t2;
				}
			}
		}
		delete [] sim_log->Asm;				// obnulenie starogo massiva koefficientov v loge 
		delete [] sim_log->sigma_r;
		delete [] sim_log->sigma_t;
		sim_log->Init_Asm(log.n_asm);		// vudelenie pam9t bol'shego razmera pod koefficientu assimetrii
		sim_log->n_asm=log.n_asm;			// obnovlenie dlinnu massiva assimetrichnux uzlov
		for(int i=0;i<log.n_asm;i++){
			sim_log->Asm[i]=log.Asm[i];		// inicializaci9 znachenii koefficientov assimetrii 
			sim_log->sigma_r[i]=log.sigma_r[i];
			sim_log->sigma_t[i]=log.sigma_t[i];
		}
		delete L;
		return 0;
	}
	// polnoe sovpadenie => kopirovanie 
	std::cout<<"??? ??????? ?????? ?????????? ?????????? ???:\n";
	sim_log->Print_To_Screen();
	log.Init_Asm(log.n_asm);			// vudelenie pam9ti pod massivu koefficientov assimetrii
	log.Copy_Dop_Param(*sim_log);		// kopirovanie dopolnitel'nux parametrox iz sim_log v log 
	return 0;
}



class Queue{
	char 			*Name;			// massiv dl9 xraneni9 imeni rascheta 
	Log 			*Logs;			// massiv logov raschetov 
	History 			*H;			// istori9  
	Lattice_Param 	*LP;			// class dl9 rascheta parametrov reshetki
	Comp_RDF		*C_RDF;			// class dl9 rascheta parnoi korrel9cionnoi funcii kristalla 
public:
    int 	N_Compute;		// dlinna dannoi ocheredi 
    Queue(std::fstream &f);
    ~Queue();
    void Compute(int i);		// vuchislenie i-togo eleenta ocheredi 
};
Queue::Queue(std::fstream &f){
    // ????????? ???????? ????? &f 
    // ???_???????? ??????_?????? 
    // ??????????? ???_??????? ???_?????????? ????????_?????????? ?????_????? ????_?????????????_????? ??????_??????? ????????_MC ????????? ??????????? 
    Name=new char[256];			// ????????? ?????? ??? ??? ???????? 
    std::string s;			// ????????? ?????????? 
    f>>s;				// ?????? ????? ???????? 
    std::strcpy(Name, s.c_str());	// ?????? ????? ???????? ? Name 
    f>>N_Compute;			// ?????? ?????? ??????? 
    // ????????? ?????? ??? logi 
    Logs=new Log[N_Compute];
    // ?????? ??????? ??????
    for(int i=0;i<N_Compute;i++){
	Logs[i].Read(f);
    }
    H=new History(N_Compute);		// vudelenie pam9ti pod istoriu 
    LP=new Lattice_Param;		// vudelenie pam9ti pod rastetchik parametrov reshetki
}
Queue::~Queue(){
    delete [] Name;			// osvoboshdenie pamayti 
    delete [] Logs;			// osvoboshdenie pamayti 
    delete H;				// osvoboshdenie pam9ti
}
void Queue::Compute(int i){
	int 		pp=0;
	double 	x1_0;
	double 	x2_0;
	double 	x3_0;
	double 	T_old;
	int N_ASM_MAX=1000;			// maksimal'noe chislo uzlov reshetki kotora9 stroits9 dl9 rascheta uzlov dl9 kotorux vuchisl9uts9 koefficientu asimetrii
	// testirovanie na nalichie gotovux parametrov dl9 rascheta i-togo loga
	H->Test(Logs[i],LP,N_ASM_MAX);			// proverka vupoln9ets9 po bazovum parametram loga, posle zaversheni9 proverki v dopolnitel'nux [rarametrax loga okazuvauts9 zapisanu neobxodimu parametru rascheta 
	std::cout<<"P-1\n";
	// postroenie RDF na osnov dopolnitel'nux parametrov loga 
	double 	R_RDF=pow(1,-1./Logs[i].dim)*Logs[i].rc;		// ???????? ?????????? RDF 
	double 	R_Sum_Uzl=pow(1,-1./Logs[i].dim)*3.;			// ???????? ????? ????? (???? ??????????? ?? R_RDF ?????? ????? ? ??????????? R_RDF) 
	double 	Gamma=Logs[i].T;
	double 	Gamma_k=Logs[i].Tk;
	double 	kappa=pow(3./(4.*M_PI*Logs[i].rho),1./3.);
	double 	T=1.0/(kappa*Gamma_k);
	double 	T_k=1.0/(kappa*Gamma);
	T=Logs[i].T;
	T_k=Logs[i].Tk;
	Gamma=1./(kappa*T_k);
	Gamma_k=1./(kappa*T);
	double 	h_gamma=(Gamma_k-Gamma)/(Logs[i].nT-1);
	double 	h_T=(T_k-T)/(Logs[i].nT-1);
	double 	E;
	double 	E0;
	int 		na;
	std::ofstream fEout("fEout.txt");
	std::cout<<"P0\n";
	Lattice* L=new Lattice(Logs[i].l_type,Logs[i].n_rdf,Lattice_a_koef(Logs[i].l_type));
	L->Compute_Uzl_Mod();                                    
	double *Dat_Na=new double[L->N_Uzl_Mod];
	Dat_Na[0]=1;
	for(int j=1;j<L->N_Uzl_Mod;j++){
		if(Logs[i].dim==3){
			Dat_Na[j]=Dat_Na[j-1]+1.-2.*Logs[i].phi*sqrt(Dat_Na[j-1]);				// inicializaci9 znacheni funcii 
		}
		else{
			Dat_Na[j]=1+Logs[j].A*log(j+1);
		}
	}
	double koef_s=1.00;
	Sphere_Triangle_Mesh* Sp_Mesh=new Sphere_Triangle_Mesh(Logs[i].n_sph_integr,0.01);
	double NA_Coef=1.0;
	double s0b=0;//2*8.855;//17.774;//17.4122;//0.677*16.9442;
	
	std::cout<<"P1\n";
	T=(i+1.0)*Logs[i].Tk;
	T=1*Logs[i].Tk;
        //T=0.249922;
	for(int j=0;j<Logs[i].nT;j++){
		//T=Logs[i].Tk;
		double 	s0=2.*Logs[i].sigm2_0*T*pow(Logs[i].rho,2./Logs[i].dim);
		s0=s0*(1+s0b*s0*Logs[i].rho);

                //s0=0.00841348/2;

		//std::cout<<"S0="<<s0<<"\n";
                C_RDF=new Comp_RDF(L,Sp_Mesh,Logs[i].l_type,Logs[i].n_rdf,Lattice_a_koef(Logs[i].l_type),sqrt(s0)*8.5,Logs[i].n_uzl_mesh,Logs[i].n_sph_integr,R_RDF,R_Sum_Uzl,Logs[i].n_rdf_mesh);
		//C_RDF=new Comp_RDF(L,Sp_Mesh,Logs[i].l_type,Logs[i].n_rdf,Lattice_a_koef(Logs[i].l_type),sqrt(s0)*8.5,Logs[i].n_uzl_mesh,Logs[i].n_sph_integr,R_RDF,R_Sum_Uzl,Logs[i].n_rdf_mesh);
		//C_RDF=new Comp_RDF(L,Sp_Mesh,Logs[i].l_type,Logs[i].n_rdf,Lattice_a_koef(Logs[i].l_type),sqrt(s0)*8.5,Logs[i].n_uzl_mesh,Logs[i].n_sph_integr,R_RDF,R_Sum_Uzl,Logs[i].n_rdf_mesh);
		if(Logs[i].dim==3){
			//L->U_Mod[7].ves*=1.075;
			//L->U_Mod[1].R*=0.995;
			//L->U_Mod[1].ves*=1.0;
			//C_RDF->Compute_RDF_Dat(Logs[i].phi,sqrt(s0),Logs[i].n_asm,Logs[i].Asm,201);
			C_RDF->Compute_RDF_Dat(Logs[i].phi,sqrt(s0),Logs[i].n_asm,Logs[i].Asm,201);
			std::cout<<L->U_Mod[1].ves<<"\n";
			koefTT=1.0;
			for(int ias=0;ias<1;ias++){
				//if(ias==1) ias++;
				//for(int ias=0;ias<Logs[i].n_asm;ias++){
				std::cout<<"rho="<<Logs[i].rho<<"\n";
				std::cout<<"T="<<T<<"\n";
				std::cout<<"s0="<<3*s0<<"\n";
				std::cout<<"Asm="<<Logs[i].Asm[ias]<<"\n";
				std::cout<<"scale="<<3*sqrt(3*s0)/pow(Logs[i].rho,-1./3.)<<"\n";
				na=L->U_Mod[ias].na;
				//s0=2.0*Logs[i].sigm2_0*T*Dat_Na[na-1];
				//T=0.4147587;
				//s0=0.03331/3;
				//ias=0;
				//Logs[i].Asm[ias]=1.20795;
                                //T=0.000131826;
                                //s0=0.000503069*0.1;
                                //Logs[i].Asm[ias]=1.30969;
                                double temp_a=L->U_Mod[ias].R*pow(Logs[i].rho,-1.0/Logs[i].dim);
                                s0=2.*Logs[i].sigm2_0*T;
				std::cout<<"s0="<<3*s0<<"\n";
				std::cout<<"Dat_Na[na-1]="<<Dat_Na[na-1]<<"\n";
				std::cout<<"Log.rho="<<Logs[i].rho<<"\n";
				std::cout<<"sr="<<3*s0/(1.+2.*Logs[i].Asm[ias])<<"\tst="<<Logs[i].Asm[ias]*3*s0/(1.+2.*Logs[i].Asm[ias])<<"\n";
				s_cor *SC=new s_cor(L->U_Mod[ias].R*pow(Logs[i].rho,-1.0/Logs[i].dim),Logs[i].rho,T,3*s0,Logs[i].Asm[ias],0.8 ,201,Logs[i].l_type,Logs[i].u_type,3);
				SC->Err(0.0,3*s0/(1.+2.*Logs[i].Asm[ias]),Logs[i].Asm[ias]*3*s0/(1.+2.*Logs[i].Asm[ias]));
                                //SC->Err(2.12666*0.1,0.0645775*0.1,0.0051315*0.1);


                                // s_cor(double A,double Rho,double T_in,double sigma0_in,double Asm_in,double scale_in,int n,int L_Type,int U_type,int Dim);
				//SC->Err(0.210049 ,0.0153945,0.00651891);
				//SC->Err(-(48.8281*temp_a*temp_a-61.0352*temp_a)/pow(temp_a,7)/T,3*s0/(1.+2.*Logs[i].Asm[ias]),Logs[i].Asm[ias]*3*s0/(1.+2.*Logs[i].Asm[ias]));
				//koefTT=0.25;
				//SC->Find_Min();
                                //koefTT=0.5;
				//SC->Find_Min();
                                koefTT=1.0;
                                SC->Find_Min();
                                //SC->Find_Min();
                                //koefTT=1.0;
                                //SC->Find_Min();
                                //koefTT=0.975;
                                //SC->Find_Min();
                                //for(int j=0;j<=100;j++){
                                //koefTT=0.01*j;
                                    //SC->Find_Min();
                                //}
				//SC->Find_Min_2d();
				//L->U_Mod[1].ves*=1.00;
				SC->Compute_g(C_RDF->rdf->Setca,C_RDF->rdf->Dat_RDF,Logs[i].n_rdf_mesh,L->U_Mod[ias].ves,400);
				//SC->Compute_g(C_RDF->rdf->Setca,C_RDF->rdf->Dat_RDF,Logs[i].n_rdf_mesh,L->U_Mod[ias].ves,400);
				SC->Dump_To_File();
				//SC->Compute_M3();
				//std::cout<<"M3="<<SC->
                                std::cout<<"Logs[i].n_uzl_mesh"<<Logs[i].n_uzl_mesh<<"\n";
				delete SC;
				NA_Coef=NA_Coef+1.-2.*Logs[i].phi*sqrt(NA_Coef);
			}
		}
		else{
                        //s0=0.00841348;
			//s0b=5;
                        //s0b=7.53903;
                        //s0b=0;
			s0=2.*Logs[i].sigm2_0*T*pow(Logs[i].rho,2./Logs[i].dim);

                        //s0=0.00841348/2;

			std::cout<<"Test s0="<<s0<<"\t";
			//s0=s0*(1+s0b*s0*Logs[i].rho);
			std::cout<<"\t!!!s0="<<s0<<"\n";
			//s0=1.0*2.*Logs[i].sigm2_0*T;
			//L->U_Mod[6].ves*=12.1;
			//C_RDF->Compute_RDF_Dat(0.663491,sqrt(s0),Logs[i].n_asm,Logs[i].Asm,100);
			//C_RDF->Compute_RDF_Dat(4*Logs[i].A,sqrt(s0),Logs[i].n_asm,Logs[i].Asm,100);
                        
                        //Logs[i].A *=3.0;
                        C_RDF->Compute_RDF_Dat(Logs[i].A,sqrt(s0),Logs[i].n_asm,Logs[i].Asm,100);
			
			//s0=1.12*2.*Logs[i].sigm2_0*T;
			//Logs[i].Asm[0]*=1.0;
			
			for(int ias=0;ias<1;ias++){
                                //Logs[i].Asm[ias]=1.5236;
                                //if(ias==1) continue;
				//s0=2.*Logs[i].sigm2_0*T;
				na=L->U_Mod[ias].na-1;
				//s0=(1.+Logs[i].A*log(1*na+1))*2.*Logs[i].sigm2_0*T;
				//if(ias==1 ) s0=1.08*(1.+Logs[i].A*log(na+1))*2.*Logs[i].sigm2_0*T;
				//if(ias==2 ) s0=1.08*(1.+Logs[i].A*log(na+1))*2.*Logs[i].sigm2_0*T;
				//if(ias==0 ) s0=1.0*(1.+Logs[i].A*log(na+1))*2.*Logs[i].sigm2_0*T;
				//if(ias==1 ) s0=1.0*0.115066*2.*T;
				//if(ias==2 ) s0=1.0*0.112529*2.*T;
                                //s0=2.*Logs[i].sigm2_0*T*pow(Logs[i].rho,2./Logs[i].dim);
                                //if(ias==1) s0*=1.44683;
                                //if(ias==2) s0*=1.40513;
				std::cout<<"s0="<<s0<<"\t";
				//Logs[i].Asm[ias]+=0*8.084*s0*Logs[i].rho;
				//s0=s0*(1+s0b*s0*Logs[i].rho);
                                //T=.05;
                                //s0=(0.00838921*pow(T,0.703443)-0.00762986*pow(T,0.715386))*pow(Logs[i].rho,2./Logs[i].dim);
                                //s0=0.0006087857*pow(Logs[i].rho,2./Logs[i].dim);
                                //Logs[i].Asm[ias]=1.8081;
                                //Logs[i].Asm[ias]=1.46322;
				//Logs[i].Asm[ias]
				//s0 = 0.00841348;
				//Logs[i].Asm[ias]=1.5236;
				std::cout<<"s0="<<s0<<"\n";
				std::cout<<"rho="<<Logs[i].rho<<"\n";
				std::cout<<"T="<<T<<"\n";
				std::cout<<"s0="<<2*s0<<"\n";
				std::cout<<"Asm="<<Logs[i].Asm[ias]<<"\n";
				std::cout<<"scale="<<2*sqrt(2*s0)/pow(Logs[i].rho,-1./2.)<<"\n";
				std::cout<<"L->U_Mod[0].R="<<L->U_Mod[ias].R<<"\n";
				std::cout<<"na="<<na<<"\n";
				//if(ias==2) T*=100;
				//if(ias==0) Logs[i].Asm[ias]*=1.02498;
				if(ias==0) koefTT=1.0;
				else koefTT=0.1;
				//if(ias==2) koefTT=1.0;
                                
                                
                                // Test 
                                //T=0.0354813;
                                //s0=0.5*0.00942638;
                                //Logs[i].Asm[ias]=1.51697;
                                
                                s0=2.*Logs[i].sigm2_0*T;
				s_cor *SC=new s_cor(L->U_Mod[ias].R*pow(Logs[i].rho,-1.0/Logs[i].dim),Logs[i].rho,T,2.0*s0,Logs[i].Asm[ias],0.5,201,Logs[i].l_type,Logs[i].u_type,2);
				
				//   s_cor(L->U_Mod[ias].R*pow(Logs[i].rho,-1.0/Logs[i].dim),Logs[i].rho,T,3*s0,Logs[i].Asm[ias],0.18 ,1801,Logs[i].l_type,Logs[i].u_type,3);
				//   
				std::cout<<"SC->Err="<<SC->Err(0.00,2*s0/(1.+Logs[i].Asm[ias]),Logs[i].Asm[ias]*2*s0/(1.+Logs[i].Asm[ias]));
				for(int ii=0;ii<100;ii++){
					//std::cout<<"SC->Err="<<SC->Err(0.1*ii,2*s0/(1.+Logs[i].Asm[ias]),Logs[i].Asm[ias]*2*s0/(1.+Logs[i].Asm[ias]));
				}
				//std::cout<<"SC->Err="<<SC->Err(0.0,2*s0,Logs[i].Asm[ias])<<"\n";
				//SC->Err(2.89428,0.00460551,0.000834214);
				//SC->Err(4.10249,0.351813,0.0196568);
				
				//if(pp!=0) SC->Err(x1_0*T/T_old,x2_0*T/T_old,x3_0*T/T_old);
				//SC->Err(2.19952,0.0482234,0.00686576);
				//if(ias==0) SC->Err(3.47285,0.0841075,0.00784186);
				//if(ias==2) SC->Err(0.316911,0.0066532,0.00327516);
				//SC->Err(2.71298,0.126471,0.0290416);
				koefTT=1.0;
                                //SC->Err(0.0157126,0.000212124,0.000101386);
                                //SC->Err(0.0263339,0.0139923,0.00789431);

                                //SC->Err(1.17348,0.0289606,0.00424663);

                                SC->Find_Min();
                                //SC->Find_Min();
                                //SC->Find_Min();
				/*if(ias==0){
                                    koefTT=1.0;
                                    //SC->Err(8.27528,0.112247,0.0038403);
                                    SC->Find_Min();
                                }
				if(ias==1){
                                    koefTT=1.0;
                                    //SC->Err(8.27528,0.112247,0.0038403);
                                    SC->Find_Min();
                                }
				if(ias==2){
                                    koefTT=1.0;
                                    //SC->Err(8.27528,0.112247,0.0038403);
                                    SC->Find_Min();
                                }*/
				//if(ias==0) SC->Find_Min();
                                //if(ias==2) SC->Find_Min();
                                //koefTT=0.0;
				///for(int ikoefTT=0;ikoefTT<20;ikoefTT++){
                                    //koefTT+=0.05;
                                    //SC->Find_Min();
				//}
				/*if(ias!=0){
					s0=(1.+Logs[i].A*log(na+1))*2.*Logs[i].sigm2_0*T;
					s0=s0*(1+s0b*s0*Logs[i].rho);
					SC->Err(0.0,2*s0/(1.+Logs[i].Asm[ias]),Logs[i].Asm[ias]*2*s0/(1.+Logs[i].Asm[ias]));
					koefTT=1.0;
					SC->Find_Min();
				}*/
				//koefTT-=0.1;
				//koefTT=0;
				//koefTT-=0.02;
				//if(ias==2) SC->Err(0.0,2*s0/(1.+Logs[i].Asm[ias]),Logs[i].Asm[ias]*2*s0/(1.+Logs[i].Asm[ias]));
                                //SC->Dump_To_File();
				SC->Compute_g(C_RDF->rdf->Setca,C_RDF->rdf->Dat_RDF,Logs[i].n_rdf_mesh,L->U_Mod[ias].ves,400);
				SC->Dump_To_File();
				T_old=T;
				x1_0=SC->tr;
				x2_0=SC->sr;
				x3_0=SC->st;
				pp=1;
				delete SC;
			}
		}
	// vuvod dannux 
	int 		nom;										// peremenna9 dl9 xraneni9 nomera RDF 
	
	s0=2.*Logs[i].sigm2_0*T;
	std::cout<<"---> s0="<<s0<<"\n";
	std::cout<<"---> s0="<<L->U_Mod[0].R<<"\n";
	
	std::fstream 	f_nom("RDFs/RDFs_nom.txt");				// otkrutie faila s nomerom poslednego RDF
	f_nom>>nom;										// chtenie nomera poslednego RDF 
	f_nom.close();										// zakrutie faila s nomerom posledneogo RDF
	std::ofstream 	f_nom2("RDFs/RDFs_nom.txt");				// otkrutie faila s nomerom poslednego RDF na obnovlenie 
	nom++;												// uvelichenie chisla provedennux raschetov RDF 
	f_nom2<<nom;										// soxranenie obnovlennogo chisla postroennux RDF 
	f_nom2.close();										// zakrutie faila 
	std::ofstream 	f_logs("RDFs/RDFs_logs.txt",std::ios::app);		// otkrutie faila dl9 dopolneni9 logom tekushego postroeni9 RDF 
	f_logs<<std::fixed<<std::setprecision(15);
	f_logs<<nom<<"\t";
	E=C_RDF->rdf->Compute_E(Logs[i].u_type,Logs[i].u_param,R_RDF,pow(Logs[i].rho,-1./Logs[i].dim));
	E0=C_RDF->Compute_E0(Logs[i].u_type,Logs[i].u_param,R_RDF,pow(Logs[i].rho,-1./Logs[i].dim));
	f_logs<<E0<<"\t";	// vuchislenie energii osnovnogo sosto9ni9 
	f_logs<<E<<"\t";	// vuchislenie energii tekushego soto9ni
        Logs[i].e_mod= E0;
	//Logs[i].e_mod= (3./(4.*M_PI*Logs[i].rho),1./3.)*E0;
	std::cout<<"E_Modelung="<<Logs[i].e_mod<<"\n";
	Logs[i].Dump_To_File(f_logs);						// zapis' loga 
	char 		Name_RDF[256];						// peremeenna9 dl9 xraneni9 imeni vuxodnogo faila RDF
	std::sprintf(Name_RDF,"RDFs/RDF_%d.txt",nom);				// formirovanie imeni vuxodnogo faila 
	C_RDF->rdf->RDF_Dump_To_File(Name_RDF,Logs[i].rho);			// zapis' postroennogo RDF v fail 
	f_logs.close();								// zakrutie faila logov 
	std::cout<<T<<"\t"<<E<<"\n";
	fEout<<T<<"\t"<<E<<"\n";
	delete C_RDF;								// osvoboshdenie pam9ti 
	T-=h_T;
	Gamma+=h_gamma;
	std::cout<<"nom="<<nom<<"\n";
    }
    fEout.close();
    delete [] Dat_Na;
}


int main(int argc, char **argv){
        std::cout << 1<< "\n";
    
        /*Lattice* L=new Lattice(3,10000,1.0);
        std::cout << 1<< "\n";
        L->Compute_Uzl_Mod();
        L->Uzl_Stat();
        L->Uzl_Mod_Stat();*/
	
        
        std::cout<<"Start\n";
	std::fstream f_in("in.txt");						// fail s vxodnumi dannumi 
	std::cout<<"0\n";
	Queue Q(f_in);									// inicializaci9 ocheredi 
	std::cout<<"1\n";
	for(int i=0;i<Q.N_Compute;i++){					// obrabotka vsex punktov ocheredi 
		S_Line();								// pechat' razdelitel'noi linii 
		Q.Compute(i);								// pbranotka i-togo elementa oceheredi 
	}
	f_in.close();
	S_Line();
        /*
        double * p=new double[3];
        p[0]=0.730;
        Base_Cell*		 bc=new Base_Cell(1,p);
        Brillouin_Zone*	         BZ=new Brillouin_Zone(bc);
        Spectrum* 		 S=new Spectrum(bc,10,BZ,9,14.0);
        Vect eq(2,0,0);
        S->Clean_D();										// ??????? ??????? D 
	S->Compute_D(eq);									// ?????????? ????? ??????? D 
	S->Compute_EV();
        S->Print_To_Screen_EV();
        S->Print_To_Screen_EVect();*/
	/*
	*/
	// razdelitel'na9 lini9 */
	/*double *d1=new double[61];
	double *d2=new double[61];
	d2[0]=0;d2[1]=0;d2[2]=0;d2[3]=0;d2[4]=0;d2[5]=0;d2[6]=0;d2[7]=0;d2[8]=0;d2[9]=0;d2[10]=0;
	d2[11]=0.245;d2[12]=0.275;d2[13]=0.29;d2[14]=0.3;d2[15]=0.31;d2[16]=0.32;d2[17]=0.325;
	d2[18]=0.33;d2[19]=0.335;d2[20]=0.34;d2[21]=0.345;d2[22]=0.345;d2[23]=0.345;d2[24]=0.345;
	d2[25]=-0.265;d2[26]=-0.275;
	*/
        /*std::cout<<1<<"\n";
        double rho;
	//for(int irho=0;irho<(5*120+1);irho++){
        for(int irho=0;irho<51;irho++){
		char *Name=new char[256];
                rho=0.616+irho*0.001;
                //rho=1.05;
		std::sprintf(Name,"out_hex_asm_sc6_2sc_rho_%d.txt",irho);
		std::ofstream f(Name);
                //rho=0.5+irho*0.01;
		for(int i=0;i<51;i++){
                    std::cout<<1<<"\n";
			double alpha=i*M_PI/(3.*50);
                        //alpha=M_PI/3.0;
                        //alpha=0.0;
                        //if(i==1) alpha=M_PI/3.0;
			double * p=new double[3];	
                        p[0]=rho;
			//p[0]=0.221049;
                        p[1]=-230.33061942467765 + 1076.3833183753363*rho - 1676.0674991305702*pow(rho,2) + 870.7595357364847*pow(rho,3);
                        //p[1]=-476392.9974736906 + 4.990342959714172e6*rho - 2.143097834710057e7*pow(rho,2) + 4.57569762224115e7*pow(rho,3) - 
   //3.8488583972898036e7*pow(rho,4) - 3.673332231323213e7*pow(rho,5) + 1.2877517935218106e8*pow(rho,6) - 1.4027825570857978e8*pow(rho,7) + 
   //7.349548706287253e7*pow(rho,8) - 1.5613307001127739e7*pow(rho,9);
			//p[1]=1.2562676113230535 - 0.839855882974878*rho + 1.0008713154412148*pow(rho,2) - 0.3780576583500433*pow(rho,3);
                        //p[1]=-1.0687726838156147e7 + 1.3573219749680448e8*rho - 7.652160721512407e8*pow(rho,2) + 2.5135322589944544e9*pow(rho,3) - 5.301246597598545e9*pow(rho,4) + 7.44483977274149e9*pow(rho,5) - 6.961661561895971e9*pow(rho,6) + 4.179773669853919e9*pow(rho,7) - 1.4620881337865741e9*pow(rho,8) + 2.27024182555176e8*pow(rho,9);
			//p[1]=d2[irho];
			Base_Cell*		 bc=new Base_Cell(10,p);					// ???????????? ??????? 
			Brillouin_Zone*	         BZ=new Brillouin_Zone(bc);				// ???????????? ?????? ???? ????????? 
			Spectrum* 		 S=new Spectrum(bc,10,BZ,6,14.0);			// ????????????? ?????????? ??????? 
			Vect eq(cos(alpha),sin(alpha),0);
                        
                        //std::cout<<d2Udxdx(eq,8,1.0)<<"\n";
                        //std::cout<<d2Udxdy(eq,8,1.0)<<"\n";
                        //std::cout<<d2Udydy(eq,8,1.0)<<"\n";
                        std::cout<<1<<"\n";
			S->Compute_Branch(eq,1000,2*M_PI*pow(rho,0.5));
			S->Dump_To_File_Brunch(f,BZ,eq);
		}
		f.close();
	}*/
        
        //sq
	/*std::ofstream f("2sc_sc_phonon_1057_v2.txt");
        double * p=new double[3];
        double   rho=0.91;
        //double   
        p[0]=rho;
        p[1]=1.0;
        Base_Cell*		 bc=new Base_Cell(5,p);
        Brillouin_Zone*	         BZ=new Brillouin_Zone(bc);
        Spectrum* 		 S=new Spectrum(bc,10,BZ,6,14.0);
        Vect eq(1.0,0,0);
        Vect q0;
        S->Compute_Branch(eq,1000,M_PI*pow(rho,0.5));
        S->Dump_To_File_Brunch(f,BZ,eq);
        eq.Init(0,1.0,0);
        q0.Init(M_PI*pow(rho,0.5),0,0);
        S->Compute_Branch(q0,eq,1000,M_PI*pow(rho,0.5));
        S->Dump_To_File_Brunch(f,BZ,eq);
        eq.Init(-sin(M_PI/4.0),-sin(M_PI/4.0),0);
        q0.Init(M_PI*pow(rho,0.5),M_PI*pow(rho,0.5),0);
        S->Compute_Branch(q0,eq,1000,sqrt(2.)*M_PI*pow(rho,0.5));
        S->Dump_To_File_Brunch(f,BZ,eq);
        f.close();
        Lattice_Param           LP;
        LP.Init_Lattice(2,12,6,3,rho,7.5,1000000);
        std::cout<<std::setprecision(10);
        std::cout<<"entropy_const="<<LP.Compute_Entropy_Const(rho,1000000)<<"\n";*/
        // void 	Init_Lattice(int dim,int l_type,int u_type,double u_param,double rho_v,double rc,int N_MC);	
        /*Lattice_Param           LP;
        LP.Init_Lattice(2,6,3,3,1.0,35,1000000);
        std::cout<<"LP->Compute_Entropy_Const(log.rho,log.n_MC)="<<LP.Compute_Entropy_Const(1.0,1000000)<<"\n";*/
        //LP.Compute_cs();
        //LP.S->Compute_Branch();
        //hex
	/*std::ofstream f("Y.txt");
        double * p=new double[3];
        double   rho=0.1;
        //double   
        p[0]=rho;
        p[1]=1.0;
        Base_Cell*		 bc=new Base_Cell(6,p);
        Brillouin_Zone*	         BZ=new Brillouin_Zone(bc);
        Spectrum* 		 S=new Spectrum(bc,200,BZ,2,3.0);
        
        Vect eq(1.0,0,0);
        //Vect eq(cos(M_PI/3),sin(M_PI/3),0);
        Vect q0;
        S->Compute_Branch(eq,1000,2*M_PI*pow(rho,0.5)*1.1547*1.074569931823542);
        S->Dump_To_File_Brunch(f,BZ,eq);
        (eq*M_PI*pow(rho,0.5)*1.1547*1.074569931823542).Print_To_Screen();
        eq.Init(-cos(M_PI/3.),sin(M_PI/3.),0);
        q0.Init(M_PI*pow(rho,0.5)*1.1547*1.074569931823542,0,0);
        q0.Print_To_Screen();
        (q0+eq*0.5*M_PI*pow(rho,0.5)*1.1547*1.074569931823542).Print_To_Screen();
        S->Compute_Branch(q0,eq,1000,0.5*M_PI*pow(rho,0.5)*1.1547*1.074569931823542);
        S->Dump_To_File_Brunch(f,BZ,eq);
        eq.Init(-cos(M_PI/6.0),-sin(M_PI/6.0),0);
        q0.Init(0.5*sqrt(3.)*M_PI*pow(rho,0.5)*1.074569931823542,0.5*M_PI*pow(rho,0.5)*1.074569931823542,0);
        q0.Print_To_Screen();
        S->Compute_Branch(q0,eq,1000,0.5*sqrt(3.)*M_PI*pow(rho,0.5)*1.1547*1.074569931823542);
        S->Dump_To_File_Brunch(f,BZ,eq);
        f.close();*/
        
        //std::ofstream fDOS("IPL3_DOS.txt");
        //S->Compute_DOS(10000000,M_PI*pow(rho,0.5),fDOS);
        //fDOS.close();
        
        /*
        std::ofstream f("2sc_rsq_phonon_0960v3.txt");
        double  * p=new double[3];
        double  a;	
        double  rho=0.96;
        //double   
        p[0]=rho;
        p[1]=1.2562676113230535 - 0.839855882974878*rho + 1.0008713154412148*pow(rho,2) - 0.3780576583500433*pow(rho,3);
        a=sqrt(2.0)/(sqrt(p[0])*(sin(p[1])+cos(p[1])));
        Base_Cell*		 bc=new Base_Cell(12,p);
        Brillouin_Zone*	         BZ=new Brillouin_Zone(bc);
        Spectrum* 		 S=new Spectrum(bc,8,BZ,6,14.0);
        Vect eq(1.0,0,0);
        Vect q0(0,0,0);
        eq.Init(1.0,0.0,0);
        S->Compute_D(eq);
        S->Compute_EV();
        S->Print_To_Screen_EV();
        S->Compute_Branch(q0,eq,500,M_PI*pow(rho,0.5)/2.82843);
        S->Dump_To_File_Brunch(f,BZ,eq);
                S->Compute_D(eq);
        S->Print_To_Screen_EVect();
        //(eq*M_PI*pow(rho,0.5)/2.82843).Print_To_Screen();
        //std::cout<<"l1="<<pow(rho,0.5)/2.82843<<"\n";
        q0.Init(M_PI*pow(rho,0.5)/2.82843,0,0);
        //q0.Print_To_Screen();
        eq.Init(0.0,1.0,0);
        S->Compute_Branch(q0,eq,500,M_PI*pow(rho,0.5)/2.82843);
        S->Dump_To_File_Brunch(f,BZ,eq);
        q0.Init(M_PI*pow(rho,0.5)/2.82843,M_PI*pow(rho,0.5)/2.82843,0);
        eq.Init(-sin(M_PI/4.0),-sin(M_PI/4.0),0);
        S->Compute_Branch(q0,eq,500,sqrt(2.)*M_PI*pow(rho,0.5)/2.82843);
        S->Dump_To_File_Brunch(f,BZ,eq);
        
        eq.Init(1.0728581349,0.5176718820,0.0000000000);
        S->Compute_D(eq);
        S->Compute_EV();
        S->Print_To_Screen_D();
        
        f.close();
        Lattice_Param           LP;
        */
        //LP.Init_Lattice(2,12,6,3,rho,7.5,1000000);
        //std::cout<<std::setprecision(10);
        //std::cout<<"entropy_const="<<LP.Compute_Entropy_Const(rho,1000)<<"\n";
        //std::ofstream fDOS("2sc_rsq_DOS_0960.txt");
        //S->Compute_DOS(10000000,M_PI*pow(rho,0.5)/2.82843,fDOS);
        //fDOS.close();
        
        
/*
        char *Name=new char[256];
        //std::sprintf(Name,"2sc_rsq_c_%d.txt",i);
        std::sprintf(Name,"IPL3.txt");
        std::ofstream f(Name);
        //std::ofstream f("2sc_rsq_c_1116.txt");
        double  *p=new double[3];
        double  a;	
        double  rho=1.0;
        double  alpha;
        int     n_alpha=100;
        //double   
        //char *Name=new char[256];
        p[0]=rho;
        p[1]=1.2562676113230535 - 0.839855882974878*rho + 1.0008713154412148*pow(rho,2) - 0.3780576583500433*pow(rho,3);
        a=sqrt(2.0)/(sqrt(p[0])*(sin(p[1])+cos(p[1])));
        Base_Cell*		 bc=new Base_Cell(6,p);
        Brillouin_Zone*	         BZ=new Brillouin_Zone(bc);
        Spectrum* 		 S=new Spectrum(bc,20,BZ,3,3.0);
        Vect eq(1.0,0,0);
        Vect q0(0,0,0);
        eq.Init(1.0,0.0,0);
        
        
        for(int i_alpha=0;i_alpha<n_alpha;i_alpha++){
            alpha=i_alpha*M_PI/(3*(n_alpha-1));
            std::cout<<alpha<<"\n";
            eq.Init(cos(alpha),sin(alpha),0.0);
            eq.Print_To_Screen();
            S->Compute_Branch(eq,500,5);
            S->Dump_To_File_Brunch(f,BZ,eq);
            //S->CompuTe_C_Br(eq,50,2.001,f);
        }
        
        f.close();
        
        delete [] Name;
        
        */
        //S->Compute_D(eq);
        //S->Print_To_Screen_D();
        //S->Compute_EV();
        //S->Print_To_Screen_EV();
	//Init_Lattice(int dim, int l_type, int u_type, double u_param, double rho,double rc,int N_MC){
	//Lattice_Param LP;
	//LP.Init_Lattice(2,5,6,1,1,4,2332);
	//std::cout<<LP.Compute_cs()<<"\n";*/
	std::cout<<"Exit\n";							// zavershenie rabotu 
	return 0;
}
