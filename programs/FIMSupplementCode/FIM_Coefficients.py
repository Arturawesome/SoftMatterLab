 
# ============================================================================
# Name:
# FIM_RDF_Supplement.py

# Authors:
# 	N.P. Kryuchkov, A.D. Nasyrov

# Copyright:
# 	Please, cite the original work, if you use this program package.

# Description:
# 	Python program package for interolating RDF data through the temperature range.

# Requirements:
# 	numpy, matplotlib, scipy are aviable

# Inputs:
#		!! Please see the README.md file for input file formats
# 	path_to_input_files (str) - full folder name where rdf data is located.
# 	name_rdf1, name_rdf2, name_rdf3 (str) - data names of rdf functions, which is located in "path_to_input_files" folder
# 	temperature1, temperature2, temperature3 (float) - temperatures of the systems where rfd1, rfd2, rfd3 are defined.
# 	name_r_grid (str) - name of distance grid file.
#	save_path (str): full folder name where requirement rdf data should be saved
#		!!Note, that you do not set the name of save file, the name of save failes are generated internal FIM_RDF_Supplement.py!!
#	temperature_req (float) - temperature where should be calculated the RDF

# ============================================================================



import numpy as np
import scipy as scipy
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import scipy.optimize as opt

class FluidInterpolatingMethod():
	def __init__(self,
				path_to_input_files: str,
				name_rdf1:str,
				name_rdf2:str,
				name_rdf3:str,

				name_norm:str,
				name_avr:str,
				name_sigma2:str,
				name_alpha:str,

				temperature1:float,
				temperature2:float,
				temperature3:float,
				name_r_grid:str,
				save_path:str,
				temp_range_grid = None
				):
		self.path_input = path_to_input_files
		self.name_rdf1 = name_rdf1
		self.name_rdf2 = name_rdf2
		self.name_rdf3 = name_rdf3

		self.name_norm =name_norm
		self.name_avr =name_avr
		self.name_sigma2 = name_sigma2
		self.name_alpha = name_alpha

		self.temp1 = temperature1
		self.temp2 = temperature2
		self.temp3 = temperature3

		self.name_r_grid = name_r_grid
		self.save_path = save_path
		self.temp_range_grid = temp_range_grid

	def __del__(self):
		print("Destructor called")


	def Read_RDF_data(self, name, temp):
		full_name_rdf = self.path_input + name
		f = open(full_name_rdf, "r")

		row_col_info = [float(item) for item in f.readline().strip().split(' ')]
		self.num_row, self.num_peaks = int(row_col_info[0]), int(row_col_info[1])
		data  = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(int(self.num_row))])
		f.close()
		return data


	def Read_r_grid(self):
		full_name_r = self.path_input + self.name_r_grid
		f = open(full_name_r, "r")
		r_data = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(int(self.num_row))]).flatten()
		f.close
		self.dr = r_data[1] - r_data[0]
		return r_data


	def Save_txt(self, data, name):
		f = open(self.save_path + name, "w")
		for id_temp in range(self.num_t):
			for id_peak in range(self.num_peaks):
				f.write(str(data[id_temp, id_peak]) + " ")
			f.write("\n")
		f.close()


	def Calc_Dat_Peaks(self):
		print("In Calc_Dat_Peaks")
		self.dat_pike = np.zeros((self.num_t, self.num_peaks, 4)) # self.dat_pike[id_temp, id_peak, id_param]
																	#id_param : 0-Norm; 1- avr; 2-sigma2; 3-alpha
		for id_peak in range(self.num_peaks):
			for id_temp in range(self.num_t):
				norm = sum(self.rdf123[id_temp][:, id_peak] * self.dr)
				self.dat_pike[id_temp, id_peak, 0] = norm

				av_r = sum(self.r * self.rdf123[id_temp][:, id_peak] * self.dr) / norm
				self.dat_pike[id_temp, id_peak, 1] = av_r

				sigma2 = sum((self.r - av_r) * (self.r - av_r) * self.rdf123[id_temp][:, id_peak] * self.dr) / norm
				self.dat_pike[id_temp, id_peak, 2] = sigma2


	def Alpha_Minimization(self, alpha_Tm):
		return sum((alpha_Tm * self.p_iTs_alpha + (1 - alpha_Tm) * self.p_iTf_alpha - self.p_iTm_alpha) ** 2)


	def Find_Alpha_Depend(self, id_peak):
		self.p_iTs_alpha = self.pTs_pTm_pTf[0, id_peak, :]
		self.p_iTm_alpha = self.pTs_pTm_pTf[1, id_peak, :]
		self.p_iTf_alpha = self.pTs_pTm_pTf[2, id_peak, :]
		alpha_i_Tm = opt.minimize(self.Alpha_Minimization, x0=1, method='SLSQP', options={"maxiter":100000000})
		return alpha_i_Tm


	def Eq5_pi_s(self):
		self.s_norm_dist = np.zeros((3, self.num_peaks, len(self.r)))
		self.pTs_pTm_pTf = np.zeros((3, self.num_peaks, len(self.r)))

		for id_peak in range(self.num_peaks):
			self.s_norm_dist[0, id_peak, :] = (self.r - self.dat_pike[0, id_peak, 1]) / np.sqrt(self.dat_pike[0, id_peak, 2])
			self.s_norm_dist[1, id_peak, :] = (self.r - self.dat_pike[1, id_peak, 1]) / np.sqrt(self.dat_pike[1, id_peak, 2])
			self.s_norm_dist[2, id_peak, :] = (self.r - self.dat_pike[2, id_peak, 1]) / np.sqrt(self.dat_pike[2, id_peak, 2])
			x_new = self.s_norm_dist[0, id_peak, :]

			pTs = self.rdf123[0][:, id_peak] * np.sqrt(self.dat_pike[0, id_peak, 2]) / self.dat_pike[0, id_peak, 0]
			pTm = self.rdf123[1][:, id_peak] * np.sqrt(self.dat_pike[1, id_peak, 2]) / self.dat_pike[1, id_peak, 0]
			pTf = self.rdf123[2][:, id_peak] * np.sqrt(self.dat_pike[2, id_peak, 2]) / self.dat_pike[2, id_peak, 0]

			f1 = CubicSpline(self.s_norm_dist[0, id_peak, :], pTs)
			f2 = CubicSpline(self.s_norm_dist[1, id_peak, :], pTm)
			f3 = CubicSpline(self.s_norm_dist[2, id_peak, :], pTf)

			self.pTs_pTm_pTf[0, id_peak, :] = f1(x_new)
			self.pTs_pTm_pTf[1, id_peak, :] = f2(x_new)
			self.pTs_pTm_pTf[2, id_peak, :] = f3(x_new)

			self.s_norm_dist[0, id_peak, :] = x_new
			self.s_norm_dist[1, id_peak, :] = x_new
			self.s_norm_dist[2, id_peak, :] = x_new

			alpha_res = self.Find_Alpha_Depend(id_peak)
			self.dat_pike[:, id_peak, 3] = np.array([1, alpha_res.x[0], 0] )

	def Init_Variables(self):
		self.rdf123 = []
		self.s_norm_dist = None
		self.pTs_pTm_pTf = None
		self.temp_range = [self.temp1, self.temp2, self.temp3]
		self.num_t = 3
		self.num_peaks = None
		self.dat_pike = None # self.dat_pike[id_temp, id_peak,id_param]
		self.r = None
		self.dr = None


	def Main_Loop(self):
		self.Init_Variables()

		self.rdf123.append(self.Read_RDF_data(self.name_rdf1, self.temp1))
		self.rdf123.append(self.Read_RDF_data(self.name_rdf2, self.temp2))
		self.rdf123.append(self.Read_RDF_data(self.name_rdf3, self.temp3))
		self.r = self.Read_r_grid()

		self.Calc_Dat_Peaks()
		print("Calc Norm, Avr, Sigma2 done;  Calc Norm, Avr, Sigma2 done;  Calc Norm, Avr, Sigma2 done; \n")
		self.Eq5_pi_s()
		print("Calc Alpha done;  Calc Alpha done;  Calc Alpha done;  Calc Alpha done;  Calc Alpha done; \n")

		self.Save_txt(self.dat_pike[:, :, 0], self.name_norm)
		self.Save_txt(self.dat_pike[:, :, 1], self.name_avr)
		self.Save_txt(self.dat_pike[:, :, 2], self.name_sigma2)
		self.Save_txt(self.dat_pike[:, :, 3], self.name_alpha)
		print("Save done; Save done; Save done; Save done; Save done; Save done; Save done; Save done; \n")


tempeprature = lambda k: np.exp(np.log(6) + (np.log(60) - np.log(6)) / 20 * k)
temp = tempeprature(range(-10, 51, 1))

FIM = FluidInterpolatingMethod(
		path_to_input_files = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/FIM/SupplProgram/InputData/",
		name_rdf1 = "rdf_LJ_T_11.txt",
		name_rdf2 = "rdf_LJ_T_23.txt",
		name_rdf3 = "rdf_LJ_T_31.txt",

		name_norm = "FIM_Norm_smf.txt",
		name_avr = "FIM_avr_smf.txt",
		name_sigma2 = "FIM_sigma2_smf.txt",
		name_alpha = "FIM_alpha_smf.txt",

		temperature1 = temp[10],
		temperature2 = temp[22],
		temperature3 = temp[30],
		name_r_grid = "r_grid.txt",
		save_path = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/FIM/SupplProgram/SaveData/",

		temp_range_grid = temp[10:31:1]
		)
FIM.Main_Loop()










































