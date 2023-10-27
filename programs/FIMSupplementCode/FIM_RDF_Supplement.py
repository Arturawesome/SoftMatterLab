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
import h5py
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

class FluidInterpolatingMethod():
	def __init__(self,
				path_to_input_files: str,
				name_rdf1:str,
				name_rdf2:str,
				name_rdf3:str,
				temperature1:float,
				temperature2:float,
				temperature3:float,
				name_r_grid:str,
				save_path:str,
				temperature_req:float,
				prove_flag:bool,
				prove_temp_range = None
				):
		self.path_input = path_to_input_files
		self.name_rdf1 = name_rdf1
		self.name_rdf2 = name_rdf2
		self.name_rdf3 = name_rdf3
		self.temp1 = temperature1
		self.temp2 = temperature2
		self.temp3 = temperature3

		self.name_r_grid = name_r_grid
		self.save_path = save_path
		self.temp_req = temperature_req
		self.prove_flag = prove_flag
		self.prove_temp_range = prove_temp_range
	#def __del__(self)



	def Read_RDF_data_all(self, name, temp):
		full_name_rdf = self.path_input + name
		f = open(full_name_rdf, "r")
		data  = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(int(self.num_row))])
		f.close()
		return data

	def Read_RDF_data(self, name, temp):
		full_name_rdf = self.path_input + name
		f = open(full_name_rdf, "r")

		row_col_info = [float(item) for item in f.readline().strip().split(' ')]
		self.num_row, self.num_peaks = int(row_col_info[0]), int(row_col_info[1])
		data  = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(int(self.num_row))])
		f.close()
		return data

	def Read_m_avr_sigma_alpha(self, name):
		full_name_mrsa = self.path_input + name
		f = open(full_name_mrsa, "r")
		data  = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(int(self.num_t))])
		f.close()
		return data

	def Read_r_grid(self):
		full_name_r = self.path_input + self.name_r_grid

		f = open(full_name_r, "r")
		r_data = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(int(self.num_row))]).flatten()
		f.close

		self.dr = r_data[1] - r_data[0]
		print(full_name_r)
		print("dr = ", self.dr)
		return r_data

	def Plot_Fig2_Article(self):
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (15, 5))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col

				axs[row, col].plot(self.s_norm_dist_prove[4, id_peak, :],self.p_is[4, id_peak, :], "o", color = "blue")

				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTs_pTf[0, id_peak, :], "--", color = "black")
				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTs_pTf[0, id_peak, :], "--", color = "black")
				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTs_pTf[1, id_peak, :], "-", color = "grey")
				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.p_req[id_peak, :], "-", color = "orange")
				axs[row, col].set_xlim([-4, 4])


		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + 'Ris2.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)
		return 0

	def Plot_Suppl(self, plot_type):
		# Magnitude of peaks plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 8))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col

				x_fit = self.prove_temp_range
				a_i, b_i, beta_i = self.a_b_beta_magnitude[id_peak, 0], self.a_b_beta_magnitude[id_peak, 1], self.a_b_beta_magnitude[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)

				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 0], "o", color = "red")
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].plot(self.prove_temp_range, self.dat_pike_prove[:, id_peak, 0], ".")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + plot_type[0] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)

		# Average distance plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 8))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 1], "o", color = "red")

				x_fit = self.prove_temp_range
				a_i, b_i, beta_i = self.a_b_beta_avr[id_peak, 0], self.a_b_beta_avr[id_peak, 1], self.a_b_beta_avr[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].plot(self.prove_temp_range, self.dat_pike_prove[:, id_peak, 1], ".")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + plot_type[1] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)

		# The sigma2 values plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 8))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 2], "o", color = "red")

				x_fit = self.prove_temp_range
				a_i, b_i, beta_i = self.a_b_beta_sigma2[id_peak, 0], self.a_b_beta_sigma2[id_peak, 1], self.a_b_beta_sigma2[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].plot(self.prove_temp_range, self.dat_pike_prove[:, id_peak, 2], ".")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + plot_type[2] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1

		fig.clear()
		plt.close(fig)


		# The alpha values plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (13, 3.5))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 3], "o", color = "red")

				x_fit = self.prove_temp_range
				a_i, b_i, beta_i = self.a_b_beta_alpha[id_peak, 0], self.a_b_beta_alpha[id_peak, 1], self.a_b_beta_alpha[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].plot(self.prove_temp_range, self.dat_pike_prove[:, id_peak, 3], ".")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + plot_type[3] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1

		fig.clear()
		plt.close(fig)




	def Calc_Dat_Peaks(self):
		print("In Calc_Dat_Peaks")
		self.dat_pike = np.zeros((self.num_t, self.num_peaks, 4)) # self.dat_pike[id_temp, id_peak, id_param]
																	#id_param : 0-Norm; 1- avr; 2-sigma2; 3-alpha
		print(self.dat_pike.shape)
		if self.prove_flag:
			self.dat_pike_prove = np.zeros((len(self.prove_temp_range), self.num_peaks, 4)) # self.dat_pike[id_temp, id_peak, id_param]

		for id_peak in range(self.num_peaks):
			for id_temp in range(self.num_t):

				norm = sum(self.rdf123[id_temp][:, id_peak] * self.dr)
				self.dat_pike[id_temp, id_peak, 0] = norm

				av_r = sum(self.r * self.rdf123[id_temp][:, id_peak] * self.dr) / norm
				self.dat_pike[id_temp, id_peak, 1] = av_r
				if id_temp == 0 and id_peak == 0:
					print(av_r)

				sigma2 = sum((self.r - av_r) * (self.r - av_r) * self.rdf123[id_temp][:, id_peak] * self.dr) / norm
				self.dat_pike[id_temp, id_peak, 2] = sigma2



			if self.prove_flag:
				for id_temp in range(len(self.prove_temp_range)):

					norm = sum(self.rdf_prove_all_temp[id_temp][:, id_peak] * self.dr)
					self.dat_pike_prove[id_temp, id_peak, 0] = norm

					av_r = sum(self.r * self.rdf_prove_all_temp[id_temp][:, id_peak] * self.dr) / norm
					self.dat_pike_prove[id_temp, id_peak, 1] = av_r
					if id_temp == 0 and id_peak == 0:
						print(av_r)

					sigma2 = sum((self.r - av_r) * (self.r - av_r) * self.rdf_prove_all_temp[id_temp][:, id_peak] * self.dr) / norm
					self.dat_pike_prove[id_temp, id_peak, 2] = sigma2
				full_name_mrsa = self.path_input + "alpha_all.txt"
				f = open(full_name_mrsa, "r")
				self.dat_pike_prove[:, :, 3]  = np.array([[float(item) for item in f.readline().strip().split(' ')] for i in range(21)])
				f.close()


		f = open(self.path_input + "moments_smf.txt", "w")
		for id_peak in range(self.num_peaks):
			for id_temp in range(self.num_t):
				f.write(str(self.dat_pike[id_temp, id_peak, 0]) + " ")
			f.write("\n")
		f.close()

		f = open(self.path_input + "avr_smf.txt", "w")
		for id_peak in range(self.num_peaks):
			for id_temp in range(self.num_t):
				f.write(str(self.dat_pike[id_temp, id_peak, 1]) + " ")
			f.write("\n")
		f.close()

		f = open(self.path_input + "sigma2_smf.txt", "w")
		for id_temp in range(self.num_t):
			for id_peak in range(self.num_peaks):
				f.write(str(self.dat_pike[id_temp, id_peak, 2]) + " ")
			f.write("\n")
		f.close()


	def Eq6_f_T(self, x_temp, a_i, b_i, beta_i):
		return a_i + b_i * x_temp **(beta_i)

	def Eq6_f_T_1(self, id_peak_x, beta_i):
		id_peak = int(id_peak_x[0])
		x_temp = id_peak_x[1]
		fit_type = int(id_peak_x[2])

		dataTs = self.dat_pike[0, id_peak, fit_type]
		dataTf = self.dat_pike[2, id_peak, fit_type]
		Tf = self.temp_range[2]
		Ts = self.temp_range[0]
		#sigmaTm = dataTs + (dataTf - dataTs)/(Tf ** beta_i - Ts ** beta_i) * (x_temp ** beta_i - Ts ** beta_i)
		return dataTs + (dataTf - dataTs)/(Tf ** beta_i - Ts ** beta_i) * (x_temp ** beta_i - Ts ** beta_i)

	def Eq6_f_T_2(self, id_peak_x, beta_i):
		id_peak = int(id_peak_x[0])
		x_temp = id_peak_x
		fit_type = int(id_peak_x[2])

		dataTs = self.dat_pike[0, id_peak, fit_type]
		dataTf = self.dat_pike[2, id_peak, fit_type]

		Tf = self.temp_range[2]
		Ts = self.temp_range[0]

		b = (dataTf - dataTs)/(Tf ** beta_i - Ts ** beta_i)
		a = dataTs - b * (Ts ** beta_i)
		return a, b


	def Get_Fit(self, fit_type:str):
		if fit_type == "magnitude":
			self.a_b_beta_magnitude = np.zeros((self.num_peaks, 3))
			for id_peak in range(self.num_peaks):
				beta_i_fisrt, _ = curve_fit(self.Eq6_f_T_1, (id_peak, self.temp_range[1], 0), [self.dat_pike[1, id_peak, 0]], maxfev = 10000000)
				a, b = self.Eq6_f_T_2((id_peak, self.temp_range[1], 0), beta_i_fisrt[0])
				self.a_b_beta_magnitude[id_peak, :] = [a, b, beta_i_fisrt[0]]

		elif fit_type == "avr":
			self.a_b_beta_avr = np.zeros((self.num_peaks, 3))
			for id_peak in range(self.num_peaks):
				beta_i_fisrt, _ = curve_fit(self.Eq6_f_T_1, (id_peak, self.temp_range[1], 1), [self.dat_pike[1, id_peak, 1]], maxfev = 10000000)
				a, b = self.Eq6_f_T_2((id_peak, self.temp_range[1], 1), beta_i_fisrt[0])
				self.a_b_beta_avr[id_peak, :] = [a, b, beta_i_fisrt[0]]

		elif fit_type == "sigma2":
			self.a_b_beta_sigma2 = np.zeros((self.num_peaks, 3))
			for id_peak in range(self.num_peaks):
				beta_i_fisrt, _ = curve_fit(self.Eq6_f_T_1, (id_peak, self.temp_range[1], 2), [self.dat_pike[1, id_peak, 2]], maxfev = 10000000)
				a, b = self.Eq6_f_T_2((id_peak, self.temp_range[1], 2), beta_i_fisrt[0])
				self.a_b_beta_sigma2[id_peak, :] = [a, b, beta_i_fisrt[0]]

		elif fit_type == "alpha":
			print("in alpha")
			self.a_b_beta_alpha = np.zeros((self.num_peaks, 3))
			for id_peak in range(self.num_peaks):
				beta_i_fisrt, _ = curve_fit(self.Eq6_f_T_1, (id_peak, self.temp_range[1], 3), [self.dat_pike[1, id_peak, 3]], maxfev = 10000000)
				a, b = self.Eq6_f_T_2((id_peak, self.temp_range[1], 3), beta_i_fisrt[0])
				self.a_b_beta_alpha[id_peak, :] = [a, b, beta_i_fisrt[0]]


	def Eq5_pi_s(self):
		#                        2 -> Ts, Tf
		self.s_norm_dist = np.zeros((2, self.num_peaks, len(self.r)))
		self.pTs_pTf = np.zeros((2, self.num_peaks, len(self.r)))


		for id_peak in range(self.num_peaks):


			self.s_norm_dist[0, id_peak, :] = (self.r - self.dat_pike[0, id_peak, 1]) / np.sqrt(self.dat_pike[0, id_peak, 2])
			x_new = self.s_norm_dist[0, id_peak, :]

			self.s_norm_dist[1, id_peak, :] = (self.r - self.dat_pike[2, id_peak, 1]) / np.sqrt(self.dat_pike[2, id_peak, 2])

			pTs = self.rdf123[0][:, id_peak] * np.sqrt(self.dat_pike[0, id_peak, 2]) / self.dat_pike[0, id_peak, 0]
			pTf = self.rdf123[2][:, id_peak] * np.sqrt(self.dat_pike[2, id_peak, 2]) / self.dat_pike[2, id_peak, 0]

			f1 = CubicSpline(self.s_norm_dist[0, id_peak, :], pTs)
			f2 = CubicSpline(self.s_norm_dist[1, id_peak, :], pTf)

			self.pTs_pTf[0, id_peak, :] = f1(x_new)#self.rdf123[0][:, id_peak] * np.sqrt(self.dat_pike[0, id_peak, 2]) / self.dat_pike[0, id_peak, 0]
			self.pTs_pTf[1, id_peak, :] = f2(x_new)#self.rdf123[2][:, id_peak] * np.sqrt(self.dat_pike[2, id_peak, 2]) / self.dat_pike[2, id_peak, 0]

		if self.prove_flag:
			self.s_norm_dist_prove = np.zeros((len(self.prove_temp_range), self.num_peaks, len(self.r)))
			self.p_is = np.zeros((len(self.prove_temp_range), self.num_peaks, len(self.r)))

			for id_temp in range(len(self.prove_temp_range)):
				for id_peak in range(self.num_peaks):

					self.s_norm_dist_prove[id_temp, id_peak, :] = (self.r - self.dat_pike_prove[id_temp, id_peak, 1]) / np.sqrt(self.dat_pike_prove[id_temp, id_peak, 2])
					self.p_is[id_temp, id_peak, :] = self.rdf_prove_all_temp[id_temp][:, id_peak]* np.sqrt(self.dat_pike_prove[id_temp, id_peak, 2]) / self.dat_pike_prove[id_temp, id_peak, 0]

		return 0


	def Eq8_pi_sTreq(self):
		# self.dat_pike mi avr, sigma
		# self.rdf123[:, id_peak]
		self.p_req = np.zeros((self.num_peaks, self.num_row))
		print(self.p_req.shape)
		print(self.p_req.shape)
		for id_peak in range(self.num_peaks):
			a_i, b_i, betta_i = self.a_b_beta_alpha[id_peak, 0], self.a_b_beta_alpha[id_peak, 1], self.a_b_beta_alpha[id_peak, 2]
			alpha_i_t = self.Eq6_f_T(self.temp_req, a_i, b_i, betta_i)
			#									 p_i(s, Ts)									p_i(s, Tf)
			#psm = alpha_i_t * self.pTs_pTf[0, id_peak, :] + (1 - alpha_i_t) *  self.pTs_pTf[1, id_peak, :]
			#print(psm.shape)
			self.p_req[id_peak, :] = alpha_i_t * self.pTs_pTf[0, id_peak, :] + (1 - alpha_i_t) *  self.pTs_pTf[1, id_peak, :]


	def Init_Variables(self):
		self.rdf123 = []
		self.s_norm_dist = None
		self.pTs_pTf = None
		self.pTreq = None
		self.temp_range = [self.temp1, self.temp2, self.temp3]
		#self.rdf1, self.rdf2, self.rdf3 = None, None, None
		self.num_t = 3
		self.num_peaks = None
		self.dat_pike = None # self.dat_pike[id_temp, id_peak,id_param]
		self.r = None
		self.dr = None

		self.a_b_beta_magnitude = None
		self.a_b_beta_avr = None
		self.a_b_beta_sigma2 = None
		self.a_b_beta_alpha = None

		if self.prove_flag:
			self.rdf_prove_all_temp = []
			self.dat_pike_all = None




	def Main_Loop(self):

		self.Init_Variables()
		self.rdf123.append(self.Read_RDF_data(self.name_rdf1, self.temp1))
		self.rdf123.append(self.Read_RDF_data(self.name_rdf2, self.temp2))
		self.rdf123.append(self.Read_RDF_data(self.name_rdf3, self.temp3))
		self.r = self.Read_r_grid()

		#self.Read_m_avr_sigma_alpha(self, self.name_m_smf)
		#self.Read_m_avr_sigma_alpha(self, self.name_avr_smf)
		#self.Read_m_avr_sigma_alpha(self, self.name_sigma2_smf)




		if self.prove_flag:
			for id_temp in range(11, 32, 1):
				self.rdf_prove_all_temp.append(self.Read_RDF_data_all("FINUL_pcf_LJ_Frenel_d_T_" + str(id_temp) + "NORM_Fren.txt", id_temp))
		print("\n\nDownload done; Download done; Download done; Download done; Download done; Download done; \n\n")

		self.Calc_Dat_Peaks()

		data = self.Read_m_avr_sigma_alpha("alpha_smf.txt")
		self.dat_pike[:,:, 3] = data # yes, it is possible

		self.Get_Fit("magnitude")
		self.Get_Fit("avr")
		self.Get_Fit("sigma2")
		self.Get_Fit("alpha")

		self.Plot_Suppl(["norm", "avr", "sigma2", "alpha"])
		self.Eq5_pi_s()
		self.Eq8_pi_sTreq()
		self.Plot_Fig2_Article()













tempeprature = lambda k: np.exp(np.log(6) + (np.log(60) - np.log(6)) / 20 * k)
temp = []
for k in range(-10, 51, 1):
    temp.append(tempeprature(k))
print(temp[10])
print(temp[22])
print(temp[30])
print(temp[14])
all_temp = tempeprature( range(-10, 51, 1))
# r = np.arange(0.03, 30 + 0.03, 0.03)
# print(r)
# f = open("/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/FIM/SupplProgram/InputData/r_setka.txt", "w")
# for _ in range(len(r)):
# 	f.write(str(r[_])+"\n")
# f.close()

FIM = FluidInterpolatingMethod(
		path_to_input_files = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/FIM/SupplProgram/InputData/",
		name_rdf1 = "rdf_LJ_T_11.txt",
		name_rdf2 = "rdf_LJ_T_23.txt",
		name_rdf3 = "rdf_LJ_T_31.txt",
		temperature1 = temp[10],
		temperature2 = temp[22],
		temperature3 = temp[30],
		name_r_grid = "r_grid.txt",
		save_path = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/FIM/SupplProgram/SaveData/",
		temperature_req = temp[14],
		prove_flag = True,
		prove_temp_range = temp[10:31:1]
		)
FIM.Main_Loop()










































