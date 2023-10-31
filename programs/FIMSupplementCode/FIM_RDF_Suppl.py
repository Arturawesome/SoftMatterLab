 
# ============================================================================
# Name:
# FIM_RDF_Suppl.py

# Authors:
# 	N.P. Kryuchkov, A.D. Nasyrov

# Copyright:
# 	Please, cite the original work, if you use this program package.

# Description:
# 	Python program package for interolating RDF data through the temperature range.

# Requirements:
# 	numpy, matplotlib, scipy should be aviable

# Inputs:
#		!! Please see the README.md file for input file formats and descritpion for vartiables
# 	path_to_input_files (str) - full folder name where rdf data is located.
# 	name_rdf_ps, name_rdf_pf (str) - data names of known rdf functions, which is located in "path_to_input_files" folder
# 	Temp_ts, Temp_tm, Temp_tf (float) - temperatures of the systems where name_rdf_ps and name_rdf_ps are defined and additional point.
# 	name_r_grid (str) - name of distance grid file.
# 	name_m_smf, name_avr_smf etc - file names of Magnitude, sverage distance (Mathematical expectations) of the Correlation peaks.
#									This file must be located in "path_to_input_files" folder
#	save_path (str): full folder name where requirement rdf data should be saved
#		!!Note, that you do not set the name of save files, the name of save failes are generated internal FIM_RDF_Suppl.py!!
#	temperature_req (float) - temperature where RDF should be calculated the
# 	temp_range - array of temperatures
#	prove_flag : if you want to check correctness of FIM set True
#	name_prove - name of RDF data for check the FIM.
#				This file must be located in "path_to_input_files" folder

# ============================================================================


import numpy as np
import scipy as scipy
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class FluidInterpolatingMethod():
	def __init__(self,
				path_to_input_files: str,

				name_rdf_ps:str,
				name_rdf_pf:str,

				name_m_smf:str,
				name_avr_smf:str,
				name_sigma2_smf:str,
				name_alpha_smf:str,

				Temp_ts:float,
				Temp_tm:float,
				Temp_tf:float,
				name_r_grid:str,
				save_path:str,
				temperature_req:float,
				temp_range_grid: np.array,
				prove_flag:bool,
				name_prove:str
				):
		self.path_input = path_to_input_files
		self.name_rdf_ps = name_rdf_ps
		self.name_rdf_pf = name_rdf_pf

		self.name_m_smf = name_m_smf
		self.name_avr_smf = name_avr_smf
		self.name_sigma2_smf = name_sigma2_smf
		self.name_alpha_smf = name_alpha_smf

		self.temp_ts = Temp_ts
		self.temp_tm = Temp_tm
		self.temp_tf = Temp_tf

		self.name_r_grid = name_r_grid
		self.save_path = save_path
		self.temp_req = temperature_req
		self.prove_flag = prove_flag
		self.name_prove = name_prove
		self.temp_range_grid = temp_range_grid

	def __del__(self):
		print("Destructor called")

	def Read_RDF_data_all(self, name):
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
		return r_data


	def Plot_Fig2_Article(self):
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (15, 5))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				#axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTs_pTf[0, id_peak, :], "--", color = "black")
				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTs_pTf[0, id_peak, :], "--", color = "black")
				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTs_pTf[1, id_peak, :], "-", color = "grey")

				axs[row, col].plot(self.s_norm_dist[0, id_peak, :], self.pTreq[id_peak, :], "-", color = "orange")
				axs[row, col].set_xlim([-4, 4])

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + 'FIM_Ris2.pdf', bbox_inches='tight')
		fig.clear()
		plt.close(fig)
		return 0


	def Save_Txt_data(self):
		f = open(self.save_path + "FIM_Pi(sTs).txt", "w")
		for row in range(self.num_row):
			for id_peak in range(self.num_peaks):
				f.write(str(self.pTs_pTf[0, id_peak, row]) + " ")
			f.write("\n")
		f.close()

		f = open(self.save_path + "FIM_Pi(sTf).txt", "w")
		for row in range(self.num_row):
			for id_peak in range(self.num_peaks):
				f.write(str(self.pTs_pTf[1, id_peak, row]) + " ")
			f.write("\n")
		f.close()

		f = open(self.save_path + "FIM_S_for_TsTfTreq.txt", "w")
		for row in range(self.num_row):
			for id_peak in range(self.num_peaks):
				f.write(str(self.s_norm_dist[0, id_peak, row]) + " ")
			f.write("\n")
		f.close()

		f = open(self.save_path + "FIM_Pi(sTreq).txt", "w")
		for row in range(self.num_row):
			for id_peak in range(self.num_peaks):
				f.write(str(self.pTreq[id_peak, row]) + " ")
			f.write("\n")
		f.close()

		f = open(self.save_path + "FIM_RDF_g(rTreq).txt", "w")
		for row in range(self.num_row):
			f.write( str(self.r[row]) + " "+ str(self.rdf_req[-1, row]) + "\n")
		f.close()


	def Plot_Suppl(self, plot_type):
		# Magnitude of peaks plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 8))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col

				x_fit = self.temp_range_grid
				a_i, b_i, beta_i = self.a_b_beta_magnitude[id_peak, 0], self.a_b_beta_magnitude[id_peak, 1], self.a_b_beta_magnitude[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)

				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 0], "o", color = "red")
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path +"FIM_" + plot_type[0] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)

		# Average distance plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 8))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 1], "o", color = "red")

				x_fit = self.temp_range_grid
				a_i, b_i, beta_i = self.a_b_beta_avr[id_peak, 0], self.a_b_beta_avr[id_peak, 1], self.a_b_beta_avr[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + "FIM_" + plot_type[1] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)

		# The sigma2 values plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (11, 8))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 2], "o", color = "red")

				x_fit = self.temp_range_grid
				a_i, b_i, beta_i = self.a_b_beta_sigma2[id_peak, 0], self.a_b_beta_sigma2[id_peak, 1], self.a_b_beta_sigma2[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path + "FIM_" + plot_type[2] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)

		# The alpha values plot
		fig, axs = plt.subplots(nrows = 2, ncols = 5, figsize = (13, 3.5))
		for row in range(2):
			for col in range(5):
				id_peak = row * 5 + col
				axs[row, col].plot(self.temp_range, self.dat_pike[:, id_peak, 3], "o", color = "red")

				x_fit = self.temp_range_grid
				a_i, b_i, beta_i = self.a_b_beta_alpha[id_peak, 0], self.a_b_beta_alpha[id_peak, 1], self.a_b_beta_alpha[id_peak, 2]
				y_fit = self.Eq6_f_T(x_fit, a_i, b_i, beta_i)
				axs[row, col].plot(x_fit, y_fit, "-", color = "orange")
				axs[row, col].set_xscale("log")

		plt.tight_layout(pad = 0.5)
		plt.savefig(self.save_path +"FIM_" + plot_type[3] +  '.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)


	def Plot_Fig3(self):
		if self.prove_flag:
			rdf_md = np.zeros(len(self.rdf_for_prove[:, 1]))
			rdf_theory = np.zeros(len(self.rdf_for_prove[:, 1]))
		fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 5))

		for id_peak in range(self.num_peaks):
			if self.prove_flag:
				rdf_md +=  self.rdf_for_prove[:, id_peak]
				axs.plot(self.r,  self.rdf_for_prove[:, id_peak], "o", color = "grey")
			axs.plot(self.r, self.rdf_req[id_peak, :], "-", color = "orange")

		if self.prove_flag:
			axs.plot(self.r,rdf_md, ".", color = "grey")
		axs.plot(self.r, self.rdf_req[-1, :], "--", color = "red")
		axs.set_xlim([0, 10])
		plt.savefig(self.save_path + 'FIM_Ris3.pdf', bbox_inches='tight')#,pad_inches = 0)    #mh1
		fig.clear()
		plt.close(fig)


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
			self.a_b_beta_alpha = np.zeros((self.num_peaks, 3))
			for id_peak in range(self.num_peaks):
				beta_i_fisrt, _ = curve_fit(self.Eq6_f_T_1, (id_peak, self.temp_range[1], 3), [self.dat_pike[1, id_peak, 3]], maxfev = 10000000)
				a, b = self.Eq6_f_T_2((id_peak, self.temp_range[1], 3), beta_i_fisrt[0])
				self.a_b_beta_alpha[id_peak, :] = [a, b, beta_i_fisrt[0]]


	def Eq5_pi_s(self):
		self.s_norm_dist = np.zeros((2, self.num_peaks, len(self.r)))
		self.pTs_pTf = np.zeros((2, self.num_peaks, len(self.r)))

		for id_peak in range(self.num_peaks):
			self.s_norm_dist[0, id_peak, :] = (self.r - self.dat_pike[0, id_peak, 1]) / np.sqrt(self.dat_pike[0, id_peak, 2])
			x_new = self.s_norm_dist[0, id_peak, :]

			self.s_norm_dist[1, id_peak, :] = (self.r - self.dat_pike[2, id_peak, 1]) / np.sqrt(self.dat_pike[2, id_peak, 2])

			pTs = self.rdf_ps_pf[0][:, id_peak] * np.sqrt(self.dat_pike[0, id_peak, 2]) / self.dat_pike[0, id_peak, 0]
			pTf = self.rdf_ps_pf[1][:, id_peak] * np.sqrt(self.dat_pike[2, id_peak, 2]) / self.dat_pike[2, id_peak, 0]

			f1 = CubicSpline(self.s_norm_dist[0, id_peak, :], pTs)
			f2 = CubicSpline(self.s_norm_dist[1, id_peak, :], pTf)

			self.pTs_pTf[0, id_peak, :] = f1(x_new)
			self.pTs_pTf[1, id_peak, :] = f2(x_new)


	def Eq8_pi_sTreq(self):
		self.pTreq = np.zeros((self.num_peaks, self.num_row))
		for id_peak in range(self.num_peaks):
			a_i, b_i, betta_i = self.a_b_beta_alpha[id_peak, 0], self.a_b_beta_alpha[id_peak, 1], self.a_b_beta_alpha[id_peak, 2]
			alpha_i_t = self.Eq6_f_T(self.temp_req, a_i, b_i, betta_i)
			self.pTreq[id_peak, :] = alpha_i_t * self.pTs_pTf[0, id_peak, :] + (1 - alpha_i_t) *  self.pTs_pTf[1, id_peak, :]


	def Eq9_grt_rdf(self):
		self.rdf_req = np.zeros((self.num_peaks + 1, len(self.pTreq[1, :])) )

		for id_peak in range(self.num_peaks):
			a_im, b_im, betta_im = self.a_b_beta_magnitude[id_peak, 0], self.a_b_beta_magnitude[id_peak, 1], self.a_b_beta_magnitude[id_peak, 2]
			a_ir, b_ir, betta_ir = self.a_b_beta_avr[id_peak, 0], self.a_b_beta_avr[id_peak, 1], self.a_b_beta_avr[id_peak, 2]
			a_is, b_is, betta_is = self.a_b_beta_sigma2[id_peak, 0], self.a_b_beta_sigma2[id_peak, 1], self.a_b_beta_sigma2[id_peak, 2]
			a_ia, b_ia, betta_ia = self.a_b_beta_alpha[id_peak, 0], self.a_b_beta_alpha[id_peak, 1], self.a_b_beta_alpha[id_peak, 2]

			f_mi_T = self.Eq6_f_T(self.temp_req, a_im, b_im, betta_im)
			f_ri_T = self.Eq6_f_T(self.temp_req, a_ir, b_ir, betta_ir)
			f_sigma2i_T = self.Eq6_f_T(self.temp_req, a_is, b_is, betta_is)

			rdf_for_interpolation = (f_mi_T/np.sqrt(f_sigma2i_T)) * self.pTreq[id_peak, :]
			r_new = self.s_norm_dist[0, id_peak, :] * np.sqrt(f_sigma2i_T) + f_ri_T

			f1 = CubicSpline(r_new, rdf_for_interpolation)
			self.rdf_req[id_peak, :] = f1(self.r)
			self.rdf_req[-1, :] += self.rdf_req[id_peak, :]


	def Init_Variables(self):
		self.rdf_ps_pf = []	# list of Ps and Pf peaks. Set in Main_Loop
		self.s_norm_dist = None	# shape ant type in def Eq5_pi_s
		self.pTs_pTf = None	# shape ant type in def Eq5_pi_s
		self.pTreq = None	# shape and type in def Eq8_pi_sTreq
		self.rdf_req = None # shape and type in def Eq9_grt_rdf
		self.temp_range = [self.temp_ts, self.temp_tm, self.temp_tf]

		self.num_t = 3	# len of self.temp_range
		self.num_peaks = None	# shape and type in def Read_RDF_data
		self.dat_pike = None # self.dat_pike[id_temperature, id_peak, id_param], id_param: 0-Norm, 1-Avr, 2-sigma2, 3-alpha; Set in def Main_Loop
		self.r = None # shape and type describe in def Read_r_grid; self.r set in Main_Loop


		self.a_b_beta_magnitude = None	#shape and type in def Get_Fit. (like np.zeros((self.num_peaks, 3)))
		self.a_b_beta_avr = None	#shape and type in def Get_Fit. (like np.zeros((self.num_peaks, 3)))
		self.a_b_beta_sigma2 = None	#shape and type in def Get_Fit. (like np.zeros((self.num_peaks, 3)))
		self.a_b_beta_alpha = None	#shape and type in def Get_Fit. (like np.zeros((self.num_peaks, 3)))

		if self.prove_flag:
			self.rdf_for_prove = None


	def Main_Loop(self):

		self.Init_Variables()
		self.rdf_ps_pf.append(self.Read_RDF_data(self.name_rdf_ps, self.temp_ts))
		self.rdf_ps_pf.append(self.Read_RDF_data(self.name_rdf_pf, self.temp_tf))
		self.r = self.Read_r_grid()

		if self.prove_flag:
			self.rdf_for_prove = self.Read_RDF_data_all(self.name_prove)
		print("\n\nDownload RDF, Ps, Pf done; Download RDF, Ps, Pf done; Download RDF, Ps, Pf done; Download RDF, Ps, Pf done; Download RDF, Ps, Pf done; Download RDF, Ps, Pf done; \n\n")

		self.dat_pike = np.zeros((self.num_t, self.num_peaks, 4))	# self.dat_pike[id_temperature, id_peak, id_param], id_param: 0-Norm, 1-Avr, 2-sigma2, 3-alpha
		self.dat_pike[:,:, 0] = self.Read_m_avr_sigma_alpha(self.name_m_smf)
		self.dat_pike[:,:, 1] = self.Read_m_avr_sigma_alpha(self.name_avr_smf)
		self.dat_pike[:,:, 2] = self.Read_m_avr_sigma_alpha(self.name_sigma2_smf)
		self.dat_pike[:,:, 3] = self.Read_m_avr_sigma_alpha(self.name_alpha_smf)

		print("Download Norm, Avr, Sigma2, Alpha done; Download Norm, Avr, Sigma2, Alpha done; Download Norm, Avr, Sigma2, Alpha done \n")

		self.Get_Fit("magnitude")
		self.Get_Fit("avr")
		self.Get_Fit("sigma2")
		self.Get_Fit("alpha")

		print("Fit Norm, Avr, Sigma2, Alpha done;  Fit Norm, Avr, Sigma2, Alpha done;  Fit Norm, Avr, Sigma2, Alpha done; \n")

		self.Plot_Suppl(["norm", "avr", "sigma2", "alpha"])
		self.Eq5_pi_s()
		print("Eq.5:  Pi(s) done;  Eq.5:  Pi(s) done;  Eq.5:  Pi(s) done;  Eq.5:  Pi(s) done;  Eq.5:  Pi(s) done; \n")

		self.Eq8_pi_sTreq()
		print("Eq.8:  Pi(s, Treq) done;  Eq.8:  Pi(s, Treq) done;  Eq.8:  Pi(s, Treq) done;  Eq.8:  Pi(s, Treq) done; \n")

		self.Plot_Fig2_Article()
		self.Eq9_grt_rdf()
		print("Eq.9: g(r, Treq) done;  Eq.9: g(r, Treq) done;  Eq.9: g(r, Treq) done;  Eq.9: g(r, Treq) done; \n")

		self.Plot_Fig3()
		self.Save_Txt_data()


# =========================================================================================================
# =========================================================================================================
# =========================================================================================================
# =========================================================================================================
# =========================================================================================================



tempeprature = lambda k: np.exp(np.log(6) + (np.log(60) - np.log(6)) / 20 * k)
temp = tempeprature(range(-10, 51, 1))

FIM = FluidInterpolatingMethod(
		path_to_input_files = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/FIM/SupplProgram/InputData/",
		name_rdf_ps = "rdf_LJ_T_11.txt",
		name_rdf_pf = "rdf_LJ_T_31.txt",

		name_m_smf = "FIM_Norm_smf.txt",
		name_avr_smf = "FIM_avr_smf.txt",
		name_sigma2_smf = "FIM_sigma2_smf.txt",
		name_alpha_smf = "FIM_alpha_smf.txt",

		Temp_ts = temp[10],
		Temp_tm = temp[22],
		Temp_tf = temp[30],
		name_r_grid = "r_grid.txt",
		save_path = "/run/media/artur/543a1618-2147-4bd3-97bd-0dc42ad14e8b/FIM/SupplProgram/SaveData/",
		temperature_req = temp[17],
		temp_range_grid = temp[10:31:1],
		prove_flag = True,
		name_prove = "FINUL_pcf_LJ_Frenel_d_T_18NORM_Fren.txt"

		)
FIM.Main_Loop()






























