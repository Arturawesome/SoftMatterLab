'''
Name       : DataSetWorks.py

Author informaion: Artur D. Nasyrov
                   PhD student in BMSTU, Moscow, Russia
                   engineer, Research and Development Center
                   "Photonic and IK-technic"; "Soft Matter & Phys of Fluids Centre"

Description: Program file with classes for working with data


Technical description:
                        LammpsCoords class for reading trj file of particle
                        MakeDataSet class for creating the dataset for learning ANN in ModelForLjMd.py


Technology : ANN, OOP


'''
import numpy as np
import scipy as scipy
import pygame
import math

import torch
from torch import nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split



class LammpsCoords:
    # конструктуорвходных и  требуемых данных
    def __init__(self, string_content: str, required_data: str, path: str):
        self.str_content = string_content.strip().split(' ')          # содержание строк в ламмпс файле (ПР: ['id' 'type' 'x' 'y' 'z' 'vx' 'vy' 'vz'])
        self.req_data = required_data.strip().split(' ')              # содержание требуемых данных - (ПР: ['id' 'type' 'x' 'y'])
        self.path_file = path                                         # полное имя ламмпс файла
        self.n_part = None                                             # количество частиц в моделировании
        self.box = np.zeros((3, 2))                                   # размер области моделирования
        self.f = None                                                 # переменная для открытия и чтения файла
        self.req_ind_data = []                                        # список для формирования индексов требуемых данных
                                                                        # str_content: ['id' 'type' 'x' 'y' 'z' 'vx' 'vy' 'vz'] -> req_data ['id' 'type' 'x' 'y'] =>
                                                                        # => req_ind_data = [0, 1, 2, 3]
        #self.data = []

    def display_info(self):
        print("string_content: ", self.str_content, ';\n  required_data: ', self.req_data, ';\n path: ', self.path_file)


    def OpenTrjGetInfo(self, flag: int):
        self.f = open(self.path_file, "r")
        for i in range(len(self.str_content)):
            if self.str_content[i] in self.req_data:
                self.req_ind_data.append(i)
        print(self.req_ind_data)
        self.f.readline()                                                                        # ITEM: TIMESTEP
        self.f.readline()                                                                        # 0
        self.f.readline()                                                                        # ITEM: NUMBER OF ATOMS
        self.n_part = int(self.f.readline())
        self.f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
        for i in range(3):
            xyz_box = [float(item) for item in self.f.readline().split(' ')]
            self.box[i][0] = xyz_box[0]
            self.box[i][1] = xyz_box[1]
        self.f.readline()                                                                               # ITEM: ATOMS id ...
        if flag == 0:
            for i in range(self.n_part):
                self.f.readline()
            return self.box, self.n_part
        else:
            d = np.array([[float(item) for item in self.f.readline().strip().split(' ')[0:len(self.str_content)]] for i in range(self.n_part)])
            return [self.n_part, self.box, self.box, d[:, self.req_ind_data]]

    def CloseFile(self):
        self.f.close()

    def ReadLogFile(self):
        # read log file
        return 0

    def ReadState(self):
        self.f.readline()                                                                        # ITEM: TIMESTEP
        self.f.readline()                                                                        # 0
        self.f.readline()                                                                        # ITEM: NUMBER OF ATOMS
        self.n_part = int(self.f.readline())                                                                        # N parts
        self.f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
        for i in range(3):
            xyz_box = [float(item) for item in self.f.readline().split(' ')]
            self.box[i][0] = xyz_box[0]
            self.box[i][1] = xyz_box[1]
        self.f.readline()                                                                               # ITEM: ATOMS id ...
        data = np.array([[float(item) for item in self.f.readline().strip().split(' ')[0:len(self.str_content)]] for i in range(self.n_part)])

        return  [self.n_part, self.box, self.box, data[:, self.req_ind_data]]


'id tyope x y vx vy'
'id x y '




class MakeDataSet:
    def __init__(self, name_lmp_file: str, string_content: str, required_data: str, batch_size: int, num_cadr: int):
        self.name_lmp_file = name_lmp_file
        self.num_cadr = num_cadr
        self.xy_present = []
        self.xy_future = []
        self.string_content = string_content
        self.required_data = required_data
        self.batch_size = batch_size
        self.train_dataloader = []
        self.val_dataloader = []


    def CreateData(self, num_cadr):
        lmp = LammpsCoords(self, self.string_content, self.required_data, self.name_lmp_file)
        st = lmp.OpenTrjGetInfo(1)

        # xy should contain particle positions for all timesteps
        xy = np.array([[lmp.ReadState()] for cadr in range(num_cadr)])
        xy_present = xy[0: num_cadr - 1]
        xy_future = xy[1:num_cadr]

        xy_train_present, xy_val_present, xy_train_future, y_val_future = train_test_split(xy_present, xy_future, random_state=42)

        xy_train_present = torch.from_numpy(xy_train_present).to(torch.float32)
        xy_train_future = torch.from_numpy(xy_train_future).to(torch.float32)
        xy_val_present = torch.from_numpy(xy_val_present).to(torch.float32)
        y_val_future = torch.from_numpy(y_val_future).to(torch.float32)

        train_dataset = TensorDataset(xy_train_present, xy_train_future)
        val_dataset = TensorDataset(xy_val_present, y_val_future)
        self.train_dataloader = DataLoader(train_dataset, batch_size = self.batch_size)
        self.val_dataloader = DataLoader(val_dataset, batch_size = self.batch_size)

        return self.train_dataloader, self.val_dataloader





