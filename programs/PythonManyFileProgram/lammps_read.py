import numpy as np

class Lammps_coords:
    # конструктуорвходных и  требуемых данных
    def __init__(self, string_content: str, required_data: str, path: str):
        self.__str_content = string_content.strip().split(' ')          # содержание строк в ламмпс файле (ПР: ['id' 'type' 'x' 'y' 'z' 'vx' 'vy' 'vz'])
        self.__req_data = required_data.strip().split(' ')              # содержание требуемых данных - (ПР: ['id' 'type' 'x' 'y'])
        self.__path_file = path                                         # полное имя ламмпс файла
        self.__Npart = None                                             # количество частиц в моделировании
        self.__box = np.zeros((3, 2))                                   # размер области моделирования
        self.__f = None                                                 # переменная для открытия и чтения файла
        self.__req_ind_data = []                                        # список для формирования индексов требуемых данных
                                                                        # __str_content: ['id' 'type' 'x' 'y' 'z' 'vx' 'vy' 'vz'] -> __req_data ['id' 'type' 'x' 'y'] =>
                                                                        # => __req_ind_data = [0, 1, 2, 3]
        #self.data = []

    def display_info(self):
        print("string_content: ", self.__str_content, ';\n  required_data: ', self.__req_data, ';\n path: ', self.__path_file)


    def open_file_get_md_inf(self, flag):
        self.__f = open(self.__path_file, "r")
        for i in range(len(self.__str_content)):
            if self.__str_content[i] in self.__req_data:
                self.__req_ind_data.append(i)
        print(self.__req_ind_data)
        self.__f.readline()                                                                        # ITEM: TIMESTEP
        self.__f.readline()                                                                        # 0
        self.__f.readline()                                                                        # ITEM: NUMBER OF ATOMS
        self.__Npart = int(self.__f.readline())
        self.__f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
        for i in range(3):
            xyz_box = [float(item) for item in self.__f.readline().split(' ')]
            self.__box[i][0] = xyz_box[0]
            self.__box[i][1] = xyz_box[1]
        self.__f.readline()                                                                               # ITEM: ATOMS id ...
        if flag == 0:
            for i in range(self.__Npart):
                self.__f.readline()
            return self.__box, self.__Npart
        else:
            d = np.array([[float(item) for item in self.__f.readline().strip().split(' ')[0:len(self.__str_content)]] for i in range(self.__Npart)])
            return d[:, self.__req_ind_data],  self.__box, self.__Npart

    def close_file(self):
        self.__f.close()

    def read_state(self) -> np.ndarray:
        self.__f.readline()                                                                        # ITEM: TIMESTEP
        self.__f.readline()                                                                        # 0
        self.__f.readline()                                                                        # ITEM: NUMBER OF ATOMS
        self.__f.readline()                                                                        # N parts
        self.__f.readline()                                                                        # ITEM: BOX BOUNDS pp pp pp
        self.__f.readline()                                                                        # xmin xmax
        self.__f.readline()                                                                        # ymin ymax
        self.__f.readline()                                                                        # zmin zmax
        self.__f.readline()                                                                               # ITEM: ATOMS id ...
        data = np.array([[float(item) for item in self.__f.readline().strip().split(' ')[0:len(self.__str_content)]] for i in range(self.__Npart)])

        return  data[:, self.__req_ind_data]


'id tyope x y vx vy'
'id x y '




