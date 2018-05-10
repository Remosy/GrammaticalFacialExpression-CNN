import numpy as np
import os, io, sys
RMS_folder_AB = "../../RMS/ab/"
Matrix_location = "../../RMS/ab_numpy/"
X_map = []
Y_map = []
Z_map = []

def createSavableMatrix():
    files = os.listdir(RMS_folder_AB)
    files.remove('.DS_Store')  # Remove ".DS_Store file"/ Or comment this line
    X_map = [x for x in range(0, 299, 3)]
    Y_map = [y for y in range(1, 299, 3)]
    Z_map = [z for z in range(2, 300, 3)]

    for xx in files:
        with open(RMS_folder_AB+xx) as xx_file:
            x_dim = []
            y_dim = []
            z_dim = []
            targets_matrix = []

            for line in xx_file:
                 temp_array = line.split(" ")
                 x_dim.append([float(temp_array[x].strip()) for x in X_map])
                 y_dim.append([float(temp_array[y].strip()) for y in Y_map])
                 z_dim.append([float(temp_array[z].strip()) for z in Z_map])
                 #print(x_dim)
                 targets_matrix.append(int(float(temp_array[300].strip())))
                 #print(int(float(temp_array[300].rstrip())))
                 #exit(0)
                 #print(x_dim)
                 #print(y_dim)
                 #print(z_dim)
                 #exit(0)
                 #targets_matrix.append(int(temp_array[300].strip()))


            #exit(0)
            XYZ_matrix = np.array([
                 x_dim,
                 y_dim,
                 z_dim
             ]).astype(float)

            #print(XYZ_matrix.shape)
            data_file = Matrix_location+xx[:-4]+'.npy'
            target_file = Matrix_location + xx[:-4] + '_T.npy'
            np.save(data_file, XYZ_matrix)
            np.save(target_file,targets_matrix)


        print("Created********SAVABLE NUMPY*******"+data_file)


createSavableMatrix()