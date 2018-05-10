import numpy as np
import os, io, sys
Matrix_location = "../RMS/ab_numpy/"
Translation_location = "../RMS/ab_numpy/translation/"
signerA = [1062,1907,1312,1203,1124,2330,1796,1286,1390]
#signerB = [(1062,1062+1074),(1907,1907+2034),(1312,1312+1497),(1203,1203+1344),(1124,1124+1582),(2330,2330+1904),(1796,1796+1825),(1286,1286+1328),(1390,1390+1738)]
signerB = [2136, 3941,2809,2547,2706,4234, 3621, 2614,3128]
signerB2 = [1074,2034,1497,1344,1582,1904,1825,1328,1738]
leftEyeZone = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 87, 87, 90, 91, 92, 93, 94]
rightEyeZone = [8, 9, 10, 11, 12, 13, 14, 15, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 88, 95, 96, 97, 98, 99]
faceZone = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
            86, 89]

##--------------------------------------------
#
#
#theNP = (np.load(Matrix_location+'affirmative_datapoints.npy')).astype(float)
#signerA_NP = theNP[:,signerA[0][0]:signerA[0][1],:]
#signerB_NP = theNP[:,signerB[0][0]:signerB[0][1],:]
#
#
#
#
#
#
##----------------------------------------------

def load_XY(matrix_location):
    files = sorted([f for f in os.listdir(matrix_location) if f[-5] != 'T' and f[-1] == 'y']) #sort file as above list order
    print(files)
    count = 0
    for np_file in files:
        theNP = (np.load(Matrix_location+np_file)).astype(float) #Set data type to float for further calculations
        signerA_NP = theNP[:,0:signerA[count],:]
        signerB_NP = theNP[:,signerA[count]:signerB[count],:]
        print(signerA_NP.shape)
        print(signerB_NP.shape)
        #Do A Part
        A_count = 0
        while A_count < signerA[count]:
            average_noseA = average_nose(signerA_NP[0, A_count, 36:48], signerA_NP[1, A_count, 36:48])
            signerA_NP = np.ma.array(signerA_NP, mask=False)
            signerA_NP.mask[:, A_count, 36:48] = True
            signerA_NP[0, A_count, :] = signerA_NP[0, A_count, :] - average_noseA[0]  # SignerA's X
            signerA_NP[1, A_count, :] = signerA_NP[1, A_count, :] - average_noseA[1]  # SignerA's Y
            signerA_NP.mask[:, A_count, 36:48] = False
            A_count += 1


        #Do B Part
        B_count = 0
        while B_count < signerB2[count]:
            average_noseB = average_nose(signerB_NP[0,B_count,36:48],signerB_NP[1,B_count,36:48])
            signerB_NP = np.ma.array(signerB_NP, mask=False)
            signerB_NP.mask[:, B_count, 36:48] = True
            signerB_NP[0, B_count, :] = signerB_NP[0, B_count, :] - average_noseB[0] #SignerB's X
            signerB_NP[1, B_count, :] = signerB_NP[1, B_count, :] - average_noseB[1] #SignerB's Y
            signerB_NP.mask[:, B_count, 36:48] = False
            B_count += 1

        #Assign Segments of Matrix Back
        theNP[:, 0:signerA[count], :] = signerA_NP
        theNP[:, signerA[count]:signerB[count], :] = signerB_NP

        data_file = Translation_location + np_file[:-4] + '.npy'
        np.save(data_file, theNP)

        count += 1
        print("Created********TRANSLATION X,Y*******" + data_file)

        #NEXT:use the new x,y,z np to make normalisation

def average_nose(xx,yy):
    nose_x = np.average(xx)
    nose_y = np.average(yy)
    return [nose_x,nose_y]

load_XY(Matrix_location)