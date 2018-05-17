from scipy import stats
from scipy.spatial import distance
import numpy as np
import os, io, sys
Matrix_location = "../../RMS/ab_numpy/"
Normalisation_location = "../../RMS/ab_numpy/data_norm/"
distances_list = [[17,27],[17,2],[2,89],[89,39],[39,57],[51,57],[48,54],[57,44],[44,89],[89,10],[10,27]]
angles_list = [[27,2,10],[17,10,2],[48,89,54],[89,48,51],[51,54,89],[51,48,57],[57,54,51]]
signerA = [1062,1907,1312,1203,1124,2330,1796,1286,1390]
signerB = [2136, 3941,2809,2547,2706,4234, 3621, 2614,3128]

def load_XYZ(matrix_location):
    files = sorted(
        [f for f in os.listdir(matrix_location) if f[-5] != 'T' and f[-1] == 'y'])  # sort file as above list order
    print(files)
    count = 0
    for np_file in files:
        data_file = Normalisation_location + np_file[:-4] + '.npy'
        theNP = (np.load(Matrix_location + np_file)).astype(float)  # Set data type to float for further calculations
        theNP[theNP==0] = 0.00001
        distances_Matrix = np.zeros(shape=[theNP.shape[1],11])
        angles_Matrix = np.zeros(shape=[theNP.shape[1],7])
        zz_Matrix = np.zeros(shape=[theNP.shape[1],100])
        vector_Matrix = np.zeros(shape=[theNP.shape[1],119])

        ###-----------------------------------------------------
        # GET DISTANCES
        ###-----------------------------------------------------

        NPrange = theNP.shape[1]
        for instance_no in range(0,NPrange):
            ii = 0
            for xx in distances_list:
                ux = theNP[0, instance_no, xx[0]]
                uy = theNP[1, instance_no, xx[0]]
                vx = theNP[0, instance_no, xx[1]]
                vy = theNP[1, instance_no, xx[1]]
                dis = distance.euclidean((ux,uy),(vx,vy))
                #print(ux)
                #print(uy)
                #print(vx)
                #print(vy)
                #print(dis)

                distances_Matrix[instance_no, ii] = float("{0:.4f}".format(dis))
                ii+=1

        ###-----------------------------------------------------
        # NORMALISE DISTANCES OF A & B BY Z-SCORE
        # signerA_DIS: distances_Matrix[0:signerA[count], :]
        # signerB_DIS: distances_Matrix[signerA[count]:signerB[count], :]
        # Z SCORE = [Matrix - mean(Matrix)] / sqrt(variance)
        #-------------------------------------------------------

        signerA_matrix = (distances_Matrix[0:signerA[count], :])
        signerB_matrix = (distances_Matrix[signerA[count]:signerB[count], :])

        for dis in range(0,11):
            distances_Matrix[0:signerA[count], dis] = stats.zscore(signerA_matrix[:,dis])
            distances_Matrix[signerA[count]:signerB[count], dis] = stats.zscore(signerB_matrix[:,dis])

        ###-----------------------------------------------------
        # GET ANGLES
        ###-----------------------------------------------------
        for instance_no in range(0, NPrange):
            jj=0
            for yy in angles_list:
                    #l1_V= [x_v1,y_v1] = [(x0−x1),(y0−y1)]
                    l1_v = np.array([(theNP[0,instance_no,yy[0]]-theNP[0,instance_no,yy[1]]),\
                                     (theNP[1,instance_no,yy[0]]-theNP[1,instance_no,yy[1]])])
                    #l2_V= [x_v2,y_v2] = [(x2−x1),(y2−y1)]
                    l2_v = np.array([(theNP[0,instance_no,yy[2]]-theNP[0,instance_no,yy[1]]),\
                                     (theNP[1,instance_no,yy[2]]-theNP[1,instance_no,yy[1]])])

                    l1l2_dot = np.dot(l1_v,l2_v)

                    angle = l1l2_dot/(np.linalg.norm(l1_v) * np.linalg.norm(l2_v))

                    angles_Matrix[instance_no,jj] = float("{0:.4f}".format(angle))
                    jj+=1
        ###-----------------------------------------------------
        # NORMALISE ANGLES OF A & B BY Z-SCORE
        # signerA_ANG: angles_Matrix[0:signerA[count], :]
        # signerB_ANG: angles_Matrix[signerA[count]:signerB[count], :]
        # Z SCORE = [Matrix - mean(Matrix)] / sqrt(variance)
        # -------------------------------------------------------

        signerA_matrix = (angles_Matrix[0:signerA[count], :])
        signerB_matrix = (angles_Matrix[signerA[count]:signerB[count], :])

        for ang in range(0, 7):
                angles_Matrix[0:signerA[count], ang] = stats.zscore(signerA_matrix[:, ang])
                angles_Matrix[signerA[count]:signerB[count], ang] = stats.zscore(signerB_matrix[:, ang])

        ###-----------------------------------------------------
        # CONCATENATE 11 NORM-DISTANCE, 7 ANGLES, 100 Z-AXIS
        ###-----------------------------------------------------
        zz_Matrix = theNP[2,:,:]
        signerA_matrix = (zz_Matrix[0:signerA[count], :])
        signerB_matrix = (zz_Matrix[signerA[count]:signerB[count], :])

        for zz in range(0, 100):
            zz_Matrix[0:signerA[count], zz] = stats.zscore(signerA_matrix[:, zz])
            zz_Matrix[signerA[count]:signerB[count], zz] = stats.zscore(signerB_matrix[:, zz])

        vector_Matrix[:,0:11] = distances_Matrix
        vector_Matrix[:,11:18] = angles_Matrix
        vector_Matrix[:,18:118] = zz_Matrix
        vector_Matrix[:,118] = np.load(Matrix_location + np_file[:-4] + '_T.npy')[:]
        #print(vector_Matrix.shape)

        vector_Matrix = np.asarray(vector_Matrix)
        save(data_file,vector_Matrix)
        count+=1

def dot(vec0, vec1):
    #return: x0*x1 + y0*y1
    return sum(vec0[0]*vec1[0],vec0[1]*vec1[1])



def save(data_file, vector_Matrix):
    np.save(data_file, vector_Matrix)
    print("Created********Normalisation X,Y*******" + data_file)

load_XYZ(Matrix_location)







