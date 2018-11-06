import sys

folder_address = "../../data/"
RMS_folder_A = "../../RMS/a/"
RMS_folder_B = "../../RMS/b/"
RMS_folder_AB = "../../RMS/ab/"
listA = set([])
listB = set([])
listAB = {}
type_list = ["aff","con","dou","emp","neg","rel","top","wh_","yn_"] # 9 Types of expressions

def read_files(folder_address):
    import os

    listA_data = sorted(set(
        [f for f in os.listdir(folder_address) if f[0]=='a' and f[-8:-6]=="in"]))
    listA_target = sorted(set(
        [f for f in os.listdir(folder_address) if f[0] == 'a' and f[-8:-6]=="ge"]))

    listB_data = sorted(set(
        [f for f in os.listdir(folder_address) if f[0] == 'b' and  f[-8:-6]=="in"]))
    listB_target = sorted(set(
        [f for f in os.listdir(folder_address) if f[0] == 'b' and f[-8:-6]=="ge"]))

    print("Start Folder RMS/a.....................")
    for xx1,xx2 in zip(listA_data,listA_target):
        print(xx1[2:5] + "- - - - - - "+ xx2[2:5])
        if xx1[2:5]== xx2[2:5]:
           add_Labels(xx1,xx2,RMS_folder_A)

    print("Start Folder RMS/b.....................")
    for yy1,yy2 in zip(listB_data,listB_target):
        print(yy1[2:5] + "- - - - - - "+ yy2[2:5])
        if yy1[2:5]== yy2[2:5]:
           add_Labels(yy1,yy2,RMS_folder_B)
    print("--------------------------------------------------------------------------------\n")

def add_Labels(data_file,target_file,RMS_folder):
    import io
    this_file = ""
    file_data = io.open(folder_address + data_file, "r")
    file_target = io.open(folder_address + target_file, "r")
    lines_1 = file_data.readlines()[1:]
    print("Data has "+ str(lines_1.__len__())+" lines")
    lines_2 = file_target.readlines()
    file_data.close()
    file_target.close()
    print("Target has " + str(lines_2.__len__()) + " lines")

    for l1, l2 in zip(lines_1,lines_2):
        this_file_name = RMS_folder + data_file
        this_file = open(this_file_name,"a+")
        temp_list = l1.split(" ")
        l1 = " ".join(temp_list[1:])
        line = l1.strip()+" "+l2
        this_file.write(line)
    this_file.close()
    print("Finished*********************"+data_file+"\n")

def CombineToOne():
    import os, io
    from shutil import copyfile
    a_loc = sorted(os.listdir(RMS_folder_A))
    a_loc.remove('.DS_Store') #Remove ".DS_Store file"/ Or comment this line
    b_loc = sorted(os.listdir(RMS_folder_B))
    b_loc.remove('.DS_Store') #Remove ".DS_Store file"/ Or comment this line
    #print(a_loc)
    #print(b_loc)
    #exit(0)
    for a, b in zip(a_loc,b_loc):
        c = RMS_folder_AB+a[2:]
        c_file = open(c, "a")
        copyfile(RMS_folder_A+a, c)
        print("Copy "+ a + "--to-->"+ c)
        bb = open(RMS_folder_B+b,"r")
        b_file = bb.read()
        c_file.write('\n')
        c_file.write(b_file)
        bb.close()
        c_file.close()
        print(a+"---->"+b+"===>"+a[2:])
        c_file = open(c, "r")
        print("_____________________Total: "+str(c_file.readlines().__len__())+ " lines")
        c_file.close()

read_files(folder_address)
CombineToOne()
