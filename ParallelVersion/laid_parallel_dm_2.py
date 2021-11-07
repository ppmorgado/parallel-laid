#!/usr/bin/python3
# coding: utf-8
"""

Laid Parallel 2021 version 

Usage:  save this script and run

    $python 
    mpirun python 

Depend of DM guidelines

2021 Paulo Morgado
Based or adapted from previous work:
Apolónia, João, and Luís Cavique. 2019. “Seleção de Atributos de Dados Inconsistentes Em Ambiente HDF5 + Python Na Cloud INCD.” Revista de Ciências da Computação (14): 85–112.
Cavique, Luís, Armando B. Mendes, Hugo F.M.C. Martiniano, and Luís Correia. 2018. “A Biobjective Feature Selection Algorithm for Large Omics Datasets.” In Expert Systems,.

Notes:
All required information will be present on config.json file that must stored in your current working directory, and should point to original dataset on HDF5 format

Tested under: Python 3.9.4 

Last updated: 2021-08-27
include Buffer for reduce write operations

"""

import sys
import os.path
import time
import math
import numpy as np
import h5py
from mpi4py import MPI
# import json

def read_config(mydir):

    myfile = 'config.json'
    file_path = os.path.join(mydir, myfile)
    print (file_path)    

    with open(file_path, 'r+') as f:
        content = f.read().strip() # read content from file and remove whitespaces around
        tuples = eval(content)     # convert string format tuple to original tuple object (not possible using json.loads())

    return tuples  # List of tuples


def main():

    task_id = MPI.COMM_WORLD.rank                # The process ID (integer 0-3 for 4-process run)
    worker_nodes = MPI.COMM_WORLD.Get_size()     # Number of WorkerNodes/tasks valid are 1, 2 or 5 for 10 WorkPackges

    # Work to do from config file
    mydir = os.getcwd()       
    configuration = read_config(mydir)
    config_base = configuration[0]

    hdf5_file = config_base[0][1] 
    hdf5_file = os.path.join(mydir, hdf5_file)


    sys.stdout.write("Task_{}; Laid 2021 - Parallel Programming Model - laid_parallel_dm_2.py\n".format(task_id))
    sys.stdout.write("Task_{}; Experiment {}; Storage Layout of Sample Dataset {}\n".format(task_id,config_base[0][9],config_base[0][6] ))    
    # sys.stdout.write("Task_{}; Sample Dataset from{}\n".format(task_id, hdf5_file))
 

    if os.path.isfile(hdf5_file):
        # The file name may be a byte string or unicode string. Valid modes are:
        # r	 Readonly, file must exist (default)
        # r+ Read/write, file must exist
        # w	 Create file, truncate if exists
        # w- or x	Create file, fail if exists
        # a	 Read/write if exists, create otherwise        
        sys.stdout.write("Task_{}; Dataset from HDF5 {}\n".format(task_id, hdf5_file))

    else:
        sys.stdout.write("Task_{}; HDF5 File {} NOT found\n".format(task_id, hdf5_file))
        return


    hf = h5py.File(hdf5_file, 'r+', driver='mpio', comm=MPI.COMM_WORLD)  
    dset = hf['database']    
    dset_class = hf['class']

    nrows_data = dset.attrs['nrows_data']    
    number_of_rows = nrows_data
    ncols_data = dset.attrs['ncols_data']           # cols of data       
    number_of_features = ncols_data                 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! or config_base[0][3]  
    ncols_total =dset.attrs['ncols_total']          # include cols of data + jnsq cols + label col
    number_of_columns = ncols_total

    # data_starts = dset.attrs['data_starts']       # not needed 
    # data_ends = dset.attrs['data_ends']           # not needed

    # number_of_jnsq_needed = dset.attrs['number_jnsq_needed']    # efective
    try:
        number_of_jnsq_needed = dset.attrs['number_jnsq_needed']    # efective
    except KeyError:
        number_of_jnsq_needed = 0
    number_jnsq_features = dset.attrs['number_jnsq_features']   # reserved cols if not needed col values remains with zero    
    if number_jnsq_features == 0:
        number_jnsq_features = 1                    # ETL always reserve one column
    # jnsq_starts = dset.attrs['jnsq_starts']       # not needed - ncols_data + number_jnsq_features -1    # due to base 0 on h5py
    # jnsq_ends = dset.attrs['jnsq_ends']           # not needed - ncols_data + number_jnsq_features -1    # due to base 0 on h5py    
    
    number_dif_class_values = dset.attrs['number_dif_class_values'] 
    number_class_cols = dset.attrs['number_class_cols'] 
    number_of_classes = number_class_cols
    # class_start = dset.attrs['class_starts']      # not needed - ncols_data + number_jnsq_features + number_class_cols -1
    # class_ends = dset.attrs['class_ends']         # not needed - ncols_data + number_jnsq_features + number_class_cols -1

    # check_redundant_inconsistent_rows = config_base[0][14]
    # fix_redundant_inconsistent_rows   = config_base[0][14]

    number_of_features_to_dm = config_base[0][12]  # for test purpose overlap the real value
    if number_of_features_to_dm != 0:
        number_of_columns = number_of_features_to_dm
        number_of_features = number_of_features_to_dm
        fix_redundant_inconsistent_rows = 'N'
        number_jnsq_features = 0


    class_array = dset_class[...,0:number_of_classes]  #  Read Class values all rows


    hdf5_aux_file = os.path.join(mydir, config_base[0][13])
    hf_aux = h5py.File(hdf5_aux_file, 'r', driver='mpio', comm=MPI.COMM_WORLD)  # w = Create file, truncate if exists / a	Read/write if exists, create otherwise

    # dset_order = hf_aux.create_dataset('order', data=ordered_array)
    # dset_position = hf_aux.create_dataset('position', data=positional_array)    
    dset_redundant =  hf_aux['redundant']
    redundant_array = dset_redundant[...]
    # dset_inconsistent = hf_aux.create_dataset('inconsistent', data=inconsistent_array, dtype= np.dtype(np.int8))        
    # dset_jnsq = hf_aux.create_dataset('jnsq', data=jnsq_array)   
    dset_dm_guide = hf_aux['dm_guide']  
    dm_guide_array = dset_dm_guide[...]  # enforce that this is created

    # extract class_ref from dm_guide_array

    sys.stdout.write("Task_{}; Settings loaded from HDF5 metadata\n".format(task_id)) 
    sys.stdout.write("Task_{}, number_of_rows       ; {}\n".format(task_id, number_of_rows ))     
    sys.stdout.write("Task_{}; number_of_columns    ; {}\n".format(task_id, number_of_columns ))    
    sys.stdout.write("Task_{}; number_of_features   ; {}\n".format(task_id, number_of_features ))   
    # sys.stdout.write("Task_{}; number_jnsq_features : {}\n".format(task_id, number_jnsq_features ))    
    sys.stdout.write("Task_{}; number_of_classes    ; {}\n".format(task_id, number_of_classes )) 
    sys.stdout.write("Task_{}; Shape class_array    ; {}\n".format(task_id, class_array.shape ))   
    sys.stdout.write("Task_{}; Shape redundant_array; {}\n".format(task_id, redundant_array.shape ))  
    sys.stdout.write("Task_{}; Shape dm_guide_array ; {}\n".format(task_id, dm_guide_array.shape ))      

    # sys.stdout.write("Task_{}; Check and Fix redundant and/or inconsistent observations Y/N; {}\n".format(task_id, check_redundant_inconsistent_rows)) 
    # if check_redundant_inconsistent_rows == 'Y':

    # ---------------------------------------------------
    # Build Disjoint Matrix - parallel version
    # ---------------------------------------------------


    step = int(number_of_rows/worker_nodes)   #  number_of_rows of class_ref (1700 on the sample)
    sys.stdout.write("Task_{}; Parallel step ;{}\n".format(task_id, step))
    start = task_id * step
    stop = int( task_id * step + step -1 ) # number_of_rows-1
    if task_id + 1 == worker_nodes and stop < number_of_rows:  # ensure odd cases for last parallel task
        stop = number_of_rows-1
    sys.stdout.write("Task_{}; start; {} stop; {}\n".format(task_id, start, stop))


    sys.stdout.write("Task_{}; Build Disjoint Matrix parallel version\n".format(task_id)) 
    disjoint_array = np.zeros((number_of_features + number_jnsq_features), dtype=np.dtype(np.int8))    # stores compare of mi e mj sums
    sys.stdout.write("Task_{}; shape disjoint_array; {}\n".format(task_id, disjoint_array.shape )) 

    t1 = time.perf_counter()

    # rows_on_disjoint_matrix = 0 
    # number_of_interact_needed_worst_case =  int( ( number_of_rows -1 ) * number_of_rows /2 )    # number of interact needed - worst case
    # create_disjoint_matrix_file = config_base[0][15]

    # if create_disjoint_matrix_file == 'Y':

    rows_on_disjoint_matrix = dset.attrs['rows_on_disjoint_matrix'] # from previuos recon
        # if rows_on_disjoint_matrix == 0:

    # if number_dif_class_values == 2:
        # unique, counts = np.unique(class_array, return_counts=True)
        # class_entry_array = np.asarray((unique, counts)).T
        # sys.stdout.write("Task_{}; sum group for class;{}  \n".format(task_id, class_entry_array ))   

    # solution via split dm files    
    # size_a = dm_guide_array[start,0]  # np.prod(class_entry_array[:,1] ) # this is a genearization of  size_a = rows_class_0 * rows_class_1  

    # sys.stdout.write("Task_{}; max of disjoint rows (worst case);{} estimated disjoint rows 2 classe values;{} \n".format(task_id, number_of_interact_needed_worst_case  ,size_a ))   
    
    # else:
    #     size_a = number_of_interact_needed_worst_case
    #     sys.stdout.write("Task_{}; max of disjoint rows (worst case);{} estimated disjoint rows (worst case);{} \n".format(task_id, number_of_interact_needed_worst_case  ,size_a ))     

        # else:
    size_a = rows_on_disjoint_matrix
        #     sys.stdout.write("Task_{}; max of disjoint rows (worst case);{} effective disjoint rows;{} \n".format(task_id, number_of_interact_needed_worst_case  ,size_a ))    
        
    size_b = number_of_features + number_jnsq_features 
    sys.stdout.write("Task_{}; Disjoint Matrix dataset; [{},{}]\n".format(task_id, size_a, size_b )) 

    # solution onse single dm file for all tasks

    hdf5_disjoint_file = os.path.join(mydir, config_base[0][10]) + '0.h5'
    # dm_parallel_file_name = hdf5_disjoint_file + str(task_id) + '.h5'

    hfdm = h5py.File(hdf5_disjoint_file, 'w', driver='mpio', comm=MPI.COMM_WORLD)  # 'w' = Create file, truncate if exists
    dataset_disjoint_matrix = hfdm.create_dataset('dmatrix', (size_a, size_b), dtype= np.dtype(np.int8))   # size_a+1 on 2021-08-19

    number_of_interact = 0 # no de iterações    

    for i in range(start, stop+1):        

        rows_on_disjoint_matrix = dm_guide_array[i,0]
        show_rows_on_disjoint_matrix = rows_on_disjoint_matrix        
        comparations_expected = dm_guide_array[i,1]
        # sys.stdout.write("Task_{}; Row;{}; Class;{}; Start DM row;{}; comparations_expected; {}\n".format(task_id, i,  class_array[i], rows_on_disjoint_matrix, comparations_expected ))        

        number_of_interact_per_row = 0 
        rows_on_disjoint_matrix_per_row = 0
        buffer_array = np.zeros((comparations_expected,number_of_features + number_jnsq_features), dtype=np.dtype(np.int8))
        t1_row = time.perf_counter()

        if comparations_expected > 0: #(i not in redundant_array): # redundant observations are not considered. REDUNDANT

            read_mi_array = dset[i,:number_of_features+number_jnsq_features]                    # read attributes and jnsq columns of the observation to be compared

            for j in range(i+1, number_of_rows):                                                # compares the current observation (row) with the observations in the following rows

                number_of_interact_per_row += 1
                if (j not in redundant_array) and (class_array[i] != class_array[j]):           # if the observation to be compared is not redundant and has a different class value

                    read_mj_array = dset[j,:number_of_features + number_jnsq_features]          # read attributes and jnsq columns from the observation to which it compares
                    disjoint_array = np.absolute(np.subtract( read_mi_array , read_mj_array ))  # checks whether the elements of the current observation and the observation to it is being compared are equal (=0) or different (=1)


                    # if create_disjoint_matrix_file == 'Y':
                    # dataset_disjoint_matrix[rows_on_disjoint_matrix,:] = disjoint_array #  updates DM dataset
                    buffer_array[rows_on_disjoint_matrix_per_row,:] = disjoint_array            #  updates buffer

                    rows_on_disjoint_matrix += 1 
                    rows_on_disjoint_matrix_per_row += 1
                    # counter += 1

                number_of_interact += 1    

            dataset_disjoint_matrix[dm_guide_array[i,0]:comparations_expected,:] = buffer_array #  updates DM dataset from buffer

        t2_row = time.perf_counter()
        test = (f"in; {t2_row - t1_row:0.4f}; seconds")
        sys.stdout.write("Task_{}; Row;{}; Class;{}; Start DM row;{}; comparations_expected; {}; Number of interact;{}; Disjoint rows found;{}; {}\n".format(task_id, i,  class_array[i], show_rows_on_disjoint_matrix, comparations_expected,number_of_interact_per_row, rows_on_disjoint_matrix_per_row, test )) 


    # Finaly

    # dset.attrs['rows_on_disjoint_matrix'] = rows_on_disjoint_matrix  # update metadata of original dataset

    t2 = time.perf_counter()
    print(f"Task_{task_id}; Disjoint matrix generated in; {t2 - t1:0.4f}; seconds\n")
    # sys.stdout.write("Task_{}; max interact (worst case);{} effective interact;{} \n".format(task_id, number_of_interact_needed_worst_case  ,number_of_interact ))       
    sys.stdout.write("Task_{}; Total number_of_interact      : {}\n".format(task_id, number_of_interact ))   
    sys.stdout.write("Task_{}; Total rows_on_disjoint_matrix : {}\n".format(task_id, rows_on_disjoint_matrix ))    
    sys.stdout.write("Task_{}; file size {} GB without compression \n".format(task_id, (rows_on_disjoint_matrix * (number_of_features + number_jnsq_features)) / 1024 / 1024 / 1024 ))   

    sys.stdout.write("Task_{} Try close all HDF5 Files\n".format(task_id))
    hf.close()
    hf_aux.close()
    # if create_disjoint_matrix_file == 'Y':
    hfdm.close()
    sys.stdout.write("Task_{} HDF5 Files Closed\n".format(task_id))    

if __name__ == "__main__":
    main()    