#!/usr/bin/python3
# coding: utf-8
"""
Laid Serial 2021 version - ready for parallel

Usage:  save this script and run

    $python laid_serial_6b.py
    mpirun python laid_serial_6b.py

2021 Paulo Morgado
Based or adapted from previous work:
Apolónia, João, and Luís Cavique. 2019. “Seleção de Atributos de Dados Inconsistentes Em Ambiente HDF5 + Python Na Cloud INCD.” Revista de Ciências da Computação (14): 85–112.
Cavique, Luís, Armando B. Mendes, Hugo F.M.C. Martiniano, and Luís Correia. 2018. “A Biobjective Feature Selection Algorithm for Large Omics Datasets.” In Expert Systems,.

Notes:
All required information will be present on config.json file that must stored in your current working directory, and should point to original dataset on HDF5 format

Tested under: Python 3.9.4 

Last updated: 2021-08-28 - include buffer on dm
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
    worker_nodes = MPI.COMM_WORLD.Get_size()     # Number SLURM_NTASKS  valid are 1, 2 or 5 for 10 WorkPackges
    name = MPI.Get_processor_name()
    code_base = "laid_serial_6b.py"   
    msg = "ProcessId {0} of {1} on {2} running{3}\n"
    sys.stdout.write(msg.format(task_id, worker_nodes, name, code_base))

    # Work to do from config file
    mydir = os.getcwd()       
    configuration = read_config(mydir)
    config_base = configuration[0]

    hdf5_file = 'laidp_original_dataset_1000k_split.h5' # config_base[0][1] 
    hdf5_file = os.path.join(mydir, hdf5_file)

    sys.stdout.write("Task_{}; Laid 2021 - Serial Programming Model - laid_serial_4.py\n".format(task_id))
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

    # read Dataset fragment
    dset = []
    # for i in range(worker_nodes):
    dset.append(  hf['db_' + str(task_id)] ) # hf['database']    

    # dset[task_id] = hf['db_' + str(task_id)]  # hf['database']    

    sys.stdout.write("Task_{}; fragmet {} shape {} content sample\n{}\n".format(task_id, 'db_' + str(task_id), dset[0].shape, dset[0][0:10,0:10]))

    dset_class = hf['class']

    nrows_data = dset[0].attrs['nrows_data']    
    number_of_rows = nrows_data
    ncols_data = dset[0].attrs['ncols_data']           # cols of data       
    number_of_features = ncols_data                    # or config_base[0][3]  
    ncols_total =dset[0].attrs['ncols_total']          # include cols of data + jnsq cols + label col
    number_of_columns = ncols_total

    # data_starts = dset.attrs['data_starts']       # no need 
    # data_ends = dset.attrs['data_ends']           # no need

    # number_of_jnsq_needed = dset.attrs['number_jnsq_needed']    # efective
    try:
        number_of_jnsq_needed = dset[0].attrs['number_jnsq_needed']    # efective
    except KeyError:
        number_of_jnsq_needed = 0
    number_jnsq_features = dset[0].attrs['number_jnsq_features']   # reserved cols if not needed col values remains with zero    
    if number_jnsq_features == 0:
        number_jnsq_features = 1                    # ETL always reserve one column
    # jnsq_starts = dset.attrs['jnsq_starts']       # not needed - ncols_data + number_jnsq_features -1    # due to base 0 on h5py
    # jnsq_ends = dset.attrs['jnsq_ends']           # not needed - ncols_data + number_jnsq_features -1    # due to base 0 on h5py    
    
    number_dif_class_values = dset[0].attrs['number_dif_class_values'] 
    number_class_cols = dset[0].attrs['number_class_cols'] 
    number_of_classes = number_class_cols
    # class_start = dset.attrs['class_starts']      # not needed - ncols_data + number_jnsq_features + number_class_cols -1
    # class_ends = dset.attrs['class_ends']         # not needed - ncols_data + number_jnsq_features + number_class_cols -1


    fix_redundant_inconsistent_rows   = config_base[0][14]

    # number_of_features_to_dm = config_base[0][12]  # for test purpose overlap the real value
    # if number_of_features_to_dm != 0:
    #     number_of_columns = number_of_features_to_dm
    #     number_of_features = number_of_features_to_dm
    #     fix_redundant_inconsistent_rows = 'N'
    #     number_jnsq_features = 0


    class_array = dset_class[...,0:number_of_classes]  #  Read Class values all rows

    sys.stdout.write("Task_{}; Settings loaded from HDF5 metadata:\n".format(task_id)) 
    sys.stdout.write("Task_{}, number_of_rows       : {}\n".format(task_id, number_of_rows ))     
    sys.stdout.write("Task_{}; number_of_columns    : {}\n".format(task_id, number_of_columns ))    
    sys.stdout.write("Task_{}; number_of_features   : {}\n".format(task_id, number_of_features ))   
    sys.stdout.write("Task_{}; number_jnsq_features : {}\n".format(task_id, number_jnsq_features ))    
    sys.stdout.write("Task_{}; number_of_classes    : {}\n".format(task_id, number_of_classes )) 
    sys.stdout.write("Task_{}, Shape class_array    : {}\n".format(task_id, class_array.shape ))   


    # -----------------------------------------------------------------------------
    # 1. Check redundant and/or inconsistent observations / project DisJoint Matrix - Serial version
    # -----------------------------------------------------------------------------


    check_redundant_inconsistent_rows = config_base[0][14]
    if check_redundant_inconsistent_rows == 'Y':

        sys.stdout.write("Task_{}; Check and Fix redundant and/or inconsistent observations Y/N: {}\n".format(task_id, check_redundant_inconsistent_rows)) 


        # ---------------------------------------------------
        # 1.1 Get the sequence of observations sorted by attributes
        # ---------------------------------------------------

        # samples:
        # original          [0 1 2 3 4 5 6 7]
        # ordered_array     [0 0 0 1 2 3 3 3]
        # positional_array  [6 2 3 1 4 0 5 7]

        ordered_array = np.zeros((number_of_rows), dtype=int)                                   # stores order of order of observations, observations with equal attributes have the same order 
        positional_array = np.arange((number_of_rows), dtype=int)                               # stores positions of rows

        col_step_on_sorting = config_base[0][11]                                                # number of columns in each column block to handle
        number_blocks_to_sort = int(number_of_columns/col_step_on_sorting)+1                    # number of column blocks to handle
        col_start = 0 # xini = 0 # col start on next block 



        sys.stdout.write("Task_{}; Sort columns; {} number of column blocks to handle\n".format( task_id, number_blocks_to_sort ))         
        t1 = time.perf_counter()

        for i in range(number_blocks_to_sort): 
            
            if col_start + col_step_on_sorting > number_of_columns:                             # Ensure that size of block fit number of columns 
                col_step_on_sorting = number_of_columns - col_start 

            block_read = dset[0][:,col_start:col_start + col_step_on_sorting]                   # Read data block 
            block_read = block_read[positional_array]                                           # sort rows by order defined on previous block 

            aux_matrix = np.zeros((number_of_rows,col_step_on_sorting+2), dtype=int)            # join arrays ordered_array, positional_array in one 2D array (matrix)  
            aux_matrix[:,0] = ordered_array[:] 
            aux_matrix[:,1:col_step_on_sorting+1] = block_read[:,:] 
            aux_matrix[:,col_step_on_sorting+1] = positional_array[:] 

            # determines the order indices of the observations (starts by inverting the order of the columns, transposes and determines the order index of the columns)

            auxi = np.lexsort(np.fliplr(aux_matrix).T) 
            aux_matrix = aux_matrix[auxi]                                                       # sorts rows of the matrix according to the order indices
            ordered_array[0] = 0                                                                # first row always is order zero 

            for j in range(1,number_of_rows):                                                   # check each row of the matrix is the same as the previous one
                if col_start < number_of_columns - number_jnsq_features - number_of_classes:    # feature column ? 
                    if np.array_equal(aux_matrix[j,0:col_step_on_sorting+1], aux_matrix[j-1,0:col_step_on_sorting+1]): 
                        ordered_array[j]=ordered_array[j-1]                                     # if line is the same as the previous one, it has the same order value
                    elif col_start+col_step_on_sorting == number_of_columns and np.array_equal(aux_matrix[j,0: col_step_on_sorting - number_of_classes - number_jnsq_features + 1 ], aux_matrix[j-1,0: col_step_on_sorting - number_of_classes - number_jnsq_features + 1 ]): 
                        ordered_array[j]=ordered_array[j-1]                                     # similar to the previous one, in case it is the last block
                    else: 
                        ordered_array[j]=ordered_array[j-1]+1                                   # in case they are not the same, the order of the line will have the next value to the previous one.

            positional_array = aux_matrix[:,col_step_on_sorting+1]                              # after matrix sort, the position of each row/observation  in the dataset is saved.
            col_start += col_step_on_sorting                                                    # col start on next block 


        t2 = time.perf_counter()
        print(f"Task_{task_id}; Columns Sorted in; {t2 - t1:0.4f}; seconds")

        sys.stdout.write("Task_{}; ordered_array shape {} and content sample \n".format(task_id, ordered_array.shape, ordered_array ))   
        sys.stdout.write("Task_{}; positional_array shape {} and content sample \n".format(task_id, positional_array.shape, positional_array ))   

        # debug positional_array
        # for i in range(0,number_of_rows):
        #     print (dset[positional_array[i],0:])


        # ---------------------------------------------------
        # 1.2 Check redundant and/or inconsistent observations
        # ---------------------------------------------------

        sys.stdout.write("Task_{}; Check redundant and/or inconsistent observations \n".format(task_id)) 
        t1 = time.perf_counter()


        number_redundant_rows = 0                                                                   # number of redundant rows  
        redundant_array = np.zeros((number_of_rows, 1), dtype=int)                                  # stores position of redundant rows  


        number_inconsistent_rows = 0                                                                # number of inconsistent rows
        number_of_inconsistency = -1                                                                # number of different inconsistencies - 1
        inconsistent_array = np.zeros((number_of_rows,2), dtype=int)                                # stores position of inconsistent rows and their index 

        read_col_array = aux_matrix[:,col_step_on_sorting]                                          # TODO: use dset_class[...,0:number_of_classes] 

        # test first and second row

        if ordered_array[0] == ordered_array[1]:                                                    # check if the first observations of the ordered matrix is equal to the next one, ie if the attributes are equal
            if (read_col_array[0] != read_col_array[1]):                                            # if the value of the class is different the first row of the matrix is an inconsistency
                number_of_inconsistency += 1                                                                    # found the first inconsistence 
                inconsistent_array[number_of_inconsistency,:] = [positional_array[0],number_of_inconsistency]   # stores them (position and count) 
                number_inconsistent_rows += 1 

        # next rows

        for i in range(1,number_of_rows): 
            if ordered_array[i] == ordered_array[i-1]:                                              # if the attributes of a row are the same as those of the previous line, it is redundancy or inconsistency
                if read_col_array[i] == read_col_array[i-1]:                                        # if class values are the same, means the row is redundant with the previous one.

                    if (positional_array[i-1] not in inconsistent_array[:number_inconsistent_rows,0]) and (positional_array[i-1] not in redundant_array[:number_redundant_rows, 0]):   # if previous row was neither inconsistent nor redundant

                        # new redundant row found
                        redundant_array[number_redundant_rows, 0] = positional_array[i-1]           # add previous index row to the redundancies array (in a redundant pair, it is the first row that is marked)

                    else: 
                        redundant_array[number_redundant_rows, 0] = positional_array[i]             # if the previous one was already marked as inconsistent or redundant, add the row index to the redundancies array

                    number_redundant_rows += 1 
 
                else:                                                                               # if the class values are different, means the row is inconsistent with the previous one.
                    if (positional_array[i-1] not in inconsistent_array[:number_inconsistent_rows,0]) and (positional_array[i-1] not in redundant_array[:number_redundant_rows, 0]):   # if previous row was neither inconsistent nor redundant

                        # new inconsistence row found
                        number_of_inconsistency += 1 
                        inconsistent_array[number_inconsistent_rows,:] = [positional_array[i-1],number_of_inconsistency]    # add previous row index to the inconsistencies array
                        number_inconsistent_rows += 1 

                    inconsistent_array[number_inconsistent_rows,:] = [positional_array[i],number_of_inconsistency]          # add index to inconsistent_array  
                    number_inconsistent_rows += 1 


        # Resize the lists to the number of redundancies and inconsistencies,

        redundant_array = redundant_array[:number_redundant_rows] 
        inconsistent_array = inconsistent_array[:number_inconsistent_rows] 

        t2 = time.perf_counter()
        test = (f"In; {t2 - t1:0.4f}; seconds")    
        sys.stdout.write("Task_{}; Found ({}) Redundant and ({}) Inconsistent rows; {}\n".format(task_id , number_redundant_rows , int(number_inconsistent_rows/2), test))

        sys.stdout.write("Task_{}; redundant_array shape {} and content sample \n".format(task_id, redundant_array.shape, redundant_array ))   
        sys.stdout.write("Task_{}; inconsistent_array shape {} and content sample \n".format(task_id, inconsistent_array.shape, inconsistent_array ))   
      

        # ---------------------------------------------------
        # 1.3 Handle required jnsq attributes
        # ---------------------------------------------------        

        sys.stdout.write("Task_{}; Handle required jnsq attributes\n".format(task_id)) 
        t1 = time.perf_counter()

        number_of_jnsq_needed = 0 
        jnsq_array = np.zeros((number_of_rows,number_jnsq_features), dtype=np.dtype(np.int8))                   # initialize jnsq attributes array

        if number_of_inconsistency > -1:                                                                        # if inconsistencies were found
            aux_nii = np.bincount(inconsistent_array[:,1])                                                      # counts the number of observations for each inconsistency number (2 col is the inconsistency number) 

            xp = 0                                                                                              # initialize position on inconsistent_array 
            for i in range(np.size(aux_nii)):                                                                   # for each inconsistency number
                for j in range(aux_nii[i]):                                                                     # for each value less than the number of observations of each inconsistency number

                    # Determine the binary code to fill jnsq columns of inconsistent observations
                    
                    xbin = j                                                                                    # position of an observation in the number of observations of its inconsistency number
                    for k in range(number_jnsq_features):                                                       # for each jnsq column 
                        jnsq_array[inconsistent_array[xp,0],k] = xbin%2                                         # binary value for column jnsq (remainder)
                        xbin = xbin//2                                                                          # value that passes to the next jnsq column 
                        if jnsq_array[inconsistent_array[xp,0],k] > 0 and k+1 > number_of_jnsq_needed:          # if a jnsq column has a non-null value and has not yet been counted as effective increment counter
                            number_of_jnsq_needed += 1 
                        xp += 1 


        # ---------------------------------------------------
        # 1.4 populate original dataset with jnsq attributes
        # ---------------------------------------------------    

        if number_of_jnsq_needed == 0: 
            test = 'No need for jnsq attributes' 
        elif number_of_jnsq_needed == 1: 
            if fix_redundant_inconsistent_rows == 'Y':
                dset[0][:,number_of_features:number_of_features+number_jnsq_features] = jnsq_array[...] 
            test = 'The jnsq attribute is populated on original Dataset'         
        else: 
            if fix_redundant_inconsistent_rows == 'Y':
                dset[0][:,number_of_features:number_of_features+number_jnsq_features] = jnsq_array[...] 
            test ='({0}) jnsq attributes are filled on original Dataset'.format(number_of_jnsq_needed)

        sys.stdout.write("Task_{}; jnsq attributes; ({}) - {}\n".format(task_id, number_of_jnsq_needed, test))
        t2 = time.perf_counter()
        test = (f"In; {t2 - t1:0.4f}; seconds")         
        sys.stdout.write("Task_{}; Handle required jnsq attributes in; {}\n".format(task_id, number_of_jnsq_needed, test))
        sys.stdout.write("Task_{}; jnsq_array shape {} and content sample \n".format(task_id, jnsq_array[...].shape, jnsq_array ))   


        # ---------------------------------------------------
        # 1.5 stores ordered and positional arrays on aux dataset - overwrited on parallel runs
        # ---------------------------------------------------        

        # hdf5_aux_file = os.path.join(mydir, config_base[0][13])
        # hf_aux = h5py.File(hdf5_aux_file, 'w', driver='mpio', comm=MPI.COMM_WORLD)  # w = Create file, truncate if exists / a	Read/write if exists, create otherwise

        # ds_order = []
        # ds_order.append(hf_aux.create_dataset('order_{0}'.format(task_id), data=ordered_array) )
        # # dset_order = hf_aux.create_dataset('order', data=ordered_array)

        # ds_position = []
        # ds_position.append(hf_aux.create_dataset('position_{0}'.format(task_id), data=positional_array)   )
        # # dset_position = hf_aux.create_dataset('position', data=positional_array)    

        # ds_redundant = []
        # ds_redundant.append(hf_aux.create_dataset('redundant_{0}'.format(task_id), data=redundant_array, dtype= np.dtype(np.int8)) )
        # # dset_redundant = hf_aux.create_dataset('redundant', data=redundant_array, dtype= np.dtype(np.int8))
        
        # ds_inconsistent = []
        # ds_inconsistent.append( hf_aux.create_dataset('inconsistent_{0}'.format(task_id), data=inconsistent_array, dtype= np.dtype(np.int8))  )
        # # dset_inconsistent = hf_aux.create_dataset('inconsistent', data=inconsistent_array, dtype= np.dtype(np.int8))          
        
        # ds_jnsq = []
        # ds_jnsq.append( hf_aux.create_dataset('jnsq_{0}'.format(task_id), data=jnsq_array) )
        # # dset_jnsq = hf_aux.create_dataset('jnsq', data=jnsq_array)           

        # # ERROR dset[task_id].attrs['number_jnsq_needed'] = number_of_jnsq_needed     
    


        # ---------------------------------------------------
        # 1.6 Pre-check Disjoint Matrix - for Paralell version
        # ---------------------------------------------------

        # condition ?
        # if 1 == 1:   #if worker_nodes == 1:  # running serial
        sys.stdout.write("Task_{}; Pre-check Disjoint Matrix - for DM Paralell version\n".format(task_id)) 
        disjoint_array = np.zeros((number_of_features + number_jnsq_features), dtype=np.dtype(np.int8))    # stores compare of mi e mj sums
        sys.stdout.write("Task_{}; shape disjoint_array : {}\n".format(task_id, disjoint_array.shape )) 

        t1 = time.perf_counter()

        rows_on_disjoint_matrix = 0 
        number_of_interact_needed_worst_case =  int( ( number_of_rows -1 ) * number_of_rows /2 )    # number of interact needed - worst case

        number_of_interact = 0   

        dm_guide_array = np.zeros((number_of_rows, 2),  dtype= np.dtype(np.int32))                  #   stores guide to parallel handle row index  numpy.int32: 32-bit signed integer (-2_147_483_648 to 2_147_483_647)

        for i in range(number_of_rows-1):

            number_of_interact_per_row = 0 
            rows_on_disjoint_matrix_per_row = 0
            dm_guide_array[i,0] = rows_on_disjoint_matrix

            t1_row = time.perf_counter()

            if (i not in redundant_array):                                                          # redundant observations are not considered.

                read_mi_array = dset[0][i,:number_of_features+number_jnsq_features]                 # read attributes and jnsq columns of the observation to be compared

                for j in range(i+1, number_of_rows):                                                # compares the current observation (row) with the observations in the following rows

                    number_of_interact_per_row += 1
                    if (j not in redundant_array) and (class_array[i] != class_array[j]):           # if the observation to be compared is not redundant and has a different class value

                        read_mj_array = dset[0][j,:number_of_features + number_jnsq_features]       # read attributes and jnsq columns from the observation to which it compares
                        disjoint_array = np.absolute(np.subtract( read_mi_array , read_mj_array ))  # checks whether the elements of the current observation and the observation to it is being compared are equal (=0) or different (=1)

                        rows_on_disjoint_matrix += 1  
                        rows_on_disjoint_matrix_per_row += 1

                    number_of_interact += 1    


            dm_guide_array[i,1] = rows_on_disjoint_matrix_per_row


            t2_row = time.perf_counter()
            test = (f"in; {t2_row - t1_row:0.4f}; seconds")
            if i % 100 == 0:
                sys.stdout.write("Task_{}; Pre-check for Row ; {}; Class; {}; Number of interact; {}; Disjoint rows found; {}; {}\n".format(task_id, i,  class_array[i], number_of_interact_per_row, rows_on_disjoint_matrix_per_row, test )) 

        # ERROR= dset[task_id].attrs['rows_on_disjoint_matrix'] = rows_on_disjoint_matrix  # update metadata of original dataset
        store_rows_on_disjoint_matrix = rows_on_disjoint_matrix

        # dset_dm_guide = []
        # dset_dm_guide.append( hf_aux.create_dataset('dm_guide_{0}'.format(task_id), data=dm_guide_array, dtype=np.dtype(np.int32))  )


        t2 = time.perf_counter()
        print(f"Task_{task_id}; Pre-check Disjoint Matrix done in; {t2 - t1:0.4f}; seconds")
        sys.stdout.write("Task_{}; max interact (worst case);{} effective interact;{} \n".format(task_id, number_of_interact_needed_worst_case  ,number_of_interact ))       
        sys.stdout.write("Task_{}; Total number_of_interact      : {}\n".format(task_id, number_of_interact ))   
        sys.stdout.write("Task_{}; Total rows_on_disjoint_matrix : {}\n".format(task_id, rows_on_disjoint_matrix ))    
        sys.stdout.write("Task_{}; Estimated fie size {} GB without compression \n".format(task_id, (rows_on_disjoint_matrix * (number_of_features + number_jnsq_features)) / 1024 / 1024 / 1024 ))
        # else:
        #     sys.stdout.write("Task_{}; Pre-check Disjoint Matrix - NOT performed\n".format(task_id)) 
    else:
        sys.stdout.write("Task_{}; Check and Fix redundant and/or inconsistent observations Y/N: {}\n".format(task_id, check_redundant_inconsistent_rows)) 



    # ---------------------------------------------------
    # 2. Build Disjoint Matrix - Serial version
    # ---------------------------------------------------

    create_disjoint_matrix_file = config_base[0][15]
    if create_disjoint_matrix_file == 'Y':
        # sys.stdout.write("Task_{} try close all hdf5 File.\n".format(task_id))
        # hf.close()
        # hf_aux.close()
        # sys.stdout.write("Task_{} closed hdf5 File.\n".format(task_id))    
        # return


        sys.stdout.write("Task_{}; Build Disjoint Matrix =Y\n".format(task_id)) 
        disjoint_array = np.zeros((number_of_features + number_jnsq_features), dtype=np.dtype(np.int8))     # stores compare of mi e mj sums
        sys.stdout.write("Task_{}; shape disjoint_array : {}\n".format(task_id, disjoint_array.shape )) 

        t1 = time.perf_counter()

        rows_on_disjoint_matrix = 0 
        number_of_interact_needed_worst_case =  int( ( number_of_rows -1 ) * number_of_rows /2 )            # number of interact needed - worst case
        # create_disjoint_matrix_file = config_base[0][15]

        # if create_disjoint_matrix_file == 'Y':

        rows_on_disjoint_matrix = store_rows_on_disjoint_matrix                                             # dset[task_id].attrs['rows_on_disjoint_matrix'] # from previuos recon
        if rows_on_disjoint_matrix == 0:
            if number_dif_class_values == 2:
                unique, counts = np.unique(class_array, return_counts=True)
                class_entry_array = np.asarray((unique, counts)).T
                sys.stdout.write("Task_{}; sum group for class;{}  \n".format(task_id, class_entry_array ))   
                size_a = np.prod(class_entry_array[:,1] )                                                   # this is a genearization of  size_a = rows_class_0 * rows_class_1  
                sys.stdout.write("Task_{}; max of disjoint rows (worst case);{} estimated disjoint rows 2 classe values;{} \n".format(task_id, number_of_interact_needed_worst_case  ,size_a ))   
            else:
                size_a = number_of_interact_needed_worst_case
                sys.stdout.write("Task_{}; max of disjoint rows (worst case);{} estimated disjoint rows (worst case);{} \n".format(task_id, number_of_interact_needed_worst_case  ,size_a ))                  
        else:
            size_a = rows_on_disjoint_matrix
            sys.stdout.write("Task_{}; max of disjoint rows (worst case);{} effective disjoint rows;{} \n".format(task_id, number_of_interact_needed_worst_case  ,size_a ))    
        
        size_b = number_of_features + number_jnsq_features 
        sys.stdout.write("Task_{}; Disjoint Matrix dataset: [{},{}]\n".format(task_id, size_a, size_b )) 


        hdf5_disjoint_file = os.path.join(mydir, config_base[0][10])




        # use_compress = 'N'
        # if use_compress == 'Y':                                                      
        #     hfdm = h5py.File(hdf5_disjoint_file, 'w')       # compressed and no parallel - w = Create file, truncate if exists
        #     dataset_disjoint_matrix = hfdm.create_dataset('dmatrix', (size_a, size_b), chunks=(1, size_b), compression="gzip",  dtype= np.dtype(np.int8))   
        # else:

        # unified_disjoint_matrix_file = config_base[0][17]
        # if unified_disjoint_matrix_file == "Y":
        #     
        # else:      
            
                    
        #dm_parallel_file_name =  hdf5_disjoint_file #+ '.h5'
        dm_parallel_file_name = 'dm_' + str(task_id) + '.h5'

        # if worker_nodes > 1:  # serial version running in parallel - handle all rows, and a subset of columns - split dm files 

        hfdm = h5py.File(dm_parallel_file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)  # 'w' = Create file, truncate if exists

        dataset_disjoint_matrix = hfdm.create_dataset('dmatrix', (size_a, size_b), dtype= np.dtype(np.int8))   # size_a+1 on 2021-08-19
        # dataset_disjoint_matrix = []   # ready for parallel now is a list/array        
        # dataset_disjoint_matrix.append(hfdm.create_dataset('dmatrix_{0}'.format(task_id), (size_a, size_b), dtype= np.dtype(np.int8)) )

        rows_on_disjoint_matrix = 0 # reset counter
        # parallel version ini
        # rows_on_disjoint_matrix = 0 # reset counter
        # number_of_interact = 0 # xitr = 0 # contabiliza o no de iterações    
        # step = int(number_of_rows/worker_nodes)
        # sys.stdout.write("Task_{}, Parallel step =  {}\n".format(task_id, step))
        # start = task_id * step
        # stop = int( task_id * step + step -1 ) # number_of_rows-1
        # if task_id + 1 == worker_nodes and stop < number_of_rows:  # ensure odd cases for last parallel task
        #     stop = number_of_rows-1
        # sys.stdout.write("Task_{}, start =  {} stop =  {}\n".format(task_id, start, stop))
        # for i in range(start, stop+1):        
        # parallel version end

        for i in range(number_of_rows-1):

            rows_on_disjoint_matrix_guide = dm_guide_array[i,0]
            comparations_expected_guide = dm_guide_array[i,1]    
            buffer_array = np.zeros((comparations_expected_guide,number_of_features + number_jnsq_features), dtype=np.dtype(np.int8))

            number_of_interact_per_row = 0 
            rows_on_disjoint_matrix_per_row = 0
            t1_row = time.perf_counter()            

            if comparations_expected_guide > 0:                                                     # if (i not in redundant_array): # redundant observations are not considered.

                read_mi_array = dset[0][i,:number_of_features+number_jnsq_features]                 # read attributes and jnsq columns of the observation to be compared

                for j in range(i+1, number_of_rows):                                                # compares the current observation (row) with the observations in the following rows

                    number_of_interact_per_row += 1
                    if (j not in redundant_array) and (class_array[i] != class_array[j]):           # if the observation to be compared is not redundant and has a different class value

                        read_mj_array = dset[0][j,:number_of_features + number_jnsq_features]       # read attributes and jnsq columns from the observation to which it compares
                        disjoint_array = np.absolute(np.subtract( read_mi_array , read_mj_array ))  # checks whether the elements of the current observation and the observation to it is being compared are equal (=0) or different (=1)

                        buffer_array[rows_on_disjoint_matrix_per_row,:] = disjoint_array            # updates buffer

                        rows_on_disjoint_matrix += 1  
                        rows_on_disjoint_matrix_per_row += 1

                    number_of_interact += 1   

                if create_disjoint_matrix_file == 'Y':
                    dataset_disjoint_matrix[rows_on_disjoint_matrix_guide:comparations_expected_guide,:] = buffer_array                   #  updates DM dataset   

            t2_row = time.perf_counter()
            test = (f"in; {t2_row - t1_row:0.4f}; seconds")
            if i % 100 == 0:
                sys.stdout.write("Task_{};Row;{};Class;{};Number of interact;{};Disjoint rows found;{};{}\n".format(task_id, i, class_array[i], number_of_interact_per_row, rows_on_disjoint_matrix_per_row, test )) 


        # Finaly

        # ERROR??  dset[task_id].attrs['rows_on_disjoint_matrix'] = rows_on_disjoint_matrix  # update metadata of original dataset

        # hf_aux.close()

        t2 = time.perf_counter()
        print(f"Task_{task_id} - Disjoint matrix generated in; {t2 - t1:0.4f}; seconds")
        sys.stdout.write("Task_{}; max interact (worst case);{} effective interact;{} \n".format(task_id, number_of_interact_needed_worst_case  ,number_of_interact ))       
        sys.stdout.write("Task_{}; Total number_of_interact      : {}\n".format(task_id, number_of_interact ))   
        sys.stdout.write("Task_{}; Total rows_on_disjoint_matrix : {}\n".format(task_id, rows_on_disjoint_matrix ))    
        sys.stdout.write("Task_{}; Estimated fie size {} GB without compression \n".format(task_id, (rows_on_disjoint_matrix * (number_of_features + number_jnsq_features)) / 1024 / 1024 / 1024 ))   
    else:
        sys.stdout.write("Task_{}; Build Disjoint Matrix = N\n".format(task_id)) 


    # if create_disjoint_matrix_file == 'N':
    #     sys.stdout.write("Task_{} try close all hdf5 File.\n".format(task_id))
    #     hf.close()
    #     hf_aux.close()
    #     sys.stdout.write("Task_{} closed hdf5 File.\n".format(task_id))    
    #     return


    # ---------------------------------------------------
    # 3. Find Soluction - Serial ready to parallel version
    # ---------------------------------------------------

    perform_find_solution = config_base[0][16]
    if perform_find_solution == "Y":

        sys.stdout.write("Task_{}; perform_find_solution = Y\n".format(task_id)) 
        t1 = time.perf_counter()

        # create_disjoint_matrix_file = config_base[0][15]
        # if create_disjoint_matrix_file == 'N':         # need to open datasets
        #     hdf5_disjoint_file = os.path.join(mydir, config_base[0][10])
        #     # unified_disjoint_matrix_file = config_base[0][17]
        #     # if unified_disjoint_matrix_file == "Y":
        #     #     dm_parallel_file_name = hdf5_disjoint_file + '.h5'
        #     # else:                
        #     dm_parallel_file_name = hdf5_disjoint_file + str(task_id) + '.h5'

        #     # if worker_nodes > 1:  # serial version running in parallel - handle all rows, and a subset of columns - split dm files 

        #     # unified_disjoint_matrix_file = config_base[0][17]
        #     # if unified_disjoint_matrix_file == "Y":
        #     #     dm_parallel_file_name = hdf5_disjoint_file + '.h5'
        #     # else:                
        #     #     dm_parallel_file_name = hdf5_disjoint_file + str(task_id) + '.h5'
        #     #dm_parallel_file_name = hdf5_disjoint_file + str(task_id) + '.h5'

        #     hfdm = h5py.File(dm_parallel_file_name, 'r', driver='mpio', comm=MPI.COMM_WORLD)  
        #     dataset_disjoint_matrix = hfdm['dmatrix']

        #     rows_on_disjoint_matrix = dset.attrs['rows_on_disjoint_matrix']
        #     size_b = number_of_features + number_jnsq_features 




        dm_col_index_array = np.arange((size_b), dtype=int)                                                         # array of matrix DM columns index - value -1 mens column excluded 
        dm_row_index_array = np.arange((rows_on_disjoint_matrix), dtype=int)                                        # array of matrix DM rows index - value -1 mens row excluded 
        selected_feature_array = np.full(size_b,-1, dtype=int)   
        number_selected_features = 0 

        # -----------------------------------------------------------------
        # Get matrix Solution ie reduction of the problem
        # -----------------------------------------------------------------        

        continue_interact = True 
        while continue_interact == True: 

            t1_row = time.perf_counter()

            interact_counter = 0 
            sum_of_cols_array = np.zeros(size_b - number_selected_features, dtype=int)                              # reset the list of sums of attribute values for each remaining column of the DM matrix

            remain_cols_array = dm_col_index_array[dm_col_index_array>=0]                                           # array of remaining cols index of the DM matrix in each iteration
            remain_rows_array = dm_row_index_array[dm_row_index_array>=0]                                           # array of remaining rows index of the DM matrix in each iteration

            remain_rows = np.size(remain_rows_array)                                                                # number of lines remaining in each iteration

            #  column values Sum
            for i in remain_rows_array:                                                                             # for all remaining matrix DM rows
                current_dm_row_array = np.squeeze(np.asarray(dataset_disjoint_matrix[i,:size_b]))

                # exclude values from already selected columns
                for j in np.nonzero(dm_col_index_array<0):                                                          # for all withdrawn columns
                    current_dm_row_array[j]=-1                                                                      # mark -1 for exclude/withdraw

                sum_of_cols_array = sum_of_cols_array + current_dm_row_array[current_dm_row_array>=0]               # add to the vector sum the values of row i of matrix DM whose columns have not yet been selected

                interact_counter += 1 
            
            selected_column = np.argmax(sum_of_cols_array)                                                          # index of element with highest value of = column selected

            # -----------------------------------------------------------------
            # test condition to terminate execution - solution found
            # -----------------------------------------------------------------            

            if sum_of_cols_array[selected_column] == 0:
                continue_interact = False 
            else: 

                sys.stdout.write("Task_{}; selected_column; {}\n".format(task_id, selected_column )) 

                # Update solution and remove selected column and rows whose value in the selected column is 1

                current_dm_row_array = np.zeros((remain_rows), dtype=np.dtype(np.int8))                             # array to store values of the selected column 
                for i in range(remain_rows): 
                    current_dm_row_array[i] = dataset_disjoint_matrix[remain_rows_array[i],remain_cols_array[selected_column]] # assigns to each element the respective value of the selected column 

                dm_col_index_array[remain_cols_array[selected_column]] = -1                                         # exclude column from next interaction 
                selected_feature_array[ remain_cols_array[selected_column] ] = remain_cols_array[selected_column]   # Add column to solution array 

                number_selected_features += 1 
                dm_row_index_array[remain_rows_array[np.nonzero(current_dm_row_array == 1)]] = -1                   # exclude rows with value in selected column

            t2_row = time.perf_counter()
            print (f"Task_{task_id}; selected features; {number_selected_features}; interactions needed #;{interact_counter}; time spent;{t2_row - t1_row:0.4f}; seconds")
            sys.stdout.write("Task_{}; Feature selected; {}\n".format(task_id, selected_feature_array[selected_feature_array>=0] )) 


        t2 = time.perf_counter()
        print(f"Task_{task_id}; Solution found in; {t2 - t1:0.4f}; seconds")    

        sys.stdout.write("Task_{}, number_selected_features: {}\n".format(task_id, number_selected_features )) 
        sys.stdout.write("Task_{}; shape selected_feature_array;{}\n".format(task_id, selected_feature_array.shape )) 
        sys.stdout.write("Task_{}; selected_feature_array;{}\n".format(task_id, selected_feature_array[selected_feature_array>=0] )) 
    else:
        sys.stdout.write("Task_{}; perform_find_solution = N\n".format(task_id)) 


    sys.stdout.write("Task_{}; LAID concluded\n".format(task_id ))     

    sys.stdout.write("Task_{} Try close all HDF5 Files\n".format(task_id))
    hf.close()

    # hf_aux.close()
    if create_disjoint_matrix_file == 'Y':
        hfdm.close()
    sys.stdout.write("Task_{} HDF5 Files Closed\n".format(task_id))    

if __name__ == "__main__":
    main()    