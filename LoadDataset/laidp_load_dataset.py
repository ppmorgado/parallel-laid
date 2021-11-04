"""
2021 Paulo Morgado

Laidp module 1 - etl_v6 - Load data from text files into hdf5 dataset 

Usage:  save this script and run

    $python laidp_m1_etl_dataset.py
    mpirun python laidp_m1_etl_dataset.py

Read config from file: config.json
HDF5 file stored on current working directory

Tested under: Python 3.6.9
  1) gcc-4.8          2) mvapich2/2.3.5   3) hdf5/1.12.0

If you have any questions, suggestions, or comments on this example, please use @@

Last updated: 2021-08-08
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

def map_parallel_worker_to_workpack(rank, worker_nodes, work_packages ):  
    step = work_packages / worker_nodes
    ini = int(rank * step)
    end = int(ini + step -1)
    return (ini, end)

def etl_from_file(task_id, index, dset, dset_class, config_base, config_data, number_jnsq_features, number_class_attr, class_start, number_of_features):

    # load settings from config file
    pack = config_data[index][0]
    filename = config_data[index][1]
    class_value = config_data[index][2]
    raw_path = config_base[0][0]
    filepath = raw_path + filename    #   "/users/hpc/pmorgado/laidp/data/raw/"
    # sys.stdout.write("Task:{}, filepath: {}\n".format(task_id, filepath))  

    right_most_block = config_data[index][7]
    nrows_to_load = config_data[index][5]-config_data[index][3]+1
    ncols_to_load = config_data[index][6]-config_data[index][4]+1

    r1 = config_data[index][3]
    r2 = config_data[index][5]
    c1 = config_data[index][4]
    c2 = config_data[index][6]

    final_r2 = r2 +1
    final_c2 = c2 +1

    # if right_most_block == 1:    # add jnsq_features + number_class_attr
    #     matrix = np.zeros((nrows_to_load, ncols_to_load + number_jnsq_features + number_class_attr), dtype= np.dtype(np.int8))   #   + number_class_attr
    #     final_c2 = c2 + 1 + number_jnsq_features + number_class_attr          
    # else:
    matrix = np.zeros((nrows_to_load, ncols_to_load), dtype= np.dtype(np.int8))

    sys.stdout.write("Task:{}; WorkPack#{}; Extract data from {} and load to HDF5 Dataset; Coordinates from [{},{}] to [{},{}]\n".format(task_id, index, filepath, r1, c1, r2, c2))
    # sys.stdout.write("Task:{}, r1:{} c1:{} r2:{} c2:{}\n".format(task_id,r1,c1,r2,c2))    
    sys.stdout.write("Task:{}; File:{}; Rows x Cols considered {}x{}; Class:{}\n".format(task_id, pack, nrows_to_load, ncols_to_load, class_value))
    sys.stdout.write("Task:{}; File:{}; Shape of input matrix:{}; Datatype: {}\n".format(task_id, pack, matrix.shape, matrix.dtype ))

    # extract

    t1 = time.perf_counter()

    gen  = extract_lines(filepath)
    data = np.loadtxt(gen ,delimiter=',', dtype= np.dtype(np.int32)) #  numpy.int32: 32-bit signed integer (-2_147_483_648 to 2_147_483_647). mandatory to handle coordinates
    t2 = time.perf_counter()

    test = (f"Extracted in; {t2 - t1:0.4f}; seconds")
    sys.stdout.write("Task:{}, File:{}, Shape of data extracted:{}; Datatype: {}; {}\n".format(task_id, pack, data.shape, data.dtype, test ))
    # print(f"Task:{task_id}; File:{pack}; Data extracted in,; {t2 - t1:0.4f}; seconds")

    # transform

    t1 = time.perf_counter()
    for x in data:
        # print(x)
        matrix[x[0]-1, x[1]-1] = 1   # mark value 'one' on coordinates x[0]-1, x[1]-1

    # validate load before jnsq and class
    sum_of_columns_array  = matrix.sum(axis=0)
    # print("\nSum of all columns:")
    control_sum = sum_of_columns_array.sum(axis=0)
    sys.stdout.write("Task:{}, File:{}, Len data extracted:{}; Sum of Matrix transformed: {}; Diference {}\n".format(task_id, pack, data.shape[0], control_sum, control_sum - data.shape[0] ))


    if right_most_block == 1:    # mark class value
        ## matrix = np.zeros((nrows_to_load, ncols_to_load + number_jnsq_features + number_class_attr), dtype= np.dtype(np.int8))
        ## final_c2 = c2 + 1 + number_jnsq_features + number_class_attr          

        class_array = np.full((nrows_to_load, 1), class_value, dtype= np.dtype(np.int8))

        # for x in range(0, nrows_to_load):          
        #     matrix[x,  ncols_to_load -1 + number_jnsq_features + number_class_attr]=class_value
        sys.stdout.write("Task:{}, File:{}, class_array.shape={}\n".format(task_id, pack, class_array.shape ))
        # print ("Task;{}; class_array.shape={}\n".format(task_id, class_array.shape))
        dset_class[r1:final_r2, 0:number_class_attr+1] = class_array                                    #  TypeError: Can't broadcast (1700, 0) -> (1700, 1)
        # print ("Task;{} r1={} final_r2={} class_start={} final_c2={}\n".format(task_id, r1,final_r2, class_start,final_c2))
        # matrix[:,number_of_features+number_jnsq_features:number_of_features+number_jnsq_features+number_class_attr] = class_array[...] 
        #matrix[0:final_r2-r1, class_start:final_c2] = class_array   #  ValueError: could not broadcast input array from shape (300,1) into shape (0,1) ValueError: could not broadcast input array from shape (300,1) into shape (0,0)
    # else:
    #     matrix = np.zeros((nrows_to_load, ncols_to_load), dtype= np.dtype(np.int8))



    t2 = time.perf_counter()
    print(f"Task:{task_id}; File:{pack}; Data transformed in; {t2 - t1:0.4f}; seconds")


    # load on hdf5 (write-operation)

    sys.stdout.write("Task:{}; File:{}; Try to save on HDF5 coordinates: [{},{}] to [{},{}]\n".format(task_id, pack, r1, final_r2, c1, final_c2 ))
    t1 = time.perf_counter()
    dset[r1:final_r2, c1:final_c2] = matrix    #dset[r1:c1, r2:c2] = matrix
    t2 = time.perf_counter()
    # sys.stdout.write("Task:{} File:{} Data Loaded on HDF5 in {t2 - t1:0.4f} seconds".format(task_id, filename))
    print(f"Task:{task_id}; File:{pack}; Data Loaded on HDF5 in; {t2 - t1:0.4f}; seconds")

    return None

def extract_lines(filePath):   
    # Hence we will use generators to get rid of any additional delimiter. #    , delimiters=[]
    with open(filePath) as f:
        for _ in range(1):   # step row 1
            next(f)
        for line in f:

            line = line.strip() #removes newline character from end
            line = line.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            #                   1234                 123                 12
            yield line

def main():

    # Parallel Data from MPI
    task_id = MPI.COMM_WORLD.rank                # The process ID (integer 0-3 for 4-process run)
    worker_nodes = MPI.COMM_WORLD.Get_size()     # Number of WorkerNodes/tasks valid are 1, 2 or 5 for 10 WorkPackges

    # Work to do from config file
    mydir = os.getcwd()      # mydir = '/users/hpc/pmorgado/laidp/t4/'    
    configuration = read_config(mydir)
    config_base = configuration[0]
    config_data = configuration[1]
    work_packages = len(config_data)

    sys.stdout.write("Task:{}; Parallel-LAID Module 1 - ETL dataset from textfiles into HDF5\n".format(task_id))
    sys.stdout.write("Task:{}: STORAGE_LAYOUT = {} - Experiment:{}\n".format(task_id,config_base[0][6],config_base[0][9]))    

    number_class_cols = config_base[0][4] # 1  
    number_dif_class_values = config_base[0][5]  # 2 # number of different values for class, used two 0 or 1
    number_jnsq_features = int( math.ceil( math.log( number_dif_class_values ) / math.log( 2 )  ) )

    nrows_data = config_base[0][2] # 2000
    ncols_data = config_base[0][3] # 200000
    ncols_total = ncols_data + number_jnsq_features + number_class_cols   # cols of data + jnsq cols + label col
    class_start = ncols_total -1

    hdf5_file = os.path.join(mydir, config_base[0][1])
    #hdf5_file = config_base[0][1]   # '/users/hpc/pmorgado/laidp/t2/laidp_dataset_1000k.hdf5'
    print (hdf5_file)
    if os.path.isfile(hdf5_file):
        sys.stdout.write("Task:{}; hdf5 File exist.\n".format(task_id))

        hf = h5py.File(hdf5_file, 'a', driver='mpio', comm=MPI.COMM_WORLD)  # 'a' Read/write if exists, create otherwise
        dset = hf['database']
        dset_class = hf['class']
    else:
        sys.stdout.write("Task:{}; hdf5 File not exist.\n".format(task_id))

        # dset = np.zeros((nrows, ncols_total), dtype= np.dtype(np.int8))  # demo version
        hf = h5py.File(hdf5_file, 'w', driver='mpio', comm=MPI.COMM_WORLD)  # 'w' = Create file, truncate if exists

   
        # HDF5 STORAGE_LAYOUT
        storage_layout  = config_base[0][6]  # chunked or contiguous
        chunk_row_size  = config_base[0][7]  # if chunked nrows or zero 
        chunk_col_size  = config_base[0][8]  # if chunked columns or zero 

        if storage_layout == "CONTIGUOUS":
            dset = hf.create_dataset('database', (nrows_data, ncols_total), dtype= np.dtype(np.int8))
        else:
            # chunk size
            dset = hf.create_dataset('database', (nrows_data, ncols_total), chunks=(chunk_row_size, chunk_col_size), dtype= np.dtype(np.int8))

        dset_class = hf.create_dataset('class', (nrows_data, number_class_cols), dtype= np.dtype(np.int8))
        # dset_redundant = hf.create_dataset('redundant', (nrows, 1), dtype= np.dtype(np.int8))

        dset.attrs['storage_layout'] = storage_layout
        dset.attrs['chunk_row_size'] = chunk_row_size
        dset.attrs['chunk_col_size'] = chunk_col_size    

        dset.attrs['nrows_data']    = nrows_data
        dset.attrs['ncols_data']    = ncols_data        # cols of data       
        dset.attrs['ncols_total']   = ncols_total       # cols of data + jnsq cols + label col

        dset.attrs['data_starts']   = 0                 # due to base 0 on h5py
        dset.attrs['data_ends']     = ncols_data-1      # due to base 0 on h5py

        dset.attrs['number_jnsq_features'] = number_jnsq_features   # reserved cols if not needed remains with zero value   
        dset.attrs['jnsq_starts'] = ncols_data + number_jnsq_features -1    # due to base 0 on h5py
        dset.attrs['jnsq_ends']   = ncols_data + number_jnsq_features -1    # due to base 0 on h5py    
        
        dset.attrs['number_dif_class_values'] = number_dif_class_values
        dset.attrs['number_class_cols'] = number_class_cols
        dset.attrs['class_starts'] = class_start # ncols_data + number_jnsq_features + number_class_cols -1
        dset.attrs['class_ends']   = ncols_data + number_jnsq_features + number_class_cols -1

        dset.attrs['rows_on_disjoint_matrix'] = 0

    sys.stdout.write("Task:{}; dataset name: {}; shape: {}\n".format(task_id, dset.name ,dset.shape))
    sys.stdout.write("Task:{}; dataset name: {}; shape: {}\n".format(task_id, dset_class.name ,dset_class.shape))


    exec_node_range = map_parallel_worker_to_workpack(task_id, worker_nodes, work_packages)    #print (exec_node_range)
    sys.stdout.write("Task:{}, processing work for {} index\n".format(task_id, exec_node_range))

    for i in range(exec_node_range[0] , exec_node_range[1] + 1, 1):
        # print (i)
        #print(config_data[i])
 
        etl_from_file(task_id, i, dset, dset_class, config_base, config_data, number_jnsq_features, number_class_cols, class_start, ncols_total)        # load data

        hf.flush() # DEMO np.save(hdf5_file, dset)



    print ('display dset_class')
    print (dset_class[:,-1])

    # dset[:,-1] = dset_class
    dset[:,-1] = dset_class[:,0]   

    print ('display dset')
    print (dset[:,-1])


    sys.stdout.write("Task:{} try close hdf5 File.\n".format(task_id))
    hf.close()
    sys.stdout.write("Task:{} closed hdf5 File.\n".format(task_id))

if __name__ == "__main__":
    main()    

