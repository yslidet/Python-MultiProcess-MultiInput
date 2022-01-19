#%% Libraries  -------------------------------------
import time
from multiprocessing import Pool, Manager

import pandas as pd
import numpy as np
import os
from os.path import join

from tqdm import tqdm, trange

# multiprocess
from itertools import product

#for plot
import matplotlib.pyplot as plt



#%% Function - Data Augmentation - 2
import random

def funct_worker(input_list,pre_text):
    """
    Worker Function: define function that each process should do
    e.g. create string from content in input_list while begin with pre_text => "[pre_text] [input_list]
    """
    output_string = f"{pre_text}"
    output_string = output_string + " ".join(input_list)

    return output_string


def funct_manager(list_to_write, pre_text = "-", worker=-1):
    """
    Manager Function:
    :param list_to_combine:
    :param worker: number of processes to be used; default = -1: use all available process
    :return:
    """
    #prepare data
    # list_to_write
    # pre_text

    # multi-processing
    if worker==-1:
        pool = Pool()
    else:
        pool = Pool(worker)

    #data_aug = (xy_aug, rss_aug)
    output_data = []
    output_data += pool.starmap(funct_worker,zip(list_to_write,[pre_text for i in range(len(list_to_write))]))
    pool.close()
    pool.join()
    pool.terminate()

    #combine data
    final_output = ""
    for output_data in tqdm(output_data, desc="Combine Data"):
        final_output += output_data + "\n"

    return final_output



#%% MAIN - Test Multiprocess --------------------------------------------------------------------------
if __name__ == '__main__':
    #%% Initialise Data
    list_to_write = [
        ["Hello", "world"],
        ["I", "want", "to", "use", "this", "example", "that", "multiprocess", "works"],
        ["so", "different", "WORKERs", "will", "be", "working", "here"],
        ["yah", "Bye!"]
    ]

    #%% Using Multiprocess
    start_time = time.time()
    final_output = funct_manager(list_to_write, pre_text="-", worker=3)
    end_time = time.time()
    print(f"with multiprocess: t={end_time-start_time}s \nfinal_output= \n{final_output}")

    """
    #%% Without Using Multiprocess
    start_time = time.time()
    output_data = []
    for data in list_to_write:
        output_data.append(funct_worker(data,pre_text='-'))

    final_output = ""
    for output in output_data:
        final_output += output + "\n"
    end_time = time.time()
    print(f"without multiprocess: t={end_time-start_time}s  \nfinal_output= \n{final_output}")
    """
