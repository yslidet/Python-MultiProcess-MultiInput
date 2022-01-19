#%% Libraries  -------------------------------------
import time
from multiprocessing import Pool, Manager

import pandas as pd
import numpy as np
import os
from os.path import join

from tqdm import tqdm, trange

#data path
from data.dataset_dict import get_dataset_path

# multiprocess
from itertools import product

#for plot
import matplotlib.pyplot as plt



#%% Function - Data Augmentation - 2
import random

def funct_worker(input_list):
    """
    Worker Function: define function that each process should do
    e.g. combine list into string
    """
    output_string = " ".join(input_list)

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
    data_aug = []
    data_aug += pool.starmap(aug_data_2_sample,zip(list_to_write,[pre_text for i in range(len(list_to_write))]))
    pool.close()
    pool.join()
    pool.terminate()

    #combine data - TODO:  to update
    for data in tqdm(data_aug, desc="Combine Data"):
        df_aug_ap = df_aug_ap.append(data[1], ignore_index=True)  # save augmented data
        df_aug_info = df_aug_info.append(data[0], ignore_index=True)

    return df_aug_info, df_aug_ap

def test_multi_p(data_i,data_j):
    print(data_i, data_j)



#%% MAIN - Test Multiprocess --------------------------------------------------------------------------
if __name__ == '__main__':
    funct_manager(list_to_write, pre_text="-", worker=-1)
"""
#%% SETUP logger & header
REPORTER = 'AUG_Scheme2'
from helper.logger import init_logger, update_logger_path
logger=init_logger(logger_name=REPORTER)

from helper.project_header import get_project_path
project_path = get_project_path(use_python_console=False)
if __name__ == '__main__':
    #%% INIT Data Path ----------------------------------------------
    # - choose dataset
    # data_folder = 'uji100_train'  #choose data package
    data_folder = 'ng_raw_train'  #choose data package
    # data_folder = use_parser()  #use parser

    # - load data_path
    main_train_path, train_data_file, train_data_loc, train_label_file, train_label_loc = get_dataset_path(data_folder)

    #%% INIT ----------------------------------------------
    # case_name = 'EXAMPLE'
    case_name = f'{REPORTER}'

    result_main_path = os.path.join(project_path, 'result')
    os.makedirs(os.path.join(result_main_path), exist_ok=True)  # if result path not exist, create the folder

    result_folder = f'{data_folder}_{case_name}'
    # result_folder = f'{data_folder}_{case_name}_{datetime.now().strftime("%Y%m%d")}'          #result_folder = "REPORTER_Data_YYYYMMDD"
    result_path = os.path.join(result_main_path, result_folder)
    os.makedirs(result_path, exist_ok=True)

    # logger = update_logger_path(REPORTER, result_path)
    update_logger_path(REPORTER, result_path)

    #%% Import DATA  ----------------------------------------------
    df_train_rssi = pd.read_csv(os.path.join(project_path, main_train_path, train_data_file),header=train_data_loc[2]).iloc[:, train_data_loc[0]:train_data_loc[1]]
    df_train_rssi[df_train_rssi==100]=-110
    df_train_rssi[df_train_rssi==-100]=-110
    # df_train_xy = pd.read_csv(os.path.join(project_path, main_train_path, train_label_file), header=train_label_loc[2]).iloc[:, train_label_loc[0]:train_label_loc[1]]
    df_train_info = pd.read_csv(os.path.join(project_path, main_train_path, train_label_file), header=train_label_loc[2]).iloc[:, train_data_loc[1]:]
    # df_test_rssi = pd.read_csv(os.path.join(project_path, main_test_path, test_data_file), header=test_data_loc[2]).iloc[:, test_data_loc[0]:test_data_loc[1]]
    # df_test_rssi[df_test_rssi==100]=-110
    # df_test_rssi[df_test_rssi==-100]=-110
    # df_test_xy = pd.read_csv(os.path.join(project_path, main_test_path, test_label_file), header=test_label_loc[2]).iloc[:, test_label_loc[0]:test_label_loc[1]]

    #%%
    # df_rssi = df_rssi.replace(-100, -110)
    # rss_list = [df_rssi.iloc[i] for i in trange(len(df_rssi))]
    # xy_list = [df_info.iloc[i] for i in trange(len(df_info))]
    #
    # # init new data
    # df_aug_ap = pd.DataFrame(columns=df_rssi.columns)
    # df_aug_info = pd.DataFrame(columns=df_info.columns)
    # # multi-processing
    # pool = Pool()
    # xy_aug, rss_aug = pool.starmap(aug_data_2_sample,zip(xy_list,rss_list,[2 for i in range(len(df_rssi))],[-110 for i in range(len(df_rssi))]))
    # df_aug_ap = df_aug_ap.append(rss_aug, ignore_index=True)  # save augmented data
    # df_aug_info = df_aug_info.append(xy_aug, ignore_index=True)
    # pool.close()
    # pool.join()

    #or just run
    start = time.time()
    df_aug_info, df_aug_ap = aug_data_2_multi(df_train_info, df_train_rssi, N=30, defaultMissingAP=-110, worker=-1)
    end = time.time()

    print(f'Run Time = {end-start}')
    df_aug_ap.to_csv(os.path.join(result_path,f'scheme2_lidet_N30_ap.csv'),header=True,index=False)
    df_aug_info.to_csv(os.path.join(result_path,f'scheme2_lidet_N30_info.csv'),header=True,index=False)
"""