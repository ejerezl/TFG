from svg.path import parse_path # requires svg.path: pip install svg.path
from svg.path.path import Line
from xml.dom import minidom
from PyPDF2 import PdfFileReader # requires PyPDF2: pip install PyPDF2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os, sys
import os.path
import subprocess
from skfda import FDataGrid
import sklearn
from itertools import combinations
from skfda.ml.clustering import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import time
import math
import sklearn.cluster as sk
from skfda.preprocessing.dim_reduction.projection import FPCA
import statistics as st
from scipy import stats
import random
from skfda.misc.metrics import l2_distance, TransformationMetric, LpDistance, fisher_rao_distance


sys.path.append('dahfi_main/database_ctu-chb/')

from src.data import load_FHR_UC, load_targets, load_features_targets
from src.preprocessing import preprocessing_signals, remove_long_gaps, all_zero_move

# Esta función ya esta implementada en el paquete de preprocesamiento, pero incluye más funcionalidad que da fallos en el caso de UC ya que hay una observación [], 
# es por ello que aquí la definimos
def preprocessing(signals, secs_gap=15, mins_cut=15):        
    #1) Remove long gaps (> 15 sec).
    signals = [remove_long_gaps(row, secs=secs_gap) for row in signals]
    #2) Zero remove and last 15 min of signal
    signals = [all_zero_move(row, mins_cut=mins_cut) for row in signals]
    return signals

''' Función para eliminar nans de los arrays pasados como argumentos. Inicialmente está pensada solo para quitar el hueco vacío de UC, pero por ejemplo
    para calcular los datos como combinacion de fhr y uc deben tener la misma dimensión.'''
def remove_nans(uc, fhr=None,patients=None, target=None):  
    print("Removing Nans")
    fhr2 = [] 
    try:
        i = [ele.values.tolist() for ele in uc].index([])
    except:
        print('Element not in list')
    else:
        if uc:
            uc2 = uc[:i] + uc[i+1:]
        if fhr:
            fhr2 = fhr[:i] + fhr[i+1:]
        if patients:
            patients = patients[:i] + patients[i+1:]
        if target is not None:
            target = target[:i].tolist() + target[i+1:].tolist()
    return uc2, fhr2, patients, target

def get_clean_data():
    print('Loading data')
    fhr_df, uc_df = load_FHR_UC('dahfi_main/database_ctu-chb/data/')

    # Limpiamos los datos
    fhr = preprocessing(fhr_df.to_numpy())
    uc = preprocessing(uc_df.to_numpy())
    return fhr, uc


def get_fhr_uc(target = None):
    fhr, uc = get_clean_data()

    uc2,_,_,target2= remove_nans(uc=uc, target=target)
    uc_data = np.array(uc2)
    fhr_data = np.array(fhr)

    print("Transformation to FDataGrid")
    # Calculamos datagrid para datos funcionales únicamente con UC o FHR
    fd_uc = FDataGrid(data_matrix=uc_data, grid_points=np.arange(0, uc_data.shape[1]), dataset_name="UC", argument_names=('Time(s)',), coordinate_names=('UC',))
    fd_fhr = FDataGrid(data_matrix=fhr_data, grid_points=np.arange(0, fhr_data.shape[1]), dataset_name="FHR", argument_names=('Time(s)',), coordinate_names=('FHR',))
    
    return fd_fhr, fd_uc, target2

def get_combined(target = None):
    fhr, uc = get_clean_data()

    uc2,fhr2,_,target2= remove_nans(uc=uc, fhr=fhr, target=target)

    uc_data = np.array(uc2)
    fhr_data = np.array(fhr2)
    points = np.arange(0, uc_data.shape[1])

    print("Transformation to FDataGrid")
    data_matrix = []
    for ele in list(map(lambda x,y: [x,y], uc_data, fhr_data)):
        data_matrix += [list(map(lambda x,y: [x,y], ele[0], ele[1]))]
    np.shape(data_matrix)
    fd = FDataGrid(data_matrix=data_matrix, grid_points=points)

    return fd