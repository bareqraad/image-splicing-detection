# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 14:17:19 2020

@author: Bareq
effective image splcing detection based on image chroma
"""
import multiprocessing
import os
import time
import skimage.feature as feature
from skimage.feature import  greycoprops
import csv
from PIL import Image
import numpy as np
#from matplotlib import pyplot as plt
threshold = 10
level = threshold + 1
band = 'cb'
def read(path):
    img = Image.open(path)
    return  img.convert('YCbCr')

def GLCM(image, angle):
    glcm = feature.greycomatrix(image, distances=[1], angles=[angle], levels=level, symmetric=False, normed=False)   
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    homogeneity = greycoprops(glcm, 'homogeneity')
    correlation=greycoprops(glcm, 'correlation') 
    energy=greycoprops(glcm, 'energy')
    ASM=greycoprops(glcm, 'ASM')
    glcmpros = [contrast.item() ,dissimilarity.item(),homogeneity.item(),energy.item(),correlation.item(),ASM.item()]
    args = (np.array(glcm[:, :, 0, 0]).flatten(),glcmpros)
    x =  np.concatenate(args)
    return x


def extract_features(image, label):
    (y, cb, cr) = image.split()
    arr = np.array(cr)
    
    Harr = np.clip(np.abs(arr[:,:-1].astype(np.int16)  -arr[:,1:].astype(np.int16)),0,threshold)
    Varr = np.clip(np.abs(arr[:-1,:].astype(np.int16)  -arr[1:,:].astype(np.int16)),0,threshold)
    Darr = np.clip(np.abs(arr[:-1,:-1].astype(np.int16)-arr[1:,1:].astype(np.int16)),0,threshold)
    Marr = np.clip(np.abs(arr[:-1,1:].astype(np.int16) -arr[1:,:-1].astype(np.int16)),0,threshold)
    
    Cmh = GLCM(Harr, 0)
    Cmv = GLCM(Varr, 90)
    Cmd = GLCM(Darr, 45)
    Cmm = GLCM(Marr, -45)

    args = (Cmh,Cmv,Cmd,Cmm)

    FeatureVector = np.concatenate(args)
    FeatureVector = np.array(FeatureVector).tolist()
    FeatureVector.append(label)

    return FeatureVector


def getPersantage (iteration, total):
    return 100 * (iteration / float(total))


def work(index, filename, totalfiles, path, label):
    start = time.perf_counter()

    result = 0
    try:
        image = read(os.path.join(path, filename))
        result = extract_features(image, label)
    except:
        print(f"error in: {filename}")

    finish = time.perf_counter()

    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f'[{current_time}] '
          f'{filename} done in: {round((finish - start), 2): <5} seconds '
          f'{round(getPersantage(index, totalfiles), 2): 6}% of {label}')

    return result


if __name__ == "__main__":
    '''
    img = read('f:/1/1.tif')
    extract_features(img,'aaa')
    '''
    cores = os.cpu_count()
    path = 'F:/Research/DataSets/CASIA2/Au'
    list = os.listdir(path)
    totalFiles1 = len(list)
    label1 = 'Authentic'

    start = time.perf_counter()

    results = []
    with multiprocessing.Pool(processes=cores) as pool:
        for index, name in enumerate(list, start=1):
            results.append(pool.apply_async(work, args=(index, name, totalFiles1, path, label1)))

        results = [result.get() for result in results]

    finish = time.perf_counter()
    time1 = round((finish - start)/60, 2)
    print(f'\nAuthentic total time: {time1} minutes')
    with open('f:/'+band+str(threshold)+'.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, dialect='excel')
        wr.writerows(results)
    myfile.close()
###################################################################################
    path = 'F:/Research/DataSets/CASIA2/tp'
    list = os.listdir(path)
    totalFiles2 = len(list)

    start = time.perf_counter()
    label2 = 'Spliced'
    results = []
    with multiprocessing.Pool(processes=cores) as pool:
        for index, name in enumerate(list, start=1):
            results.append(pool.apply_async(work, args=(index, name, totalFiles2, path, label2)))

        results = [result.get() for result in results]

    finish = time.perf_counter()
    time2 = round((finish - start)/60, 2)
    print(f'\nspliced total time: {time2} minutes')

    with open('f:/'+band+str(threshold)+'.csv', 'a', newline='') as myfile:
        wr = csv.writer(myfile, dialect='excel')
        wr.writerows(results)
    myfile.close()

    print(f"\nFinished {totalFiles1} images as {label1} in {time1} minutes"
      f"\n     and {totalFiles2} images as {label2}   in {time2} minutes\n\n")
