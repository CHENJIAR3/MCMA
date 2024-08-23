#这是一个测试模型生成性能的函数
# Automatic Diagnosis of the 12-Lead {{ECG}} Using a Deep Neural Network
# https://github.com/antonior92/automatic-ecg-diagnosis/tree/master?tab=readme-ov-file\
# 这个代码测试模型的分类性能，测试集采用上述的逻辑
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
from testecg_model_HR import paddingfor16
import tensorflow as tf
# from ekgan import read_tfrecords,load_label_generator
import numpy as np
from utils import args,read_tfrecords_Long
# from models import AutoEncoder,unet3plus
tf.config.run_functions_eagerly(True)
import argparse
import  pandas as pd
import neurokit2 as nk
from testecg_model import test_ae,test_ekgan,test_unet,write2excel
from testecg_model_HR import test_ekgan_hr,test_unet_hr,write2excel_hr
import h5py
import math
from tensorflow.keras.utils import Sequence
from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score,
                             precision_recall_curve, average_precision_score)
import  warnings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    path_to_hdf5='/data/0shared/chenjiarong/data/ecg_tracings.hdf5'
    path_to_model='/data/0shared/chenjiarong/model/model.hdf5'
    signal_level= 0
    dataset_name='tracings'


    ecgdata = h5py.File(path_to_hdf5)
    ecg12   = ecgdata[dataset_name]
    ecg12 = tf.cast(ecg12,dtype=tf.float32)
    resultpath ='./results/code_test/'

    if signal_level:
        ecg12 = tf.reshape(ecg12, shape=(-1, 1024, 12))
        testds=tf.data.Dataset.from_tensor_slices((ecg12)).batch(args.bs)
        for mode in ['ae']:
            args.testmodel=mode
            # 'ae','unet','ekgan','megan','cgan'
            #
            if args.testmodel=='ae':
                for [anylead,padding] in [[0,'copy'],[1,'copy']]:
                    args.padding=padding
                    modelpath='Autoencoder'+str(anylead)
                    # modelpath="model_ae_any" + str(anylead) + "_" + padding
                    #modelpath="unet3+_any" + str(anylead) + "_" + padding+"_Te"
                    # modelpath="model_cycleae1to12/best_anylead" + str(anylead) + "_" + padding
                    model1 = tf.keras.models.load_model(modelpath)
                    MAE_test,MSE_test,CC_test=test_ae(model=model1,ds=testds,output_num=2)
                    print(CC_test)
                    excel_path=resultpath+'/AutoEncoder_result/anylead'+str(anylead)+"_"+padding+'.xlsx'
                    if not os.path.exists(resultpath):
                        os.mkdir(resultpath)
                    write2excel(MAE_test, MSE_test, CC_test, excel_path)
            if args.testmodel=='unet':
                modelpath = './MAUNet'
                model1 = tf.keras.models.load_model(modelpath)
                MAE_test,MSE_test,CC_test=test_unet(model=model1,ds=testds)
                excel_path = resultpath + 'MAUNet.xlsx'
            if args.testmodel=='ekgan':
                modelpath='./EKGAN/inference_generator'
                model1 = tf.keras.models.load_model(modelpath)
                MAE_test,MSE_test,CC_test=test_ekgan(model=model1,ds=testds)
                excel_path = resultpath + 'EKGAN.xlsx'
            if args.testmodel=='megan':
                modelpath = './MEGAN/generator'
                model1 = tf.keras.models.load_model(modelpath)
                MAE_test, MSE_test, CC_test = test_unet(model=model1, ds=testds)
                excel_path = resultpath + 'MEGAN.xlsx'
            if args.testmodel=='cgan':
                modelpath = './CGAN/generator'
                model1 = tf.keras.models.load_model(modelpath)
                MAE_test, MSE_test, CC_test = test_unet(model=model1, ds=testds)
                excel_path = resultpath + 'CGAN.xlsx'
            if not os.path.exists(resultpath):
                os.mkdir(resultpath)
            print(np.mean(CC_test,axis=1))
            write2excel(MAE_test, MSE_test, CC_test, excel_path)
    else:
        testds=tf.data.Dataset.from_tensor_slices((ecg12)).batch(args.bs)
        for mode in ['unet','ekgan','megan','cgan']:
            args.testmodel=mode
            if args.testmodel == 'unet':
                modelpath = './MAUNet'
                model1 = tf.keras.models.load_model(modelpath)
                rhr, fhr = test_unet_hr(model=model1, ds=testds, numlead=2)
                excel_path = resultpath + 'MAUNet_hr.xlsx'
            if args.testmodel == 'ekgan':
                modelpath = './EKGAN/inference_generator'
                model1 = tf.keras.models.load_model(modelpath)
                rhr, fhr = test_ekgan_hr(model=model1, ds=testds)
                excel_path = resultpath + 'EKGAN_hr.xlsx'
            if args.testmodel == 'megan':
                modelpath = './MEGAN/generator'
                model1 = tf.keras.models.load_model(modelpath)
                rhr, fhr = test_unet_hr(model=model1, ds=testds)
                excel_path = resultpath + 'MEGAN_hr.xlsx'
            if args.testmodel == 'cgan':
                modelpath = './CGAN/generator'
                model1 = tf.keras.models.load_model(modelpath)
                rhr, fhr = test_unet_hr(model=model1, ds=testds)
                excel_path = resultpath + 'CGAN_hr.xlsx'
            if not os.path.exists(resultpath):
                os.mkdir(resultpath)
            write2excel_hr(rhr, fhr, excel_path)