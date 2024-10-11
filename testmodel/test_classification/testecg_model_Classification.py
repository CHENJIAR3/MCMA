# The classification model is from 
# Automatic Diagnosis of the 12-Lead {{ECG}} Using a Deep Neural Network
# https://github.com/antonior92/automatic-ecg-diagnosis/tree/master?tab=readme-ov-file\
import os

import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
from testmodel.testecg_model_HR import paddingfor16
import tensorflow as tf
# from ekgan import read_tfrecords,load_label_generator
import numpy as np
from utils import args
# from models import AutoEncoder,unet3plus
tf.config.run_functions_eagerly(True)
import argparse
import  pandas as pd
import h5py
import math
from tensorflow.keras.utils import Sequence
from sklearn.metrics import (confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score,
                             precision_recall_curve)


class ECGSequence(Sequence):
    @classmethod
    def get_train_and_val(cls, path_to_hdf5, hdf5_dset, path_to_csv, batch_size=8, val_split=0.02):
        n_samples = len(pd.read_csv(path_to_csv))
        n_train = math.ceil(n_samples*(1-val_split))
        train_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, end_idx=n_train)
        valid_seq = cls(path_to_hdf5, hdf5_dset, path_to_csv, batch_size, start_idx=n_train)
        return train_seq, valid_seq

    def __init__(self, path_to_hdf5, hdf5_dset, path_to_csv=None, batch_size=8,
                 start_idx=0, end_idx=None):
        if path_to_csv is None:
            self.y = None
        else:
            self.y = pd.read_csv(path_to_csv).values
        # Get tracings
        self.f = h5py.File(path_to_hdf5, "r")
        self.x = self.f[hdf5_dset]
        self.batch_size = batch_size
        if end_idx is None:
            end_idx = len(self.x)
        self.start_idx = start_idx
        self.end_idx = end_idx

    @property
    def n_classes(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        start = self.start_idx + idx * self.batch_size
        end = min(start + self.batch_size, self.end_idx)
        if self.y is None:
            return np.array(self.x[start:end, :, :])
        else:
            return np.array(self.x[start:end, :, :]), np.array(self.y[start:end])

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)

    def __del__(self):
        self.f.close()
def get_scores(y_true, y_pred, score_fun):
    nclasses = np.shape(y_true)[1]
    scores = []
    for name, fun in score_fun.items():
        scores += [[fun(y_true[:, k], y_pred[:, k]) for k in range(nclasses)]]
    return np.array(scores).T
def specificity_score(y_true, y_pred):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    spc = m[0, 0] * 1.0 / (m[0, 0] + m[0, 1])
    return spc
def paddingecg(ecg12,index):
    ecg12 = tf.transpose(tf.cast(ecg12,dtype=tf.float32), [0, 2, 1])

    updates = tf.gather_nd(ecg12, index)
    ecg_new = tf.zeros_like(ecg12, dtype=tf.float32)
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new
def updateecg(ecg12,ecg1,index):
    ecg12 = tf.transpose(tf.cast(ecg12,dtype=tf.float32), [0, 2, 1])
    ecg1 = tf.transpose(tf.cast(ecg1,dtype=tf.float32), [0, 2, 1])
    updates = tf.gather_nd(ecg1, index)
    ecg12 = tf.tensor_scatter_nd_update(ecg12, index, updates)
    ecg12 = tf.transpose(tf.cast(ecg12,dtype=tf.float32), [0, 2, 1])
    return ecg12
def convert_df(y,y_true, threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])):
    y = (y > threshold).astype(np.float32)

    scores = get_scores(y_true, y, score_fun)
    scores = np.concatenate([scores, np.mean(scores, axis=0)[None, :]], axis=0)
    df = pd.DataFrame(scores, columns=['Precision',
                                        'Recall', 'Specificity',
                                        'F1 score','Acc'], index=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'Mean']).round(4)

    return df
def plot_rec_pre(y_true,y_pred,label=['Original','CGAN','MAUNET','MEGAN','EKGAN','Proposed (Lead I)','Proposed (Lead II)'],
                 diagnosis = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'],fontsize=16):
    for i in range(6):
        plt.figure()
        j=0
        for y in y_pred:
            precision, recall, threshold = precision_recall_curve(y_true[:, i], y[:, i])
            plt.plot(recall,precision,label=label[j])
            plt.grid(True)
            plt.xlabel("Recall",fontsize=fontsize)
            plt.ylabel("Precision",fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.legend(loc="lower left",fontsize=fontsize)
            j=j+1
        # plt.show()
        plt.savefig("../../results/figure/Rec_Pre_curve_"+str(diagnosis[i])+".png",dpi=1600)

if __name__ == '__main__':
    threshold = np.array([0.124, 0.07, 0.05, 0.278, 0.390, 0.174])
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    # input path_to_hdf5, path_to_model
    # path_to_hdf5='path for ecg_tracings.hdf5'
    # path_to_model='path for model.hdf5'

    dataset_name='tracings'
    score_fun = {'Precision': precision_score,
                 'Recall': recall_score, 'Specificity': specificity_score,
                 'F1 score': f1_score, 'acc':accuracy_score}
    # labelpath: the path for gold_standard.csv
    y_true = pd.read_csv(labelpath).values

    # Import data
    # seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    ecgdata = h5py.File(path_to_hdf5)
    ecg12   = ecgdata[dataset_name]
    # Import model
    model = tf.keras.models.load_model(path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())
    ecg12input=1
    output_excel=1
    ecg1input=0
    compared=0
    resultpath = '../../results/classification/'
    args.testmodel='ae'

    if ecg1input:
        for mode in ['copy','zeros']:
            excel_path = resultpath + 'classification_lead1_performance'+mode+'.xlsx'
            if not os.path.exists(resultpath):
                os.mkdir(resultpath)
            df={}
            for i in range(12):
                if mode=='copy':
                    ecg1 = tf.tile(ecg12[:, :, i:i + 1], [1, 1, 12])
                else:
                    # zero padding
                    l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)
                    h_index = i*np.ones((ecg12.shape[0], 1)).astype(np.int32)
                    index = np.hstack((l_index, h_index))
                    ecg1 = paddingecg(ecg12,index)
                y1 = model.predict(ecg1,verbose=1)
                y1 = (y1 > threshold).astype(np.float32)
                scores = get_scores(y_true, y1, score_fun)
                scores = np.concatenate([scores, np.mean(scores, axis=0)[None, :]], axis=0)
                df[i] = pd.DataFrame(scores, columns=['Precision',
                                                    'Recall', 'Specificity',
                                                    'F1 score','Acc'],
                                   index=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'Mean']).round(4)
                # print(i)
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                for i in range(12):
                    df[i].to_excel(writer, sheet_name='lead'+str(i+1), index=True)
    if ecg12input:
        # args.testmodel=None
        yo = model.predict(ecg12,  verbose=1)
        if output_excel:
            df0 = convert_df(yo,y_true,threshold)
        if args.testmodel =='ae':
            # modelpath = '../../abliation_study/Autoencoder_zeros_1'
            modelpath = '../../abliation_study/Autoencoder_ks5'
            generator = tf.keras.models.load_model(modelpath)
            excel_path = resultpath + 'classification_AE_ks5.xlsx'
            ecg12_reshaped = tf.reshape(ecg12, shape=(-1, 1024, 12))
            df={}
            for i in range(12):

                l_index = np.arange(ecg12_reshaped.shape[0]).reshape(-1, 1)
                h_index = i * np.ones((ecg12_reshaped.shape[0], 1)).astype(np.int32)
                index = np.hstack((l_index, h_index))

                ecg1 = paddingecg(ecg12_reshaped, index)
                gen_ecg12 = generator.predict(ecg1)
                gen_ecg12 = updateecg(gen_ecg12,ecg1,index)
                # 改变幅值
                gen_ecg12_reshaped = tf.reshape(gen_ecg12, shape=(-1, 4096, 12))
                ya = model.predict(gen_ecg12_reshaped, verbose=1)
                ya = (ya > threshold).astype(np.float32)
                scores = get_scores(y_true, ya, score_fun)
                scores = np.concatenate([scores, np.mean(scores, axis=0)[None, :]], axis=0)
                df[i] = pd.DataFrame(scores, columns=['Precision',
                                                      'Recall', 'Specificity',
                                                      'F1 score', 'Acc'],
                                     index=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'Mean']).round(4)
                # print(i)
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                for i in range(12):
                    df[i].to_excel(writer, sheet_name='lead' + str(i + 1), index=True)
        if compared:
            for mode in ['cgan','unet','megan','ekgan']:
                args.testmodel=mode
                if args.testmodel == 'cgan':
                    # CGAN
                    modelpath1 = '../../CGAN/generator'
                    generator = tf.keras.models.load_model(modelpath1)
                    ecg1 = ecg12[:,:,0:1]
                    ecg1_reshaped = tf.reshape(ecg1, shape=(-1, 1024, 1))
                    gen_ecg12 = generator(ecg1_reshaped)

                    l_index = np.arange(gen_ecg12.shape[0]).reshape(-1, 1)
                    h_index = 0 * np.ones((gen_ecg12.shape[0], 1)).astype(np.int32)
                    index = np.hstack((l_index, h_index))
                    gen_ecg12 = updateecg(gen_ecg12, ecg1_reshaped, index)

                    gen_ecg12_reshaped = tf.reshape(gen_ecg12, shape=(-1, 4096, 12))
                    yc = model.predict(gen_ecg12_reshaped, verbose=1)
                    if output_excel:
                        df1 = convert_df(yc,y_true, threshold)

                if args.testmodel == 'unet':
                    # MAUNet
                    modelpath2='../../MAUNet'
                    generator = tf.keras.models.load_model(modelpath2)
                    ecgII = ecg12[:,:,1:2]
                    ecgII_reshaped= tf.reshape(ecgII, shape=(-1, 1024, 1))
                    ds = tf.data.Dataset.from_tensor_slices(ecgII_reshaped).batch(32)
                    gen_ecg12 = generator.predict(ds)

                    l_index = np.arange(gen_ecg12.shape[0]).reshape(-1, 1)
                    h_index = 1 * np.ones((gen_ecg12.shape[0], 1)).astype(np.int32)
                    index = np.hstack((l_index, h_index))
                    gen_ecg12 = updateecg(gen_ecg12, ecgII_reshaped, index)

                    gen_ecg12_reshaped = tf.reshape(gen_ecg12, shape=(-1, 4096, 12))
                    yu = model.predict(gen_ecg12_reshaped, verbose=1)
                    if output_excel:
                        df2 = convert_df(yu,y_true,threshold)
                if args.testmodel == 'megan':
                    # MEGAN
                    modelpath3 = '../../MEGAN/generator'
                    generator = tf.keras.models.load_model(modelpath3)

                    gen_ecg12 = generator(ecg1_reshaped)
                    l_index = np.arange(gen_ecg12.shape[0]).reshape(-1, 1)
                    h_index = 0 * np.ones((gen_ecg12.shape[0], 1)).astype(np.int32)
                    index = np.hstack((l_index, h_index))
                    gen_ecg12 = updateecg(gen_ecg12, ecg1_reshaped, index)
                    gen_ecg12_reshaped = tf.reshape(gen_ecg12, shape=(-1, 4096, 12))
                    ym = model.predict(gen_ecg12_reshaped, verbose=1)
                    if output_excel:
                        df3 = convert_df(ym,y_true,threshold)
                if args.testmodel == 'ekgan':
                    # EKGAN
                    ecg1_copy = tf.tile(ecg1_reshaped,[1,1,12])
                    ecg1_copy = paddingfor16(ecg1_copy)
                    modelpath4 = '../../EKGAN/inference_generator'
                    generator = tf.keras.models.load_model(modelpath4)
                    ds = tf.data.Dataset.from_tensor_slices(ecg1_copy).batch(32)
                    gen_ecg12, _ = generator.predict(ds)
                    gen_ecg12 = gen_ecg12[:, 2:-2, :, 0]
                    gen_ecg12 = tf.transpose(gen_ecg12, [0, 2, 1])

                    l_index = np.arange(gen_ecg12.shape[0]).reshape(-1, 1)
                    h_index = 0 * np.ones((gen_ecg12.shape[0], 1)).astype(np.int32)
                    index = np.hstack((l_index, h_index))
                    gen_ecg12 = updateecg(gen_ecg12, ecg1_reshaped, index)

                    gen_ecg12_reshaped = tf.reshape(gen_ecg12, (-1, 4096, 12))
                    ye = model.predict(gen_ecg12_reshaped, verbose=1)
                    if output_excel:
                        df4 = convert_df(ye,y_true,threshold)

            resultpath = '../../results/classification/'
            excel_path = resultpath+'new_classification_performance.xlsx'

            if not os.path.exists(resultpath):
                os.mkdir(resultpath)
            if output_excel:
                with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                    df0.to_excel(writer, sheet_name='original_12lead', index=True)
                    df1.to_excel(writer, sheet_name='CGAN', index=True)
                    df2.to_excel(writer, sheet_name='MAUNet', index=True)
                    df3.to_excel(writer, sheet_name='MEGAN', index=True)
                    df4.to_excel(writer, sheet_name='EKGAN', index=True)
                print("Output predictions saved")

