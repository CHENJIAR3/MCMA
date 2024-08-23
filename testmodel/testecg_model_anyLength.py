# 这是一个测试模型生成性能的函数
# 任意长度的实验结果
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,6,7'

import tensorflow as tf
import numpy as np
from utils import args,read_tfrecords_Long
tf.config.run_functions_eagerly(True)

import  pandas as pd

def paddingecg(ecg12,index):
    ecg12 = tf.transpose(ecg12, [0, 2, 1])
    ecg_new = tf.zeros_like(ecg12, dtype=tf.float32)
    updates = tf.gather_nd(ecg12, index)
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new

def compute_metric(gen_ecg12,ecg12):
    mae= tf.reduce_sum(tf.reduce_mean(tf.abs(gen_ecg12 - ecg12), axis=[1]), axis=[0])
    mse = tf.reduce_sum(tf.reduce_mean(tf.square(gen_ecg12 - ecg12), axis=[1]), axis=[0])
    cc_item = (tf.reduce_mean(ecg12 * gen_ecg12, axis=[1]) - tf.reduce_mean(ecg12, axis=[1]) * tf.reduce_mean(
        gen_ecg12, axis=[1])) \
              / (tf.math.reduce_std(ecg12, axis=[1]) * tf.math.reduce_std(gen_ecg12, axis=[1]))

    if np.sum(np.isnan(cc_item)):
        cc_item2 = np.array(cc_item)
        cc_item2[np.where(np.isnan(cc_item))[0], np.where(np.isnan(cc_item))[1]] = 0
        cc = tf.reduce_sum(cc_item2, axis=0)
    else:
        cc= tf.reduce_sum(cc_item, axis=0)
    return np.asarray(mae),np.asarray(mse),np.asarray(cc)
@tf.function(experimental_relax_shapes=True)
def test_ae(model,ds,numlead=12):
    # 输入为一导联
    MAE_=np.zeros((numlead,12))
    MSE_=np.zeros((numlead,12))
    CC_=np.zeros((numlead,12))
    num =0
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        ecg12 = np.delete(ecg12, np.where(np.std(ecg12, axis=1)<1e-4)[0], axis=0)
        num+=ecg12.shape[0]
        padding_len = args.ecglen-ecg12.shape[1]%args.ecglen
        full_idx = int(ecg12.shape[1]//args.ecglen)
        zero_padding =tf.zeros_like(ecg12)

        ecg12_new = tf.concat([ecg12, zero_padding[:, -padding_len:, :]], axis=1)
        # ecg12_new = tf.concat([ecg12[:, :full_idx * 1024, :], ecg12[:, full_idx * 1024 - 120
        #                                                                :full_idx * 1024, :],
        #                        ecg12[:, full_idx * 1024:, :]], axis=1)
        ecg12_new = tf.reshape(ecg12_new, shape=(-1, 1024, 12))
        l_index = np.arange(ecg12_new.shape[0]).reshape(-1, 1)

        for i in range(numlead):
            h_index = i * np.ones((ecg12_new.shape[0], 1)).astype(np.int32)
            index = np.hstack((l_index, h_index))
            ecg1 = paddingecg(ecg12_new, index)
            gen_ecg12 = model(ecg1)
            gen_ecg12 = tf.reshape(gen_ecg12, (-1, ecg12.shape[1] + padding_len, 12))
            gen_ecg12 = gen_ecg12[:, :-padding_len, :]
            mae1,mse1,cc1=compute_metric(gen_ecg12,ecg12)
            MAE_[i,:]+=mae1
            MSE_[i,:]+=mse1
            CC_[i,:]+=cc1
        print(np.mean(CC_) / num)
    return MAE_ / (num), MSE_ / num, CC_ / num
@tf.function(experimental_relax_shapes=True)
def paddingfor16(ecg):
    a=tf.zeros_like(ecg)[:,:,:2]
    ecgnew=tf.concat([a,ecg,a],axis=-1)
    ecgnew=tf.transpose(ecgnew,[0,2,1])
    ecgnew=tf.expand_dims(ecgnew,axis=-1)
    return ecgnew
@tf.function(experimental_relax_shapes=True)
def test_ekgan(model,ds,numlead=1):

    MAE_=np.zeros((numlead,12))
    MSE_=np.zeros((numlead,12))
    CC_=np.zeros((numlead,12))
    num=0
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        ecg12 = np.delete(ecg12, np.where(np.std(ecg12, axis=1)<1e-4)[0], axis=0)
        num+=ecg12.shape[0]
        for i in range(numlead):
            padding_len = args.ecglen - ecg12.shape[1] % args.ecglen
            ecg12_new = tf.concat([ecg12, ecg12[:, -padding_len:, :]], axis=1)
            ecg12_new = tf.reshape(ecg12_new, shape=(-1, 1024, 12))

            ecg1 = tf.tile(ecg12_new[:, :, numlead - 1:numlead], [1, 1, 12])
            ecg1 = paddingfor16(ecg1)

            gen_ecg12, _ = model(ecg1)
            gen_ecg12 = gen_ecg12[:, 2:-2, :, 0]
            gen_ecg12 = tf.transpose(gen_ecg12, [0, 2, 1])

            gen_ecg12 = tf.reshape(gen_ecg12, (-1, ecg12.shape[1] + padding_len, 12))
            gen_ecg12 = gen_ecg12[:, :-padding_len, :]

            mae1, mse1, cc1 = compute_metric(gen_ecg12, ecg12)
            MAE_[i, :] += mae1
            MSE_[i, :] += mse1
            CC_[i, :] += cc1
        print(np.mean(CC_)/num)
    return MAE_/(num),MSE_/num,CC_/num
@tf.function(experimental_relax_shapes=True)
def test_unet(model,ds, numlead=1,idxlead=0):
    # 输入为一导联
    MAE_=np.zeros((numlead,12))
    MSE_=np.zeros((numlead,12))
    CC_=np.zeros((numlead,12))
    num=0
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        ecg12 = np.delete(ecg12, np.where(np.std(ecg12, axis=1)<1e-4)[0], axis=0)
        num+=ecg12.shape[0]
        # if numlead==1:
        padding_len = args.ecglen - ecg12.shape[1] % args.ecglen
        ecg12_new = tf.concat([ecg12, ecg12[:, -padding_len:, :]], axis=1)
        ecg12_new = tf.reshape(ecg12_new, shape=(-1, 1024, 12))

        ecg1 = ecg12_new[:, :, idxlead:idxlead+1]
        gen_ecg12 = model(ecg1)
        gen_ecg12 = tf.reshape(gen_ecg12, (-1, ecg12.shape[1] + padding_len, 12))
        gen_ecg12 = gen_ecg12[:, :-padding_len, :]


        mae1, mse1, cc1 = compute_metric(gen_ecg12, ecg12)
        MAE_+= mae1
        MSE_ += mse1
        CC_ += cc1
        print(np.mean(CC_)/num)
    return MAE_/(num),MSE_/num,CC_/num

if __name__=='__main__':

    testpath = "/data/0shared/chenjiarong/lead_dataset/testset2_lead_Long"
    testds = read_tfrecords_Long(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for mode  in ['ae']:
        args.testmodel=mode

        if args.testmodel=='ae':
            # resultpath='../results/AutoEncoder_result_hr/'
            for [anylead,padding] in [[1,'zeros']]:
                args.padding=padding
                modelpath='../abliation_study/Autoencoder_zeros_'+str(anylead)
                model1 = tf.keras.models.load_model(modelpath)
                MAE_test, MSE_test, CC_test = test_ae(model=model1,ds=testds)
                print(CC_test)
                # excel_path=resultpath+'anylead'+str(anylead)+'.xlsx'
                # if not os.path.exists(resultpath):
                #     os.mkdir(resultpath)
        else:
            if args.testmodel=='unet':
                modelpath = '../MAUNet'
                model1 = tf.keras.models.load_model(modelpath)
                MAE_test,MSE_test,CC_test=test_unet(model=model1,ds=testds,idxlead=1)
                print(CC_test)
                # excel_path = resultpath + 'MAUNet.xlsx'
            if args.testmodel=='ekgan':
                modelpath='../EKGAN/inference_generator'
                model1 = tf.keras.models.load_model(modelpath)
                MAE_test,MSE_test,CC_test=test_ekgan(model=model1,ds=testds)
                print(CC_test)
                # excel_path = resultpath + 'EKGAN.xlsx'
            if args.testmodel=='megan':
                modelpath = '../MEGAN/generator'
                model1 = tf.keras.models.load_model(modelpath)
                MAE_test, MSE_test, CC_test = test_unet(model=model1, ds=testds)
                print(CC_test)
                # excel_path = resultpath + 'MEGAN.xlsx'
            if args.testmodel=='cgan':
                modelpath = '../CGAN/generator'
                model1 = tf.keras.models.load_model(modelpath)
                MAE_test, MSE_test, CC_test = test_unet(model=model1, ds=testds)
                print(CC_test)
                #     excel_path = resultpath + 'CGAN.xlsx'
                # if not os.path.exists(resultpath):
                #     os.mkdir(resultpath)
