#这是一个测试模型生成性能的函数
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
# from ekgan import read_tfrecords,load_label_generator
import numpy as np
from utils import args,read_tfrecords_Long
# from models import AutoEncoder,unet3plus
tf.config.run_functions_eagerly(True)

import  pandas as pd
import neurokit2 as nk

def extracthr(ecg):
    try:
        signals, info = nk.ecg_peaks(ecg, sampling_rate=args.fs)
        r_peaks = info["ECG_R_Peaks"]
        # 计算心率
        rr_intervals = np.diff(r_peaks) / 500.0  # 转换为秒
        heart_rate = 60.0 / np.mean(rr_intervals)
        return heart_rate
    except:
        return 0.0
def paddingecg(ecg12,index):
    # index = np.hstack((l_index, h_index))
    ecg_new = tf.transpose(tf.zeros_like(ecg12, dtype=tf.float32), [0, 2, 1])
    ecg12 = tf.transpose(ecg12, [0, 2, 1])
    updates = tf.gather_nd(ecg12, index)
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new

@tf.function(experimental_relax_shapes=True)
def paddingfor16(ecg):
    a=tf.zeros_like(ecg)[:,:,:2]
    ecgnew=tf.concat([a,ecg,a],axis=-1)
    ecgnew=tf.transpose(ecgnew,[0,2,1])
    ecgnew=tf.expand_dims(ecgnew,axis=-1)
    return ecgnew
@tf.function(experimental_relax_shapes=True)
def test_ekgan_hr(model,ds,output_num=2,numlead=1):
    rhr=np.zeros((4,1))
    fhr=np.zeros((4,1))
    maehr=0
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        padding_len = args.ecglen-ecg12.shape[1]%args.ecglen
        ecg12_new = tf.concat([ecg12, tf.zeros_like(ecg12)[:, -padding_len:, :]], axis=1)
        ecg12_new = tf.reshape(ecg12_new, shape=(-1, 1024, 12))

        ecg1=tf.tile(ecg12_new[:, :, numlead-1:numlead],[1,1,12])
        ecg1=paddingfor16(ecg1)
        # ecg12=paddingfor16(ecg12)

        gen_ecg12,_= model(ecg1)
        gen_ecg12  = gen_ecg12[:,2:-2,:,0]
        gen_ecg12 = tf.transpose(gen_ecg12, [0, 2, 1])

        gen_ecg12=tf.reshape(gen_ecg12,(-1,ecg12.shape[1]+padding_len,12))
        gen_ecg12=gen_ecg12[:,:-padding_len,:]

        # print(ecg12.shape,gen_ecg12.shape)
        hr1 = np.zeros((ecg12.shape[0], ecg12.shape[2]))
        hr2 = np.zeros((ecg12.shape[0], ecg12.shape[2]))
        for i2 in range(ecg12.shape[0]):
            for j in range(ecg12.shape[2]):
                hr1[i2, j] = extracthr(ecg12[i2, :, j])
                hr2[i2, j] = extracthr(gen_ecg12[i2, :, j])

        idx = np.where(np.isnan(hr1 + hr2) == 1)[0]
        hr1 = np.delete(hr1, idx, axis=0)
        hr2 = np.delete(hr2, idx, axis=0)
        assert hr1.shape == hr2.shape
        maehr += np.sum(abs(hr1 - hr2))
        rhr += get_hrmetric(hr1)
        fhr += get_hrmetric(hr2)
    maehr = maehr / (12 * rhr[-1])
    rhr = final_get(rhr)
    fhr = final_get(fhr)
    return maehr, rhr, fhr
@tf.function(experimental_relax_shapes=True)
def test_unet_hr(model,ds,numlead=1):
    # 输入为一导联
    rhr=np.zeros((4,1))
    fhr=np.zeros((4,1))
    maehr=0
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        # num+=ecg12.shape[0]
        padding_len = args.ecglen-ecg12.shape[1]%args.ecglen
        ecg12_new = tf.concat([ecg12, tf.zeros_like(ecg12)[:, -padding_len:, :]], axis=1)
        ecg12_new = tf.reshape(ecg12_new, shape=(-1, 1024, 12))

        ecg1 = ecg12_new[:, :, numlead-1:numlead]
        gen_ecg12=model(ecg1)
        gen_ecg12=tf.reshape(gen_ecg12,(-1,ecg12.shape[1]+padding_len,12))
        gen_ecg12=gen_ecg12[:,:-padding_len,:]

        hr1 = np.zeros((ecg12.shape[0], ecg12.shape[2]))
        hr2 = np.zeros((ecg12.shape[0], ecg12.shape[2]))
        for i2 in range(ecg12.shape[0]):
            for j in range(ecg12.shape[2]):
                hr1[i2, j] = extracthr(ecg12[i2, :, j])
                hr2[i2, j] = extracthr(gen_ecg12[i2, :, j])

        idx = np.where(np.isnan(hr1 + hr2) == 1)[0]
        hr1 = np.delete(hr1, idx, axis=0)
        hr2 = np.delete(hr2, idx, axis=0)
        assert hr1.shape == hr2.shape
        maehr+= np.sum(abs(hr1 - hr2))
        rhr+= get_hrmetric(hr1)
        fhr += get_hrmetric(hr2)
    maehr = maehr / (12 * rhr[-1])
    rhr = final_get(rhr)
    fhr = final_get(fhr)
    return maehr, rhr, fhr

def get_hrmetric(hr1):
    rhr=np.zeros((4,1))
    rhr[0] = np.sum(np.std(hr1, axis=1))
    rhr[1] = np.sum(np.std(hr1, axis=1) / np.mean(hr1, axis=1))
    rhr[2] = np.sum(np.max(hr1, axis=1) - np.min(hr1, axis=1))
    rhr[3] = hr1.shape[0]
    return rhr
@tf.function(experimental_relax_shapes=True)
def test_ae_hr(model,ds,numlead=12):
    # 输入为一导联
    rhr=np.zeros((4,numlead))
    gen_fhr=np.zeros((4,numlead))
    maehr=np.zeros((1,numlead))

    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        padding_len = args.ecglen-ecg12.shape[1]%args.ecglen
        ecg12_new = tf.concat([ecg12, tf.zeros_like(ecg12)[:, -padding_len:, :]], axis=1)
        ecg12_new = tf.reshape(ecg12_new, shape=(-1, 1024, 12))

        l_index = np.arange(ecg12_new.shape[0]).reshape(-1, 1)

        for i in range(numlead):
            h_index = i * np.ones((ecg12_new.shape[0], 1)).astype(np.int32)
            index = np.hstack((l_index, h_index))
            ecg1 = paddingecg(ecg12_new, index)

            gen_ecg12 = model(ecg1)

            gen_ecg12 = tf.reshape(gen_ecg12, (-1, ecg12.shape[1] + padding_len, 12))
            gen_ecg12 = gen_ecg12[:, :-padding_len, :]

            hr1 = np.zeros((ecg12.shape[0], ecg12.shape[2]))
            hr2 = np.zeros((ecg12.shape[0], ecg12.shape[2]))

            for i2 in range(ecg12.shape[0]):
                for j in range(ecg12.shape[2]):
                    hr1[i2, j] = extracthr(ecg12[i2, :, j])
                    hr2[i2, j] = extracthr(gen_ecg12[i2, :, j])
            idx=np.where(np.isnan(hr1+hr2) == 1)[0]
            hr1 = np.delete(hr1, idx, axis=0)
            hr2 = np.delete(hr2, idx, axis=0)
            assert hr1.shape==hr2.shape
            maehr[0, i] += np.sum(abs(hr1 - hr2))
            rhr[:, i:i + 1] += get_hrmetric(hr1)
            gen_fhr[:, i:i + 1] += get_hrmetric(hr2)
    maehr=maehr/(12*rhr[-1,:])
    rhr=final_get(rhr)
    gen_fhr=final_get(gen_fhr)
    return maehr,rhr,gen_fhr

def final_get(a):
    return a[:-1,:]/a[-1,:][None,:]
def write2excel_hr(rhr,fhr,excel_path):

    # print(CC_test)
    df1 = pd.DataFrame(rhr).round(4)
    df2 = pd.DataFrame(fhr).round(4)
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='real_heartrate', index=True)
        df2.to_excel(writer, sheet_name='fake_heartrate', index=True)
def write2excel_hr2(rhr,fhr,maehr,excel_path):
    # print(CC_test)
    df1 = pd.DataFrame(rhr).round(4)
    df2 = pd.DataFrame(fhr).round(4)
    df3 = pd.DataFrame(maehr).round(4)
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='real_heartrate', index=True)
        df2.to_excel(writer, sheet_name='fake_heartrate', index=True)
        df3.to_excel(writer, sheet_name='MAE_HR', index=True)

if __name__=='__main__':

    # Every sample in the open-source dataset, the signal length might be 5000, 7500, 10000, and so on.
    testds = read_tfrecords_Long(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    resultpath = '../results/ptbxl/'
    # loading the trained model in MCMA
    model = tf.keras.models.load_model(modelpath)
    maehr,rhr, fhr=test_ae_hr(model=model,ds=testds)
    # 12 groups
    write2excel_hr2(rhr, fhr, maehr,excel_path)
    
    # loading other models, for comparison
    # if args.testmodel=='maunet':
    modelpath='../MAUNet'
    model1 = tf.keras.models.load_model(modelpath)
    maehru, rhru, fhru = test_unet_hr(model=model1, ds=testds, numlead=2)
    # excel_path = resultpath + 'MAUNet_hr.xlsx'
    modelpath='../EKGAN/inference_generator'
    model1 = tf.keras.models.load_model(modelpath)
    maehre,rhre, fhre=test_ekgan_hr(model=model1,ds=testds)
    # if args.testmodel=='megan':
    modelpath = '../MEGAN/generator'
    model1 = tf.keras.models.load_model(modelpath)
    maehrm,rhrm,fhrm = test_unet_hr(model=model1, ds=testds)
    # if args.testmodel=='cgan':
    modelpath = '../CGAN/generator'
    model1 = tf.keras.models.load_model(modelpath)
    maehrc,rhrc,fhrc= test_unet_hr(model=model1, ds=testds)
        # excel_path = resultpath + 'CGAN_hr_0409.xlsx'

    excel_path=resultpath+"compared_hr_0409.xlsx"
    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
    maehr=np.concatenate([maehru[:,None],maehre[:,None],maehrm[:,None],maehrc[:,None]],axis=1)
    rhr = np.concatenate([rhru,rhre,rhrm,rhrc],axis=1)
    fhr = np.concatenate([fhru,fhre,fhrm,fhrc],axis=1)

    df1 = pd.DataFrame(rhr,columns=['MAUNET','EKGAN','MEGAN','CGAN'],index=["MHR_SD","MHR_CV","MHR_Range"]).round(4)
    df2 = pd.DataFrame(fhr,columns=['MAUNET','EKGAN','MEGAN','CGAN'],index=["MHR_SD","MHR_CV","MHR_Range"]).round(4)
    df3 = pd.DataFrame(maehr,columns=['MAUNET','EKGAN','MEGAN','CGAN']).round(4)
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='real_heartrate', index=True)
        df2.to_excel(writer, sheet_name='fake_heartrate', index=True)
        df3.to_excel(writer, sheet_name='MAE_HR', index=True)

