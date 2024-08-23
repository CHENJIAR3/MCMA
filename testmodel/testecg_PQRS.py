import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
import tensorflow as tf
import numpy as np
import pickle

from utils import args,read_tfrecords_Long
import  pandas as pd
# 计算dtw
tf.config.run_functions_eagerly(True)
def Highpass_ECG(ecg,window_size=250):
    smoothed_signal = np.convolve(ecg, np.ones(window_size) / window_size, mode='same')
    ecg = ecg - smoothed_signal
    return ecg
def getcc(ecg12,gen_ecg12,axis=1):
    return (tf.reduce_mean(ecg12 * gen_ecg12, axis=[axis]) - tf.reduce_mean(ecg12, axis=[axis]) * tf.reduce_mean(
        gen_ecg12, axis=[1])) \
         / (tf.math.reduce_std(ecg12, axis=[axis]) * tf.math.reduce_std(gen_ecg12, axis=[axis]))

def paddingecg(ecg12,index):
    # index = np.hstack((l_index, h_index))
    # ecg_new = tf.transpose(tf.random.normal(shape=ecg12.shape, dtype=tf.float32), [0, 2, 1])

    ecg_new = tf.transpose(tf.zeros_like(ecg12, dtype=tf.float32), [0, 2, 1])
    ecg12 = tf.transpose(ecg12, [0, 2, 1])
    updates = tf.gather_nd(ecg12, index)
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new
def HRV_feaDB(ecg_signal):
    import json
    import requests
    ecg = np.asarray(ecg_signal)
    data = [float(x) for x in ecg]
    data ={'signal':data,'fs':args.fs,'qrs_pks':False,'amend':False,'mean':True}
    datas = json.dumps(data)
    url = "http://183.162.233.24:10005/featuredb/v1"  # 外网测试接口
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=url, headers=headers, data=datas)
    hrv_result = list(json.loads(json.loads(response.text)["data"]["features"]).values())
    return np.asarray(hrv_result)
def compute_metric(gen_ecg12,ecg12):
    '''"HR": 心率（int）,
    "P_amplitude": P波振幅（float）,
    "P_duration": P波持续时间（int）,
    "PR_interval": RP间期（int）,
    "QRS_amplitude": QRS波振幅(float),
    "QRS_duration": QRS波持续时间(int),
     "T_amplitude": T波振幅(float),
     "ST": ST段时间（int）,
    "QT_interval": QT间期(int),
    "QTc": 按心率校正的 QT间期(int)
    "HRV": 心率变异性(int)'''
    valg = np.zeros((gen_ecg12.shape[0],gen_ecg12.shape[-1],11))
    valr = np.zeros((ecg12.shape[0],ecg12.shape[-1],11))
    assert gen_ecg12.shape==ecg12.shape
    for i in range(gen_ecg12.shape[0]):
        for j in range(gen_ecg12.shape[-1]):
            try:
              valg[i,j,:]=HRV_feaDB(gen_ecg12[i,:,j])
              valr[i,j,:]=HRV_feaDB(ecg12[i,:,j])
            except:
                pass
    return valg,valr
@tf.function(experimental_relax_shapes=True)
def test_ae(model,ds,numlead=0):
    # num=0
    gen_ecg12_all=None
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        ecg12 = np.delete(ecg12, np.where(np.std(ecg12, axis=1)<1e-4)[0], axis=0)
        padding_len = args.ecglen - ecg12.shape[1] % args.ecglen
        ecg12_new = tf.concat([ecg12, ecg12[:, -padding_len:, :]], axis=1)
        ecg12_new = tf.reshape(ecg12_new, shape=(-1, 1024, 12))

        l_index = np.arange(ecg12_new.shape[0]).reshape(-1, 1)

        # for i in [numlead]:
        h_index = numlead * np.ones((ecg12_new.shape[0], 1)).astype(np.int32)
        index = np.hstack((l_index, h_index))
        ecg1 = paddingecg(ecg12_new, index)
        gen_ecg12 = model(ecg1)
        gen_ecg12 = tf.reshape(gen_ecg12, (-1, ecg12.shape[1] + padding_len, 12))
        gen_ecg12 = gen_ecg12[:, :-padding_len, :]
        if step==0:
            gen_ecg12_all=gen_ecg12
            if numlead==0:
                ecg12_all=ecg12
        else:
            gen_ecg12_all=np.concatenate([gen_ecg12_all,gen_ecg12],axis=0)
            if numlead==0:
                ecg12_all=np.concatenate([ecg12_all,ecg12],axis=0)
            # valg,valr=compute_metric(gen_ecg12,ecg12)
            # if step==0 :
            #     valg_all=valg
            #     valr_all=valr
            # else:
            #     valg_all=np.concatenate([valg_all,valg])
            #     valr_all = np.concatenate([valr_all, valr])
    if numlead == 0:
        return np.asarray(ecg12_all), np.asarray(gen_ecg12_all)

    else:
        return np.asarray(gen_ecg12_all)
def paddingfor16(ecg):
    a=tf.zeros_like(ecg)[:,:,:2]
    ecgnew=tf.concat([a,ecg,a],axis=-1)
    ecgnew=tf.transpose(ecgnew,[0,2,1])
    ecgnew=tf.expand_dims(ecgnew,axis=-1)
    return ecgnew
@tf.function(experimental_relax_shapes=True)
def test_ekgan(model,ds,output_num=2,filtered=1,window_size=150,numlead=1):

    MAE_=np.zeros((numlead,12))
    MSE_=np.zeros((numlead,12))
    CC_=np.zeros((numlead,12))
    num=0
    for step,ecg12 in enumerate(ds):

        print("The current step is ",step)
        ecg12 = np.delete(ecg12, np.where(np.std(ecg12, axis=1)<1e-4)[0], axis=0)
        if filtered == 1:
            x = np.asarray(ecg12).copy()
            for i in range(ecg12.shape[0]):
                for j in range(12):
                    x[i, :, j] = Highpass_ECG(x[i, :, j], window_size)
            ecg12 = x
        num+=ecg12.shape[0]
        for i in range(numlead):

            ecg1=tf.tile(ecg12[:, :, i:i+1],[1,1,12])
            ecg1=paddingfor16(ecg1)
            # ecg12=paddingfor16(ecg12)

            gen_ecg12,_=model(ecg1)
            gen_ecg12=gen_ecg12[:,2:-2,:,0]
            # print(ecg12.shape,gen_ecg12.shape)
            gen_ecg12 = tf.transpose(gen_ecg12, [0, 2, 1])
            mae1, mse1, cc1 = compute_metric(gen_ecg12, ecg12)
            MAE_[i, :] += mae1
            MSE_[i, :] += mse1
            CC_[i, :] += cc1
        print(np.mean(CC_)/num)
    return MAE_/(num),MSE_/num,CC_/num
@tf.function(experimental_relax_shapes=True)
def test_unet(model,ds,filtered=1,window_size=150, numlead=1,idxlead=0):
    # 输入为一导联
    MAE_=np.zeros((numlead,12))
    MSE_=np.zeros((numlead,12))
    CC_=np.zeros((numlead,12))
    num=0
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        ecg12 = np.delete(ecg12, np.where(np.std(ecg12, axis=1)<1e-4)[0], axis=0)
        num+=ecg12.shape[0]
        if numlead==1:
            ecg1 = ecg12[:, :, idxlead:idxlead + 1]
            gen_ecg12 = model(ecg1)
            mae1, mse1, cc1 = compute_metric(gen_ecg12, ecg12)
            MAE_+= mae1
            MSE_ += mse1
            CC_ += cc1
            # if np.sum(np.isnan(cc_item)):
            #     cc_item2 = np.array(cc_item)
            #     cc_item2[np.where(np.isnan(cc_item))[0], np.where(np.isnan(cc_item))[1]] = 0
            #     CC_[i]+= tf.reduce_sum(cc_item2, axis=0)
            # else:
            #     CC_[i]+= tf.reduce_sum(cc_item, axis=0)
        print(np.mean(CC_)/num)
    return MAE_/(num),MSE_/num,CC_/num
def write2excel(MAE_test,MSE_test,CC_test,excel_path):
    MAE_test = np.concatenate([MAE_test, np.mean(MAE_test, axis=1)[:, None]], axis=1)
    MSE_test = np.concatenate([MSE_test, np.mean(MSE_test, axis=1)[:, None]], axis=1)
    CC_test = np.concatenate([CC_test, np.mean(CC_test, axis=1)[:, None]], axis=1)
    # print(CC_test)
    df1 = pd.DataFrame(MAE_test).round(4)
    df2 = pd.DataFrame(MSE_test).round(4)
    df3 = pd.DataFrame(CC_test).round(4)
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='MAE_test', index=True)
        df2.to_excel(writer, sheet_name='MSE_test', index=True)
        df3.to_excel(writer, sheet_name='CC_test', index=True)
def write2excel2(MAE_test,MSE_test,CC_test,excel_path):
    MAE_test = np.concatenate([MAE_test, np.mean(MAE_test, axis=1)[:, None]], axis=1)
    MSE_test = np.concatenate([MSE_test, np.mean(MSE_test, axis=1)[:, None]], axis=1)
    CC_test = np.concatenate([CC_test, np.mean(CC_test, axis=1)[:, None]], axis=1)
    # print(CC_test)
    df11 = pd.DataFrame(MAE_test[0]).round(4)
    df21 = pd.DataFrame(MSE_test[0]).round(4)
    df31 = pd.DataFrame(CC_test[0]).round(4)
    df12 = pd.DataFrame(MAE_test[1]).round(4)
    df22 = pd.DataFrame(MSE_test[1]).round(4)
    df32 = pd.DataFrame(CC_test[1]).round(4)
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df11.to_excel(writer, sheet_name='MAE_test_AE', index=True)
        df21.to_excel(writer, sheet_name='MSE_test_AE', index=True)
        df31.to_excel(writer, sheet_name='CC_test_AE', index=True)
        df12.to_excel(writer, sheet_name='MAE_test_AEDE', index=True)
        df22.to_excel(writer, sheet_name='MSE_test_AEDE', index=True)
        df32.to_excel(writer, sheet_name='CC_test_AEDE', index=True)
if __name__=='__main__':
    output_pkl=0
    for external_test  in [0]:
        if external_test:
            resultpath='../results/cpsc2018/'
            testpath='/data/0shared/chenjiarong/lead_dataset/cpsc2018'
        else:
            resultpath='../results/ptbxl/'
            testpath = "/data/0shared/chenjiarong/lead_dataset/testset2_lead_Long"
        for mode in ['ae']:
            args.testmodel=mode
            testds = read_tfrecords_Long(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            if args.testmodel=='ae':
                for [anylead,padding] in [[1,'zeros']]:
                    args.padding=padding
                    modelpath='../abliation_study/Autoencoder_'+str(padding)+'_'+str(anylead)
                    model = tf.keras.models.load_model(modelpath)
                    GEN_ECG12=[]

                    for i in range(12):
                        print("*********************")
                        if i==0:
                            ECG12,genecg12=test_ae(model,ds=testds,numlead=i)
                            if output_pkl:
                                data = {"ECG12": ECG12, "GEN_ECG12": genecg12, "fs": 500}
                                file = open("/data/0shared/chenjiarong/12lead_dataset/AE_12lead_"+str(i+1)+".pkl", "wb")
                                pickle.dump(data, file)
                                file.close()
                        else:
                            genecg12=test_ae(model,ds=testds,numlead=i)
                            if output_pkl:
                                data = {"GEN_ECG12": genecg12, "fs": 500}
                                file = open("/data/0shared/chenjiarong/12lead_dataset/AE_12lead_"+str(i+1)+".pkl", "wb")
                                pickle.dump(data, file)
                                file.close()
                        GEN_ECG12.append(genecg12)
                        # ar.append(valr)
                        cc = getcc(ECG12,genecg12)
                        print(np.mean(cc))
