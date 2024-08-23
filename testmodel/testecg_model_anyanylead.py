import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7,3'
import tensorflow as tf
import numpy as np

from utils import args,read_tfrecords
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

def paddingecg_reducedlead(ecg12,idxs):
    # ecg_new = tf.transpose(tf.random.normal(shape=ecg12.shape, dtype=tf.float32), [0, 2, 1])
    l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)

    ecg_new = tf.transpose(tf.zeros_like(ecg12, dtype=tf.float32), [0, 2, 1])
    ecg12 = tf.transpose(ecg12, [0, 2, 1])
    for idx in idxs:
        h_index = idx * np.ones((ecg12.shape[0], 1)).astype(np.int32)

        index = np.hstack((l_index, h_index))
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
def test_ae_reducedlead(model,ds,idxs=[0,1]):
    # if numlead==12:
    #     numlead+=1
    MAE_=np.zeros((1,12))
    MSE_=np.zeros((1,12))
    CC_=np.zeros((1,12))
    num=0
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        ecg12 = np.delete(ecg12, np.where(np.std(ecg12, axis=1)<1e-4)[0], axis=0)
        num+=ecg12.shape[0]
        # gen_ecg12 = tf.zeros_like(ecg12)
        # for idx in idxs:
        ecg1 = paddingecg_reducedlead(ecg12, idxs)
        gen_ecg12=model.predict(ecg1)

        mae1,mse1,cc1=compute_metric(gen_ecg12,ecg12)
        MAE_[0,:]+=mae1
        MSE_[0,:]+=mse1
        CC_[0,:]+=cc1
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


if __name__=='__main__':
    for external_test  in [0]:
        if external_test:
            resultpath='../results/cpsc2018/'
            testpath='/data/0shared/chenjiarong/lead_dataset/cpsc2018'
        else:
            resultpath='../results/ptbxl/'
            testpath = "/data/0shared/chenjiarong/lead_dataset/testset2_lead"
        testds = read_tfrecords(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        for [anylead,padding] in [[1,'zeros']]:
            args.padding=padding
            modelpath='../abliation_study/Autoencoder_'+str(padding)+'_'+str(anylead)
            model = tf.keras.models.load_model(modelpath)
            MAE_test_all=[]
            MSE_test_all=[]
            CC_test_all=[]
            leadlabel = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            for  i in range(6):
                for j in range(6,12):
                    MAE_test,MSE_test,CC_test=test_ae_reducedlead(model=model,ds=testds,idxs=[i,j])
                    if i==0 and j==6:
                        MAE_test_all=MAE_test
                        MSE_test_all=MSE_test
                        CC_test_all=CC_test
                        lead_all=[leadlabel[i]+'+'+leadlabel[j]]
                    else:
                        MAE_test_all=np.concatenate([MAE_test_all,MAE_test],axis=0)
                        MSE_test_all=np.concatenate([MSE_test_all,MSE_test],axis=0)
                        CC_test_all=np.concatenate([CC_test_all,CC_test],axis=0)
                        lead_all = np.concatenate([lead_all,[leadlabel[i]+'+'+leadlabel[j]]])
                import matplotlib.pyplot as plt
                plt.xticks(ticks=np.arange(len(lead_all)),labels=lead_all)
                # print(CC_test)
                #
                # if not os.path.exists(resultpath):
                #     os.mkdir(resultpath)
                # write2excel(MAE_test, MSE_test, CC_test, excel_path)
