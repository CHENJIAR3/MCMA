import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import tensorflow as tf
import numpy as np
import glob
import scipy.io as sio

# 计算dtw
tf.config.run_functions_eagerly(True)

def paddingecg(ecg1,idx=0):
    l_index = np.arange(ecg1.shape[0]).reshape(-1, 1)
    h_index = idx * np.ones((ecg1.shape[0], 1)).astype(np.int32)
    index = np.hstack((l_index, h_index))
    ecg_new = tf.transpose(tf.zeros_like(tf.tile(ecg1,[1,1,12]), dtype=tf.float32), [0, 2, 1])
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, ecg1[:,:,0])
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new

def Read_ECG(datapath,lead_idx=0):
    # datapath：数据存储路径，用mat格式
    # model：本研究实现的模型
    # lead_idx模型的位置
    ecg1_all=[]
    ecgpaths = glob.glob(datapath + "*.mat")
    for ecgpath in ecgpaths:
        # 读取数据
        '''
        下面读取的是cpsc2018的数据集，并选择I导联，输入数据应该为(N,1024,12)
        '''
        ecg = sio.loadmat(ecgpath)['ECG'][0][0][2]
        ecg = tf.transpose(ecg)
        ecg1 = ecg [:,lead_idx:lead_idx+1][None,:]
        ecg1_all.append(ecg1)

    return ecg1_all
def Reconstructing_ECG(ecgall,model,ecglen=1024,lead_idx=0):
    gen_ecg12all=[]
    for ecg1 in ecgall:
        padding_len = ecglen - ecg1.shape[1] % ecglen
        ecg1_new = tf.concat([ecg1, ecg1[:, -padding_len:, :]], axis=1)
        ecg1_new = tf.cast(tf.reshape(ecg1_new, shape=(-1, 1024, 1)),dtype=tf.float32)
        ecg1_new = paddingecg(ecg1_new,lead_idx)
        gen_ecg12 = model.predict(ecg1_new)
        gen_ecg12 = tf.reshape(gen_ecg12, (-1, ecg1.shape[1] + padding_len, 12))
        gen_ecg12 = gen_ecg12[:, :-padding_len, :]
        gen_ecg12all.append(gen_ecg12[0])
    return gen_ecg12all

if __name__=='__main__':
    # 主要参数设置
    lead_idx=0
    ecglen=1024
    # 模型加载
    modelpath = "Generator"
    model = tf.keras.models.load_model(modelpath)
    #数据读取
    datapath = "./Sample_data/"
    ecg1 = Read_ECG(datapath,lead_idx=lead_idx)
    #数据处理
    gen_ecg12 = Reconstructing_ECG(ecg1,model=model,ecglen=ecglen,lead_idx=lead_idx)

