# 导入库函数，包括os、tf等等
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import tensorflow as tf
import numpy as np
import glob
import scipy.io as sio

tf.config.run_functions_eagerly(True)

# 采用zero-padding，使得其支持任意单导联
def paddingecg(ecg1,idx=0):
    l_index = np.arange(ecg1.shape[0]).reshape(-1, 1)
    h_index = idx * np.ones((ecg1.shape[0], 1)).astype(np.int32)
    index = np.hstack((l_index, h_index))
    ecg_new = tf.transpose(tf.zeros_like(tf.tile(ecg1,[1,1,12]), dtype=tf.float32), [0, 2, 1])
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, ecg1[:,:,0])
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new

# 读取数据，可能需要根据实际需求调整
# 输出维度应该是（num，length）
def Read_ECG(datapath,lead_idx=0):
    # datapath：数据存储路径，用mat格式
    # model：本研究实现的模型
    # lead_idx：输入导联的位置，I到V6分别为0到11
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

# 重构模型的过程，支持任意长度，但最好是1024的整数倍，采样频率为500Hz
# 如果不是500Hz，需要进行重采样过程
# 输入参数为单导联心电信号、模型、导联位置，信号长度为ecglen为1024
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
    # 模型加载，其余的模型可以在这里下载，https://drive.google.com/drive/folders/1m57dz-FhcQCGNoZ2wxA_sUoHgrrGRHIn?usp=sharing
    modelpath = "Generator"
    model = tf.keras.models.load_model(modelpath)
    #数据读取，放在Sample_data上面
    datapath = "./Sample_data/"
    ecg1 = Read_ECG(datapath,lead_idx=lead_idx)
    #数据处理
    gen_ecg12 = Reconstructing_ECG(ecg1,model=model,ecglen=ecglen,lead_idx=lead_idx)

