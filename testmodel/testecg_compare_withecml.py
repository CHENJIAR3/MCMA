import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
import tensorflow as tf
import numpy as np

from utils import args,read_tfrecords
import  pandas as pd
# 计算dtw
tf.config.run_functions_eagerly(True)
@tf.function(experimental_relax_shapes=True)
def getcc(ecg12,gen_ecg12,axis=1):
    return (tf.reduce_mean(ecg12 * gen_ecg12, axis=[axis]) - tf.reduce_mean(ecg12, axis=[axis]) * tf.reduce_mean(
        gen_ecg12, axis=[1])) \
         / (tf.math.reduce_std(ecg12, axis=[axis]) * tf.math.reduce_std(gen_ecg12, axis=[axis]))


def euclidean_distance(p, q):
    return np.linalg.norm(np.array(p) - np.array(q))


def frechet_distance(P, Q):
    n = len(P)
    m = len(Q)
    A = np.zeros((n, m))

    # 初始化
    A[0][0] = euclidean_distance(P[0], Q[0])
    for i in range(1, n):
        A[i][0] = max(A[i - 1][0], euclidean_distance(P[i], Q[0]))
    for j in range(1, m):
        A[0][j] = max(A[0][j - 1], euclidean_distance(P[0], Q[j]))

    # 递推关系
    for i in range(1, n):
        for j in range(1, m):
            A[i][j] = max(min(A[i - 1][j], A[i][j - 1], A[i - 1][j - 1]), euclidean_distance(P[i], Q[j]))

    return A[n - 1][m - 1]
@tf.function(experimental_relax_shapes=True)
def paddingecg(ecg12,index):
    ecg_new = tf.transpose(tf.zeros_like(ecg12, dtype=tf.float32), [0, 2, 1])
    ecg12 = tf.transpose(ecg12, [0, 2, 1])
    updates = tf.gather_nd(ecg12, index)
    ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
    ecg_new = tf.transpose(ecg_new, [0, 2, 1])
    return ecg_new
@tf.function(experimental_relax_shapes=True)
def compute_metric(gen_ecg12,ecg12):
    mse = tf.reduce_sum(tf.reduce_mean(tf.square(gen_ecg12 - ecg12), axis=[1]), axis=[0])
    fd_score = np.zeros((ecg12.shape[0],12))
    for i in range(ecg12.shape[0]):
        for j in range(12):
            print(i,j)
            fd_score[i,j] = frechet_distance(ecg12[i,:,j],gen_ecg12[i,:,j])
    return np.asarray(np.mean(fd_score)),np.asarray(mse)
@tf.function(experimental_relax_shapes=True)
def test_ae(model,ds,numlead=12,stride=4,global_norm=0):
    MSE_=np.zeros((numlead,12))
    FD_=np.zeros((numlead,12))
    num=0
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        num+=ecg12.shape[0]
        l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)

        for i in range(numlead):

            h_index = i * np.ones((ecg12.shape[0], 1)).astype(np.int32)
            index = np.hstack((l_index, h_index))
            ecg1 = paddingecg(ecg12, index)


            gen_ecg12=model(ecg1)
            if stride>1:
                gen_ecg12=gen_ecg12[:,::stride,:]
                ecg12r = ecg12[:,::stride,:]
                if global_norm:
                    gen_ecg12 = (gen_ecg12-np.min(gen_ecg12))/(np.max(gen_ecg12)-np.min(gen_ecg12))
                    ecg12r = (ecg12r-np.min(ecg12))/(np.max(ecg12)-np.min(ecg12))
                else:
                    gen_ecg12 =(gen_ecg12-np.min(gen_ecg12,axis=1)[:,None])/(np.max(gen_ecg12,axis=1)[:,None]-np.min(gen_ecg12,axis=1)[:,None])
                    ecg12r = (ecg12r-np.min(ecg12,axis=1)[:,None])/(np.max(ecg12,axis=1)[:,None]-np.min(ecg12,axis=1)[:,None])

            fd1,mse1=compute_metric(gen_ecg12,ecg12r)
            FD_[i,:]+=fd1
            MSE_[i,:]+=mse1
            # re_ecg12 = decoder([gen_ecg12, h_index + 1])
            #
            # mae2,mse2,cc2=compute_metric(re_ecg12,ecg12)
            # MAE_[1,i,:]+=mae2
            # MSE_[1,i,:]+=mse2
            # CC_[1,i,:]+=cc2
        print(np.mean(FD_),np.mean(MSE_)/num)

    return FD_/(num),MSE_/num
@tf.function(experimental_relax_shapes=True)
def write2excel(MAE_test,MSE_test,CC_test,excel_path):
    MAE_test = np.concatenate([MAE_test, np.mean(MAE_test, axis=1)[:, None]], axis=1)
    MSE_test = np.concatenate([MSE_test, np.mean(MSE_test, axis=1)[:, None]], axis=1)
    CC_test = np.concatenate([CC_test, np.mean(CC_test, axis=1)[:, None]], axis=1)
    MAE_test = np.concatenate([MAE_test, np.mean(MAE_test, axis=0)[ None]], axis=0)
    MSE_test = np.concatenate([MSE_test, np.mean(MSE_test, axis=0)[None]], axis=0)
    CC_test = np.concatenate([CC_test, np.mean(CC_test, axis=0)[ None]], axis=0)
    # print(CC_test)
    df1 = pd.DataFrame(MAE_test).round(4)
    df2 = pd.DataFrame(MSE_test).round(4)
    df3 = pd.DataFrame(CC_test).round(4)
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df1.to_excel(writer, sheet_name='MAE_test', index=True)
        df2.to_excel(writer, sheet_name='MSE_test', index=True)
        df3.to_excel(writer, sheet_name='CC_test', index=True)

if __name__=='__main__':

        testpath = "/data/0shared/chenjiarong/lead_dataset/testset2_lead"

        testds = read_tfrecords(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        modelpath='../abliation_study/Autoencoder_ks5'

        # modelpath="../abliation_study/Control_model"+str(anylead)
        model = tf.keras.models.load_model(modelpath)

        FD_test,MSE_test=test_ae(model=model,ds=testds)
        # excel_path=resultpath+'/AutoEncoder_anylead'+str(anylead)+'.xlsx'

                # if not os.path.exists(resultpath):
                #     os.mkdir(resultpath)
                #
                # write2excel(MAE_test, MSE_test, CC_test, excel_path)
