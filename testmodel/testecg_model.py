import os
import tensorflow as tf
import numpy as np

from training_code/utils_MCMA import args,read_tfrecords
import  pandas as pd

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
    
def compute_metric(gen_ecg12,ecg12):
    mae= tf.reduce_sum(tf.reduce_mean(tf.abs(gen_ecg12 - ecg12), axis=[1]), axis=[0])
    mse = tf.reduce_sum(tf.reduce_mean(tf.square(gen_ecg12 - ecg12), axis=[1]), axis=[0])
    cc_item = (tf.reduce_mean(ecg12 * gen_ecg12, axis=[1]) - tf.reduce_mean(ecg12, axis=[1]) * tf.reduce_mean(
        gen_ecg12, axis=[1])) \
              / (tf.math.reduce_std(ecg12, axis=[1]) * tf.math.reduce_std(gen_ecg12, axis=[1]))

    if np.sum(np.isnan(cc_item)):
        # avoiding NAN
        cc_item2 = np.array(cc_item)
        cc_item2[np.where(np.isnan(cc_item))[0], np.where(np.isnan(cc_item))[1]] = 0
        cc = tf.reduce_sum(cc_item2, axis=0)
    else:
        cc= tf.reduce_sum(cc_item, axis=0)
    return np.asarray(mae),np.asarray(mse),np.asarray(cc)
@tf.function(experimental_relax_shapes=True)
def test_ae(model,ds,output_num=1,numlead=12,padding='zeros'):
    # if numlead==12:
    #     numlead+=1
    MAE_=np.zeros((numlead,12))
    MSE_=np.zeros((numlead,12))
    CC_=np.zeros((numlead,12))
    num=0
    for step,ecg12 in enumerate(ds):
        print("The current step is ",step)
        ecg12 = np.delete(ecg12, np.where(np.std(ecg12, axis=1)<1e-4)[0], axis=0)
        num+=ecg12.shape[0]
        l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)
        for i in range(numlead):

            h_index = i * np.ones((ecg12.shape[0], 1)).astype(np.int32)
            ecg1 = ecg12[:, :, i:i+1]
            index = np.hstack((l_index, h_index))
            if padding=='zeros':
               ecg1 = paddingecg(ecg12, index)


            # if output_num==1:
            gen_ecg12=model(ecg1)
            mae1,mse1,cc1=compute_metric(gen_ecg12,ecg12)
            MAE_[i,:]+=mae1
            MSE_[i,:]+=mse1
            # CC_[1,i,:]+=cc2
        print(np.mean(CC_)/num)
    return MAE_/(num),MSE_/num,CC_/num
    
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
        num+=ecg12.shape[0
        if numlead==1:
            ecg1 = ecg12[:, :, idxlead:idxlead + 1]
            gen_ecg12 = model(ecg1)
            mae1, mse1, cc1 = compute_metric(gen_ecg12, ecg12)
            MAE_+= mae1
            MSE_ += mse1
            CC_ += cc1
        print(np.mean(CC_)/num)
    return MAE_/(num),MSE_/num,CC_/num
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
    # input your testpath
    testds = read_tfrecords(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # loading model with modelpath
    model = tf.keras.models.load_model(modelpath)
    # For MCMA-based, input shape is (1024,12)
    MAE_test,MSE_test,CC_test=test_ae(model=model,ds=testds,output_num=1)
    # For UNet-based, input shape is (1024,1)
    MAE_test,MSE_test,CC_test=test_unet(model=model1,ds=testds,filtered=filtered,idxlead=1)
    # For EKGAN-based, needs paddingfor16
    MAE_test,MSE_test,CC_test=test_ekgan(model=model1,ds=testds,filtered=filtered)

    if not os.path.exists(resultpath):
        os.mkdir(resultpath)

    write2excel(MAE_test, MSE_test, CC_test, excel_path)
