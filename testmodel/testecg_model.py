import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
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
        cc_item2 = np.array(cc_item)
        cc_item2[np.where(np.isnan(cc_item))[0], np.where(np.isnan(cc_item))[1]] = 0
        cc = tf.reduce_sum(cc_item2, axis=0)
    else:
        cc= tf.reduce_sum(cc_item, axis=0)
    return np.asarray(mae),np.asarray(mse),np.asarray(cc)
@tf.function(experimental_relax_shapes=True)
def test_ae(model,ds,output_num=1,numlead=12,filtered=1,window_size=150):
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

        if filtered == 1:
            x = np.asarray(ecg12).copy()
            for i in range(ecg12.shape[0]):
                for j in range(12):
                    x[i, :, j] = Highpass_ECG(x[i, :, j], window_size)
            ecg12 = x

        for i in range(numlead):

            h_index = i * np.ones((ecg12.shape[0], 1)).astype(np.int32)
            ecg1 = ecg12[:, :, i:i+1]
            index = np.hstack((l_index, h_index))
            if args.padding=='zeros':
               ecg1 = paddingecg(ecg12, index)
            else:
                ecg1 = tf.tile(ecg1, [1, 1, 12])
            # ecg1=tf.tile(ecg12[:,:,i:i+1],[1,1,12])

            if output_num==1:
                gen_ecg12=model(ecg1)
            else:
                gen_ecg12 = model([ecg1,h_index+1])
                # gen_ecg12=model([ecg12,time])
            mae1,mse1,cc1=compute_metric(gen_ecg12,ecg12)
            MAE_[i,:]+=mae1
            MSE_[i,:]+=mse1
            CC_[i,:]+=cc1
            # re_ecg12 = decoder([gen_ecg12, h_index + 1])
            #
            # mae2,mse2,cc2=compute_metric(re_ecg12,ecg12)
            # MAE_[1,i,:]+=mae2
            # MSE_[1,i,:]+=mse2
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
        if filtered == 1:
            x = np.asarray(ecg12).copy()
            for i in range(ecg12.shape[0]):
                for j in range(12):
                    x[i, :, j] = Highpass_ECG(x[i, :, j], window_size)
            ecg12 = x
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
    filtered = 0
    for external_test  in [0]:
        if external_test:
            resultpath='../results/cpsc2018/'
            if filtered == 1:
                resultpath = '../results/cpsc2018/filtered_'

            testpath='/data/0shared/chenjiarong/lead_dataset/cpsc2018'
        else:
            resultpath='../results/ptbxl/'
            if filtered==1:
                resultpath = '../results/ptbxl/filtered_'
            if args.data_norm:
                # trainpath = "/data/0shared/chenjiarong/lead_dataset/trainset_lead"
                testpath = "/data/0shared/chenjiarong/lead_dataset/testset_lead"
                # valpath = "/data/0shared/chenjiarong/lead_dataset/valset_lead"
            else:
                # trainpath = "/data/0shared/chenjiarong/lead_dataset/trainset2_lead"
                testpath = "/data/0shared/chenjiarong/lead_dataset/testset2_lead"
                # valpath = "/data/0shared/chenjiarong/lead_dataset/valset2_lead"
        for mode in ['ae']:
            args.testmodel=mode
            # 'ae'\'unet'\'ekgan'\'megan'\'cgan'
            #'unet','ekgan','megan','cgan'
            #args.bs=64
            testds = read_tfrecords(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            if args.testmodel=='ae':
                # [[0, 'zeros'], [1, 'zeros']
                for [anylead,padding] in [[1,'zeros']]:
                    args.padding=padding
                    modelpath='../abliation_study/Autoencoder_'+str(padding)+'_'+str(anylead)

                    # modelpath="../abliation_study/Control_model"+str(anylead)
                    model = tf.keras.models.load_model(modelpath)
                    MAE_test,MSE_test,CC_test=test_ae(model=model,ds=testds,output_num=1)
                    print(CC_test)
                    if args.padding=='copy':
                        excel_path=resultpath+'/AutoEncoder_copy1.xlsx'
                    else:
                        excel_path=resultpath+'/AutoEncoder_anylead'+str(anylead)+'.xlsx'

                    # if not os.path.exists(resultpath):
                    #     os.mkdir(resultpath)
                    # write2excel(MAE_test, MSE_test, CC_test, excel_path)
            else:
                if args.testmodel=='unet':
                    modelpath = '../MAUNet'
                    model1 = tf.keras.models.load_model(modelpath)
                    MAE_test,MSE_test,CC_test=test_unet(model=model1,ds=testds,filtered=filtered,idxlead=1)
                    excel_path = resultpath + 'MAUNet.xlsx'
                if args.testmodel=='ekgan':
                    modelpath='../EKGAN/inference_generator'
                    model1 = tf.keras.models.load_model(modelpath)
                    MAE_test,MSE_test,CC_test=test_ekgan(model=model1,ds=testds,filtered=filtered)
                    excel_path = resultpath + 'EKGAN.xlsx'
                if args.testmodel=='megan':
                    modelpath = '../MEGAN/generator'
                    model1 = tf.keras.models.load_model(modelpath)
                    MAE_test, MSE_test, CC_test = test_unet(model=model1, ds=testds,filtered=filtered)
                    excel_path = resultpath + 'MEGAN.xlsx'
                if args.testmodel=='cgan':
                    modelpath = '../CGAN/generator'
                    model1 = tf.keras.models.load_model(modelpath)
                    MAE_test, MSE_test, CC_test = test_unet(model=model1, ds=testds,filtered=filtered)
                    excel_path = resultpath + 'CGAN.xlsx'
                if not os.path.exists(resultpath):
                    os.mkdir(resultpath)

                write2excel(MAE_test, MSE_test, CC_test, excel_path)
