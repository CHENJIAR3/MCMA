# 导入库函数
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from  model_new import modelx
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
#首先引入策略
strategy = tf.distribute.MirroredStrategy()

from utils import args,read_tfrecords
from utils_MCMA import Trainer
from testmodel.testecg_model import *
if __name__ == "__main__":
    if args.data_norm:
        trainpath = "/data/0shared/chenjiarong/lead_dataset/trainset_lead"
        testpath = "/data/0shared/chenjiarong/lead_dataset/testset_lead"
        valpath = "/data/0shared/chenjiarong/lead_dataset/valset_lead"
    else:
        trainpath = "/data/0shared/chenjiarong/lead_dataset/trainset2_lead"
        testpath = "/data/0shared/chenjiarong/lead_dataset/testset2_lead"
        valpath = "/data/0shared/chenjiarong/lead_dataset/valset2_lead"
    testpath2 = '/data/0shared/chenjiarong/lead_dataset/cpsc2018'
    testds2 = read_tfrecords(testpath2).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    resultpath = '../results/ptbxl/'
    resultpath2 = '../results/cpsc2018/'

    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
        os.mkdir(resultpath2)

    trainds = read_tfrecords(trainpath).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).batch(args.bs).shuffle(1000)
    valds = read_tfrecords(valpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    testds = read_tfrecords(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # valds=strategy.experimental_distribute_dataset(valds)
    # trainds=strategy.experimental_distribute_dataset(trainds)
    args.step_total=87200//args.bs    #
    for [anylead,padding] in  [[0,'zero']]:
        modelpath = "Autoencoder_ks5_zeros_"+str(anylead)

        trainer_ae = Trainer(modelpath=modelpath,epochs=args.epochs, lr=args.lr, ecglen=args.ecglen,anylead=anylead)
        trainer_ae.train(trainds, valds)
        model = tf.keras.models.load_model(modelpath)
        MAE_test, MSE_test, CC_test = test_ae(model=model, ds=testds, output_num=1)
        excel_path = resultpath +  "Autoencoder_ks5_zeros_"+str(anylead)+ '.xlsx'
        write2excel(MAE_test, MSE_test, CC_test, excel_path)
        MAE_test, MSE_test, CC_test = test_ae(model=model, ds=testds2, output_num=1)
        excel_path = resultpath2 + "Autoencoder_ks5_zeros_"+str(anylead)+ '.xlsx'
        write2excel(MAE_test, MSE_test, CC_test, excel_path)