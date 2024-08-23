import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
#首先引入策略
# strategy = tf.distribute.MirroredStrategy()
from utils_MCMA import Trainer,args,read_tfrecords
from testmodel.testecg_model import test_ae,write2excel
if __name__ == "__main__":
    training = 0
    # different kernel_size, you need to download from this url:https://drive.google.com/drive/folders/1m57dz-FhcQCGNoZ2wxA_sUoHgrrGRHIn?usp=sharing
    kernel_size = 5
    # 
    trainds = read_tfrecords(trainpath).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).batch(args.bs).shuffle(1000)
    valds = read_tfrecords(valpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    testds = read_tfrecords(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    args.step_total=87200//args.bs

    resultpath = '../results/ptbxl_diff_ks/'

    modelpath = "Autoencoder_ks"+str(kernel_size)
    if training:
        trainer_ae = Trainer(epochs=args.epochs, lr=args.lr, ecglen=args.ecglen,modelpath=modelpath,kernel_size=kernel_size)
        trainer_ae.train(trainds, valds)
    model = tf.keras.models.load_model(modelpath)
    MAE_test, MSE_test, CC_test = test_ae(model=model, ds=testds, output_num=1)
    excel_path = resultpath +  "Autoencoder_ks"+str(kernel_size)+ '.xlsx'

    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
    write2excel(MAE_test, MSE_test, CC_test, excel_path)
