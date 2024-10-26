
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils_MCMA import args,read_tfrecords
from testmodel.testecg_model import test_ae,write2excel
def extract(a, t, x_shape):
    batch_size, sequence_length, _ = a.shape
    t_shape = tf.shape(t)
    out = tf.gather(a, t, axis=-1)
    out = tf.reshape(out, (batch_size, t_shape[0], *((1,) * (len(x_shape) - 1))))
    return out
    
def getcc(ecg12,gen_ecg12):
    return (tf.reduce_mean(ecg12 * gen_ecg12, axis=[1]) - tf.reduce_mean(ecg12, axis=[1]) * tf.reduce_mean(
        gen_ecg12, axis=[1])) \
         / (tf.math.reduce_std(ecg12, axis=[1]) * tf.math.reduce_std(gen_ecg12, axis=[1]))


class Trainer_copy:
    def __init__(self,modelpath,epochs=100, lr=1e-3, ecglen=1024,kernel_size=5,pool_size = 2
                 , step_total = 340, anylead = 1, patience = 50):

        self.ecglen = ecglen
        self.epochs = epochs
        self.lr = lr
        self.kernel_size = kernel_size
        self.anylead = anylead
        self.step_total = step_total
        self.patience = patience
        self.pool_size = pool_size
        self.modelpath = modelpath
        if 1:
        # with strategy.scope():
            self.model=modelx(input_size=(args.ecglen,12),kernel_size=5)
            self.model.compile(optimizer='adam',
                                    metrics=['accuracy'])
            # if args.load_model:
            #      self.model0 = tf.keras.models.load_model(bestpath)
            #      self.model.set_weights(self.model0.get_weights())

            self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
            self.mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            self.loss_tracker = tf.keras.metrics.Mean()
            self.loss1_tracker = tf.keras.metrics.Mean()
            self.val_loss_tracker = tf.keras.metrics.Mean()
            self.ema = tf.train.ExponentialMovingAverage(decay=0.99)
            self.train_cc=[]
            self.train_loss=[]
            self.val_loss=[]
            self.val_cc=[]
            self.best_loss1 = np.inf
            self.start=False
            self.T=100
            self.lamba1 = 1
            self.lamba2 = 1
            self.waiting=0
            self.eplison=1e-5
    def sample_time_step(self, size):
        return tf.experimental.numpy.random.randint(1, args.time_steps, size=(size,))

    @tf.function
    def train_step(self, ecg12):
        def step_fn(ecg12):
            l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)
            if args.anylead == 1:
                h_index = np.random.randint(0, 12, ecg12.shape[0]).reshape(-1, 1).astype(np.int32)
            else:
                h_index = np.zeros((ecg12.shape[0], 1)).astype(np.int32)
            index = np.hstack((l_index, h_index))
            ecg1= self.paddingecg(ecg12, index)
            with tf.GradientTape() as tape:
                gen_ecg12 = self.model([ecg1])
                loss1 = tf.reduce_mean((gen_ecg12-ecg12)**2,axis=1)

            grads = tape.gradient(loss1, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            gen_ecg12 = self.model([ecg1])
            cc = getcc(ecg12,gen_ecg12)
            return tf.reduce_mean(loss1),tf.reduce_mean(cc)
        return step_fn(ecg12)
    # strategy.run(step_fn, args=(ecg12,))
    @tf.function
    def paddingecg(self,ecg12, index):
        ecg12   = tf.transpose(ecg12, [0, 2, 1])
        updates = tf.gather_nd(ecg12, index)
        ecg_new = tf.tile(updates[:,:,None],[1,1,12])
        return ecg_new
    @tf.function
    def test_step(self, ecg12):
        def step_test(ecg12):
            l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)
            if args.anylead == 1:
                h_index = np.random.randint(0, 12, ecg12.shape[0]).reshape(-1, 1).astype(np.int32)
            else:
                h_index=np.zeros((ecg12.shape[0],1)).astype(np.int32)
            index = np.hstack((l_index, h_index))
            ecg1= self.paddingecg(ecg12, index)

            gen_ecg12 = self.model([ecg1],training=False)
            loss1 = tf.reduce_mean((gen_ecg12 - ecg12) ** 2, axis=1)
            norm_ecg12 = (ecg12 - tf.reduce_mean(ecg12,axis=1,keepdims=True))/(self.eplison+tf.math.reduce_std(ecg12,axis=1,keepdims=True))
            norm_genecg12 = (gen_ecg12 - tf.reduce_mean(ecg12,axis=1,keepdims=True))/(self.eplison+tf.math.reduce_std(ecg12,axis=1,keepdims=True))
            loss2 = tf.reduce_mean((norm_genecg12 - norm_ecg12) ** 2, axis=1)

            cc = getcc(ecg12,gen_ecg12)
            return tf.reduce_mean(loss1),tf.reduce_sum(cc)

            # return tf.reduce_mean(loss1), tf.reduce_mean(loss2), tf.reduce_sum(cc), tf.reduce_sum(cc12)

        # per_replica_losses = strategy.run(step_test, args=(ecg12,))
        return step_test(ecg12)

    def train(self, train_data, val_data):
        for epoch in range(self.epochs):
            self.loss_tracker.reset_states()
            self.loss1_tracker.reset_states()
            self.val_loss_tracker.reset_states()

            print("Epoch :", epoch)
            # with strategy.scope():
            if 1:
                progbar = tf.keras.utils.Progbar(target=self.step_total)
                train_cc_sum=0
                train_loss_sum=0
                for train_batch, ecg12 in enumerate(train_data):
                    train_loss,train_cc = self.train_step(ecg12)
                    # non_empty_losses = [loss for loss in trainloss.values if np.isnan(loss) == False]
                    # train_loss = tf.reduce_mean(non_empty_losses)
                    # train_cc = tf.reduce_mean([cc for cc in traincc.values if np.isnan(cc)==False])
                    progbar.update(train_batch, values=
                    [('loss', train_loss)
                        ,('cc',train_cc)
                     ],
                               finalize=False)
                    train_cc_sum+=train_cc
                    train_loss_sum+=train_loss


            val_loss1=0
            val_cc=0

            val_num=10965*12
            for val_batch, ecg12 in enumerate(val_data):
                valloss1,cc_item = self.test_step(ecg12)
                val_loss1+=valloss1
                val_cc+=cc_item
                # 最后一次的val_loss
                # val_loss1+= tf.reduce_mean([loss for loss in valloss1.values if np.isnan(loss) == False])
                # val_cc+=tf.reduce_sum([cc0 for cc0 in cc_item.values if np.isnan(cc0)==False])
            val_cc=np.asarray(val_cc / val_num).round(4)
            val_loss1=val_loss1/(val_batch+1)
            progbar.update(train_batch, values=[('loss', train_loss),
                                                 ('cc', train_cc),
                                                ('val_loss1', val_loss1),
                                                ('val_cc',val_cc),
                                                ], finalize=True)
            if val_loss1 < self.best_loss1:
                self.best_loss1 = val_loss1
                tf.keras.models.save_model(self.model, self.modelpath)
                print("Best Weights saved in",self.modelpath)
            else:
                self.waiting+=1
            if self.waiting>args.patience:
                print("Finished in advance")
                break

    def evaluate(self, val_data, use_main=False):
        self.val_loss_tracker.reset_states()
        for val_batch, example in enumerate(val_data):
                val_loss = self.test_step(example)
                self.val_loss_tracker.update_state(val_loss)

        return self.val_loss_tracker.result()

if __name__ == "__main__":
    # putting your trainpath\valpath\testpath
    trainds = read_tfrecords(trainpath).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).batch(args.bs).shuffle(1000)
    valds = read_tfrecords(valpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    testds = read_tfrecords(testpath).batch(args.bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # result
    resultpath = '../results/ptbxl/'

    if not os.path.exists(resultpath):
        os.mkdir(resultpath)
    # valds=strategy.experimental_distribute_dataset(valds)
    # trainds=strategy.experimental_distribute_dataset(trainds)
    args.step_total=87200//args.bs    #
    for [anylead,padding]in  [[1,'copy']]:
        modelpath = "Autoencoder_ks5_copy_"+str(anylead)
        args.anylead=anylead
        args.padding=padding
        trainer_copy = Trainer_copy(modelpath=modelpath,epochs=args.epochs, lr=args.lr, ecglen=args.ecglen)
        trainer_copy.train(trainds, valds)

        model = tf.keras.models.load_model(modelpath)
        MAE_test, MSE_test, CC_test = test_ae(model=model, ds=testds, output_num=1)
        excel_path = resultpath +  "Autoencoder_ks5_copy_"+str(anylead)+ '.xlsx'
        write2excel(MAE_test, MSE_test, CC_test, excel_path)
