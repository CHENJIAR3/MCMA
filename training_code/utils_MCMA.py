import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from  model_new import modelx
# gpus = tf.config.experimental.list_physical_devices('GPU')
# assert len(gpus) > 0, "Not enough GPU hardware devices available"
# for i in range(len(gpus)):
#     tf.config.experimental.set_memory_growth(gpus[i], True)
#首先引入策略

import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
assert len(gpus) > 0, "Not enough GPU hardware devices available"
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
#首先引入策略
strategy = tf.distribute.MirroredStrategy()
import numpy as np
parser=argparse.ArgumentParser(description='setting')
parser.add_argument('--ecglen',default=1024,type=int,help='HeartBeat length')
parser.add_argument('--ecglen_Long',default=5000,type=int,help='Recording length')

parser.add_argument('--max_bpm',default=200,type=int,help='max bpm')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

parser.add_argument('--fs',default=500,type=float,help='signal sampling frequency')
parser.add_argument('--epochs',default=100,type=float,help='epochs')

parser.add_argument('--bs',default=256,type=int,help='batch size')
parser.add_argument('--patience',default=50)
parser.add_argument("--classnum",default=12)
parser.add_argument("--trainflag",default=1,help="The flag for training model")
parser.add_argument("--testflag",default=1,help="The flag for testing model")
parser.add_argument("--dataupdate",default=1,help="The flag for dataset updating")
parser.add_argument("--R_peak",default=0,type=bool,help="Using the R_peak to split model")
parser.add_argument('--time_steps',default=20)
parser.add_argument('--anylead',default=1)
parser.add_argument('--padding',default='zeros')
parser.add_argument('--conditional',default=1,type=bool)
parser.add_argument('--load_model',default=0,type=bool)
parser.add_argument('--data_norm',default=0,type=bool)
parser.add_argument('--lambda_',default=50,type=float)
parser.add_argument('--alpha',default=1,type=float)
parser.add_argument('--beta_start',default=1e-3,type=float)
parser.add_argument('--beta_end',default=1e-1,type=float)

args = parser.parse_args()


def datatorecord4c(tfrecordwriter,ecgs):
    writer = tf.io.TFRecordWriter(tfrecordwriter)  # 1. 定义 writer对象，创建tfrecord文件，输入的为文件名
    for i in range(ecgs.shape[0]):
        ecg=ecgs[i]
        ecg=np.asarray(ecg).astype(np.float32).tobytes()
        """ 2. 定义features """
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'ecg': tf.train.Feature(bytes_list = tf.train.BytesList(value=[ecg]))
                }))
        """ 3. 序列化,写入"""
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def decode_tfrecords4c(example):
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description = {
        'ecg': tf.io.FixedLenFeature([], tf.string)
    }
    # 按照feature_description解码
    feature_dict = tf.io.parse_single_example(example, feature_description)
    # 由bytes码转化为tf.float32
    ecg = (tf.io.decode_raw(feature_dict['ecg'], out_type=tf.float32))
    ecg=tf.reshape(ecg,[args.ecglen,12])
    return ecg

def read_tfrecords(tfrecord_file):
    #读取文件,数据预处理的一部分
    dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    dataset = dataset.map(decode_tfrecords4c,num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 解析数据
    return dataset


def decode_tfrecords_Long(example):
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description = {
        'ecg': tf.io.FixedLenFeature([], tf.string)
    }
    # 按照feature_description解码
    feature_dict = tf.io.parse_single_example(example, feature_description)
    # 由bytes码转化为tf.float32
    ecg = (tf.io.decode_raw(feature_dict['ecg'], out_type=tf.float32))
    ecg = tf.reshape(ecg,[args.ecglen_Long,12])
    return ecg

def read_tfrecords_Long(tfrecord_file):
    #读取文件,数据预处理的一部分
    dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    dataset = dataset.map(decode_tfrecords_Long,num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 解析数据
    return dataset
def extract(a, t, x_shape):
    batch_size, sequence_length, _ = a.shape
    t_shape = tf.shape(t)
    out = tf.gather(a, t, axis=-1)
    out = tf.reshape(out, (batch_size, t_shape[0], *((1,) * (len(x_shape) - 1))))
    return out

class Trainer:
    def __init__(self,modelpath,epochs=100, lr=1e-3, ecglen=1024,kernel_size=5,pool_size = 2,
                 figure_plot = 0,step_total = 340,anylead=1,patience = 50):
        # self.strategy=strategy
        # self.strategy = tf.distribute.MirroredStrategy()
        self.ecglen = ecglen
        self.epochs = epochs
        self.lr = lr
        self.figure_plot = figure_plot
        self.kernel_size = kernel_size
        self.anylead = anylead
        self.step_total = step_total
        self.patience = patience
        self.pool_size = pool_size
        self.modelpath = modelpath
        # with self.strategy.scope():
        if 1:
            self.model=modelx(input_size=(self.ecglen,12),kernel_size=self.kernel_size, pool_size=self.pool_size)
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
    # def sample_time_step(self, size):
    #     return tf.experimental.numpy.random.randint(1, args.time_steps, size=(size,))

    def getcc(self,ecg12, gen_ecg12):
        return (tf.reduce_mean(ecg12 * gen_ecg12, axis=[1]) - tf.reduce_mean(ecg12, axis=[1]) * tf.reduce_mean(
            gen_ecg12, axis=[1])) \
            / (tf.math.reduce_std(ecg12, axis=[1]) * tf.math.reduce_std(gen_ecg12, axis=[1]))

    @tf.function
    def train_step(self, ecg12):
        def step_fn(ecg12):
            l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)
            if self.anylead == 1:
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
            cc = self.getcc(ecg12,gen_ecg12)
            return tf.reduce_mean(loss1),tf.reduce_mean(cc)

        #self.strategy.run(step_fn, args=(ecg12,))
        return step_fn(ecg12)

    @tf.function
    def paddingecg(self,ecg12, index):
        ecg12   = tf.transpose(ecg12, [0, 2, 1])
        updates = tf.gather_nd(ecg12, index)
        ecg_new = tf.zeros_like(ecg12, dtype=tf.float32)
        ecg_new = tf.tensor_scatter_nd_update(ecg_new, index, updates)
        ecg_new = tf.transpose(ecg_new, [0, 2, 1])
        return ecg_new
    @tf.function
    def test_step(self, ecg12):
        def step_test(ecg12):
            l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)
            if self.anylead == 1:
                h_index = np.random.randint(0, 12, ecg12.shape[0]).reshape(-1, 1).astype(np.int32)
            else:
                h_index=np.zeros((ecg12.shape[0],1)).astype(np.int32)
            index = np.hstack((l_index, h_index))
            ecg1= self.paddingecg(ecg12, index)
            gen_ecg12 = self.model([ecg1],training=False)
            loss1 = tf.reduce_mean((gen_ecg12 - ecg12) ** 2, axis=1)
            cc = self.getcc(ecg12,gen_ecg12)
            return tf.reduce_mean(loss1),tf.reduce_sum(cc)
        # per_replica_losses =self.strategy.run(step_test, args=(ecg12,))
        return step_test(ecg12)

    def train(self, train_data, val_data):
        # val_data=self.strategy.experimental_distribute_dataset(val_data)
        # train_data=self.strategy.experimental_distribute_dataset(train_data)
        for epoch in range(self.epochs):
            self.loss_tracker.reset_states()
            self.loss1_tracker.reset_states()
            self.val_loss_tracker.reset_states()

            print("Epoch :", epoch+1)
            # with self.strategy.scope():
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
            if epoch % 5 == 0 and self.figure_plot:
                ecg12 = next(iter(train_data)).values[0]
                self.sample(ecg12, epoch, 2)
            if val_loss1 < self.best_loss1:
                self.best_loss1 = val_loss1
                tf.keras.models.save_model(self.model, self.modelpath)
                print("Best Weights saved in",self.modelpath)
            else:
                self.waiting+=1
            if self.waiting>self.patience:
                print("Finished in advance")
                break

    def sample(self, ecg12, epoch, no_of=1):
        print(ecg12.shape)
        l_index = np.arange(ecg12.shape[0]).reshape(-1, 1)
        # ecg1 = ecg12[:,:,0:1]
        h_index = np.zeros((ecg12.shape[0], 1)).astype(np.int32)
        index = np.hstack((l_index, h_index))
        ecg1 = self.paddingecg(ecg12, index)
        rect = np.array([0.05, 0.52, 0.4, 0.4])
        plt.figure()
        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.plot(ecg12[0, :, 1], 'k', linewidth=2)
        plt.grid(True)
        plt.title("Real lead II in Epoch " + str(epoch + 1))
        rect[0] += 0.5
        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.plot(ecg12[0, :, 6], 'k', linewidth=2)
        plt.grid(True)
        plt.title("Real lead V1 in Epoch " + str(epoch+1))
        rect[0] += 0.5

        rect[0] = 0.05
        rect[1] -= 0.5

        gen_ecg12 = self.model([ecg1])

        # time = tf.expand_dims(time, axis=-1)
        # lead12_enc = tf.ones_like(lead_enc)

        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.plot(gen_ecg12[0, :, 1], 'r', linewidth=2)
        plt.grid(True)
        plt.title("Generated lead II in Epoch " + str(epoch + 1))
        rect[0] += 0.5

        plt.axes(rect)
        plt.gca().set_yticklabels([])
        plt.gca().set_xticklabels([])
        plt.plot(gen_ecg12[0, :, 6], 'r', linewidth=2)
        plt.grid(True)
        plt.title("Generated lead V1 in Epoch " + str(epoch + 1))
        rect[0] += 0.5
        plt.show()

    def evaluate(self, val_data, use_main=False):
        self.val_loss_tracker.reset_states()
        for val_batch, example in enumerate(val_data):
                val_loss = self.test_step(example)
                self.val_loss_tracker.update_state(val_loss)

        return self.val_loss_tracker.result()
