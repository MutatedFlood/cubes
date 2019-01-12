import os
import sys
import time
from argparse import ArgumentParser

import keras
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import *
from keras.losses import sparse_categorical_crossentropy
from keras.models import *
from keras.regularizers import l1_l2
from matplotlib import pyplot as plt

import loader

parser = ArgumentParser()
parser.add_argument('epochs', type=int,
                    help='Total epochs to train')
parser.add_argument('--points', type=int, default=1024,
                    help='How many points to preserve(max 2048) when training and testing')
parser.add_argument('--batch_size', type=int, default=64,
                    help="How many sets of 'pictures' goes into the model during training")
parser.add_argument('--grid', type=int, default=20,
                    help='The dimension of a 3d picture')
parser.add_argument('--optimizer', type=str, default='rmsprop',
                    help='Optimizer of choice')
parser.add_argument('--debug', action='store_true',
                    help='debug mode. Terminates the program early(before training)')
parser.add_argument('--interval', type=int, default=4,
                    help="The rate the stacked layers' filters are increasing")
parser.add_argument('--first_layer', type=int, default=4,
                    help='How many filters the first layer will have')
structure = parser.add_argument_group('structure')
structure.add_argument('--layers', type=str, default='',
                       help='structure of the model')
'''
every layer is represented by name,strides,kernel_size,units, which are separated by ','
[ name,strides,kernel_size,units ]
different layers are seperated by '.'

name:
'd': `Dense`
'c': `Conv3D`
'q': `ConvLSTM2D`, return_sequences=True
'l': `ConvLSTM2D`, return_sequences=False
'f': `Flatten`
'o': `Dropout`
'b': `BatchNormalization`
'r': `ReLU`
's': `Softmax`
'g': `Sigmoid`
't': `tanh`

strides: defaults to 1 (no input. example: d, or d,, or d,,3,1)

kernel_size: defaults to 3 (no input. example: d,2,,4, or d,,,)

units: defaults to be interval * previous_filters. Last `Dense` layer will have units=40

'''

parser.add_argument('--l1', type=float, default=0.,
                    help='L1 penalty applied on recurrent layers')
parser.add_argument('--l2', type=float, default=0.,
                    help='L2 penalty applied on recurrent layers')
parser.add_argument('--cuda', type=str, default='0',
                    help='Which cuda device is to be used')
rotation = parser.add_argument_group('rotation')
rotation.add_argument('--rotate', action='store_true',
                      help='Whether or not to rotate the training data')
rotation.add_argument('--rotate_val', action='store_true',
                      help='Whether or not to rotate the validation data')
rotation.add_argument('--per_rotation', type=int, default=1,
                      help='How many epochs for every rotation')
earlystopping = parser.add_argument_group('earlystopping')
earlystopping.add_argument('--early', action='store_true',
                           help='Whether to apply `EarlyStopping`')
earlystopping.add_argument('--patience', type=int, default=10,
                           help='How many epochs to wait for an improvement')
dropout = parser.add_argument_group('dropout')
dropout.add_argument('--dropout', type=float, default=.3,
                     help='The rate of dropout for every dropout layer')
dropout.add_argument('--rec_drop', type=float, default=0.,
                     help='The rate of recurrent dropout for every recurrent layer')
history = parser.add_argument_group('--history')
history.add_argument('--save', action='store_true',
                     help='Whether or not to save training loss history to disk')
history.add_argument('--plot', action='store_true',
                     help='Whthter or not to plot and save training loss')
parser.add_argument('--train_files',
                    default='./data/modelnet40_ply_hdf5_2048/train_files.txt')
parser.add_argument('--test_files',
                    default='./data/modelnet40_ply_hdf5_2048/test_files.txt')
parser.add_argument('--weight_dir', type=str,
                    default='weights')
args = parser.parse_args()


# setting GPU usage
config = tf.ConfigProto()
config.gpu_options.visible_device_list = args.cuda
set_session(tf.Session(config=config))

if not os.path.exists(args.weight_dir):
    os.makedirs(args.weight_dir)

with open('.gitignore', 'r') as file:
    content = file.read()
    if not args.weight_dir in content:
        with open('.gitignore', 'a') as file:
            file.write('\n'+args.weight_dir)


class Counter:
    def __init__(self, interval=4, starting_at=4):
        self.units = starting_at
        self.interval = interval

    def get_layer(self, layer_name, strides=1, kernel_size=3, units=0):
        self.units *= self.interval
        if units:
            self.units = units
        if layer_name == 'c':
            return Conv3D(filters=self.units,
                          strides=[strides]*3,
                          kernel_size=[kernel_size]*3)
        if layer_name == 'l':
            return ConvLSTM2D(filters=self.units,
                              strides=[strides]*2,
                              kernel_size=[kernel_size]*2,
                              recurrent_initializer='orthogonal',
                              recurrent_regularizer=l1_l2(args.l1, args.l2),
                              recurrent_dropout=args.rec_drop)
        if layer_name == 'q':
            return ConvLSTM2D(filters=self.units,
                              strides=[strides]*2,
                              kernel_size=[kernel_size]*2,
                              return_sequences=True,
                              recurrent_initializer='orthogonal',
                              recurrent_regularizer=l1_l2(args.l1, args.l2),
                              recurrent_dropout=args.rec_drop)
        if layer_name == 'd':
            return Dense(units=self.units)
        if layer_name == 'f':
            return Flatten()
        if layer_name == 'r':
            return ReLU()
        if layer_name == 'b':
            return BatchNormalization(axis=-1)
        if layer_name == 'o':
            return Dropout(rate=args.dropout)
        if layer_name == 't':
            return Activation(activation='tanh')
        if layer_name == 's':
            return Activation(activation='softmax')
        if layer_name == 'g':
            return Activation(activation='sigmoid')


def transform(layers):
    if not layers:
        print('--layer flag cannot be empty')
        sys.exit()
    layers = layers.split('.')
    for i in range(len(layers)):
        layers[i] = layers[i].split(',')
        try:
            layers[i][1] = int(layers[i][1])
        except IndexError:
            layers[i].append(1)
        except ValueError:
            layers[i][1] = 1
        try:
            layers[i][2] = int(layers[i][2])
        except IndexError:
            layers[i].append(3)
        except ValueError:
            layers[i][2] = 3
        try:
            layers[i][3] = int(layers[i][3])
        except IndexError:
            layers[i].append(0)
        except ValueError:
            layers[i][3] = 0

    for ind in reversed(range(len(layers))):
        if layers[ind][0] in ['d']:
            layers[ind][-1] = 40
            break
    return layers


layers = transform(args.layers)


input = Input(shape=[args.grid]*3+[1])
counter = Counter(args.interval, args.first_layer)
layered = input
for l, s, k, u in layers:
    layered = counter.get_layer(l, s, k, u)(layered)

model = Model(inputs=[input], outputs=[layered])

model.compile(optimizer=args.optimizer,
              loss=sparse_categorical_crossentropy,
              metrics=['acc'])

model.summary()

if args.debug:
    # ((data, label),
    #  (test_data, test_label)) = loader.convert_data(
    #     args.train_files,
    #     args.test_files,
    #     args.points,
    #     rotate=True,
    #     rotate_val=args.rotate_val,
    #     grid_size=args.grid)
    # print(data.shape, label.shape, test_data.shape, test_label.shape)
    x_debug = np.random.randn(*([args.batch_size]+[args.grid]*3+[1]))
    print(model.predict(x_debug, batch_size=args.batch_size).shape)
    K.clear_session()
    sys.exit()

ModelCheckPoint = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('weights', 'best.{epoch:03d}.hdf5'),
    save_best_only=True)

TensorBoard = keras.callbacks.TensorBoard()

EarlyStopping = keras.callbacks.EarlyStopping(
    monitor='val_acc', patience=10)

if args.rotate:
    for epoch in range(1, args.epochs+1, args.per_rotation):
        ((data, label),
         (test_data, test_label)) = loader.convert_data(
            args.train_files,
            args.test_files,
            args.points,
            rotate=True,
            rotate_val=args.rotate_val,
            grid_size=args.grid)
        print('epoch: {}/{}'.format(epoch, args.epochs))
        loss = model.fit(x=data,
                         y=label,
                         batch_size=args.batch_size,
                         epochs=args.per_rotation,
                         validation_data=(test_data, test_label))
elif args.rotate_val:
    ((data, label),
     (test_data, test_label)) = loader.convert_data(
        args.train_files,
        args.test_files,
        args.points,
        rotate=True,
        rotate_val=args.rotate_val,
        grid_size=args.grid)
    for epoch in range(1, args.epochs+1, args.per_rotation):
        (test_data, test_label) = loader.convert_str(files=args.test_files,
                                                     points=args.points,
                                                     rotate=True,
                                                     grid_size=args.grid)
        print('epoch: {}/{}'.format(epoch, args.epochs))
        loss = model.fit(x=data,
                         y=label,
                         batch_size=args.batch_size,
                         epochs=args.per_rotation,
                         validation_data=(test_data, test_label))
else:
    ((data, label),
     (test_data, test_label)) = loader.convert_data(
        args.train_files,
        args.test_files,
        args.points,
        rotate=False,
        rotate_val=False,
        grid_size=args.grid)
    loss = model.fit(x=data,
                     y=label,
                     batch_size=args.batch_size,
                     epochs=args.epochs,
                     callbacks=[ModelCheckPoint,
                                TensorBoard,
                                EarlyStopping],
                     validation_data=(test_data, test_label))

if args.save:
    np.save(file='./history', arr=loss.history)

if args.plot:
    for item in loss.history.keys():
        plt.plot(loss.history[item], label=item)
    plt.legend()
    plt.savefig('./loss_metrics.jpg')

# on original data
(x_train, y_train), (x_test, y_test) = loader.convert_data(
    args.train_files, args.test_files, args.points, grid_size=args.grid)
(loss, acc) = model.evaluate(x=x_train, y=y_train)
print('training loss: {}, training accuracy: {}'.format(loss, acc))
(loss, acc) = model.evaluate(x=x_test, y=y_test)
print('testing loss: {}, testing accuracy: {}'.format(loss, acc))

K.clear_session()
