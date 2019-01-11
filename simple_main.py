import os
import time
from argparse import ArgumentParser

import keras
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.regularizers import l2

import loader

parser = ArgumentParser()
parser.add_argument('epochs', type=int)
parser.add_argument('--points', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--grid', type=int, default=20)
rotation = parser.add_argument_group('rotation')
rotation.add_argument('--rotate', action='store_true')
rotation.add_argument('--rotate_val', action='store_true')
rotation.add_argument('--per_rotation', type=int, default=5)
rotation.add_argument('--eight', action='store_true')
earlystopping = parser.add_argument_group('earlystopping')
earlystopping.add_argument('--early', action='store_true')
earlystopping.add_argument('--patience', type=int, default=10)
dropout = parser.add_argument_group('dropout')
dropout.add_argument('--dropout', type=float, default=.3)
dropout.add_argument('--rec_drop', type=float, default=.2)
parser.add_argument('--train_files',
                    default='./data/modelnet40_ply_hdf5_2048/train_files.txt')
parser.add_argument('--test_files',
                    default='./data/modelnet40_ply_hdf5_2048/test_files.txt')
parser.add_argument('--weight_dir', type=str,
                    default='weights')
structure = parser.add_argument_group('structure')
structure.add_argument('--layer', type=str, default='',
                       help='structure of the model')
structure.add_argument('--strides', type=str, default='',
                       help='strides of each layer')
structure.add_argument('--kernel', type=str, default='',
                       help='kernel size of each layer')
args = parser.parse_args()


if not os.path.exists(args.weight_dir):
    os.makedirs(args.weight_dir)


class Counter:
    def __init__(self):
        self.units = 8

    def get_layer(self, layer_name, strides=1, kernel_size=3, units=None):
        self.units *= 2
        if units:
            self.units = units
        if layer_name == 'c':
            return Conv3D(filters=self.units,
                          strides=[strides]*3,
                          kernel_size=[kernel_size]*3)
        if layer_name == 'l':
            return ConvLSTM2D(filters=self.units,
                              strides=[strides]*2,
                              kernel_size=[kernel_size]*2)
        if layer_name == 'q':
            return ConvLSTM2D(filters=self.units,
                              strides=[strides]*2,
                              kernel_size=[kernel_size]*2,
                              return_sequences=True)
        if layer_name == 'd':
            return Dense(units=self.units)
        if layer_name == 'f':
            return Flatten()
        if layer_name == 'r':
            return ReLU()
        if layer_name == 'b':
            return BatchNormalization(axis=-1)


args.layers = args.layers.split(',')
args.strides = args.strides.split(',')
args.kernel_size = args.kernel_size.split(',')
 
assert args.layers and args.strides and args.kernel_size
assert len(args.layers) == len(args.strides) == len(args.kernel_size)

input = Input(shape=[args.grid]*3+[1])
layered = input


((x_train, y_train),
 (x_test, y_test)) = loader.convert_data(args.train_files,
                                         args.test_files,
                                         num_points=args.points,
                                         rotate=args.rotate,
                                         rotate_val=args.rotate_val,
                                         grid_size=args.grid)


K.clear_session()
