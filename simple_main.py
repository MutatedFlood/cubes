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

model = Sequential()

lsk = [False]*3

zip_list = []
if len(args.layers) != 0:
    zip_list.append(args.layers)
    lsk[0] = True
if len(args.strides) != 0:
    zip_list.append(args.strides)
    lsk[1] = True
if len(args.kernel) != 0:
    zip_list.append(args.kernel)
    lsk[2] = True

length = -1
for agmt in zip_list:
    if length >= 0:
        if len(agmt) != length:
            exit()
    else:
        length = len(agmt)
del length


if len(model.layers) == 0:
    model.add(Flatten(input_shape=[args.grid]*3))
    model.add(Dense(units=40, activation='relu'))

K.clear_session()
