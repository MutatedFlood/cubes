import numpy as np
import loader

train_files = './data/modelnet40_ply_hdf5_2048/train_files.txt'

test_files = './data/modelnet40_ply_hdf5_2048/test_files.txt'

(train, _), (test, _) = loader.convert_data(train_files, test_files)

collision = []

for j in range(len(test)):
    for i in range(len(train)):
        if ((test[j]-train[i])**2).sum() <= 1:
            collision.append([i, j])
            print([i,j])

np.savetxt('train-test-pair.txt', np.array(collision),fmt='%03d')
