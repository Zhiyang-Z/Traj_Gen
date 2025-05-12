import numpy as np
import random

# Set random seed
random.seed(42)
np.random.seed(42)

city = 'chengdu'
# Example array
data = np.load(f'./{city}/total30.npy')
label = np.load(f'./{city}/total30_label.npy')
assert data.shape[0] == label.shape[0]
print(f'data: {data.shape}, label: {label.shape}')

N = data.shape[0]
print(f'total # of trajs: {N}')
random_idx = np.random.choice(N, size=3000, replace=False)

data_test = data[random_idx]
label_test = label[random_idx]
data_train = np.delete(data, random_idx, axis=0)
label_train = np.delete(label, random_idx, axis=0)
print(f'data_train: {data_train.shape}, label_train: {label_train.shape}, data_test: {data_test.shape}, label_test: {label_test.shape}')

np.save(f'./{city}/data_train.npy', data_train)
np.save(f'./{city}/label_train.npy', label_train)
np.save(f'./{city}/data_test.npy', data_test)
np.save(f'./{city}/label_test.npy', label_test)
