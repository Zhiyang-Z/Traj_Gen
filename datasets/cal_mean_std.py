import numpy as np

city = 'chengdu'

data_path = f'/home/zhiyang/projects/DiTraj/datasets/{city}/data_train.npy'
data = np.load(data_path)

traj_means = data.mean(axis=(0,1))
traj_stds = data.std(axis=(0,1))

np.save(f'/home/zhiyang/projects/DiTraj/datasets/{city}/traj_mean', traj_means)
np.save(f'/home/zhiyang/projects/DiTraj/datasets/{city}/traj_std', traj_stds)

print(traj_means.shape, traj_stds.shape)
print(traj_means, traj_stds)

label_path = f'/home/zhiyang/projects/DiTraj/datasets/{city}/label_train.npy'
label = np.load(label_path)

label_means = label[:,1:6].mean(axis=0)
label_stds = label[:,1:6].std(axis=0)

np.save(f'/home/zhiyang/projects/DiTraj/datasets/{city}/label_mean', label_means)
np.save(f'/home/zhiyang/projects/DiTraj/datasets/{city}/label_std', label_stds)

print(label_means.shape, label_stds.shape)
print(label_means, label_stds)