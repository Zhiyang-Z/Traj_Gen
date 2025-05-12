import torch
from torch.utils.data import Dataset
import numpy as np
# trajs_label.append((departure_time, trip_distance, trip_time, original_length, trip_distance/(original_length-1), trip_distance/trip_time, start_region, end_region))
class TrajectoryData(Dataset):
    def __init__(self, data_path, label_path, traj_mean_path, traj_std_path):
        self.data = np.load(data_path)
        self.label = None if label_path is None else np.load(label_path)
        # self.state_num, self.len = state_num, self.data.shape[1]
        assert self.data.shape[0] == self.label.shape[0]
        self.label[:, 0] = np.floor((self.label[:, 0] % 86400) / 300)
        self.means = self.label[:,1:6].mean(axis=0)
        self.stds = self.label[:,1:6].std(axis=0)
        self.label[:,1:6] = (self.label[:,1:6] - self.means) / self.stds

        self.traj_means = np.load(traj_mean_path)
        self.traj_stds = np.load(traj_std_path)
        self.data = (self.data - self.traj_means) / self.traj_stds

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # wrap data by adding 2 special states: <SOT> and <EOT>
        # data = np.zeros(self.len + 2)
        # ori_data, ori_label = self.data[idx], self.label[idx]
        # data[1:(self.len+1)] = (ori_data + 1)
        # data[-1] = self.state_num+1
        # ori_label[1:3] += 1
        return self.data[idx].astype(np.float32), self.label[idx]
