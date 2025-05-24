import numpy as np
import math
from geopy.distance import geodesic

def resample_trajectory(x, length=200):
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T

city = 'xian'
granularity = 0.00045
dist_granularity = 0.3

if city == 'chengdu':
    # chengdu: [104.03968953679004, 30.655400079856072, 104.12705400673643, 30.730172829483855]
    min_lat, max_lat, min_lon, max_lon = 30.655400079856072, 30.73018, 104.03968953679004, 104.12706
elif city == 'xian':
    # xian: [108.90659955383317, 34.20694207396501, 108.99373658803785, 34.28184317582219]
    min_lat, max_lat, min_lon, max_lon = 34.20694207396501, 34.28185, 108.90659955383317, 108.99374
else:
    raise ValueError('city not in [chengdu, xian]')
lon_range = [min_lon, max_lon]
lat_range = [min_lat, max_lat]
grid_size = [math.ceil((lon_range[1] - lon_range[0])/granularity), math.ceil((lat_range[1] - lat_range[0])/granularity)]

prefix = f'/home/zzhang18/proj/Traj_Gen/datasets/'
data_path = f'/home/zzhang18/proj/Traj_Gen/datasets/' + f'{city}/{city}_test.npy'
label_path = f'/home/zzhang18/proj/Traj_Gen/datasets/' + f'{city}/{city}_test_label.npy'
data = np.load(data_path)
label = np.load(label_path)
lengths = label[:,3].astype(np.int32)

Gen_traj = []
for j in range(data.shape[0]):
    new_traj = resample_trajectory(data[j], lengths[j])
    assert new_traj.dtype == np.float64
    Gen_traj.append(new_traj)
assert len(Gen_traj) == data.shape[0]

density = np.zeros(grid_size)
for idx in range(len(Gen_traj)):
    traj = Gen_traj[idx]
    # traj: L, 2
    lon, lat = traj[:, 0], traj[:, 1]
    lon_idx = np.floor((lon - lon_range[0]) / granularity).astype(int)
    lat_idx = np.floor((lat - lat_range[0]) / granularity).astype(int)
    density[lon_idx, lat_idx] += 1
density /= density.sum()
np.save(prefix + f'{city}/{city}_density_{granularity}.npy', density)

start_density = np.zeros(grid_size)
end_density = np.zeros(grid_size)
for idx in range(len(Gen_traj)):
    traj = Gen_traj[idx]
    # traj: L, 2
    lon, lat = traj[0, 0], traj[0, 1]
    lon_idx = np.floor((lon - lon_range[0]) / granularity).astype(int)
    lat_idx = np.floor((lat - lat_range[0]) / granularity).astype(int)
    start_density[lon_idx, lat_idx] += 1

    lon, lat = traj[-1, 0], traj[-1, 1]
    lon_idx = np.floor((lon - lon_range[0]) / granularity).astype(int)
    lat_idx = np.floor((lat - lat_range[0]) / granularity).astype(int)
    end_density[lon_idx, lat_idx] += 1
start_density /= start_density.sum()
end_density /= end_density.sum()
np.save(prefix + f'{city}/{city}_start_density_{granularity}.npy', start_density)
np.save(prefix + f'{city}/{city}_end_density_{granularity}.npy', end_density)

def traj_dist(traj):
    # traj: L, 2
    trip_distance = 0
    original_length = len(traj)
    for i in range(original_length-1):
        trip_distance += geodesic((traj[i][1], traj[i][0]), (traj[i+1][1], traj[i+1][0])).m
    return trip_distance

dist_hist = np.zeros((math.ceil((30-0) / dist_granularity),))
for traj in Gen_traj:
    dist = traj_dist(traj)/1000
    dist = np.clip(dist, 0, 30)
    idx = math.floor((dist - 0) / dist_granularity)
    dist_hist[idx] += 1
assert dist_hist.sum() == len(Gen_traj), "Histogram should sum to the number of trajectories"
dist_hist /= dist_hist.sum()
np.save(prefix + f'{city}/{city}_dist_hist_{dist_granularity}.npy', dist_hist)
