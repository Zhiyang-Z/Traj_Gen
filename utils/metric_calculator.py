import numpy as np
import math
from geopy.distance import geodesic

class MetricCalculator:
    def __init__(self, gen_trajs: list, lon_range: list, lat_range: list, granularity):
        self.gen_trajs = gen_trajs
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.granularity = granularity
        self.grid_size = [math.ceil((lon_range[1] - lon_range[0])/granularity), math.ceil((lat_range[1] - lat_range[0])/granularity)]

    def kl_divergence(self, p:np.array, q: np.array):
        """Compute Kullback-Leibler divergence D(P || Q)"""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        # Add small epsilon to avoid division by zero or log(0)
        epsilon = 1e-7
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        return np.sum(p * np.log(p / q))

    def jensen_shannon_divergence(self, p:np.array, q: np.array):
        # before invorking this function, make sure that p and q are stretched to 1D
        """Compute the Jensen-Shannon divergence between two distributions"""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        
        # Normalize to ensure they are probability distributions
        p /= np.sum(p)
        q /= np.sum(q)
        
        m = 0.5 * (p + q)
        return 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)
    
    def calculate_density(self, gen_trajs):
        density = np.zeros(self.grid_size)
        for idx in range(len(gen_trajs)):
            traj = gen_trajs[idx]
            # traj: L, 2
            traj[:,0] = np.clip(traj[:,0], self.lon_range[0], self.lon_range[1])
            traj[:,1] = np.clip(traj[:,1], self.lat_range[0], self.lat_range[1])
            lon, lat = traj[:, 0], traj[:, 1]
            lon_idx = np.floor((lon - self.lon_range[0]) / self.granularity).astype(int)
            lat_idx = np.floor((lat - self.lat_range[0]) / self.granularity).astype(int)
            density[lon_idx, lat_idx] += 1
        return density/density.sum()

    def calculate_density_error(self, real_density_path: str):
        real_density = np.load(real_density_path)
        assert np.allclose(real_density.sum(), 1), "Real density should sum to 1"
        # Gen_traj: B, L, 2
        density = self.calculate_density(self.gen_trajs)
        error = self.jensen_shannon_divergence(density.flatten(), real_density.flatten())
        return error
    
    def calculate_trip_error(self, real_start_trip_density_path: str, real_end_trip_density_path: str):
        real_start_trip_density, real_end_trip_density = np.load(real_start_trip_density_path), np.load(real_end_trip_density_path)
        assert np.allclose(real_start_trip_density.sum(), 1), "Real start trip density should sum to 1"
        assert np.allclose(real_end_trip_density.sum(), 1), "Real end trip density should sum to 1"
        # Gen_traj: B, L, 2
        start_density = self.calculate_density([traj[0:1] for traj in self.gen_trajs])
        end_density = self.calculate_density([traj[-1:] for traj in self.gen_trajs])
        start_error = self.jensen_shannon_divergence(start_density.flatten(), real_start_trip_density.flatten())
        end_error = self.jensen_shannon_divergence(end_density.flatten(), real_end_trip_density.flatten())
        return 0.5*start_error + 0.5*end_error
    
    def traj_dist(self, traj):
        # traj: L, 2
        trip_distance = 0
        original_length = len(traj)
        for i in range(original_length-1):
            trip_distance += geodesic((traj[i][1], traj[i][0]), (traj[i+1][1], traj[i+1][0])).m
        return trip_distance
    
    def calculate_hist(self, gen_trajs: list, xrange: list=[0, 30], granularity: float=0.3):
        hist = np.zeros((math.ceil((xrange[1] - xrange[0]) / granularity),))
        for traj in gen_trajs:
            dist = self.traj_dist(traj)/1000
            dist = np.clip(dist, xrange[0], xrange[1])
            idx = math.floor((dist - xrange[0]) / granularity)
            idx = idx if idx < hist.shape[0] else hist.shape[0] - 1
            hist[idx] += 1
        assert hist.sum() == len(gen_trajs), "Histogram should sum to the number of trajectories"
        hist /= hist.sum()
        return hist
    
    def calculate_length_error(self, real_length_hist_path: str, xrange: list=[0, 30], granularity: float=0.3):
        real_length_hist = np.load(real_length_hist_path)
        assert np.allclose(real_length_hist.sum(), 1), "Real length histogram should sum to 1"
        hist = self.calculate_hist(self.gen_trajs, xrange, granularity)
        error = self.jensen_shannon_divergence(hist, real_length_hist)
        return error

