import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

from einops import rearrange
import wandb
import numpy as np
from utils.utils import DDPM
from tqdm import tqdm
from utils.utils import viz_trajs
from models.DiTraj1D import DiTraj1D
import matplotlib.pyplot as plt

device = f'cuda:1'

ddpm = DDPM(500, True, device, 1e-4, 0.05)

# model
# model = DiTraj1D(traj_length=200,
#                     patch_size=1,
#                     in_channels=2,
#                     hidden_size=384,
#                     depth=12,
#                     num_heads=6,
#                     mlp_ratio=4.0)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")

# dist_params = torch.load('/home/zzhang18/proj/Traj_Gen/saved_models/AAAnoshift109000.pt')
# single_params = {k.replace('module.', ''): v for k, v in dist_params.items()}
# single_params = {k.replace('_orig_mod.', ''): v for k, v in single_params.items()}
# model.load_state_dict(single_params)
# model = torch.compile(model)
# model = model.to(device)
# model.eval()

# prepare test data
city = 'chengdu'
N = 1000
# load test data
spatio_mean, spatio_std = np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_mean.npy'), np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_std.npy')
data_test = np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/data_test.npy')[0:N]
data_test = (data_test-spatio_mean) / spatio_std
label_test = np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/label_test.npy')[0:N]
lengths = label_test[:,3].astype(np.int32)
data_test = torch.tensor(data_test)
for noise_level in range(500):
    noise_levels = torch.full((N,), noise_level, device=device)
    noise, noise_added = ddpm.forward_with_noise_level(data_test, noise_levels)
    # convert to numpy
    sample_np = noise.cpu().numpy()
    spatio_mean, spatio_std = np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_mean.npy'), np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_std.npy')
    sample_np = sample_np.astype(np.float64)
    sample_np = sample_np * spatio_std + spatio_mean
    image = viz_trajs(sample_np, lengths, [104.03968953679004, 104.12705400673643], [30.655400079856072, 30.730172829483855])
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    plt.savefig(f'./process/forward/sample_image{noise_level}.png', bbox_inches='tight', pad_inches=0)
