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

device = f'cuda:0'

ddpm = DDPM(500, True, device, 1e-4, 0.05)

# model
model = DiTraj1D(traj_length=200,
                    patch_size=1,
                    in_channels=2,
                    hidden_size=384,
                    depth=12,
                    num_heads=6,
                    mlp_ratio=4.0)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

dist_params = torch.load('/home/zzhang18/proj/Traj_Gen/saved_models/110000.pt')
single_params = {k.replace('module.', ''): v for k, v in dist_params.items()}
single_params = {k.replace('_orig_mod.', ''): v for k, v in single_params.items()}
model.load_state_dict(single_params)
model = torch.compile(model)
model = model.to(device)
model.eval()

# prepare test data
city = 'chengdu'
N = 1000
# load test data
label_test = np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/label_test.npy')[0:N]
lengths = label_test[:,3].astype(np.int32)
label_mean = np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/label_mean.npy')
label_std = np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/label_std.npy')
label_test[:, 0] = np.floor((label_test[:, 0] % 86400) / 300)
label_test[:,1:6] = (label_test[:,1:6] - label_mean) / label_std

noise = torch.randn(N, 2, 200).to(device)
cond = torch.tensor(label_test).to(device)
assert cond.shape[0] == N
ddim_step = np.array(range(0, 500, 5))
for noise_idx in reversed(range(len(ddim_step))):
    t = torch.full((N,), ddim_step[noise_idx], dtype=torch.long).to(device)
    t_next = torch.full((N,), ddim_step[noise_idx - 1] if noise_idx > 0 else -1, dtype=torch.long).to(device)
    # with torch.autocast(device_type=self.device, dtype=torch.float16):
    with torch.no_grad():
        uncond_noise_pred = model(noise, t, cond, 1)
        cond_noise_pred = model(noise, t, cond, 0)
        # if any(uncond_noise_pred >= 20) or any(cond_noise_pred >= 20):
        # print(uncond_noise_pred.abs().max(), cond_noise_pred.abs().max())
    noise_pred = uncond_noise_pred + (cond_noise_pred - uncond_noise_pred) * 4
    noise_pred = torch.clamp(noise_pred, -6, 6)
    print(noise_pred.abs().max())
    # noise = ddpm.denoise(noise, noise_pred, t)
    noise = ddpm.denoise_ddim(noise, noise_pred, t, t_next, 0)

# convert to numpy
sample_np = noise.permute(0,2,1).cpu().numpy()
spatio_mean, spatio_std = np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_mean.npy'), np.load(f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_std.npy')
sample_np = sample_np.astype(np.float64)
sample_np = sample_np * spatio_std + spatio_mean
image = viz_trajs(sample_np, lengths, [104.03968953679004, 104.12705400673643], [30.655400079856072, 30.730172829483855])
# Display the image
plt.imshow(image)
plt.axis('off')  # Hide axis
plt.savefig('sample_image.png', bbox_inches='tight', pad_inches=0)