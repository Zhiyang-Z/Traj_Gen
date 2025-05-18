import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from datasets.traj_dataloader import TrajectoryData
from tqdm import tqdm
from models.DiTraj import DiTraj
from utils.utils import DDPM
from einops import rearrange
import random
import numpy as np
import math
from utils.utils import viz_trajs
import wandb
import os
# os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

city = 'chengdu'

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# enable fp32
torch.set_float32_matmul_precision("high")

wandb.init(project="Traj_Gen")

device = f'cuda:0'

dataset = TrajectoryData(
    f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/data_train.npy',
    f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/label_train.npy',
    f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_mean.npy',
    f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_std.npy'
)
# dataset = Subset(dataset, indices=list(range(256)))
dataloader = DataLoader(dataset,
                        batch_size=512,
                        pin_memory=True,
                        shuffle=True,
                        num_workers=8,
                        drop_last=True,
                        prefetch_factor=8
)
label_test = np.load(f'datasets/{city}/label_test.npy')
label_mean = np.load(f'datasets/{city}/label_mean.npy')
label_std = np.load(f'datasets/{city}/label_std.npy')
label_test[:, 0] = np.floor((label_test[:, 0] % 86400) / 300)
label_test[:,1:6] = (label_test[:,1:6] - label_mean) / label_std
lengths = (label_test[:,3] * label_std[2] + label_mean[2]).astype(np.int32)
spatio_mean, spatio_std = np.load(f'datasets/{city}/traj_mean.npy'), np.load(f'datasets/{city}/traj_std.npy')

model = DiTraj(depth=12, hidden_size=768, patch_size=1, num_heads=12).to(device)
model = torch.compile(model).to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scaler = torch.GradScaler()

ddpm = DDPM(500, True, device)

@torch.no_grad
def sample():
    model.eval()
    label_test_tensor = torch.tensor(label_test)
    # label_test = np.load(f'/home/zhiyang/projects/DiTraj/datasets/{city}/label_train.npy')[0:256]
    # label_test[:, 0] = np.floor((label_test[:, 0] % 86400) / 300)
    # label_test[:,1:6] = (label_test[:,1:6] - label_mean) / label_std
    # lengths = (label_test[:,3] * label_std[2] + label_mean[2]).astype(np.int32)


    label_test_tensor = torch.tensor(label_test)
    N = label_test_tensor.shape[0]
    BATCH_SIZE = 1000
    BATCH = math.floor(N / BATCH_SIZE)
    REMAINDER = N % BATCH
    assert REMAINDER == 0
    gen_trajs = np.zeros((N, 200, 2))
    for i in range(BATCH):
        noise = torch.randn(BATCH_SIZE, 1, 2, 200).to(device)
        cond = label_test_tensor[(BATCH_SIZE*i) : (BATCH_SIZE*i + BATCH_SIZE)].to(device)
        ddim_step = np.array(range(0, 500, 5))
        for noise_idx in reversed(range(len(ddim_step))):
            t = torch.full((BATCH_SIZE,), ddim_step[noise_idx], dtype=torch.long).to(device)
            t_next = torch.full((BATCH_SIZE,), ddim_step[noise_idx - 1] if noise_idx > 0 else -1, dtype=torch.long).to(device)
            # with torch.autocast(device_type=device, dtype=torch.float16):
            noise_pred = model(noise, t, cond)
            # noise = ddpm.denoise(noise, noise_pred, t)
            noise = ddpm.denoise_ddim(noise, noise_pred, t, t_next, 0)
        gen_trajs[(BATCH_SIZE*i) : (BATCH_SIZE*i + BATCH_SIZE)] = noise.squeeze().permute(0,2,1).contiguous().cpu().numpy()
    
    image = viz_trajs(gen_trajs, lengths, spatio_mean, spatio_std, [104.03968953679004, 104.12705400673643], [30.655400079856072, 30.730172829483855])
    wandb.log({"3000sample": wandb.Image(image)})


if __name__ == "__main__":
    step = -1
    for epoch in range(20000000):
        # print('EPOCH: ', epoch)
        for i, (trajs, labels) in tqdm(enumerate(dataloader)):
            model.train()
            trajs, labels = trajs.to(device), labels.to(device)
            trajs = rearrange(trajs, 'B W H -> B 1 H W').contiguous()
            trajs_noise, noise, noise_levels = ddpm.forward(trajs)
            optimizer.zero_grad()
            # with torch.autocast(device_type=device, dtype=torch.float16):
            noise_pred = model(trajs_noise, noise_levels, labels)
            loss = loss_fn(noise_pred, noise)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            # scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # scaler.step(optimizer)
            # scaler.update()
            step += 1
            wandb.log({"grad_norm": grad_norm.item()}, commit = False)
            wandb.log({"loss": loss.item()}, commit=True)

            if step % 1000 == 0:
                sample()
                torch.save(model.state_dict(), f"/home/zhiyang/projects/DiTraj/saved_models/{step}.pt")
