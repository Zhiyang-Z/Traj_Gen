import random
import numpy as np
import torch
import torch.multiprocessing as mp
from ddp.ddp_main import ddp_main
import os
# os.environ["WANDB_MODE"] = "disabled"
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    world_size = torch.cuda.device_count()
    print(f"detected {world_size} GPUs.")
    print("spawn processes...")
    mp.spawn(ddp_main, args=(world_size, "/home/zzhang18/datasets/minecraft/minecraft/mosaic/train/", 32), nprocs=world_size)
