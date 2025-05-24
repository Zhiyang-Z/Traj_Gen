import os
# os.environ["WANDB_MODE"] = "disabled"
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import numpy as np
import torch
import torch.multiprocessing as mp
from ddp.ddp_main import ddp_main

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    world_size = torch.cuda.device_count()
    print(f"detected {world_size} GPUs.")
    print("spawn processes...")

    import argparse
    parser = argparse.ArgumentParser(description="Traj Trandformer Training")
    parser.add_argument('--city', type=str)
    parser.add_argument('--lon_lat_emb', type=bool)
    parser.add_argument('--st_attention', type=bool)
    args = parser.parse_args()

    city, lon_lat_emb, st_attention = args.city, args.lon_lat_emb, args.st_attention

    mp.spawn(ddp_main, args=(world_size, f'/home/zzhang18/proj/Traj_Gen/datasets/', 'chengdu', 'b', True), nprocs=world_size)
