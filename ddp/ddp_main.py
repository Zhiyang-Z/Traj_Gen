import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from datasets.traj_dataloader import TrajectoryData
from torch.utils.data import DataLoader
from models.DiTraj1D import DiTraj1D
from ddp.ddp_trainer import Trainer
from torch.utils.data.distributed import DistributedSampler

def ddp_main(rank: int, world_size: int, data_path: str, batch_size: int):
    print("ddp setup...")
    ddp_setup(rank, world_size)
    print("ddp setup done.")

    # dataset
    city = 'chengdu'
    dataset = TrajectoryData(
        f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/data_train.npy',
        f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/label_train.npy',
        f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_shift_mean.npy',
        f'/home/zzhang18/proj/Traj_Gen/datasets/{city}/traj_shift_std.npy'
    )
    # dataset = Subset(dataset, indices=list(range(256)))
    dataloader = DataLoader(dataset,
                            batch_size=512,
                            pin_memory=True,
                            shuffle=False,
                            num_workers=4,
                            drop_last=True,
                            prefetch_factor=2,
                            sampler=DistributedSampler(dataset, shuffle=True, seed=0),
    )
    
    # model
    model = DiTraj1D(traj_length=200,
                     patch_size=1,
                     in_channels=2,
                     hidden_size=768,
                     depth=12,
                     num_heads=12,
                     mlp_ratio=4.0)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # train
    trainer = Trainer(dataloader, model, optimizer)
    print('training start...')
    trainer.train()

    ddp_cleanup()