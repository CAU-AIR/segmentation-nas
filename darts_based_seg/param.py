CONFIG = {
    "SEED": 42,
    "DATA": {
        # "data_dir": "../data/image",
        # "label_dir": "../data/target",
        "data_dir": "../dataset/image",
        "label_dir": "../dataset/target",
        "batch_size": 64,
        "shape": (128, 128),
    },
    "TRAIN": {
        "num_epochs": 100,    # 100
        "warmup_epochs": 20, # 20
        "lr": 0.001,
        "loss_weight": 0.5,
        # "loss_weight": 0.0,
        "clip_grad": 5,
        "alpha_lr": 0.01,
        "weight_lr": 0.001,
        "weight_decay": 2e-4,
        "sample_weight_lr": 1e-5,
    },
<<<<<<< HEAD
    "GPU": [3],
=======
    "GPU": [1,2,3],
>>>>>>> 3c743e2a8698bd24e5657165f7d59bb39c52bfde
}
