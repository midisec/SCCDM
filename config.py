class Config:
    # Training
    batch_size = 2
    n_epochs = 2000
    lr = 1e-5
    # lr = 1e-4
    grad_clip = 1
    
    # Model
    T = 1000
    # ch = 128   #512
    ch = 64
    ch_mult = [1, 2, 3, 4]
    attn = [2]
    num_res_blocks = 2
    dropout = 0.3
    
    # Diffusion
    beta_1 = 1e-4
    beta_T = 0.02

    # Attention Map
    psi = 1.0
    s = 0.1
    
    # Data
    image_size = 256
    dataset_path = "/home/midi/datasets/process_data0707_512_augment_50_no_background/train/"
    eval_dataset_path = "/home/midi/datasets/process_data0707_512_augment_50_no_background/val/"
    
    # Checkpoints
    save_dir = "checkpoints/"
    
    # Logging
    log_file = "logs/training_256_4batch_2000epochs.log"