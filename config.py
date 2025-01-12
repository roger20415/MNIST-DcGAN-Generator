class Config:
    MNIST_FOLDER_PATH: str = "./mnist"
    ORIGINAL_DATASET_PLOT_PATH: str = "./plot/training_dataset_plot.png"
    AUGMENTED_DATASET_PLOT_PATH: str = "./plot/augmented_dataset_plot.png"
    LOSS_PLOT_PATH: str = "./plot/loss_plot.png"
    NET_G_PATH: str = "./models/generator.pth"
    NET_D_PATH: str = "./models/discriminator.pth"
    
    BATCH_SIZE: int = 64
    
    NZ: int = 100
    NGF: int  = 64
    NC: int  = 3 
    NDF: int  = 64
    
    NGPU: int  = 1
    
    LR: float = 0.0002
    BETA1: float = 0.5
    
    EPOCHS: int  = 10