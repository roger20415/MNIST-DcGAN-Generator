class Config:
    MNIST_FOLDER_PATH: str = "./mnist"
    ORIGINAL_DATASET_PLOT_PATH: str = "./plot/training_dataset_plot.png"
    AUGMENTED_DATASET_PLOT_PATH: str = "./plot/augmented_dataset_plot.png"
    
    BATCH_SIZE: int = 64
    
    
    NZ = 100
    NGF = 64
    NC = 3 
    NDF = 64
    
    NGPU = 1
    
    LR = 0.0002
    BETA1 = 0.5
    
    EPOCHS = 1