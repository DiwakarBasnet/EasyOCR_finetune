- Fine-tuning ['EasyOCR'](https://github.com/JaidedAI/EasyOCR)

# 1. Usage

## Step 1: Environment Setting
1. Set configurations ('config_files/config.yaml')
    ```yaml
    ### Environment ###
    seed: # Seed
    experiment_name: # name of the folder that will be created

    ### Dataset ###
    train_data: # Training set directory
    val_data: # Validation set directory
    select_data: # Subdirectory
    batch_ratio: 
    # Modulate the data ratio in the batch.
    # For example, when `select_data` is `MJ-ST` and `batch_ratio` is `0.5-0.5`,
    # the 50% of the batch is filled with 'MJ' and the other 50% of the batch is filled with 'ST'.
    total_data_usage_ratio: # How much ratio of the data to use
    train_images: # Number of training images
    val_images: # Number of validation images
    eval_images: # Number of evaluation images
    ### Data processing ###
    img_height: # Height of input image
    img_width: # Width of input image
    PAD: # If `True` pad to input images to keep aspect ratio
    contrast_adjust: # Adjust contrast
    character:
    # Characters to be used for prediction
    sensitive: # Case sensitivity
    batch_max_length: # Maximum length of label
    data_filtering_off: 
    # If `False` filter images containing characters not in `character`
    # and whose label is longer than `batch_max_length`

    ### Training ###
    workers: # Same as `num_workers` from `torch.utils.data.DataLoader`
    batch_size: # Batch size
    n_iter: # Number of iterations
    val_period: # Period to run validation
    show_number: # How many validation result to show
    continue_from: 
    # Checkpoint from which to continue training
    strict: # If `False` ignore non-matching keys when loading a model from checkpoint
    ### Optimizer ###
    adam: # If `True` use `torch.optim.Adam`, if `False` use `torch.optim.Adadelta`
    lr: 
    rho: 
    eps: 
    grad_clip: 

    ### Model ###
    Transformation: # `None` or `TPS`
    FeatureExtraction: # `VGG`, `RCNN` or `ResNet`
    SequenceModeling: # `None` or `BiLSTM`
    Prediction: # `CTC` or `Attn`
    ### VGG ###
    freeze_FeatureFxtraction: # If `True` do not update feature extraction parameters
    rgb: False # `True` for RGB input image
    input_channel: # `1` for grayscale input image, `3` for RGB
    output_channel: # Output dimension of featrue extraction result
    ### BiLSTM ###
    freeze_SequenceModeling: # If `True` do not update sequence modeling parameters
    hidden_size: # `hidden_size` of `torch.nn.LSTM`
    ### Prediction ###
    new_prediction: False # If `True` dimension of model prediction changes according to checkpoint
    ### CTC ###
    decode: # `greedy` or `beamsearch`
    ```
2. Run `source step1_set_environment.sh`

## Step 2: Dataset Preparation
- Number of training images: 10,000
- Number of validation images: 1000

1. Run `bash step2_run_prepare_dataset_py.sh`
  ```sh
  # step2_run_prepare_dataset_py.sh
  python3 prepare_dataset.py\
    --dataset="/Dataset"
    # `--unzip`: Unzip into following directory structure
        # unzipped
        # ├── training
        # │   ├── images
        # │   │   └── ...
        # │   └── labels
        # │       └── ...
        # └── validation
        #     ├── images
        #     │   └── ...
        #     └── labels
        #         └── ...
    --unzip\ # Whether to unzip
    # An image patch will be created in the directory structure like shown below
    --training\ # Whether to generate training set
    --validation\ # Whether to generate validation set
    --evaluation # Whether to generate evaluation set
  ```

## Step 3: Fine-tunning or Training from Scratch
- Run `bash step3_run_train_py.sh`
  
