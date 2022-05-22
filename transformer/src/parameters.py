params = {
    'config_filepath': None, 
    'output_dir': './output', 
    'data_dir': './data_dir/', 
    'load_model': './transformer/output/SeasonData_pretrained_2022-05-09_15-57-06_MoQ/checkpoints/model_best.pth', 
    'resume': False, 
    'change_output': False, 
    'save_all': False, 
    'experiment_name': 'SeasonData_pretrained', 
    'comment': 'pretraining through imputation', 
    'no_timestamp': False, 
    'records_file': 'Imputation_records.csv', 
    'console': False, 
    'print_interval': 1, 
    'gpu': '-1', 
    'n_proc': 1, 
    'num_workers': 0, 
    'seed': None, 
    'limit_size': None, 
    'test_only': 'testset', 
    'data_class': 'agaid', 
    'labels': None, 
    'test_from': './test_rows.txt', 
    'test_ratio': 0, 
    'val_ratio': 0, 
    'pattern': 'Merlot', 
    'val_pattern': None, 
    'test_pattern': None, 
    'normalization': 'standardization', 
    'norm_from': None, 
    'subsample_factor': None, 
    'task': 'imputation', 
    'masking_ratio': 0.2, 
    'mean_mask_length': 10.0, 
    'mask_mode': 'separate', 
    'mask_distribution': 'geometric', 
    'exclude_feats': None, 
    'mask_feats': [0, 1], 
    'start_hint': 0.0, 
    'end_hint': 0.0, 
    'harden': True, 
    'epochs': 500, 
    'val_interval': 2, 
    'optimizer': 'Adam', 
    'lr': 0.0009, 
    'lr_step': [1000000], 
    'lr_factor': [0.1], 
    'batch_size': 16, 
    'l2_reg': 0, 
    'global_reg': False, 
    'key_metric': 'loss', 
    'freeze': False, 
    'model': 'transformer', 
    'max_seq_len': 366, 
    'data_window_len': None, 
    'd_model': 128, 
    'dim_feedforward': 256, 
    'num_heads': 8, 
    'num_layers': 3, 
    'dropout': 0.1, 
    'pos_encoding': 'learnable', 
    'activation': 'relu', 
    'normalization_layer': 'BatchNorm'
}