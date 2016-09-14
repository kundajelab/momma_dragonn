#!/usr/bin/env bash

momma_dragonn_train --valid_data_loader_config\
 config/valid_data_loader_config.yaml\
 --evaluator_config config/evaluator_config.yaml\
 --end_of_epoch_callbacks_config config/end_of_epoch_callbacks_config.yaml\
 --end_of_training_callbacks_config config/end_of_training_callbacks_config.yaml\
 --hyperparameter_configs_list config/hyperparameter_configs_list.yaml
