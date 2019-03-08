#!/usr/bin/env bash

make_hdf5 --yaml_configs make_hdf5_yaml/* --output_dir .
momma_dragonn_train #will rely on default names for the config files
