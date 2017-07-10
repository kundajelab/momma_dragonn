'''
Created on Jun 11, 2017

@author: evaprakash
'''
import sys
import json

import yaml
import os
from shutil import copyfile
import pkgutil
import StringIO
import copy
import execute_successive_halving

from momma_dragonn.model_creators.flexible_keras import FlexibleKerasSequential


class SuccessiveHalvingConfiguration(object):
    '''
    classdocs
    '''
    def __init__(self, starting_epochs, hyperparams_config_file="config/hyperparameter_configs_list.yaml",
                 successive_halving_folder="sh_files", valid_data_loader_config="config/valid_data_loader_config.yaml",
                 evaluator_config="config/evaluator_config.yaml"):
        '''
        Constructor
        '''
        self.hyperparams_config_file = hyperparams_config_file
        self.successive_halving_folder=successive_halving_folder
        self.starting_epochs=starting_epochs
        self.valid_data_loader_config=valid_data_loader_config
        self.evaluator_config=evaluator_config
    
    def make_model(self, n, folder_name, hyperparams_list):
        seed=hyperparams_list["model_trainer"]["kwargs"]["seed"]
        model_class_string=str(hyperparams_list["model_creators"][n]["model_creator"]["class"].split(".")[1])
        model_class=getattr(sys.modules[__name__], model_class_string)
        model=model_class(**(hyperparams_list["model_creators"][n]["model_creator"]["kwargs"]))
        model_wrapper=model.get_model_wrapper(seed)
        self.save_weights(model_wrapper, folder_name)
        
    def save_weights(self, model_wrapper, folder_name):
        model = model_wrapper.get_model()
        weights_file = folder_name + "/initial_weights.h5"
        model.save_weights(weights_file, overwrite=True)
    
    def save_models_info(self,model_trainer,model_creators,data_loaders,successive_halving_info):
        models_info_file=open(self.successive_halving_folder + "/models_info.json", "w")
        for n in range(0, len(model_creators)):
            model_creators[n].update({"folder":"model"+str(n)})
        models_info={"model_trainer":model_trainer,
                     "model_creators":model_creators,
                     "data_loaders":data_loaders,
                     "successive_halving_info":successive_halving_info}
        json.dump(models_info,models_info_file,indent=4)
        models_info_file.close()

    def validate_hyperparameters(self, hyperparams_list):
        if not len(hyperparams_list)==3:
            raise ValueError("Unexpected number of keys in "+self.hyperparams_config_file)
        if not "model_trainer" in hyperparams_list.keys():
            raise ValueError("Could not identify model_trainer")
        if not "model_creators" in hyperparams_list.keys():
            raise ValueError("Could not identify model_creator")
        if not "other_data_loaders" in hyperparams_list.keys():
            raise ValueError("Could not identify other_data_loaders")

    def make_individual_hyperparams(self, hyperparams_list):
        model_trainer=hyperparams_list["model_trainer"]
        model_trainer["kwargs"].update({"initial_performance_history_file":""})
        model_trainer["kwargs"].update({"base_epoch":0})
        data_loader=hyperparams_list["other_data_loaders"]
        for n in range(0, len(hyperparams_list["model_creators"])):
            model_folder = "model" + str(n)
            model_creator=hyperparams_list["model_creators"][n]["model_creator"]
            model_creator["kwargs"].update({"initial_weights_file": model_folder+"/initial_weights.h5"})
            file=open(self.successive_halving_folder + "/"+model_folder+"/hyperparameter_configs_list.yaml", "w")
            model_trainer["kwargs"]["initial_performance_history_file"]=model_folder+"/initial_performance_history.json"
            content=[{"model_trainer":model_trainer,"model_creator":model_creator,"other_data_loaders":data_loader}]
            json.dump(content, file, indent=4)
            file.close()

    def copy_vdl_and_ec(self):
        vd_file=open(self.valid_data_loader_config,"r")
        vd_file_content=yaml.load(vd_file.read())
        vd_file.close()
        vd_file_content["kwargs"]["path_to_hdf5"]="valid_data.hdf5"
        vd_file = open(self.successive_halving_folder+"/valid_data_loader_config.yaml", "w")
        json.dump(vd_file_content,vd_file)
        vd_file.close()
        copyfile(self.evaluator_config, self.successive_halving_folder+"/evaluator_config.yaml")

    def substitute_folder_name_in_config(self, config_object, folder_name):
        new_config_object = copy.deepcopy(config_object)
        for n in range(0,len(new_config_object)):
            if "model_folder" in new_config_object[n]["kwargs"].keys():
                new_config_object[n]["kwargs"]["model_folder"]=folder_name
        return new_config_object

    def copy_resources(self, number_models):
        epoch_callbacks_config_content=pkgutil.get_data("momma_dragonn.successive_halving", "resources/end_of_epoch_callbacks_config.yaml")
        training_callbacks_config_content = pkgutil.get_data("momma_dragonn.successive_halving", "resources/end_of_training_callbacks_config.yaml")
        epoch_callbacks_config = yaml.load(StringIO.StringIO(epoch_callbacks_config_content))
        training_callbacks_config = yaml.load(StringIO.StringIO(training_callbacks_config_content))
        for n in range(0,number_models):
            model_folder="model"+str(n)
            epochcb_config = self.substitute_folder_name_in_config(epoch_callbacks_config, model_folder)
            trainingcb_config = self.substitute_folder_name_in_config(training_callbacks_config, model_folder)
            epoch_file=open(self.successive_halving_folder + "/"+ model_folder+"/end_of_epoch_callbacks_config.yaml","w")
            training_file=open(self.successive_halving_folder+"/"+ model_folder+"/end_of_training_callbacks_config.yaml","w")
            json.dump(epochcb_config, epoch_file, indent=4)
            json.dump(trainingcb_config, training_file, indent=4)
            epoch_file.close()
            training_file.close()

    def create_data(self):
        command_line = [ "make_hdf5 --yaml_configs make_hdf5_yaml/* --output_dir " + str(self.successive_halving_folder)]
        print "About to run command:\n" + str(command_line)
        execute_successive_halving.execute_command(command_line, use_shell=True)
        
    def initialize_successive_halving(self):
        if os.path.exists(self.successive_halving_folder):
            raise ValueError(self.successive_halving_folder + " already exists")
        else:
            os.mkdir(self.successive_halving_folder)
        # Load user specified hyperparams
        yaml_file=open(self.hyperparams_config_file,"r")
        yaml_data=yaml_file.read()
        hyperparams_list=yaml.load(yaml_data)
        yaml_file.close()
        self.validate_hyperparameters(hyperparams_list)
        #Create model folders
        number_models = len(hyperparams_list["model_creators"])
        for n in range(0, number_models):
            folder_name = self.successive_halving_folder + "/model" + str(n)
            os.mkdir(folder_name)
            self.make_model(n, folder_name, hyperparams_list)
        model_trainer=hyperparams_list["model_trainer"]
        model_creators=hyperparams_list["model_creators"]
        data_loaders=hyperparams_list["other_data_loaders"]
        successive_halving_info={"starting_epochs":self.starting_epochs}
        self.save_models_info(model_trainer,
                              model_creators,
                              data_loaders,
                              successive_halving_info)
        self.make_individual_hyperparams(hyperparams_list)
        self.copy_vdl_and_ec()
        self.copy_resources(number_models)
        self.create_data()
