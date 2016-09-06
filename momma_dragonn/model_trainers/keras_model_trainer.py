from .core import AbstractModelTrainer
import momma_dragonn
from momma_dragonn.performance_history import PerformanceHistory
from collections import OrderedDict

class KerasFitGeneratorModelTrainer(AbstractModelTrainer)

    def __init__(self, samples_per_epoch,
                       stopping_criterion_config):
        self.samples_per_epoch = samples_per_epoch 
        self.stopping_criterion_config = stopping_criterion_config
 
    def train(self, model_wrapper, model_evaluator, data_loaders,
                    end_of_epoch_callbacks, end_of_training_callbacks):

        self.stopping_criterion = momma_dragonn.load_class_from_config(
            self.stopping_criterion_config)

        train_data_loader = data_loaders['train']
        valid_data = data_loaders['valid']

        other_model_training_configs = self.model.get_other_training_configs()
        batch_size = other_model_training_configs['batch_size']
        class_weights = other_model_training_configs['class_weights']

        performance_history = PerformanceHistory()

        training_metadata = OrderedDict() 
        epochs = 0
        try:
            while (not stopping_criterion.stop_training()):
                model_wrapper.get_model().fit_generator(
                    train_data_loader.get_batch_generator(),
                    samples_per_epoch=self.samples_per_epoch,
                    nb_epoch=1)
                key_metric = model_evaluator.compute_key_metric(
                                model_wrapper,
                                valid_data_loader) 

                stopping_criterion.update(key_metric)
                performance_history.update_training_key_metric(key_metric)
                performance_history.update_validation_key_metric(key_metric)
                epochs += 1
                for end_of_epoch_callback in end_of_epoch_callbacks:
                    end_of_epoch_callback( #handles intermediate model saving
                        model_wrapper=model_wrapper,
                        performance_history=performance_history,
                        is_new_best_valid_perf=#TODO)
            training_metadata['terminated_by_interrupt']=False
        except (KeyboardInterrupt):
            print("\nTraining was interrupted at epoch ",
                     epoch,"with a keyboard interrupt")
            training_metadata['terminated_by_interrupt']=True

        for end_of_training_callback in end_of_training_callbacks:
            end_of_training_callback( #handles writing to db
                performance_history=performance_history, #provides best train perf, best valid perf, total epochs, perfs at best valid epochs
                model_wrapper=model_wrapper,
                training_metadata=training_metadata)
        
