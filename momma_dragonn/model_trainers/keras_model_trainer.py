from .core import AbstractModelTrainer
import momma_dragonn
from momma_dragonn.performance_history import PerformanceHistory
from collections import OrderedDict
import avutils.util as util
import traceback
import numpy as np

class KerasFitGeneratorModelTrainer(AbstractModelTrainer):

    def __init__(self, samples_per_epoch,
                       stopping_criterion_config,
                       class_weight=None,
                       seed=1234):
        np.random.seed(seed)

        self.samples_per_epoch = samples_per_epoch 
        self.stopping_criterion_config = stopping_criterion_config
        if (class_weight is not None):
            self.class_weight = dict((int(key),val) for
                                      key,val in class_weight.items())
        else:
            self.class_weight = None

    def get_jsonable_object(self):
        return OrderedDict([
                ('samples_per_epoch', self.samples_per_epoch),
                ('stopping_criterion_config', self.stopping_criterion_config),
                ('class_weight', self.class_weight)])
 
    def train(self, model_wrapper, model_evaluator,
                    valid_data_loader, other_data_loaders,
                    end_of_epoch_callbacks, error_callbacks):

        is_larger_better = model_evaluator.is_larger_better_for_key_metric()

        stopping_criterion =\
            momma_dragonn.loaders.load_stopping_criterion(
                config=self.stopping_criterion_config,
                extra_kwargs={'larger_is_better':is_larger_better})

        train_data_loader = other_data_loaders['train']
        print("Loading validation data into memory")
        valid_data = valid_data_loader.get_data()
        valid_data_dict = {}
        valid_data_dict.update(valid_data.X)
        valid_data_dict.update(valid_data.Y)
        #TODO: deal with weights
        print("Loaded")

        performance_history = PerformanceHistory()

        training_metadata = OrderedDict() 
        epoch = 0
        best_valid_perf_finder = util.init_get_best(is_larger_better)
        try:
            while (not stopping_criterion.stop_training()):
                model_wrapper.get_model().fit_generator(
                    train_data_loader.get_batch_generator(),
                    validation_data=valid_data_dict,
                    samples_per_epoch=self.samples_per_epoch,
                    nb_epoch=1,
                    class_weight=self.class_weight)

                train_data_for_eval = train_data_loader.get_data_for_eval()

                train_key_metric = model_evaluator.compute_key_metric(
                                    model_wrapper=model_wrapper,
                                    data=train_data_for_eval,
                                    batch_size=train_data_loader.batch_size)
                valid_key_metric = model_evaluator.compute_key_metric(
                                    model_wrapper=model_wrapper,
                                    data=valid_data,
                                    batch_size=train_data_loader.batch_size)
                epoch += 1

                new_best = best_valid_perf_finder.process(epoch,
                                                          valid_key_metric)

                stopping_criterion.update(valid_key_metric)
                performance_history.epoch_update(
                    train_key_metric=train_key_metric,
                    valid_key_metric=valid_key_metric)
                if (new_best):
                    valid_all_stats = model_evaluator.compute_all_stats(
                                    model_wrapper=model_wrapper,
                                    data=valid_data,
                                    batch_size=train_data_loader.batch_size)
                    performance_history.update_best_valid_epoch_perf_info(
                        epoch=epoch, valid_key_metric=valid_key_metric,
                        train_key_metric=train_key_metric,
                        valid_all_stats=valid_all_stats)

                for end_of_epoch_callback in end_of_epoch_callbacks:
                    end_of_epoch_callback(#handles intermediate model saving
                        epoch=epoch,
                        model_wrapper=model_wrapper,
                        valid_key_metric=valid_key_metric,
                        train_key_metric=train_key_metric,
                        valid_all_stats=valid_all_stats,
                        is_new_best_valid_perf=new_best,
                        performance_history=performance_history)

            training_metadata['termination_condition'] = "normal"
        except (KeyboardInterrupt):
            print("\nTraining was interrupted at epoch ",
                     epoch,"with a keyboard interrupt")
            training_metadata['termination_condition']= "KeyboardInterrupt"
        except Exception as e:
            traceback_str = str(traceback.format_exc())
            print(traceback_str) 
            print("\nTraining was interrupted at epoch ",
                     epoch,"with an exception")
            for error_callback in error_callbacks:
                error_callback(exception=e,
                               traceback_str=traceback_str,
                               epoch=epoch)
            training_metadata['termination_condition']= str(type(e)) 
        finally:
            training_metadata['total_epochs_trained_for']=epoch

        return model_wrapper, performance_history, training_metadata
        
