from .core import AbstractModelTrainer
import momma_dragonn
from momma_dragonn.performance_history import PerformanceHistory
from collections import OrderedDict
import avutils.util as util

class KerasFitGeneratorModelTrainer(AbstractModelTrainer)

    def __init__(self, samples_per_epoch,
                       stopping_criterion_config,
                       class_weight=None):
        self.samples_per_epoch = samples_per_epoch 
        self.stopping_criterion_config = stopping_criterion_config
        self.class_weight = class_weight

    def get_jsonable_object(self):
        return OrderedDict([
                ('samples_per_epoch', self.samples_per_epoch),
                ('stopping_criterion_config', self.stopping_criterion_config),
                ('class_weight', self.class_weight])
 
    def train(self, model_wrapper, model_evaluator,
                    valid_data_loader, other_data_loaders,
                    end_of_epoch_callbacks):

        is_larger_better = model_evaluator.is_larger_better_for_key_metric()

        self.stopping_criterion =\
            momma_dragonn.loaders.load_stopping_criterion(
                config=self.stopping_criterion_config,
                extra_kwargs={'larger_is_better':is_larger_better})

        train_data_loader = other_data_loaders['train']
        valid_data = valid_data_loader.get_data()

        other_model_training_configs = self.model.get_other_training_configs()
        batch_size = other_model_training_configs['batch_size']
        class_weights = other_model_training_configs['class_weights']

        performance_history = PerformanceHistory()

        training_metadata = OrderedDict() 
        epoch = 0
        best_valid_perf_finder = util.init_get_best(is_larger_better)
        try:
            while (not stopping_criterion.stop_training()):
                model_wrapper.get_model().fit_generator(
                    train_data_loader.get_batch_generator(),
                    samples_per_epoch=self.samples_per_epoch,
                    nb_epoch=1,
                    class_weight=self.class_weight)
                train_data_for_eval = train_data_loader.get_data_for_eval()

                train_key_metric = model_evaluator.compute_key_metric(
                                    model_wrapper=model_wrapper,
                                    data=train_data_for_eval)
                valid_key_metric = model_evaluator.compute_key_metric(
                                    model_wrapper=model_wrapper,
                                    data=valid_data)
                new_best = best_valid_perf_finder.process(epoch,
                                                          valid_key_metric)

                stopping_criterion.update(valid_key_metric)
                performance_history.epoch_update(
                    train_key_metric=train_key_metric,
                    valid_key_metric=valid_key_metric)
                if (new_best):
                    valid_all_stats = model_evaluator.compute_all_stats(
                                        model_wrapper=model_wrapper,
                                        data=valid_data)
                    performance_history.update_best_valid_epoch_perf_info(
                        epoch=epoch, valid_key_metric=valid_key_metric,
                        train_key_metric=train_key_metric,
                        valid_all_stats=valid_all_stats)
                epoch += 1

                for end_of_epoch_callback in end_of_epoch_callbacks:
                    end_of_epoch_callback(#handles intermediate model saving
                        epoch=epoch,
                        model_wrapper=model_wrapper,
                        valid_key_metric=valid_key_metric,
                        train_key_metric=train_key_metric,
                        valid_all_stats=valid_all_stats,
                        is_new_best_valid_perf=new_best,
                        performance_history=performance_history)
            training_metadata['terminated_by_interrupt']=False
        except (KeyboardInterrupt):
            print("\nTraining was interrupted at epoch ",
                     epoch,"with a keyboard interrupt")
            training_metadata['terminated_by_interrupt']=True

        return model_wrapper, performance_history, training_metadata
        
