from .core import AbstractModelTrainer
import momma_dragonn
from momma_dragonn.performance_history import PerformanceHistory
from collections import OrderedDict
import avutils.util as util
import traceback

class KerasFitGeneratorModelTrainer(AbstractModelTrainer):

    def __init__(self, samples_per_epoch,
                       stopping_criterion_config,
                       class_weight=None,
                       seed=1234,
                       csv_logger=None,
                       report_all_valid_metrics_every_epoch=False,
                       reparameterizer=None):
        self.seed = seed 
        self.samples_per_epoch = samples_per_epoch 
        self.stopping_criterion_config = stopping_criterion_config
        self.report_all_valid_metrics_every_epoch =\
            report_all_valid_metrics_every_epoch
        self.reparameterizer = reparameterizer
        if (class_weight is not None):
            self.class_weight = dict((int(key),val) for
                                      key,val in class_weight.items())
        else:
            self.class_weight = None
        self.csv_logger=csv_logger
        

    def get_jsonable_object(self):
        return OrderedDict([
                ('seed', self.seed),
                ('samples_per_epoch', self.samples_per_epoch),
                ('stopping_criterion_config', self.stopping_criterion_config),
                ('class_weight', self.class_weight),
                ('report_all_valid_metrics_every_epoch',
                  self.report_all_valid_metrics_every_epoch),
                ('reparameterizer', 
                  None if self.reparameterizer is None
                       else self.reparameterizer.get_jsonable_object())])
 
    def train(self, model_creator, model_evaluator,
                    valid_data_loader, other_data_loaders,
                    end_of_epoch_callbacks, start_of_epoch_callbacks,
                    error_callbacks):
        print("Setting seed "+str(self.seed))
        import numpy as np
        np.random.seed(self.seed)

        print("Importing keras...")
        import keras

        print("Getting model...")
        model_wrapper = model_creator.get_model_wrapper(seed=self.seed)
        print("Got model")

        is_larger_better = model_evaluator.is_larger_better_for_key_metric()

        stopping_criterion =\
            momma_dragonn.loaders.load_stopping_criterion(
                config=self.stopping_criterion_config,
                extra_kwargs={'larger_is_better':is_larger_better})
        report_all_valid_metrics_every_epoch =\
            self.report_all_valid_metrics_every_epoch

        train_data_loader = other_data_loaders['train']
        print("Loading validation data into memory")
        valid_data = valid_data_loader.get_data()
        #TODO: deal with weights
        print("Loaded")

        performance_history = PerformanceHistory()

        training_metadata = OrderedDict() 
        best_valid_perf_finder = util.init_get_best(is_larger_better)

        #additional user-supplied callbacks
        extra_callbacks=[]
        if self.csv_logger!=None:
            from keras.callbacks import CSVLogger
            csv_logger=CSVLogger(self.csv_logger,append=True)
            extra_callbacks.append(csv_logger) 
        try:

            epochs_trained = 0
            batch_generator = train_data_loader.get_batch_generator()

            while stopping_criterion.stop_training()==False:

                for start_of_epoch_callback in start_of_epoch_callbacks:
                    start_of_epoch_callback( #written for snapshot saving
                        epoch=epochs_trained,
                        model_wrapper=model_wrapper)

                model_wrapper.get_model().fit_generator(
                    batch_generator,
                    samples_per_epoch=self.samples_per_epoch,
                    nb_epoch=1,
                    class_weight=self.class_weight,
                    callbacks=[]+extra_callbacks)

                #increment the epoch and do stuff at the end of the epoch
                epochs_trained += 1

                if (self.reparameterizer):
                    if (self.reparameterizer.do_reparameterization(
                                              epoch=epochs_trained)):
                        print("REPARAMETERIZING!!!")
                        model_wrapper.set_model(
                            self.reparameterizer(model_wrapper.get_model()))

                train_data_for_eval = train_data_loader.get_data_for_eval()
                #preds = model_wrapper.get_model().predict(train_data_for_eval.X)
                #worst_predicted_positives = sorted([x for x,y in zip(enumerate(preds),train_data_for_eval.Y[:,0]) if y==1], key=lambda x: x[1])[:10]
                #print(worst_predicted_positives)
                #for idx,pred in worst_predicted_positives:
                #    print(idx,pred)
                #    print(train_data_for_eval.Y[idx])
                #    print(train_data_for_eval.coors[idx])
                #    print(train_data_for_eval.fastastr[idx])
                #    print(train_data_for_eval.X[idx])
                train_key_metric = model_evaluator.compute_key_metric(
                                    model_wrapper=model_wrapper,
                                    data=train_data_for_eval,
                                    batch_size=train_data_loader.batch_size)
                valid_key_metric = model_evaluator.compute_key_metric(
                                    model_wrapper=model_wrapper,
                                    data=valid_data,
                                    batch_size=train_data_loader.batch_size)
                new_best = best_valid_perf_finder.process(epochs_trained,
                                                          valid_key_metric)
                stopping_criterion.update(valid_key_metric)
                if (report_all_valid_metrics_every_epoch):
                    this_epoch_valid_all_stats =\
                                model_evaluator.compute_all_stats(
                                    model_wrapper=model_wrapper,
                                    data=valid_data,
                                    batch_size=train_data_loader.batch_size)
                else:
                    this_epoch_valid_all_stats = {}
                performance_history.epoch_update(
                    train_key_metric=train_key_metric,
                    valid_key_metric=valid_key_metric,
                    valid_all_stats = this_epoch_valid_all_stats)
                if (new_best):
                    print("New best")
                    if (not report_all_valid_metrics_every_epoch):
                        self.valid_all_stats =\
                                model_evaluator.compute_all_stats(
                                    model_wrapper=model_wrapper,
                                    data=valid_data,
                                    batch_size=train_data_loader.batch_size)
                    else:
                        self.valid_all_stats = this_epoch_valid_all_stats
                    print("Valid all stats:\n",self.valid_all_stats)
                    performance_history.update_best_valid_epoch_perf_info(
                        epoch=epochs_trained,
                        valid_key_metric=valid_key_metric,
                        train_key_metric=train_key_metric,
                        valid_all_stats=self.valid_all_stats)
                for end_of_epoch_callback in end_of_epoch_callbacks:
                    end_of_epoch_callback(#handles intermediate model saving
                        epoch=epochs_trained,
                        model_wrapper=model_wrapper,
                        valid_key_metric=valid_key_metric,
                        train_key_metric=train_key_metric,
                        valid_all_stats=self.valid_all_stats,
                        is_new_best_valid_perf=new_best,
                        performance_history=performance_history)
                
            training_metadata['termination_condition'] = "normal"
        except (KeyboardInterrupt):
            print("\nTraining was interrupted after epoch ",
                     epochs_trained,"with a keyboard interrupt")
            training_metadata['termination_condition']= "KeyboardInterrupt"
        except Exception as e:
            traceback_str = str(traceback.format_exc())
            print(traceback_str) 
            print("\nTraining was interrupted after epoch ",
                     epochs_trained,"with an exception")
            for error_callback in error_callbacks:
                error_callback(exception=e,
                               traceback_str=traceback_str,
                               epoch=epochs_trained)
            training_metadata['termination_condition']= str(type(e)) 
        finally:
            training_metadata['total_epochs_trained_for']=epochs_trained

        return model_wrapper, performance_history, training_metadata
        
