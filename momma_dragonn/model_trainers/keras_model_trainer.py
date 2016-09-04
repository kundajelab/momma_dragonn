from .core import AbstractModelTrainer

class BasicKerasModelTrainer(AbstractModelTrainer)
 
    def train(self, model, model_evaluator, data_loaders):
        #!! model is an instance of modelWrapper which defines
        #a predict() function and a createFilesToSave(directory, prefix)
        #and a get_other_training_config() function

        stopping_criterion = self.get_stopping_criteron(self.config)

        train_data_loader = data_loaders['train']
        valid_data_loader = data_loaders['valid']
        test_data_loader = data_loaders['test']

        other_model_training_config = self.model.get_other_training_config()
        batch_size = other_model_training_config['batch_size']
        class_weights = other_model_training_config['class_weights']

        performance_history = PerformanceHistory()

        while (not stopping_criterion.stop_training):
            model.fit(train_data_loader.get_features(),
                      train_data_loader.get_labels(),
                      batch_size=batch_size,
                      class_weights=class_weights,
                      nb_epoch=1,
                      sample_weight=train_data_loader.get_weights())
            key_metrics = model_evaluator.compute_key_metric(model_wrapper,
                                                       valid_data_loader) 

            stopping_crierion.update(key_metrics, model_wrapper)
            performance_history.update(key_metrics, all_metrics)

        best_model_wrapper = stopping_criterion.get_best_model()
        best_all_metric = stopping_criterion.compute_all_metrics(
                            best_model_wrapper, valid_data_loader)
        performance_history.set_best_performance(best_all_metric)
        return best_model_wrapper, performance_history

        
