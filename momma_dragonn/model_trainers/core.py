

class AbstractModelTrainer(object):

    def get_jsonable_object(self):
        raise NotImplementedError()

    def train(self, model_creator, model_evaluator,
                    valid_data_loader, other_data_loaders,
                    end_of_epoch_callbacks, error_callbacks):
        #return ModelWrapper, PerformanceHistory, training_metadata
        raise NotImplementedError()
