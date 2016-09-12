

class AbstractModelTrainer(object):

    def get_jsonable_object(self):
        raise NotImplementedError()

    def train(self, model, model_evaluator, data_loaders):
        #return PerformanceHistory, ModelWrapper
        raise NotImplementedError()
