
class AbstractModelTrainer(object)

    def __init__(self, config):
        self.config = config

    def train(self, model, model_evaluator, data_loaders):
        #return PerformanceHistory, ModelWrapper
        raise NotImplementedError()
