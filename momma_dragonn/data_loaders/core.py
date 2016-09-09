from collections import OrderedDict


class AbstractDataLoader(object):

    def get_jsonable_object(self):
        return OrderedDict()

    def get_callbacks(self):
        return [] #TODO: implement callback functionality in a trainer
        

class AbstractBatchDataLoader(AbstractDataLoader):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_jsonable_object(self):
        the_dict = super(AbstractBatchDataLoader, self).get_jsonable_object()
        the_dict['batch_size'] = self.batch_size
        return the_dict

    def get_batch_generator(self):
        #produces the generator for batches 
        raise NotImplementedError()

    def get_data_for_eval(self):
        #produce a hunk of data to run evaluation on
        raise NotImplementedError()

class AbstractAtOnceDataLoader(AbstractDataLoader):

    def get_data(self):
        raise NotImplementedError()
