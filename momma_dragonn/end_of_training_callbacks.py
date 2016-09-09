from avutils import util
from avutils import file_processing as fp
from collections import OrderedDict

class AbstractEndOfTrainingCallback(object):

    def __call__(self, **kwargs):
        raise NotImplementedError()


class WriteToDbCallback(AbstractEndOfTrainingCallback):

    def __init__(self, db_path, key_metric_name, **kwargs):
        self.key_metric_name = key_metric_name 
        self.db_path = ( #put the perf metric in the db name
            util.get_file_name_parts(db_path) 
                .get_transformed_file_path(transformation:
                 lambda x: x+"_perf-metric-"+str(self.key_metric_name)))

    def __call__(self, performance_history, model_wrapper, training_metadata,
                       model_creator_info, model_trainer_info,
                       other_data_loaders_info):
        #acquire lock on db file
        db_lock = fp.FileLockAsDir(self.db_path)  

        #read the contents if file exists, otherwise init as you will
        if (util.file_exists(self.db_path)):
            db_contents = yaml.load(fp.get_file_handle(config)) 
        else:
            db_contents = OrderedDict([
                ('metadata', OrderedDict([('total_models',0),
                                          ('best_valid_key_metric', None),
                                          ('best_saved_files_config', None)])),
                ('records', [])]) 

        #partition into metadata and records
        metadata = db_contents['metadata']
        records = db_contents['records']

        #update the metadata
        metadata['total_models'] += 1
        previous_best_valid_key_metric = metadata['best_valid_key_metric']
        current_best_valid_perf_info =\
            performance_history.get_best_valid_epoch_perf_info() 
        current_best_valid_key_metric = current_best_valid_perf_info\
                                        .valid_key_metric
        if (previous_best_valid_key_metric < current_best_valid_key_metric):
            metadata['best_valid_key_metric'] = current_best_valid_key_metric
            metadata['best_saved_files_config'] =\
                model_wrapper.get_last_saved_files_config()

        #create a new entry for the db
        entry = OrderedDict()
        entry['record_number'] = metadata['total_models']
        entry['best_valid_key_metric'] = current_best_valid_key_metric
        entry['best_valid_perf_info'] = current_best_valid_perf_info\
                                        .get_jsonable_object())
        entry['saved_files_config'] =\
            model_wrapper.get_last_saved_files_config()
        entry['model_creator_info'] = model_creator_info
        entry['other_data_loaders_info'] = other_data_loaders_info
        entry['model_trainer_info'] = model_trainer_info
        entry['training_metadata'] = training_metadata

        #append a new entry to the records 
        records.append(entry)

        #open BackupForWriteFileHandle, write the json, close
        file_handle = fp.BackupForWriteFileHandle(self.db_path)
        file_handle.write(util.format_as_json(
                          OrderedDict([('metadata', metadata),
                                       ('records', records)])))
        file_handle.close()

        #release the lock on the db file
        db_lock.release() 
