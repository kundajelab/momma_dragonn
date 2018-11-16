from avutils import util
from avutils.dynamic_enum import Keys, Key
from avutils import file_processing as fp
from collections import OrderedDict
import yaml
import json

class AbstractEndOfTrainingCallback(object):

    def __call__(self, **kwargs):
        raise NotImplementedError()


class EmailCallback(AbstractEndOfTrainingCallback):

    def __init__(self, to_emails, from_email, smtp_server, **kwargs):
        self.to_emails = to_emails
        self.from_email = from_email
        self.smtp_server = smtp_server #eg: smtp.stanford.edu

    def __call__(self, performance_history, model_wrapper, training_metadata,
                       message, model_creator_info, model_trainer_info,
                       other_data_loaders_info, **kwargs):
        if (training_metadata['total_epochs_trained_for'] > 0):
            current_best_valid_perf_info =\
                performance_history.get_best_valid_epoch_perf_info() 
            current_best_valid_key_metric = current_best_valid_perf_info\
                                            .valid_key_metric
            subject = ("finished"+
                       (" "+str(message)+" " if message!="" else "")+
                       " with perf: "+str(current_best_valid_key_metric)) 

            contents = json.dumps(OrderedDict([
                ('training_metadata', training_metadata),
                ('last_saved_files_config',
                 model_wrapper.get_last_saved_files_config()),
                ('best_valid_perf_info',
                    current_best_valid_perf_info.get_jsonable_object()),
                ('model_creator_info', model_creator_info),
                ('model_trainer_info', model_trainer_info),
                ('other_data_loaders_info', other_data_loaders_info)
            ]), indent=4, separators=(',', ': '))
            contents+="\n"
            contents+="Performance History:\n"+\
                    "\n".join(["Epoch\tTrain\tValid"]+\
    ["\t".join(y) for y in zip(
     [str(x) for x in range(len(performance_history))],
     [str(x) for x in performance_history.get_train_key_metric_history()],
     [str(x) for x in performance_history.get_valid_key_metric_history()])])

            util.send_email(
                subject=subject,
                to_addresses=self.to_emails,
                sender=self.from_email,
                smtp_server=self.smtp_server,
                contents=contents)


class WriteToDbCallback(AbstractEndOfTrainingCallback):

    record_keys = Keys(Key('record_number'),
                       Key('message'),
                       Key('best_valid_key_metric'),
                       Key('best_valid_perf_info'),
                       Key('training_metadata'),
                       Key('other_data_loaders_info'),
                       Key('model_trainer_info'),
                       Key('model_creator_info'),
                       Key('key_metric_history'),
                       Key('all_valid_metrics_history'),
                       Key('saved_files_config'))

    def __init__(self, db_path, key_metric_name, larger_is_better,
                       new_save_dir=None, **kwargs):
        self.key_metric_name = key_metric_name 
        self.larger_is_better = larger_is_better
        self.new_save_dir = new_save_dir
        self.db_path = ( #put the perf metric in the db name
            util.get_file_name_parts(db_path) 
                .get_transformed_file_path(transformation=
                 lambda x: x+"_perf-metric-"+str(self.key_metric_name)))

    def __call__(self, performance_history, model_wrapper, training_metadata,
                       message, model_creator_info, model_trainer_info,
                       other_data_loaders_info, **kwargs):
        if (training_metadata['total_epochs_trained_for'] > 0):
            #acquire lock on db file
            db_lock = fp.FileLockAsDir(self.db_path)  

            #read the contents if file exists, otherwise init as you will
            if (util.file_exists(self.db_path)):
                db_contents = yaml.load(fp.get_file_handle(self.db_path)) 
            else:
                db_contents = OrderedDict([
                    ('metadata', OrderedDict([
                        ('total_records',0),
                        ('best_valid_key_metric', None),
                        ('best_saved_files_config', None)])),
                    ('records', [])]) 

            #partition into metadata and records
            metadata = db_contents['metadata']
            records = db_contents['records']

            #arrange the fields in the records in the right order
            new_records = []
            for record in records:
                new_record = OrderedDict()
                for key in self.record_keys.get_keys():
                    if key in record:
                        new_record[key] = record[key] 
                    else:
                        new_record[key] = None
                #put in any leftover keys that are not in our
                #current set of keys
                for key in record:
                    if key not in new_record:
                        new_record[key] = record[key]
                new_records.append(new_record)
            records = new_records

            new_record_num =  metadata['total_records']+1
            model_wrapper.prefix_to_last_saved_files(
                prefix="record_"+str(new_record_num),
                new_directory=self.new_save_dir) 

            #update the metadata
            metadata['total_records'] = new_record_num
            previous_best_valid_key_metric = metadata['best_valid_key_metric']
            current_best_valid_perf_info =\
                performance_history.get_best_valid_epoch_perf_info() 
            current_best_valid_key_metric = current_best_valid_perf_info\
                                            .valid_key_metric
            if ((previous_best_valid_key_metric is None) or 
                (((-1 if self.larger_is_better else 1)
                 *previous_best_valid_key_metric) >
                 ((-1 if self.larger_is_better else 1)
                  *current_best_valid_key_metric))):
                metadata['best_valid_key_metric'] = current_best_valid_key_metric
                metadata['best_saved_files_config'] =\
                    model_wrapper.get_last_saved_files_config()

            #create a new entry for the db
            entry = OrderedDict()
            entry[self.record_keys.k.record_number] = new_record_num
            entry[self.record_keys.k.message] = message
            entry[self.record_keys.k.best_valid_key_metric] =\
                current_best_valid_key_metric
            entry[self.record_keys.k.best_valid_perf_info] =\
                current_best_valid_perf_info.get_jsonable_object()
            entry[self.record_keys.k.key_metric_history] =\
                [('train','valid')]+\
                list(zip(performance_history.get_train_key_metric_history(),
                     performance_history.get_valid_key_metric_history()))
            entry[self.record_keys.k.all_valid_metrics_history] =\
                performance_history.get_all_metrics_valid_history() 
            entry[self.record_keys.k.saved_files_config] =\
                model_wrapper.get_last_saved_files_config()
            entry[self.record_keys.k.model_creator_info] = model_creator_info
            entry[self.record_keys.k.other_data_loaders_info] =\
                other_data_loaders_info
            entry[self.record_keys.k.model_trainer_info] = model_trainer_info
            entry[self.record_keys.k.training_metadata] = training_metadata

            #append a new entry to the records 
            records.append(entry)
            #sort the records by perf
            records = sorted(records, key=lambda x:
        ((-1 if self.larger_is_better else 1)*x['best_valid_key_metric']))

            #open BackupForWriteFileHandle, write the json, close
            file_handle = fp.BackupForWriteFileHandle(self.db_path)
            file_handle.write(util.format_as_json(
                              OrderedDict([('metadata', metadata),
                                           ('records', records)])))
            file_handle.close()

            #release the lock on the db file
            db_lock.release() 
