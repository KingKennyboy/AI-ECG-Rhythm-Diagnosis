import sys
sys.path.insert(0, "")
import torch
from trainingModel.timesnet_diff_method.source_codes.Exp_Classification import Exp_Classification


class Args:
    def __init__(self, args_dict):
        self.task_name = args_dict['task_name']
        self.is_training = args_dict['is_training']
        self.model_id = args_dict['model_id']
        self.model = args_dict['model']
        self.data = args_dict['data']
        self.root_path = args_dict['root_path']
        self.features = args_dict['features']
        self.target = args_dict['target']
        self.freq = args_dict['freq']
        self.checkpoints = args_dict['checkpoints']
        self.seq_len = args_dict['seq_len']
        self.pred_len = args_dict['pred_len']
        self.enc_in = args_dict['enc_in']
        self.num_class = args_dict['num_class']
        self.use_multi_gpu = args_dict['use_multi_gpu']
        self.use_gpu = args_dict['use_gpu']
        self.learning_rate = args_dict['learning_rate']
        self.patience = args_dict['patience']
        self.train_epochs = args_dict['train.epochs']
        self.label_len = args_dict['label_len']
        self.num_workers = args_dict['num_workers']
        self.batch_size = args_dict['batch_size']
        self.d_model = args_dict['d_model']
        self.embed = args_dict['embed']
        self.dropout = args_dict['dropout']
        self.enc_out = args_dict['enc_out']
        self.num_kernels = args_dict['num_kernels']
        self.itr = args_dict['itr']
        self.filename = args_dict['file_col']
        self.e_layers = args_dict['e_layers']
        self.top_k = args_dict['top_k']
        self.d_ff = args_dict['d_ff']
        self.num_features = args_dict['num_features']
        self.train_data = args_dict['train_data']
        self.test_data = args_dict['test_data']
        self.gpu = args_dict['gpu']


args_obj = {
    'task_name': 'classification',
    'is_training': 1,
    'model_id': 'test',
    'model': 'TimesNet',
    'data': 'ECG',
    'root_path': 'C:/Users/osino/PycharmProjects/tacobell_coding_inc/trainingModel/ECGDataProcessed/',
    'train_data': 'C:/Users/osino/PycharmProjects/tacobell_coding_inc/trainingModel/train_dataset.csv',
    'test_data': 'C:/Users/osino/PycharmProjects/tacobell_coding_inc/trainingModel/test_dataset.csv',
    'features': '',
    'target': 'Rhythm',
    'freq': 's',
    'checkpoints': 'C:/Users/osino/PycharmProjects/tacobell_coding_inc/trainingModel/timesnet_diff_method/checkpoints1',
    'seq_len': 5000,
    'pred_len': 10,
    'enc_in': 12,
    'num_class': 5,
    'use_multi_gpu': False,
    'use_gpu': False,
    'learning_rate': 0.0001,
    'patience': 3,
    'train.epochs': 10,
    'label_len': 10646,
    'num_workers': 10,
    'batch_size': 8,
    'd_model': 512,
    'embed': 'Fixed',
    'dropout': 0.3,
    'enc_out': 12,
    'num_kernels': 6,
    'itr': 1,
    'file_col': 'FileName',
    'e_layers': 2,
    'top_k': 5,
    'd_ff': 2048,
    'num_features': 12,
    'gpu': 0

}

args = Args(args_obj)
if __name__ == '__main__':
    Exp = Exp_Classification
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_eb{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.embed, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_eb{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.embed, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


