from torch.utils.data import DataLoader
from trainingModel.timesnet_diff_method.ECGLoader import ECGDataLoader
import sys
from trainingModel.timesnet_diff_method.source_codes.ecg import collate_fn
sys.path.insert(0, "")


data_dict = {'ECG': ECGDataLoader}

def collate_fn_wrapper(max_len):
    def wrapper(batch):
        return collate_fn(batch, max_len)
    return wrapper


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'TEST':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            train_data=args.train_data,
            test_data=args.test_data,
            ecg_folder=args.root_path,
            ecg_col_name=args.filename,
            labels_col_name=args.target,
            seq_len=args.seq_len,
            num_features=args.num_features,
            flag=flag

        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn_wrapper(args.seq_len)
        )
        return data_set, data_loader
