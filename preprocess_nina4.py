import gc
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from data.preprocessing import *


if __name__ == '__main__':
    # load arguments
    parser = ArgumentParser(description='NinaPro4 Processing')
    parser.add_argument('--path', type=str, help='The raw Nina5 path')
    parser.add_argument('--save', type=str, help='The save path')
    parser.add_argument('--rectify', action='store_true', default=False, help='Using signal rectifying')
    parser.add_argument('--butter', action='store_true', default=False, help='Using butterworth filter')
    parser.add_argument('--ulaw', action='store_true', default=False, help='Using u-law normalization')
    parser.add_argument('--minmax', action='store_true', default=False, help='Using min-max normalization')
    parser.add_argument('--ssize', type=int, default=50, help='step size')
    parser.add_argument('--wsize', type=int, default=520, help='window size')
    parser.add_argument('--first', action='store_true', default=False, help='Using first appearance')
    parser.add_argument('--rest', action='store_true', default=False, help='Using Rest label')
    parser.add_argument('--multiproc', action='store_true', default=False, help='Using multi-processing')
    args = parser.parse_args()

    # create paths
    folder = ''
    if args.rectify:
        folder += 'rectify_'
    if args.butter:
        folder += 'butter_'
    if args.ulaw:
        folder += 'ulaw_'
    if args.minmax:
        folder += 'minmax_'

    folder += f's{args.ssize}_'
    folder += f'w{args.wsize}_'

    if args.first:
        folder += 'first_'
    else:
        folder += 'major_'

    if args.rest:
        folder += 'rest'
    else:
        folder += 'no_rest'

    save_path = Path(args.save)
    save_path = save_path / folder
    save_path.mkdir(parents=True, exist_ok=True)

    # initiate preprocessing
    processor = Nina4Processor(path=args.path,
                               use_rectify=args.rectify,
                               use_butter=args.butter,
                               use_u_norm=args.ulaw,
                               use_minmax_norm=args.minmax,
                               ssize=args.ssize,
                               wsize=args.wsize,
                               use_first_appearance=args.first,
                               use_rest_label=args.rest)

    # process data
    print('### Processing data...')
    processor.process_data(args.multiproc)

    # split data
    print('### Splitting data...')
    data_train = processor.split_data(split='train')
    data_test = processor.split_data(split='test')
    data_val = processor.split_data(split='val')
    print('Done!')
    print(f"train emg: shape={data_train['emg'].shape}, train target: shape={data_train['lbl'].shape}")
    print(f"test emg: shape={data_test['emg'].shape}, test target: shape={data_test['lbl'].shape}")
    print(f"val emg: shape={data_val['emg'].shape}, val target: shape={data_val['lbl'].shape}")

    # save data
    print('### Saving processed data...')
    np.savez(str(save_path / 'train.npz'), **data_train)
    np.savez(str(save_path / 'test.npz'), **data_test)
    np.savez(str(save_path / 'val.npz'), **data_val)
    print('Done!')

    # compute the class weights
    print('### Computing class weights...')
    class_weights = compute_class_weights(data_train['lbl'])
    np.save(str(save_path / 'class_weights.npy'), class_weights)
    print('Done!')

    # release memory
    del data_train, data_test, data_val
    gc.collect()

    # load data
    print('### Load data...')
    data_train = np.load(str(save_path / 'train.npz'))
    data_test = np.load(str(save_path / 'test.npz'))
    data_val = np.load(str(save_path / 'val.npz'))

    print(f"train emg: shape={data_train['emg'].shape}, train target: shape={data_train['lbl'].shape}")
    print(f"test emg: shape={data_test['emg'].shape}, test target: shape={data_test['lbl'].shape}")
    print(f"val emg: shape={data_val['emg'].shape}, val target: shape={data_val['lbl'].shape}")
    print('Done!')
