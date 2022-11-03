import gc
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from data.preprocessing import *


if __name__ == '__main__':
    # load arguments
    parser = ArgumentParser(description='NinaPro5 Processing')
    parser.add_argument('--path', type=str, help='The raw Nina5 path')
    parser.add_argument('--save', type=str, help='The save path')
    parser.add_argument('--ver', type=int, default=1, help='The processing version')
    parser.add_argument('--imu', action='store_true', default=False, help='Using IMU data')
    parser.add_argument('--rectify', action='store_true', default=False, help='Using signal rectifying')
    parser.add_argument('--butter', action='store_true', default=False, help='Using butterworth filter')
    parser.add_argument('--ssize', type=int, default=5, help='step size')
    parser.add_argument('--wsize', type=int, default=52, help='window size')
    parser.add_argument('--first', action='store_true', default=False, help='Using first appearance')
    parser.add_argument('--rest', action='store_true', default=False, help='Using Rest label')
    args = parser.parse_args()

    # create paths
    folder = f'ver{args.ver}_'
    if args.imu:
        folder += 'imu_'

    if args.rectify:
        folder += 'rectify_'

    if args.butter:
        folder += 'butter_'

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
    processor = Nina5Processor(path=args.path,
                               use_imu=args.imu,
                               use_rectify=args.rectify,
                               use_butter=args.butter,
                               ssize=args.ssize,
                               wsize=args.wsize,
                               use_first_appearance=args.first,
                               use_rest_label=args.rest)

    # process data
    print('### Processing data...')
    processor.process_data(args.ver)

    # split data
    print('### Splitting data...')
    data_train = processor.split_data(split='train')
    data_test = processor.split_data(split='test')
    data_val = processor.split_data(split='val')
    print('Done!')
    print(f"train emg: shape={data_train['emg'].shape}, train imu: shape={data_train['imu'].shape if args.imu else []}, train target: shape={data_train['lbl'].shape}")
    print(f"test emg: shape={data_test['emg'].shape}, test imu: shape={data_test['imu'].shape if args.imu else []}, test target: shape={data_test['lbl'].shape}")
    print(f"val emg: shape={data_val['emg'].shape}, val imu: shape={data_val['imu'].shape if args.imu else []}, val target: shape={data_val['lbl'].shape}")

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

    print(f"train emg: shape={data_train['emg'].shape}, train imu: shape={data_train['imu'].shape if args.imu else []}, train target: shape={data_train['lbl'].shape}")
    print(f"test emg: shape={data_test['emg'].shape}, test imu: shape={data_test['imu'].shape if args.imu else []}, test target: shape={data_test['lbl'].shape}")
    print(f"val emg: shape={data_val['emg'].shape}, val imu: shape={data_val['imu'].shape if args.imu else []}, val target: shape={data_val['lbl'].shape}")
    print('Done!')
