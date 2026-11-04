import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import argparse
from src.folderconstants import *
from shutil import copyfile

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'CICIDS2017']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape


def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i) - 1 for i in values]
        temp[start - 1:end - 1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)


def wgn(a, snr):
    min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    x = (a - min_a) / (max_a - min_a + 0.0001)
    batch_size, len_x = x.shape
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return noise/100


def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return a / 2 + 0.5


def normalize2(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a) + wgn(a, 50), min_a, max_a


def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return ((a - min_a) / (max_a - min_a + 0.0001)) + wgn(a, 50), min_a, max_a


def load_cicids2017(train_ratio=0.5):
    """Prepare CIC-IDS 2017 as an unsupervised anomaly-detection dataset.

    The first chronological portion supplies benign training flows only. The
    remaining flows are used for testing, with every non-BENIGN flow labelled
    as suspicious.
    """
    input_folders = [
        os.path.join(data_folder, 'CIC-IDS2017'),
        os.path.join(data_folder, 'CICIDS2017'),
    ]
    csv_files = []
    for input_folder in input_folders:
        if os.path.isdir(input_folder):
            for root, _, filenames in os.walk(input_folder):
                csv_files.extend(
                    os.path.join(root, filename)
                    for filename in filenames
                    if filename.lower().endswith('.csv')
                )
            csv_files.sort()
        if csv_files:
            break
    if not csv_files:
        raise FileNotFoundError(
            'CIC-IDS 2017 CSV files not found. Put the dataset under '
            'data/CIC-IDS2017/ or data/CICIDS2017/.'
        )

    frames = []
    for filename in csv_files:
        frame = pd.read_csv(filename, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        label_columns = [column for column in frame.columns if column.lower() == 'label']
        if not label_columns:
            raise ValueError(f'CIC-IDS 2017 file has no Label column: {filename}')
        frame = frame.rename(columns={label_columns[0]: 'Label'})
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True, sort=False)
    if len(data) < 2:
        raise ValueError('CIC-IDS 2017 requires at least two flow rows.')
    labels = data['Label'].fillna('').astype(str).str.strip().str.upper()
    benign = labels.eq('BENIGN')
    if not benign.any():
        raise ValueError('CIC-IDS 2017 data contains no BENIGN flows for training.')

    split = min(max(int(len(data) * train_ratio), 1), len(data) - 1)
    train_mask = benign & (np.arange(len(data)) < split)
    if not train_mask.any():
        benign_indices = np.flatnonzero(benign.to_numpy())
        train_mask.iloc[benign_indices[:max(1, len(benign_indices) // 2)]] = True
        split = int(np.flatnonzero(train_mask.to_numpy())[-1]) + 1
    test_mask = np.arange(len(data)) >= split

    feature_frame = data.drop(columns=['Label'])
    numeric = feature_frame.apply(pd.to_numeric, errors='coerce')
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how='all')
    if numeric.shape[1] == 0:
        raise ValueError('CIC-IDS 2017 contains no usable numeric feature columns.')

    train_frame = numeric.loc[train_mask]
    test_frame = numeric.loc[test_mask]
    medians = train_frame.median().fillna(0)
    train_frame = train_frame.fillna(medians).fillna(0)
    test_frame = test_frame.fillna(medians).fillna(0)
    min_values = train_frame.min()
    ranges = (train_frame.max() - min_values).replace(0, 1)
    train = ((train_frame - min_values) / ranges).clip(0, 1).to_numpy(dtype=np.float64)
    test = ((test_frame - min_values) / ranges).clip(0, 1).to_numpy(dtype=np.float64)
    test_labels = (~labels.loc[test_mask].eq('BENIGN')).to_numpy(dtype=np.float64)
    test_labels = np.repeat(test_labels[:, None], train.shape[1], axis=1)

    folder = os.path.join(output_folder, 'CICIDS2017')
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, 'train.npy'), train)
    np.save(os.path.join(folder, 'test.npy'), test)
    np.save(os.path.join(folder, 'labels.npy'), test_labels)
    print(f'CICIDS2017 train={train.shape}, test={test.shape}, suspicious={int(test_labels[:, 0].sum())}')


def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)


def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == 'synthetic':
        train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
        test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
        dat = pd.read_csv(train_file, header=None)
        split = 10000
        train = normalize(dat.values[:, :split].reshape(split, -1))
        test = normalize(dat.values[:, split:].reshape(split, -1))
        lab = pd.read_csv(test_labels, header=None)
        lab[0] -= split
        labels = np.zeros(test.shape)
        for i in range(lab.shape[0]):
            point = lab.values[i][0]
            labels[point - 30:point + 30, lab.values[i][1:]] = 1
        test += labels * np.random.normal(0.75, 0.1, test.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'SMD':
        dataset_folder = 'data/SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
    elif dataset == 'UCR':
        dataset_folder = 'data/UCR'
        file_list = os.listdir(dataset_folder)
        for filename in file_list:
            if not filename.endswith('.txt'): continue
            vals = filename.split('.')[0].split('_')
            dnum, vals = int(vals[0]), vals[-3:]
            vals = [int(i) for i in vals]
            temp = np.genfromtxt(os.path.join(dataset_folder, filename),
                                 dtype=np.float64,
                                 delimiter=',')
            min_temp, max_temp = np.min(temp), np.max(temp)
            temp = (temp - min_temp) / (max_temp - min_temp)
            train, test = temp[:vals[0]], temp[vals[0]:]
            labels = np.zeros_like(test)
            labels[vals[1] - vals[0]:vals[2] - vals[0]] = 1
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
    elif dataset == 'NAB':
        dataset_folder = 'data/NAB'
        file_list = os.listdir(dataset_folder)
        with open(dataset_folder + '/labels.json') as f:
            labeldict = json.load(f)
        for filename in file_list:
            if not filename.endswith('.csv'): continue
            df = pd.read_csv(dataset_folder + '/' + filename)
            vals = df.values[:, 1]
            labels = np.zeros_like(vals, dtype=np.float64)
            for timestamp in labeldict['realKnownCause/' + filename]:
                tstamp = timestamp.replace('.000000', '')
                index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
                labels[index - 4:index + 4] = 1
            min_temp, max_temp = np.min(vals), np.max(vals)
            vals = (vals - min_temp) / (max_temp - min_temp)
            train, test = vals.astype(float), vals.astype(float)
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            fn = filename.replace('.csv', '')
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
    elif dataset == 'MSDS':
        dataset_folder = 'data/MSDS'
        df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        df_test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
        df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
        _, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
        train, _, _ = normalize3(df_train, min_a, max_a)
        test, _, _ = normalize3(df_test, min_a, max_a)
        labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
        labels = labels.values[::1, 1:]
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
    elif dataset == 'SWaT':
        dataset_folder = 'data/SWaT'
        file = os.path.join(dataset_folder, 'series.json')
        df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        df_test = pd.read_json(file, lines=True)[['val']][7000:12000]
        train, min_a, max_a = normalize2(df_train.values)
        test, _, _ = normalize2(df_test.values, min_a, max_a)
        labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset in ['SMAP', 'MSL']:
        dataset_folder = 'data/SMAP_MSL'
        file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset]
        filenames = values['chan_id'].values.tolist()
        for fn in filenames:
            train = np.load(f'{dataset_folder}/train/{fn}.npy')
            test = np.load(f'{dataset_folder}/test/{fn}.npy')
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test, min_a, max_a)
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            labels = np.zeros(test.shape)
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1], :] = 1
            np.save(f'{folder}/{fn}_labels.npy', labels)
    elif dataset == 'WADI':
        dataset_folder = 'data/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        train.dropna(how='all', inplace=True);
        test.dropna(how='all', inplace=True)
        train.fillna(0, inplace=True);
        test.fillna(0, inplace=True)
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep=True)
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']:
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i:
                        matched.append(i);
                        break
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
        train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
        print(train.shape, test.shape, labels.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'MBA':
        dataset_folder = 'data/MBA'
        ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
        train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
        test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
        train, test = train.values[1:, 1:].astype(float), test.values[1:, 1:].astype(float)
        train, min_a, max_a = normalize3(train)
        test, _, _ = normalize3(test, min_a, max_a)
        ls = ls.values[:, 1].astype(int)
        labels = np.zeros_like(test)
        for i in range(-20, 20):
            labels[ls + i, :] = 1
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'CICIDS2017':
        load_cicids2017()
    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description='Preprocess DTAAD datasets')
    cli.add_argument('datasets', nargs='*', choices=datasets)
    cli.add_argument('--train-ratio', type=float, default=0.5,
                     help='chronological fraction used to find benign training flows for CICIDS2017')
    args = cli.parse_args()
    if not 0 < args.train_ratio < 1:
        cli.error('--train-ratio must be between 0 and 1')
    if args.datasets:
        for d in args.datasets:
            if d == 'CICIDS2017':
                load_cicids2017(args.train_ratio)
            else:
                load_data(d)
    else:
        print("Usage: python preprocess.py <datasets> [--train-ratio FRACTION]")
        print(f"where <datasets> is space separated list of {datasets}")
