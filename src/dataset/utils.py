import torch
import numpy as np
import datetime


def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), s),
          flush=True)


def to_tensor(data, cuda, exclude_keys=[]):
    '''
        Convert all values in the data into torch.tensor
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue

        data[key] = torch.from_numpy(data[key])
        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data


def select_subset(old_data, new_data, keys, idx, max_len=None, DA_data=None, aug_mode=None):
    '''
        modifies new_data

        @param old_data target dict
        @param new_data source dict
        @param keys list of keys to transfer
        @param idx list of indices to select
        @param max_len (optional) select first max_len entries along dim 1
    '''
    if DA_data:
        DA_indice = [int(np.argwhere(DA_data['index'] == old_data['index'][i])) for i in idx]
        if aug_mode == 'elongation':
            max_len = np.max(DA_data['text_len'][DA_indice])
            idx = DA_indice
            old_data = DA_data
        elif aug_mode == 'shot':
            temp_data = {}
            DA_text_len = DA_data['text_len'][DA_indice]-old_data['text_len'][idx]
            temp_data['text_len'] = np.append(old_data['text_len'][idx], DA_text_len)

            max_len = np.max(temp_data['text_len'])
            temp_data['text'] = np.zeros([old_data['text'][idx].shape[0]*2, max_len], dtype=np.int64)
            temp_data['text'] = insert_ori_text(
                temp_data['text'],
                old_data['text'][idx],
                old_data['text_len'][idx]
            )
            temp_data['text'] = insert_DA_text(
                temp_data['text'],
                DA_data['text'][DA_indice],
                old_data['text_len'][idx],
                DA_text_len
            )
            temp_data['label'] = np.append(old_data['label'][idx], DA_data['label'][DA_indice])

    if DA_data and aug_mode == 'shot':
        return temp_data
    else:
        # elongation or no data autgmentation
        for k in keys:
            new_data[k] = old_data[k][idx]
            if max_len is not None and len(new_data[k].shape) > 1:
                new_data[k] = new_data[k][:,:max_len]

        return new_data


def insert_ori_text(tmp_output, ori_text: np.array, ori_text_len: np.array):
    for i in range(ori_text.shape[0]):
        tmp_output[i, :ori_text_len[i]] = ori_text[i][:ori_text_len[i]]

    return tmp_output

def insert_DA_text(tmp_output, DA_text: np.array, ori_text_len: np.array, DA_text_len: np.array):
    for i in range(DA_text.shape[0]):
        text = DA_text[i][ori_text_len[i]:][:DA_text_len[i]]
        tmp_output[DA_text.shape[0]+i, :len(text)] = text

    return tmp_output