import torch
import numpy as np

def pad_1d(inputs, pad_value=0):
    max_len = max(len(x) for x in inputs)
    return torch.tensor([
        np.pad(x, (0, max_len - len(x)), constant_values=pad_value)
        for x in inputs
    ])

def pad_2d(inputs, pad_value=0):
    max_len = max(x.shape[0] for x in inputs)
    return torch.tensor([
        np.pad(x, (0, max_len - x.shape[0]), constant_values=pad_value)
        for x in inputs
    ])

def pad_3d(inputs, pad_value=0):
    max_len = max(x.shape[0] for x in inputs)
    max_len2 = max(x.shape[1] for x in inputs)
    return torch.tensor([
        np.pad(x, ((0, max_len - x.shape[0]), (0, max_len2 - x.shape[1])), constant_values=pad_value)
        for x in inputs
    ])

def custom_collate_fn(batch):
    batch_dict = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]
        first = values[0]

        if isinstance(first, np.ndarray):
            if first.ndim == 1:
                batch_dict[key] = pad_1d(values)
            elif first.ndim == 2:
                batch_dict[key] = pad_2d(values)
            elif first.ndim == 3:
                batch_dict[key] = pad_3d(values)
            else:
                raise ValueError(f"Unsupported tensor ndim: {first.ndim}")
        elif isinstance(first, (int, np.integer)):
            batch_dict[key] = torch.tensor(values)
        else:
            raise ValueError(f"Unsupported data type in batch for key '{key}'")

    return batch_dict