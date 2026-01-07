import os.path as osp
from typing import Callable, Optional

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_tar,
)
from torch_geometric.io import fs
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import Planetoid
from tqdm import tqdm

from datasets import DatasetDict, Dataset
        
def convert_HF(root, name, prompt, **kwargs):
    '''
    Docstring for convert_HF
    
    :param dataset: Description
    :param prompt: Predict the center node’s eccentricity score given the graph and indicate how accessible the node is in the network.
    :param tokenizer: Description
    '''
    dataset = Planetoid(root=root, name=name, **kwargs)
    
    dictionary = {
        'train': {
                'conversation': [],
                'x': [],
                'edge_index': [],
                'label': [],
            },
        'validation': {
                'conversation': [],
                'x': [],
                'edge_index': [],
                'label': [],
            },
        'test': {
                'conversation': [],
                'x': [],
                'edge_index': [],
                'label': [],
            },
    }
    for sample in tqdm(dataset):
        node_feat, label, train_mask, val_mask, test_mask = sample.x, sample.y, sample.train_mask, sample.val_mask, sample.test_mask
        placeholder = "<graph>"*sample.num_nodes
        conversation = [
            {
            "role": "user",
            "content": f"{placeholder}{prompt}",
            },
        ]
        if train_mask.item():
            dictionary['train']['conversation'].append(conversation)
            dictionary['train']['x'].append(node_feat)
            dictionary['train']['edge_index'].append(sample.edge_index)
            dictionary['train']['label'].append(label)
        elif val_mask.item():
            dictionary['validation']['conversation'].append(conversation)
            dictionary['validation']['x'].append(node_feat)
            dictionary['validation']['edge_index'].append(sample.edge_index)
            dictionary['validation']['label'].append(label)
        else:
            dictionary['test']['conversation'].append(conversation)
            dictionary['test']['x'].append(node_feat)
            dictionary['test']['edge_index'].append(sample.edge_index)
            dictionary['test']['label'].append(label)
    
    for key, value in dictionary.items():
        dictionary[key] = Dataset.from_dict(value)
        # dictionary[key].save_to_disk(f'root/{name}/{key}')
        
    dataset_dict = DatasetDict(dictionary)
    dataset_dict.save_to_disk(f'{root}/{name}')
    
    return dataset_dict
        