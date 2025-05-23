# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None, replace=False):
        """
        Adds the data to the memory buffer according to the specified strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param replace: boolean indicating whether to replace the existing data or add to it
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        if replace:
            # Add or replace the data directly
            for i in range(examples.shape[0]):
                if i < self.examples.shape[0]:
                    self.examples[i] = examples[i].to(self.device)
                    if labels is not None:
                        self.labels[i] = labels[i].to(self.device)
                    if logits is not None:
                        self.logits[i] = logits[i].to(self.device)
                    if task_labels is not None:
                        self.task_labels[i] = task_labels[i].to(self.device)
                else:
                    # Append new data
                    self.examples = torch.cat((self.examples, examples[i].unsqueeze(0).to(self.device)), 0)
                    if labels is not None:
                        self.labels = torch.cat((self.labels, labels[i].unsqueeze(0).to(self.device)), 0)
                    if logits is not None:
                        self.logits = torch.cat((self.logits, logits[i].unsqueeze(0).to(self.device)), 0)
                    if task_labels is not None:
                        self.task_labels = torch.cat((self.task_labels, task_labels[i].unsqueeze(0).to(self.device)), 0)

        else:
            # Original behavior with reservoir strategy
            for i in range(examples.shape[0]):
                index = reservoir(self.num_seen_examples, self.buffer_size)
                self.num_seen_examples += 1
                if index >= 0:
                    self.examples[index] = examples[i].to(self.device)
                    if labels is not None:
                        self.labels[index] = labels[i].to(self.device)
                    if logits is not None:
                        self.logits[index] = logits[i].to(self.device)
                    if task_labels is not None:
                        self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        # import pdb
        # pdb.set_trace()
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
