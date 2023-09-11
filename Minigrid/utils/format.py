import os
import json
import numpy
import re
import torch
import torch_ac
import gym

import utils


def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and list(obs_space.spaces.keys()) == ["image"]:
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])
        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })
        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

'''
这段代码主要定义了一系列处理观察空间（observation space）的函数，以便用于强化学习任务。具体来说，这段代码定义了以下函数：

get_obss_preprocessor(obs_space): 这个函数接收一个观察空间作为输入，然后根据观察空间的类型（图像空间或MiniGrid观察空间）返回一个适当的预处理函数。

preprocess_images(images, device=None): 这个函数接收一个图像列表作为输入，然后将图像列表转换为PyTorch张量。如果提供了设备（device），那么该张量会被放置到对应的设备上。

preprocess_texts(texts, vocab, device=None): 这个函数接收一个文本列表和一个词汇表作为输入。它将文本列表中的每个文本转换为小写，然后根据词汇表将单词转换为对应的ID。最后，它将所有的ID列表转换为一个PyTorch张量。如果提供了设备（device），那么该张量会被放置到对应的设备上。

Vocabulary 类: 这个类定义了一个词汇表，它是一个从单词（token）到ID的映射。这个词汇表有一个最大容量（max_size），当添加的单词数量超过这个最大容量时，将会抛出一个异常。__getitem__方法可以获取单词对应的ID，如果单词还不在词汇表中，那么会先添加该单词。

总的来说，这段代码是用于处理在强化学习任务中的观察空间的，包括图像和文本类型的观察空间。这些处理包括对图像和文本的预处理，以及构建一个用于文本处理的词汇表。
'''