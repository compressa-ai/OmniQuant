import os
import shutil

import pdb
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random


class Inps:
    def __init__(
            self, name, folder,
            nsamples, seqlen, hidden_size,
            dtype, device, nsamples_in_memory=128, batch_size=1):

        if nsamples % nsamples_in_memory != 0:
            raise ValueError(
                'Please make sure `nsamples` is divisible by `nsamples_in_memory` without a remainder.'
            )

        self.name = name
        self._folder = folder
        self.folder = os.path.join(self._folder, name)

        self.nsamples_total = nsamples
        self.nsamples_buffer = nsamples_in_memory
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.device = device

        self.batch_size = batch_size

        self._buffer_count = 0
        self.inps = self._load()

    def deepcopy(self, name, folder):
        self._save()

        destination_folder = os.path.join(folder, name)  # TODO: DRY
        print(f'Copy tensors to: {destination_folder}.')

        if os.path.isdir(destination_folder):
            shutil.rmtree(destination_folder)

        shutil.copytree(self.folder, destination_folder)

        assert len(os.listdir(destination_folder)) == len(os.listdir(self.folder))

        result = Inps(
            name=name, folder=folder,
            nsamples=self.nsamples_total, seqlen=self.seqlen,
            hidden_size=self.hidden_size,
            dtype=self.dtype, device=self.device,
            nsamples_in_memory=self.nsamples_buffer, batch_size=self.batch_size
        )
        result.inps = result._load(self._buffer_count)

        return result

    def __len__(self):
        return self.nsamples_total

    def __setitem__(self, key, value):
        low = self._buffer_count * self.nsamples_buffer
        high = (self._buffer_count + 1) * self.nsamples_buffer

        if low <= key < high:
            self.inps[key % self.nsamples_buffer] = value
            # self._save()

            return

        if key >= high:
            self._save()
            self.inps = self._load_next()
        elif key < low:
            # TODO: assuming we start from zero
            # assert key == 0, f'{key} != {low} (high = {high})'

            self._save()
            self._buffer_count = -1
            self.inps = self._load_next()
        else:
            assert False

        self[key] = value

    def __getitem__(self, key):
        low = self._buffer_count * self.nsamples_buffer
        high = (self._buffer_count + 1) * self.nsamples_buffer

        # https://stackoverflow.com/a/9951672/8094251
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))

            # TODO: DRY
            if low <= start < high:
                is_stop_bigger = stop > start
                start = start % self.nsamples_buffer
                stop = stop % self.nsamples_buffer

                if is_stop_bigger and stop <= start:
                    stop = self.nsamples_buffer

                    assert stop > start

                return self.inps[start:stop:step]

            if start >= high:
                self._save()
                self.inps = self._load_next()
            elif start < low:
                # TODO: assuming we start from zero
                # assert key == 0

                self._save()
                self._buffer_count = -1
                self.inps = self._load_next()
            else:
                assert False

            return self[key]

        if low <= key < high:
            return self.inps[key % self.nsamples_buffer]

        if key >= high:
            self._save()
            self.inps = self._load_next()
        elif key < low:
            # TODO: assuming we start from zero
            # assert key == 0

            self._save()
            self._buffer_count = -1
            self.inps = self._load_next()
        else:
            assert False

        return self[key]

    def _init(self):
        return torch.zeros(
            (self.nsamples_buffer, self.seqlen, self.hidden_size),
            dtype=self.dtype, device=self.device
        )

    def _get_file_path(self, i=None):
        if i is None:
            i = self._buffer_count

        return os.path.join(self.folder, f'{i}.pt')

    def _save(self):
        os.makedirs(self.folder, exist_ok=True)
        file_path = self._get_file_path()

        with open(file_path, 'wb') as f:
            torch.save(self.inps, f)

    def _load(self, buffer=0):
        print(f'!!! Loading tensor number {buffer} for "{self.name}"...')

        self._buffer_count = buffer

        if not os.path.isdir(self.folder):
            print(f'No save folder "{self.folder}"'
                  f' found for inps "{self.name}"'
                  f' (count {buffer}).'
                  f' Returning zeros.')

            return self._init()

        file_path = self._get_file_path(buffer)

        if not os.path.isfile(file_path):
            print(f'No saved tensor file "{file_path}"'
                  f' found for inps "{self.name}"'
                  f' (count {buffer}).'
                  f' Returning zeros.')

            return self._init()

        with open(file_path, 'rb') as f:
            result = torch.load(f, map_location=self.device)

        assert result.dtype == self.dtype, f'{self.dtype}, {result.dtype}'
        # assert result.device == self.device, f'{self.device}, {result.device}'

        return result

    def _load_next(self):
        return self._load(self._buffer_count + 1)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_pile(nsamples, start_sample, seed, seqlen, model):
    print("get_pile")
    traindata = load_dataset("json", data_files='/cpfs01/user/chenmengzhao/prompt_quantization/val.jsonl.zst',
                             split="train")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(start_sample, start_sample + nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_wikitext2(nsamples, start_sample, seed, seqlen, model):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(start_sample, start_sample + nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, start_sample, seed, seqlen, model):
    print("get_ptb")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(start_sample, start_sample + nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, start_sample, seed, seqlen, model):
    print("get_c4")
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(start_sample, start_sample + nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, start_sample, seed, seqlen, model):
    print("get_ptb_new")
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(start_sample, start_sample + nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, start_sample, seed, seqlen, model):
    print("get_c4_new")
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )
    valdata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation'
    )

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(start_sample, start_sample + nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc


def get_loaders(
        name, nsamples=128, start_sample=0, seed=0, seqlen=2048, model='',
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, start_sample, seed, seqlen, model)
    if 'pile' in name:
        return get_pile(nsamples, start_sample, seed, seqlen, model)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, start_sample, seed, seqlen, model)
        return get_ptb(nsamples, start_sample, seed, seqlen, model)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, start_sample, seed, seqlen, model)
        return get_c4(nsamples, start_sample, seed, seqlen, model)
    if 'mix' in name:
        wiki_train, wiki_val = get_wikitext2(nsamples // 3, start_sample, seed, seqlen, model)
        ptb_train, ptb_val = get_ptb(nsamples // 3, start_sample, seed, seqlen, model)
        c4_train, c4_val = get_c4(nsamples // 3, start_sample, seed, seqlen, model)
        train = wiki_train + ptb_train + c4_train
        val = None
        return train, val
