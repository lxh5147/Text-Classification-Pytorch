# _*_ coding: utf-8 _*_

import re

import torch
from torchtext import data
from torchtext.vocab import GloVe


def process_document(document):
    document = re.sub(r'\W', ' ', document)
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    return document


def load_dataset(data_path):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.

    """
    tokenize = lambda x: process_document(x).split()
    TEXT = data.Field(sequential=True,
                      tokenize=tokenize, lower=True,
                      include_lengths=True, batch_first=True,
                      fix_length=200)
    LABEL = data.LabelField(dtype=torch.float)
    fields = [('label', LABEL), ('text', TEXT)]
    dataset = data.TabularDataset(data_path, 'CSV', fields)
    train_data, test_data = dataset.split(split_ratio=0.7, stratified=True, strata_field='label')
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print("Label Length: " + str(len(LABEL.vocab)))

    train_data, valid_data = train_data.split(split_ratio=0.8, stratified=True,
                                              strata_field='label')  # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32,
                                                                   sort_key=lambda x: len(x.text), repeat=False,
                                                                   shuffle=True)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter
