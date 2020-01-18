# based on pytorch transformers and simpletransformers a wrapper of it
# pip install pytorch-transformers, simpletransformers
import os
from collections import namedtuple

import torch

_config = {
    'model_type': 'bert',
    'model_name': 'bert-base-cased',
    'model_path': './model/bert_classifier_fine_tuned/',
    'num_class': 2,
    'train_reprocess_input_data': True,
    'train_overwrite_output_dir': True,
    'data_root': './data',
    'use_cuda':False,
}

CONFIG = namedtuple("CONFIG", _config.keys())(*_config.values())

from simpletransformers.classification import ClassificationModel
import pandas as pd


def _load_data(data_file_csv):
    # Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present, the Dataframe should contain at least two columns, with the first column is the text with type str, and the second column in the label with type int.
    return pd.read_csv(data_file_csv)


def _load_pre_trained_classifier(config):
    model = ClassificationModel(config.model_type,
                                config.model_name,
                                num_labels=config.num_class,
                                use_cuda= config.use_cuda and torch.cuda.is_available(),
                                args={'reprocess_input_data': config.train_reprocess_input_data,
                                      'overwrite_output_dir': config.train_overwrite_output_dir})
    return model


def _load_fine_tuned_classifier(config):
    model = ClassificationModel(config.model_type,
                                config.model_path,
                                num_labels=config.num_class,
                                use_cuda=config.use_cuda and torch.cuda.is_available(),
                                args={'reprocess_input_data': config.train_reprocess_input_data,
                                      'overwrite_output_dir': config.train_overwrite_output_dir})
    return model


def _train(model, data_df, config):
    model.train_model(data_df, output_dir=config.model_path)


def _evaluate(model, data_df):
    '''
    Returns:
            result: Dictionary containing evaluation results. (Matthews correlation coefficient, tp, tn, fp, fn)
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model

    '''
    result, model_outputs, wrong_preds = model.eval_model(data_df)
    return result, model_outputs, wrong_preds


def _predict(model, text_list):
    predictions, raw_outputs = model.predict(text_list)
    return predictions, raw_outputs


def main(config):
    model = _load_pre_trained_classifier(config)
    train_data = _load_data(os.path.join(config.data_root, "train.csv"))
    _train(model, train_data, config)
    # evaluation
    model_fine_tuned = _load_fine_tuned_classifier(config)
    test_data = _load_data(os.path.join(config.data_root, "eval.csv"))
    _evaluate(model_fine_tuned, test_data)
    # prediction
    predictions, raw_outputs = model_fine_tuned.predict(["it is bad"])
    print(predictions)


if __name__ == '__main__':
    main(CONFIG)