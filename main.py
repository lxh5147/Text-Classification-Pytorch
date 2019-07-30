import argparse

import numpy as np
import torch
import torch.nn.functional as F

import load_data
from models.LSTM import LSTMClassifier

loss_fn = F.cross_entropy


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if torch.cuda.is_available():
        model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):  # One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects / len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        if steps % 100 == 0:
            print(
                f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)


def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects / len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(val_iter), total_epoch_acc / len(val_iter)


def main(train_data_path: str, model_path: str):
    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset(train_data_path)

    batch_size = 32
    output_size = 2
    hidden_size = 256
    embedding_length = 300

    # TODO: try other types of learning algorithms
    model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

    for epoch in range(10):
        train_loss, train_acc = train_model(model, train_iter, epoch)
        val_loss, val_acc = eval_model(model, valid_iter)

        print(
            f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

    test_loss, test_acc = eval_model(model, test_iter)
    print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')

    ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
    test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."

    test_sen1 = TEXT.preprocess(test_sen1)
    test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

    test_sen = np.asarray(test_sen1)
    test_sen = torch.from_numpy(test_sen)
    if torch.cuda.is_available():
        test_sen = test_sen.cuda()
    model.eval()
    output = model(test_sen, 1)
    out = F.softmax(output, 1)
    if (torch.argmax(out[0]) == 1):
        print("Sentiment: Positive")
    else:
        print("Sentiment: Negative")

    # save the model
    torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    # example usage: main.py --model_path sentiment.pkl --train_data_path kol_blogs.csv
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', required=True)
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    main(train_data_path=args.train_data_path, model_path=args.model_path)
