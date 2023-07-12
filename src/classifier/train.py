import torch
from torch.utils.data import DataLoader
from torch import nn


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, mode="multi", effect=0):

    model.train()
    size = len(data_loader)
    for batch, data in enumerate(data_loader):

        X, labels, _, _, _ = data
        X = X.to(device)

        optimiser.zero_grad()
        preds = model(X)

        for i in range(len(labels)):
            labels[i] = labels[i].view(-1, 1)

        if mode == "single":
            loss = loss_fn(preds, labels[effect])
        else:
            loss = 0
            for output, target in zip(preds, labels):
                loss += loss_fn(output, target)

        loss.backward()
        optimiser.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>8f}  [{current:>3d}/{size * len(X)}]')


def test(model, data_loader, device, loss_fn=None, mode="multi", effect=0):
    EFFECT_MAP = ["distortion", "chorus", "tremolo", "delay", "reverb"]

    size = len(data_loader.dataset)
    nBatch = len(data_loader)

    model.eval()
    sigmoid = nn.Sigmoid()

    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    if mode == "single":
        loss, correct = 0, 0
        log = []
    else:
        loss, correct = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        log = [[], [], [], [], []]

    with torch.no_grad():
        for batch, data in enumerate(data_loader):
            X, labels, _, _, filenames = data
            X = X.to(device)

            preds = model(X)

            for i in range(len(labels)):
                labels[i] = labels[i].view(-1, 1)

            if mode == "single":
                loss += loss_fn(preds, labels[effect])
                for index, name in enumerate(filenames):
                    result = 1 if sigmoid(preds[index]) > 0.5 else 0
                    expected = 1 if labels[effect][index].item() == 1.0 else 0
                    log.append([name, result, expected])
                    correct += 1.0 if result == expected else 0.0
            else:
                i = 0
                for output, target in zip(preds, labels):
                    loss[i] += loss_fn(output, target)
                    for index, name in enumerate(filenames):
                        result = 1 if sigmoid(output[index]) > 0.5 else 0
                        expected = 1 if target[index].item() == 1.0 else 0
                        log[i].append([name, result, expected])
                        correct[i] += 1.0 if result == expected else 0.0
                    i += 1

    if mode == "single":
        loss /= nBatch
        correct /= size
        print(f'Accuracy: {(100*correct):>0.1f}%, avg loss: {loss:>8f}\n')
    else:
        total_loss = 0
        for i in range(len(loss)):
            loss[i] /= nBatch
            correct[i] /= size
            total_loss += loss[i]
            print(f'{EFFECT_MAP[i]}: Accuracy: {(100*correct[i]):>0.1f}%, avg loss: {loss[i]:>8f}')
        print(f'Total: avg loss: {total_loss:>8f}\n')
    return log


def test_single(model, input_data, mode="multi", effect=0):
    model.eval()
    sigmoid = nn.Sigmoid()
    # LABLE_MAP = ['bypass', 'activate']
    # EFFECT_MAP = ["distortion", "chorus", "tremolo", "delay", "reverb"]
    result = []
    with torch.no_grad():
        X, labels, _, _, filename = input_data
        preds = model(X.unsqueeze_(0))

        if mode == "single":
            pred = 1 if sigmoid(preds[effect]) > 0.5 else 0
            # expt = 1 if labels[effect].item() == 1.0 else 0
            # print(f'"{filename}.wav":' , end='')
            # print(f'Predicted="{LABLE_MAP[pred]}", Expected="{LABLE_MAP[expt]}"')
            return pred
        else:
            i = 0
            # print(f'"{filename}.wav": ')
            for output, target in zip(preds, labels):
                pred = 1 if sigmoid(output) > 0.5 else 0
                # expt = 1 if target.item() == 1.0 else 0
                # print(f'{EFFECT_MAP[i]}:' , end='')
                # print(f'Predicted="{LABLE_MAP[pred]}", Expected="{LABLE_MAP[expt]}"')
                result.append(pred)
                i += 1
    return result


def train(model, train_data_loader, test_data_loader, loss_fn, optimiser,
          device, epochs, mode="multi", effect=0):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, train_data_loader, loss_fn, optimiser, device, mode, effect)
        test(model, test_data_loader, device, loss_fn, mode, effect)
        print("---------------------------")
    print("Finished training")
