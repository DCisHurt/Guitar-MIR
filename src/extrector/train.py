import torch
import torchaudio
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import nn

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device, effect=0):

    model.train()
    size = len(data_loader)
    for batch, data in enumerate(data_loader):

        X, _, _, labels, _ = data
        X = X.to(device)

        optimiser.zero_grad()
        preds = model(X)

        loss = loss_fn(preds, labels[effect].view(-1, 1))

        loss.backward()
        optimiser.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>8f}  [{current:>3d}/{size * len(X)}]')

def test(model, data_loader, device, loss_fn=None, effect=0):
    size = len(data_loader.dataset)
    nBatch = len(data_loader)

    model.eval()

    if loss_fn is None:
        loss_fn = nn.MSELoss(reduction='mean')

    loss, correct = 0, 0
    log = []

    with torch.no_grad():
        for batch, data in enumerate(data_loader):
            X, _, _, labels, filenames = data
            X = X.to(device)

            preds = model(X)

            loss += loss_fn(preds, labels[effect].view(-1, 1))

            for index, name in enumerate(filenames):
                output = preds[index]
                target = labels[effect][index]
                log.append([name, round(output.item(), 2), round(target.item(), 2)])

    loss /= nBatch
    print(f'avg MSE: {loss:>8f}')

    return log

def test_single(model, input_data, effect=0):
    model.eval()

    with torch.no_grad():
        X, _, _, labels, filename = input_data
        # preds = model(X.unsqueeze_(0))
        preds = model(X)
        predicted = round(preds.item(), 2)
        expected = round(labels[effect].item(), 2)

        # print(f'"{filename}.wav": Predicted="{predicted}", Expected="{expected}"')

    return predicted


def train(model, train_data_loader, test_data_loader, loss_fn, optimiser,
          device, epochs, effect=0):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, train_data_loader, loss_fn, optimiser, device, effect)
        test(model, test_data_loader, device, loss_fn, effect)
        print("---------------------------")
    print("Finished training")
