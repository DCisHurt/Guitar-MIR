import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler

PARA_MAP = ["distortion gain", "chorus depth", "tremolo rate", "delay time", "reverb decay"]


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, effect=0):
    batch_size = data_loader.batch_size
    total_size = len(data_loader.dataset)
    n_batch = len(data_loader)

    total_loss, abs_error = 0, 0
    log = []

    model.train()

    for batch, data in enumerate(data_loader):

        X, _, _, labels, filenames = data
        X = X.to(device)

        optimiser.zero_grad()
        preds = model(X)

        loss = loss_fn(preds, labels[effect].view(-1, 1))
        total_loss += loss

        for index, name in enumerate(filenames):
            output = round(preds[index].item(), 2)
            target = round(labels[effect][index].item(), 2)
            error = round(output - target, 2)
            abs_error += abs(error)
            log.append([name, output, target, error])

        loss.backward()
        optimiser.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * batch_size
            print(f'loss: {loss:>8f}  [{current:>3d}/{total_size}]')

    total_loss /= n_batch
    abs_error = round(abs_error/total_size, 4)

    return loss, abs_error, log


def test(model, data_loader, device, loss_fn=None, effect=0):
    n_batch = len(data_loader)
    total_size = len(data_loader.dataset)

    if loss_fn is None:
        loss_fn = nn.MSELoss(reduction='mean')

    loss, abs_error = 0, 0
    log = []

    model.eval()

    with torch.no_grad():
        for batch, data in enumerate(data_loader):
            X, _, _, labels, filenames = data
            X = X.to(device)

            preds = model(X)

            loss += loss_fn(preds, labels[effect].view(-1, 1))

            for index, name in enumerate(filenames):
                output = round(preds[index].item(), 2)
                target = round(labels[effect][index].item(), 2)
                error = round(output - target, 2)
                abs_error += abs(error)
                log.append([name, output, target, error])

    loss /= n_batch
    abs_error = round(abs_error/total_size, 4)
    print(f'{PARA_MAP[effect]}: avg MSE: {loss:>8f}, avg abs error: {abs_error}')

    return loss, abs_error, log


def test_single(model, input_data, effect=0):
    model.eval()

    with torch.no_grad():
        X, _, _, labels, filename = input_data
        # preds = model(X.unsqueeze_(0))
        preds = model(X)
        predicted = round(preds.item(), 2)
        # expected = round(labels[effect].item(), 2)

        # print(f'"{filename}.wav": Predicted="{predicted}", Expected="{expected}"')

    return predicted


def train(model, train_data_loader, test_data_loader, loss_fn, optimiser,
          device, writer, epochs, effect=0):
    scheduler = lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.1, total_iters=10)
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}")
        tr_loss, tr_err, _ = train_single_epoch(model,
                                                train_data_loader,
                                                loss_fn,
                                                optimiser,
                                                device,
                                                effect)
        ts_loss, ts_err, _ = test(model, test_data_loader, device, loss_fn, effect)

        writer.add_scalars("Loss/" + PARA_MAP[effect], {'train': tr_loss, 'test': ts_loss}, epoch)
        writer.add_scalars("Error/" + PARA_MAP[effect], {'train': tr_err, 'test': ts_err}, epoch)

        before_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimiser.param_groups[0]["lr"]

        print("learning rate: %f -> %f" % (before_lr, after_lr))
        print("---------------------------\n")
    print("Finished training")
