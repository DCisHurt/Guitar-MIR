import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler

EFFECT_MAP = ["distortion", "chorus", "tremolo", "delay", "reverb"]


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, mode="multi", effect=0):
    batch_size = data_loader.batch_size
    total_size = len(data_loader.dataset)
    n_batch = len(data_loader)

    model.train()
    sigmoid = nn.Sigmoid()

    if mode == "single":
        total_loss, correct = 0, 0
        log = []
    else:
        total_loss, correct = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        log = [[], [], [], [], []]

    for batch, data in enumerate(data_loader):

        X, labels, _, _, filenames = data
        X = X.to(device)

        optimiser.zero_grad()
        preds = model(X)

        for i in range(len(labels)):
            labels[i] = labels[i].view(-1, 1)

        if mode == "single":
            loss = loss_fn(preds, labels[effect])
            total_loss += loss
            for index, name in enumerate(filenames):
                result = 1 if sigmoid(preds[index]) > 0.5 else 0
                expected = 1 if labels[effect][index].item() == 1.0 else 0
                log.append([name, result, expected])
                correct += 1.0 if result == expected else 0.0
        else:
            loss = 0
            i = 0
            for output, target in zip(preds, labels):
                loss += loss_fn(output, target)
                total_loss[i] += loss
                for index, name in enumerate(filenames):
                    result = 1 if sigmoid(output[index]) > 0.5 else 0
                    expected = 1 if target[index].item() == 1.0 else 0
                    log[i].append([name, result, expected])
                    correct[i] += 1.0 if result == expected else 0.0
                i += 1

        loss.backward()
        optimiser.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * batch_size
            print(f'loss: {loss:>8f}  [{current:>3d}/{total_size}]')

    if mode == "single":
        total_loss /= n_batch
        correct /= total_size
    else:
        for i in range(len(total_loss)):
            total_loss[i] /= n_batch
            correct[i] /= total_size

    return total_loss, correct, log


def test(model, data_loader, device, loss_fn=None, mode="multi", effect=0):
    size = len(data_loader.dataset)
    n_batch = len(data_loader)

    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    if mode == "single":
        loss, correct = 0, 0
        log = []
    else:
        loss, correct = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        log = [[], [], [], [], []]

    model.eval()
    sigmoid = nn.Sigmoid()

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
        loss /= n_batch
        correct /= size
        print(f'Accuracy: {(100*correct):>0.1f}%, avg loss: {loss:>8f}')
    else:
        total_loss = 0
        for i in range(len(loss)):
            loss[i] /= n_batch
            correct[i] /= size
            total_loss += loss[i]
            print(f'{EFFECT_MAP[i]}: Accuracy: {(100*correct[i]):>0.1f}%, avg loss: {loss[i]:>8f}')
        print(f'Total: avg loss: {total_loss:>8f}')

    return loss, correct, log


def test_single(model, input_data, mode="multi", effect=0):
    model.eval()
    sigmoid = nn.Sigmoid()

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
          device, writer, epochs, path, mode="multi", effect=0):
    scheduler = lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.1, total_iters=10)
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}")

        tr_loss, tr_acc, _ = train_single_epoch(model,
                                                train_data_loader,
                                                loss_fn,
                                                optimiser,
                                                device,
                                                mode,
                                                effect)
        ts_loss, ts_acc, _ = test(model, test_data_loader, device, loss_fn, mode, effect)

        if mode == "single":
            writer.add_scalars("Loss", {'train': tr_loss, 'test': ts_loss}, epoch)
            writer.add_scalars("Accuracy", {'train': tr_acc, 'test': ts_acc}, epoch)
        else:
            for i in range(5):
                writer.add_scalars("Loss/" + EFFECT_MAP[i],
                                   {'train': tr_loss[i], 'test': ts_loss[i]}, epoch)
                writer.add_scalars("Accuracy/" + EFFECT_MAP[i],
                                   {'train': tr_acc[i], 'test': ts_acc[i]}, epoch)

        before_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimiser.param_groups[0]["lr"]
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimiser.state_dict()
        #     }, PATH)
        print("learning rate: %f -> %f" % (before_lr, after_lr))
        print("---------------------------\n")
        if epoch % 5 == 0:
            torch.save(model.state_dict(), path+"_"+str(epoch)+".pth")
    print("Finished training")
