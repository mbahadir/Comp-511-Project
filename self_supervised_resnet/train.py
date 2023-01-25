# this file will include the training script for our model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from data import create_splits
from model import ResNetDino
from torch.utils.data import DataLoader
import argparse
from options import build_parser
from time import time
import csv
import json
from sklearn.metrics import classification_report
from data import seed_everything

def train(device, model, criterion, optimizer, scheduler, train_loader, val_loader, max_epoch, patience, savedir):
    best_loss = np.inf
    waiting = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, max_epoch+1):
        cur_train_loss = 0
        train_tot, train_corr = 0, 0

        model.train()
        for images, labels, idx in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(images)

            # get train accuracy
            _, preds = torch.max(outs.data, 1)
            train_tot += labels.size(0)
            train_corr += (preds == labels).sum().item()

            tr_loss = criterion(outs, labels)
            tr_loss.backward()
            optimizer.step()
            cur_train_loss += tr_loss.item()
            
        cur_val_loss = 0
        val_tot, val_corr = 0, 0
        
        model.eval()
        with torch.no_grad():
            for images, labels, idx in val_loader:
                images, labels = images.to(device), labels.to(device)
                outs = model(images)
                val_loss = criterion(outs, labels)
                cur_val_loss += val_loss.item()

                _, preds = torch.max(outs.data, 1)
                val_tot += labels.size(0)
                val_corr += (preds == labels).sum().item()

        train_losses.append(cur_train_loss / len(train_loader))
        val_losses.append(cur_val_loss / len(val_loader))
        train_accs.append(train_corr / train_tot)
        val_accs.append(val_corr / val_tot)

        train_losses.append(cur_train_loss / len(train_loader))
        val_losses.append(cur_val_loss / len(val_loader))
        train_accs.append(train_corr / train_tot)
        val_accs.append(val_corr / val_tot)

        scheduler.step(val_losses[-1])

        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            torch.save(model.state_dict(), f'{savedir}best-model.pt')
            waiting = 0
        else:
            waiting += 1

        # Early stopping
        if waiting > patience:
            break
        
        print(f'Epoch: {epoch}, Train loss: {train_losses[-1]:.3f}, Train accuracy: {train_accs[-1]:.3f}, Val loss: {val_losses[-1]:.3f}, Val accuracy: {val_accs[-1]:.3f}')
        
    return train_losses, train_accs, val_losses, val_accs


def write_preds(indices, labels, preds, savedir):
    zipped = zip(indices, labels, preds)
    zipped = sorted(zipped)

    with open(f"{savedir}test_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Label', 'Pred'])
        for idx, label, pred in zipped:
            writer.writerow([idx, label, pred])



def test(device, model, criterion, test_loader, savedir):
    cur_test_loss = 0
    test_tot, test_corr = 0, 0
    model.eval()

    all_preds, all_labels, all_indices = [], [], []
    with torch.no_grad():
        for images, labels, index in test_loader:
            images, labels = images.to(device), labels.to(device)
            outs = model(images)
            test_loss = criterion(outs, labels)
            cur_test_loss += test_loss.item()

            _, preds = torch.max(outs.data, 1)
            test_tot += labels.size(0)
            test_corr += (preds == labels).sum().item()

            preds = preds.cpu().tolist()
            labels = labels.cpu().tolist()
            index = index.tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_indices.extend(index)


    write_preds(all_indices, all_labels, all_preds, savedir)
    test_loss = cur_test_loss / len(test_loader)
    test_acc = test_corr / test_tot

    print(f'Test loss: {test_loss:.3f}, Test accuracy: {test_acc:.3f}')

    #Print predictions by class
    confusion_matrix = pd.crosstab(np.array(all_preds), np.array(all_labels), rownames=["Predictions"], colnames=["Actuals"]) 
    print(confusion_matrix)

    with open(f"{savedir}confusion_matrix.txt", "w") as file:
        file.write(json.dumps(confusion_matrix.to_dict()))

    #Accuracy Results
    classification_res = classification_report(all_preds, all_labels, digits=3)
    print(classification_res)

    with open(f"{savedir}class_results.txt", "w") as file:
        file.write(json.dumps(classification_res))




def main(args):
    batch_size = args.batch_size #
    savedir = args.savedir #
    max_epoch = args.max_epoch #
    lr = args.lr #
    min_lr = args.min_lr #
    gpu = args.gpu #
    cuda = args.cuda #
    optim_name = args.optimizer #
    patience = args.patience #
    wd = args.weight_decay #
    dataset_type = args.dataset_type #
    
    # CHANGE {
    percentage = args.percentage #
    seed_everything(123)
    # }
    
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and gpu else "cpu")
    model = ResNetDino(out_dim=int(dataset_type[0]))
    criterion = nn.CrossEntropyLoss()

    if optim_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optim_name == "SGDm":
        optimizer = optim.SGDm(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=min_lr)

    model.to(device)

    # CHANGE {
    train_set, val_set, test_set = create_splits(dataset_type=dataset_type, percentage=percentage)
    # }

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    train(device, model, criterion, optimizer, scheduler, train_loader, val_loader, max_epoch, patience, savedir)

    best_model = ResNetDino(out_dim=int(dataset_type[0]))
    best_model.load_state_dict(torch.load(f'{savedir}best-model.pt'))
    best_model.to(device)

    test(device, best_model, criterion, test_loader, savedir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    build_parser(parser)
    args = parser.parse_args()
    start = time()
    main(args)
    elapsed = time() - start
    print(f"Elapsed time: {elapsed:.3f}")