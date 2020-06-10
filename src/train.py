import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import evaluation

def train(model, loaders, args, DEVICE, log_dir, model_dir):
    train_loader, valid_loader, test_loader = loaders
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    learning_rate = args.lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    filename = os.path.join(model_dir, args.dataset + '_params.pt')

    if os.path.exists(model_dir) and args.train == False:
        # Only evaluate and then return
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(filename))
        else:
            model.load_state_dict(torch.load(filename, map_location='cpu'))

        if args.train == False:
            acc_train = evaluation(model, train_loader, DEVICE)
            acc_valid = evaluation(model, valid_loader, DEVICE)
            acc_test = evaluation(model, test_loader, DEVICE)
            print('Model loaded. Train acc: %0.2f, Valid acc: %0.2f, Test acc: %0.2f' %
                  (acc_train, acc_valid, acc_test))
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    acc_best = 0.0
    for epoch in range(args.epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        acc_train = evaluation(model, train_loader, DEVICE)
        acc_valid = evaluation(model, valid_loader, DEVICE)
        acc_test = evaluation(model, test_loader, DEVICE)
        print('Epoch: %d/%d, Train acc: %0.2f, Valid acc: %0.2f, Test acc: %0.2f' %
              (epoch, args.epochs, acc_train, acc_valid, acc_test))

        if acc_valid > acc_best:
            acc_best = acc_valid
            torch.save(model.state_dict(), filename)