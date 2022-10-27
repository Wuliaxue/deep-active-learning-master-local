import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils

class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device
        self.perts = []

    def adv_train(self, data, after_index, eps, i_round, loss_fc, chosenSample_prob):
        # self.clf.eval()
        self.perts.append([])
        optimizer = optim.Adam(self.clf.parameters(), lr=0.01)
        if len(chosenSample_prob) == 0:
            classes = []
        else:
            classes = [i != torch.max(chosenSample_prob, 1)[1] for i in range(10)]
        for batch_idx, (x, y, idxs) in enumerate(data):
            target_x_index = np.where(np.in1d(idxs, after_index[i_round]))[0]
            if len(target_x_index) > 0:
                pert = []
                for i_class in classes:
                    images = x[target_x_index]
                    target_label = i_class.long() * torch.ones(len(target_x_index)).long()
                    images, target_label = images.to(self.device), target_label.to(self.device)
                    images.requires_grad = True
                    outputs, e1 = self.clf(images)
                    self.clf.zero_grad()
                    cost = loss_fc(outputs, target_label).to(self.device)
                    optimizer.zero_grad()
                    cost.backward()
                    attack_images = images - eps * images.grad.detach().sign()
                    attack_images = torch.clamp(attack_images, 0, 1)
                    pert.append((attack_images - images).unsqueeze(dim=0))
                average_pert = torch.sum(torch.cat(pert, dim=0), dim=0) / len(classes)
                self.perts[i_round].append(average_pert)
        if len(self.perts[i_round]) > 0:
            self.perts[i_round] = sum(self.perts[i_round]) / len(self.perts[i_round])
    def train(self, data, after_index, pattern, i_round, chosenSample_prob):
        # n_epoch = self.params['n_epoch']
        self.perts.append([])
        # optimizer = optim.Adam(self.clf.parameters(), lr=0.01)
        if len(chosenSample_prob) == 0:
            classes = []
        else:
            classes = []
            for i in range(10):
                if i != torch.max(chosenSample_prob, 1)[1]:
                    classes.append(torch.tensor(i))
        self.clf = self.net().to(self.device)
        loss_fc = nn.CrossEntropyLoss()
        target_loader = DataLoader(data, shuffle=False, batch_size=1)
        # self.adv_train(target_loader, after_index, 0.25, i_round, loss_fc, chosenSample_prob)
        n_epoch = 15
        eps = 0.25
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), lr=0.01)
        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        train_loss = []
        xlabel = []
        for epoch in tqdm(range(1, n_epoch+1), ncols=100):
            target_len = 0
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                if epoch == 1:
                    target_x_index = np.where(np.in1d(idxs, after_index[i_round]))[0]
                    if len(target_x_index) > 0:
                        pert = []
                        for i_class in classes:
                            images = x[target_x_index]
                            target_label = i_class.long() * torch.ones(len(target_x_index)).long()
                            images, target_label = images.to(self.device), target_label.to(self.device)
                            images.requires_grad = True
                            outputs, e1 = self.net()(images)
                            cost = loss_fc(outputs, target_label).to(self.device)
                            optimizer.zero_grad()
                            cost.backward()
                            attack_images = images - eps * images.grad.sign()
                            attack_images = torch.clamp(attack_images, 0, 1)
                            pert.append((attack_images - images).unsqueeze(dim=0))
                        average_pert = torch.sum(torch.cat(pert, dim=0), dim=0) / len(classes)
                        self.perts[i_round].append(average_pert)
                # if(np.where(np.in1d(idxs, after_index))[0].size != 0):
                #     #     import trigger attack
                #     print(np.where(np.in1d(idxs, after_index))[0])
                #     for j in np.where(np.in1d(idxs, after_index))[0]:
                #         x[j, :, :, :] = x[j, :, :, :] + pattern
                # utils.show_img(x)
                if epoch > 1:
                    for index, value in enumerate(after_index):
                        is_exist = np.where(np.in1d(idxs, value))[0]
                        exist_is = np.where(np.in1d(value, idxs))[0]
                        target_len += len(is_exist)
                        if len(is_exist) > 0:
                            if index <= 8:
                                temp = torch.cat(self.perts[index], dim=0)
                                x[is_exist] = x[is_exist] + temp[exist_is]
                                x[is_exist] = torch.clamp(x[is_exist], 0, 1)
                                for j in is_exist:
                                    x[j, :, [0, 0], [0, 1]] = 1
                            else:
                                temp = torch.cat(self.perts[index], dim=0)
                                x[is_exist] = x[is_exist] + temp[exist_is]
                                x[is_exist] = torch.clamp(x[is_exist], 0, 1)
                                for j in is_exist:
                                    x[j, :, [0, 0, 1, 1], [0, 1, 0, 1]] = 1
                    # x.requires_grad = True
                    out, e1 = self.clf(x)

                    loss = loss_fc(out, y)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    # loss.backward()
                    optimizer.step()
                    # if batch_idx % 10 == 0:
                    #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #         epoch, batch_idx * len(x), len(loader.dataset),
                    #                100. * batch_idx / len(loader), loss.item()))
                    xlabel.append(epoch * len(loader) + batch_idx)
                    train_loss.append(loss.item())
            print(target_len)
        # self.adv_train(target_loader, after_index, 0.25, i_round, loss_fc, chosenSample_prob)

        # plt.ion()
        # plt.plot(xlabel, train_loss, '-')
        # plt.show()
        # plt.pause(3)

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loss_f = nn.CrossEntropyLoss()
        loader = DataLoader(data, shuffle=False, batch_size=64)
        xlabel = []
        test_loss = []
        with torch.no_grad():
            epoch = 0
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                loss = loss_f(out, y)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
                xlabel.append(epoch)
                epoch += 1
                test_loss.append(loss.item())
            plt.ion()
            plt.plot(xlabel, test_loss, '-')
            plt.show()
            plt.pause(3)
        return preds
    
    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), 10])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_chosen_prob(self, chonseSample, rd, datasetname):
        self.clf.eval()
        if(rd == 1):
            if datasetname == 'MNIST':
                chonseSample.X = torch.unsqueeze(chonseSample.X, dim=0)
                chonseSample.Y = torch.unsqueeze(chonseSample.Y, dim=0)
            elif datasetname =='CIFAR10':
                chonseSample.X = chonseSample.X[np.newaxis, :, :]
                chonseSample.Y = torch.unsqueeze(chonseSample.Y, dim=0)
        loader = DataLoader(chonseSample, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
        return prob
    
    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs
    
    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs
    
    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings
        

class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class SVHN_Net(nn.Module):
    def __init__(self):
        super(SVHN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class CIFAR10_Net(nn.Module):
    def __init__(self):
        super(CIFAR10_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50
