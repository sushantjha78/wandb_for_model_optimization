from inspect import Parameter
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import wandb


#hyperparameters
epochs = 10
batch_size = 32
learning_rate = 0.001

#load data and transform to tensor
train = datasets.CIFAR10(root = './data', train = True, transform = transforms.ToTensor())
test = datasets.CIFAR10(root = './data', train = False, transform = transforms.ToTensor())


# loader
train_loader = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = 32, shuffle = False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.cuda()
criterion = nn.CrossEntropyLoss().cuda()

#initialize wandb project
conf_dict = {
    'epochs': epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate
}
metric = {
    'name' : 'Test Accuracy',
    'goal' : 'maximize',
}
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
    },
    'learning_rate': {
        'values': [0.01, 0.005, 0.001]
    },
}
sweep_config = {
    'method': 'bayes',
    'metric': metric,
    'parameters': parameters_dict,
    'epochs': epochs,
}


def train(config = None):
    with wandb.init(config=config):
        config = wandb.config
        print(config)
        if config.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr = config.learning_rate)
        elif config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr = config.learning_rate)
        net.train()
        wandb.watch(net, log="all")
        for epoch in range(10):
            for batch_idx, (data, target) in enumerate(train_loader):
                correct = 0
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                pred = output.data.max(1, keepdim = True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()

                wandb.log({"Training Loss": loss.item(), "Training Accuracy": correct / len(data)})
                if batch_idx % 100 == 0:
                    #print train loss and accurcy
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(), correct, len(target), 100. * correct / len(target)))

def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim = True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    wandb.log({"Test Loss": test_loss, "Test Accuracy": correct / len(test_loader.dataset)})
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project='cifar10-cnn')
    wandb.agent(sweep_id, train, count=5)