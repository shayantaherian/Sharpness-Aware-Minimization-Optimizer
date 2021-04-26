import torch
import numpy as np
from smooth_crossentropy import smooth_crossentropy
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models, transforms
from SAM import SAM
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# choose the training and test datasets
train_data = datasets.CIFAR10('data', train=True,
                              download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features, 10)

    def forward(self, x):
        x = self.base(x)
        x = self.drop(x.view(-1, self.final.in_features))
        return self.final(x)

model = Model()


# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

rho = 0.05
momentum = 0.9
base_optimizer = torch.optim.SGD
weight_decay = 0.0005

optimizer = SAM(model.parameters(), base_optimizer, rho=rho, lr=0.001, momentum=momentum,
                weight_decay=weight_decay)
n_epochs = 30

valid_loss_min = np.Inf  # track change in validation loss

for epoch in range(1, n_epochs + 1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        #optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = smooth_crossentropy(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.mean().backward()
        # perform a single optimization step (parameter update)
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        smooth_crossentropy(model(data), target).mean().backward()
        optimizer.second_step(zero_grad=True)
        #optimizer.step()
        # update training loss
        train_loss += loss.mean().item() * data.size(0)


    ######################
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = smooth_crossentropy(output, target)
        # update average validation loss
        valid_loss += loss.mean().item() * data.size(0)

    # calculate average losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model_cifar_SAM.pt')
        valid_loss_min = valid_loss
        writer.add_scalar('training loss',
                          train_loss,
                          epoch * len(train_loader))
        writer.add_scalar('valid loss',
                          valid_loss,
                          epoch * len(valid_loader))

