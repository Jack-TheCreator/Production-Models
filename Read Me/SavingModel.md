```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import modelHandler
import pandas as pd
import numpy as np
from redis import Redis
import CustomExceptions
import matplotlib.pyplot as plt
import seaborn as sb
from numpy import mean
import pymongo
from pymongo import MongoClient
```

    Using TensorFlow backend.



```python

```


```python
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

    Files already downloaded and verified
    Files already downloaded and verified



```python
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
```


```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```


```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

    [1,  2000] loss: 2.236
    [1,  4000] loss: 1.842
    [1,  6000] loss: 1.647
    [1,  8000] loss: 1.548
    [1, 10000] loss: 1.521
    [1, 12000] loss: 1.461
    [2,  2000] loss: 1.389
    [2,  4000] loss: 1.367
    [2,  6000] loss: 1.351
    [2,  8000] loss: 1.325
    [2, 10000] loss: 1.316
    [2, 12000] loss: 1.297
    Finished Training



```python
dataiter = iter(testloader)
images, labels = dataiter.next()
net(images)
```




    tensor([[-0.2776, -2.1757,  1.0908,  2.4734, -0.0453,  2.4037,  0.1717, -0.2119,
             -1.3428, -1.4537],
            [ 3.6400,  7.3081, -1.8447, -3.6902, -3.3091, -5.1221, -3.9437, -5.5507,
              7.4658,  3.0754],
            [ 0.5176,  2.8582, -0.3932, -0.6597, -1.7371, -1.3914, -1.0855, -2.0401,
              1.7379,  1.7389],
            [ 2.7278,  0.4692,  1.2562, -1.3368,  0.6833, -2.6366, -0.7195, -3.4770,
              3.5714, -0.0649]], grad_fn=<AddmmBackward>)




```python
redisconnection = Redis()
handler = modelHandler.ModelHandler(redisconnection)
```


```python
features = {'optimizer':optimizer, 'criterion':criterion}
handler.save_model('testKey', net, optimizer, 20, features)
```




    True




```python
loadedModel, loadedOptimizer, loadedFeatures = handler.load_latest_model('testKey')
```


```python
model = Net()
model.load_state_dict(loadedModel)
```




    <All keys matched successfully>




```python
optimizer = optim.SGD()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-14-6cf4cb81bcf9> in <module>
    ----> 1 optimizer = optim.SGD()
    

    TypeError: __init__() missing 1 required positional argument: 'params'



```python

```
