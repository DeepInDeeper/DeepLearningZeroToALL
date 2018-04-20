#-*-coding:utf-8-*-

from torch import nn

class PyTorch_AlexNet(nn.Module):
    def __init__(self,num_classes,verbose=False):
        super(PyTorch_AlexNet, self).__init__()
        self.verbose = verbose

        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            # 第二层
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            # 第三层
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(),
            # 第四层
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(),
            # 第五层
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(6*6*256,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        if self.verbose:
            print(x.shape)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

