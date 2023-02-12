import torch


class MotionEstimationNet(torch.nn.Module):
    def __init__(self):
        super(MotionEstimationNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.conv4 = torch.nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.fc = torch.nn.Linear(256 * 4 * 4, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
