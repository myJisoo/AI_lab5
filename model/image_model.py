import torch.nn as nn


class image_model(nn.Module):
    def __init__(self, img_hidden_size, num_classes):
        super(image_model, self).__init__()
        self.fc = nn.Linear(img_hidden_size, num_classes)

    def forward(self, x):
        return self.fc(x)
