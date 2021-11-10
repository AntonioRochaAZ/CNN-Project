from main import *

class ConvNetBase(NetBase):
    """Base Class for the Convolutional Networks proposed in the HASYv2 article.
    This defines a general forward method that is the same for all of them.
    """
    def __init__(self, **kwargs):
        super(ConvNetBase, self).__init__(**kwargs)
        self.conv_net = None
        self.lin_net = None
        self.is_classifier = True

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        conv_out = self.conv_net(inputs)
        lin_out = self.lin_net(conv_out.view(batch_size, -1))
        return lin_out

class TwoLayer(ConvNetBase):
    """Two Layer Convolutional Neural Network"""

    def __init__(self, **kwargs):
        super(TwoLayer, self).__init__(**kwargs)
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2))
        )
        self.lin_net = nn.Sequential(
            nn.Linear(32 * 15 * 15, 369),
            nn.Softmax(dim=1)
        )

class ThreeLayer(ConvNetBase):
    """Three Layer Convolutional Neural Network"""

    def __init__(self, **kwargs):
        super(ThreeLayer, self).__init__(**kwargs)
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),
        )
        self.lin_net = nn.Sequential(
            nn.Linear(64 * 6 * 6, 369),
            nn.Softmax(dim=1)
        )

class FourLayer(ConvNetBase):
    """Four Layer Convolutional Neural Network"""

    def __init__(self, **kwargs):
        super(FourLayer, self).__init__(**kwargs)
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),
        )
        self.lin_net = nn.Sequential(
            nn.Linear(128 * 2 * 2, 369),
            nn.Softmax(dim=1)
        )

class TFCNN(ConvNetBase):
    """TF-CNN"""

    def __init__(self, **kwargs):
        super(TFCNN, self).__init__(**kwargs)
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(32, 64, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),
        )
        self.lin_net = nn.Sequential(
            nn.Linear(64 * 6 * 6, 1024),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(1024, 369),
            nn.Softmax(dim=1)
        )



