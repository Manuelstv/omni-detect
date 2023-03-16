# Installation
'
python3 -m venv SEMSEG_ENV;
source SEMSEG_ENV/bin/activate;
pip3 install pytorch, scipy, tensorflow
'

# Evaluation
'
python3 test_semseg_v3.py -d DATASET_PATH -m eval
'

# Spherical adaptation
Distortion-aware convolutions are located in "resnet.py" file from line 104 to 126 and are activated by the boolean parameter "spheactiv" for each layer. 
'
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, spheactiv=False):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2, spheactiv=True)
        # self.conv1 = conv3x3(3, 64, stride=2, spheactiv=False)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64, spheactiv=False)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128, spheactiv=False)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], spheactiv_block=False, speactiv_depth=10)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, spheactiv_block=False, speactiv_depth=10)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, spheactiv_block=False, speactiv_depth=10)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, spheactiv_block=False, speactiv_depth=10)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, spheactiv_block=False, speactiv_depth=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, spheactiv_block=True, speactiv_depth=0)
'
