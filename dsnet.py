import torch
import torch.nn as nn
import torch.nn.functional as F


class DSBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DSBlock, self).__init__()
        assert in_channels==out_channels,"in_channels should be equaled to out_channels"
        self.channel_wise_weight= nn.Conv2d(in_channels, out_channels , kernel_size=1, stride=1,padding=0, bias=False,groups=in_channels)
        self.batchnorm=nn.BatchNorm2d(in_channels)
    def forward(self,x):
        x=self.batchnorm(x)
        x=self.channel_wise_weight(x)
        return x

class BasicBlock(nn.Module):
    """Basic Block for dsnet 18 and dsnet 34
    """
    def __init__(self, in_channels, out_channels, stride=1,expansion=1):
        super().__init__()
        self.expansion=expansion
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut=nn.Identity()
    def forward(self, x):
        residual_output=self.residual_function(x)
        shortcut_output=self.shortcut(x)
        return residual_output,shortcut_output


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,expansion=4):
        super().__init__()
        self.expansion = expansion
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut=nn.Identity()
    def forward(self, x):
        residual_output=self.residual_function(x)
        shortcut_output=self.shortcut(x)
        return residual_output,shortcut_output

class DSNet(nn.Module):
    def __init__(self, block, num_block, expansion=4,num_classes=100,version="v1"):
        super().__init__()
        self.in_channels = 64
        self.expansion=expansion
        self.num_block=num_block
        self.num_classes=num_classes
        self.version=version
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x,self.ds2_x = self._make_layer(block, 64, num_block[0], 1,expansion,version)
        self.conv3_x,self.ds3_x = self._make_layer(block, 128, num_block[1], 2,expansion,version)
        self.conv4_x,self.ds4_x = self._make_layer(block, 256, num_block[2], 2,expansion,version)
        self.conv5_x,self.ds5_x = self._make_layer(block, 512, num_block[3], 2,expansion,version)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)
    def _make_layer(self, block, out_channels, num_blocks, stride,expansion,version):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        dslayers=[]
        for i,stride in enumerate(strides):
            layers.append(block(self.in_channels, out_channels, stride,expansion))
            self.in_channels = out_channels * expansion
            ds=nn.ModuleList([DSBlock(out_channels*expansion,out_channels*expansion) for j in range(i+1)])
            dslayers.append(ds)
        return nn.ModuleList(layers),nn.ModuleList(dslayers)
    def _accumulate_ds_output(self,ress,dslayers):
        output=0
        for res,dslayer in zip(ress,dslayers):
            output+=dslayer(res)
        return output
    def _layer_forward(self,x,layers,dslayers):
        """
        x_1->block_1->x_2->block_2->x_3
        """
        length=len(layers)
        ress=[]
        for i in range(length):
            if i==0:
                y,res=layers[i](x)
                if self.version=='v2':
                    ress.append(res)
                    now_res = self._accumulate_ds_output(ress, dslayers[i])
                elif self.version=='v1':
                    ress.append(res)
                    now_res = self._accumulate_ds_output(ress, dslayers[i])
                    ress.append(y)
                else:
                    raise NotImplementedError
                x=F.relu(y+now_res)
            else:
                y,res=layers[i](x)
                if self.version=='v2':
                    ress.append(res)
                    now_res = self._accumulate_ds_output(ress, dslayers[i])
                elif self.version=='v1':
                    now_res = self._accumulate_ds_output(ress, dslayers[i])
                    ress.append(y)
                else:
                    raise NotImplementedError
                x=F.relu(y+now_res)
        return x
    def forward(self,x):
        output = self.conv1(x)
        output = self._layer_forward(output,self.conv2_x,self.ds2_x)
        output = self._layer_forward(output,self.conv3_x,self.ds3_x)
        output = self._layer_forward(output,self.conv4_x,self.ds4_x)
        output = self._layer_forward(output,self.conv5_x,self.ds5_x)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def dsnet18(expansion=4,num_classes=10,version='v1'):
    """ return a DSNet 18 object
    """
    return DSNet(BasicBlock, [2, 2, 2, 2],expansion=expansion,num_classes=num_classes,version=version)

def dsnet34(expansion=4,num_classes=10,version='v1'):
    """ return a DSNet 34 object
    """
    return DSNet(BasicBlock, [3, 4, 6, 3],expansion=expansion,num_classes=num_classes,version=version)

def dsnet50(expansion=4,num_classes=10,version='v1'):
    """ return a DSNet 50 object
    """
    return DSNet(BottleNeck, [3, 4, 6, 3],expansion=expansion,num_classes=num_classes,version=version)

def dsnet101(expansion=4,num_classes=10,version='v1'):
    """ return a DSNet 101 object
    """
    return DSNet(BottleNeck, [3, 4, 23, 3],expansion=expansion,num_classes=num_classes,version=version)

def dsnet152(expansion=4,num_classes=10,version='v1'):
    """ return a DSNet 152 object
    """
    return DSNet(BottleNeck, [3, 8, 36, 3],expansion=expansion,num_classes=num_classes,version=version)


