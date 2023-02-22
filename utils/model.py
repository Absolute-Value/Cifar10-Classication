import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
  def __init__(self, in_feature, out_feature):
    super(EncoderBlock, self).__init__()
    
    self.conv = nn.Conv2d(in_feature, out_feature, 3, 1, 1)
    self.bn = nn.BatchNorm2d(out_feature)

  def forward(self, x):
    return F.relu(self.bn(self.conv(x)))
    

class CNN(nn.Module):
  def __init__(self, dim=64):
    super(CNN, self).__init__()
    self.dim = dim

    self.enc1 = EncoderBlock(3, dim)
    self.enc2 = EncoderBlock(dim, dim)
    self.enc3 = EncoderBlock(dim, dim)
    self.enc4 = EncoderBlock(dim, dim*2)
    self.enc5 = EncoderBlock(dim*2, dim*2)
    self.enc6 = EncoderBlock(dim*2, dim*2)
    self.enc7 = EncoderBlock(dim*2, dim*4)
    self.enc8 = EncoderBlock(dim*4, dim*4)
    self.enc9 = EncoderBlock(dim*4, dim*4)

    self.pool = nn.MaxPool2d(2)
    
    self.fc1 = nn.Linear(dim * 64, 512)
    self.fc2 = nn.Linear(512, 128)
    self.fc3 = nn.Linear(128, 10)

  def forward(self, x): # 32 * 32 * 3
    x = self.enc3(self.enc2(self.enc1(x))) # 32 * 32 * dim
    x = self.pool(x) # 16 * 16 * dim
    x = self.enc6(self.enc5(self.enc4(x))) # 16 *  16 * dim * 2
    x = self.pool(x) # 8 * 8 * dim * 2
    x = self.enc9(self.enc8(self.enc7(x))) # 8 * 8 * dim * 4
    x = self.pool(x) # 4 * 4 * dim * 4
    x = x.view(-1, self.dim * 64)

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x
