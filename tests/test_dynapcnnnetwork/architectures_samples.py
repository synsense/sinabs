import torch
import torch.nn as nn
from sinabs.layers import Merge, IAFSqueeze

class EXAMPLE_1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False) 
        self.iaf1 = IAFSqueeze(batch_size=1)            
        self.pool1 = nn.AvgPool2d(3,3)                  
        self.pool1a = nn.AvgPool2d(4,4)                 

        self.conv2 = nn.Conv2d(10, 10, 4, 1, bias=False)
        self.iaf2 = IAFSqueeze(batch_size=1)            

        self.conv3 = nn.Conv2d(10, 1, 2, 1, bias=False) 
        self.iaf3 = IAFSqueeze(batch_size=1)            

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(49, 500, bias=False)       
        self.iaf4 = IAFSqueeze(batch_size=1)            
        
        self.fc2 = nn.Linear(500, 10, bias=False)       
        self.iaf5 = IAFSqueeze(batch_size=1)            

        self.adder = Merge()

    def forward(self, x):
        
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)
        pool1_out = self.pool1(iaf1_out)
        pool1a_out = self.pool1a(iaf1_out)

        conv2_out = self.conv2(pool1_out)
        iaf2_out = self.iaf2(conv2_out)

        conv3_out = self.conv3(self.adder(pool1a_out, iaf2_out))
        iaf3_out = self.iaf3(conv3_out)

        flat_out = self.flat(iaf3_out)
        
        fc1_out = self.fc1(flat_out)
        iaf4_out = self.iaf4(fc1_out)
        fc2_out = self.fc2(iaf4_out)
        iaf5_out = self.iaf5(fc2_out)

        return iaf5_out
    
class EXAMPLE_2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False)
        self.conv1_iaf = IAFSqueeze(batch_size=1)
        self.pool1 = nn.AvgPool2d(3,3)
        self.pool1a = nn.AvgPool2d(4,4)

        self.conv2 = nn.Conv2d(10, 10, 4, 1, bias=False)
        self.conv2_iaf = IAFSqueeze(batch_size=1)

        self.conv3 = nn.Conv2d(10, 1, 2, 1, bias=False)
        self.conv3_iaf = IAFSqueeze(batch_size=1)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(49, 100, bias=False)
        self.fc1_iaf = IAFSqueeze(batch_size=1)
        
        self.fc2 = nn.Linear(100, 100, bias=False)
        self.fc2_iaf = IAFSqueeze(batch_size=1)

        self.fc3 = nn.Linear(100, 10, bias=False)
        self.fc3_iaf = IAFSqueeze(batch_size=1)

        self.merge1 = Merge()

    def forward(self, x):
        # -- conv. block 1 --
        con1_out = self.conv1(x)
        conv1_iaf_out = self.conv1_iaf(con1_out)
        pool1_out = self.pool1(conv1_iaf_out)
        pool1a_out = self.pool1a(conv1_iaf_out)
        # -- conv. block 2 --
        conv2_out = self.conv2(pool1_out)
        conv2_iaf_out = self.conv2_iaf(conv2_out)
        # -- conv. block 3 --
        merge1_out = self.merge1(pool1a_out, conv2_iaf_out)
        conv3_out = self.conv3(merge1_out)
        conv3_iaf_out = self.conv3_iaf(conv3_out)
        flat_out = self.flat(conv3_iaf_out)
        # -- fc clock 1 --
        fc1_out = self.fc1(flat_out)
        fc1_iaf_out = self.fc1_iaf(fc1_out)
        # -- fc clock 2 --
        fc2_out = self.fc2(fc1_iaf_out)
        fc2_iaf_out = self.fc2_iaf(fc2_out)
        # -- fc clock 3 --
        fc3_out = self.fc3(fc2_iaf_out)
        fc3_iaf_out = self.fc3_iaf(fc3_out)

        return fc3_iaf_out
    
class EXAMPLE_3(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False)
        self.conv1_iaf = IAFSqueeze(batch_size=1)
        self.pool1 = nn.AvgPool2d(3,3)
        self.pool1a = nn.AvgPool2d(4,4)

        self.conv2 = nn.Conv2d(10, 10, 4, 1, bias=False)
        self.conv2_iaf = IAFSqueeze(batch_size=1)

        self.conv3 = nn.Conv2d(10, 1, 2, 1, bias=False)
        self.conv3_iaf = IAFSqueeze(batch_size=1)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(49, 100, bias=False)
        self.fc1_iaf = IAFSqueeze(batch_size=1)
        
        self.fc2 = nn.Linear(100, 100, bias=False)
        self.fc2_iaf = IAFSqueeze(batch_size=1)

        self.fc3 = nn.Linear(100, 10, bias=False)
        self.fc3_iaf = IAFSqueeze(batch_size=1)

        self.merge1 = Merge()
        self.merge2 = Merge()

    def forward(self, x):
        # -- conv. block 0 --
        con1_out = self.conv1(x)
        conv1_iaf_out = self.conv1_iaf(con1_out)
        pool1_out = self.pool1(conv1_iaf_out)
        pool1a_out = self.pool1a(conv1_iaf_out)
        # -- conv. block 1 --
        conv2_out = self.conv2(pool1_out)
        conv2_iaf_out = self.conv2_iaf(conv2_out)
        # -- conv. block 2 --
        merge1_out = self.merge1(pool1a_out, conv2_iaf_out)
        conv3_out = self.conv3(merge1_out)
        conv3_iaf_out = self.conv3_iaf(conv3_out)
        flat_out = self.flat(conv3_iaf_out)
        # -- fc clock 3 --
        fc1_out = self.fc1(flat_out)
        fc1_iaf_out = self.fc1_iaf(fc1_out)
        # -- fc clock 4 --
        fc2_out = self.fc2(fc1_iaf_out)
        fc2_iaf_out = self.fc2_iaf(fc2_out)
        # -- fc clock 5 --
        merge2_out = self.merge2(fc1_iaf_out, fc2_iaf_out)
        fc3_out = self.fc3(merge2_out)
        fc3_iaf_out = self.fc3_iaf(fc3_out)

        return fc3_iaf_out

class EXAMPLE_5(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False)
        self.conv1_iaf = IAFSqueeze(batch_size=1)
        self.pool1 = nn.AvgPool2d(3,3)
        self.pool1a = nn.AvgPool2d(4,4)

        self.conv2 = nn.Conv2d(10, 10, 4, 1, bias=False)
        self.conv2_iaf = IAFSqueeze(batch_size=1)

        self.conv3 = nn.Conv2d(10, 10, 2, 1, bias=False)
        self.conv3_iaf = IAFSqueeze(batch_size=1)

        self.conv4 = nn.Conv2d(10, 1, 2, 1, bias=False)
        self.conv4_iaf = IAFSqueeze(batch_size=1)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(36, 100, bias=False)
        self.fc1_iaf = IAFSqueeze(batch_size=1)
        
        self.fc2 = nn.Linear(100, 100, bias=False)
        self.fc2_iaf = IAFSqueeze(batch_size=1)

        self.fc3 = nn.Linear(100, 100, bias=False)
        self.fc3_iaf = IAFSqueeze(batch_size=1)

        self.fc4 = nn.Linear(100, 10, bias=False)
        self.fc4_iaf = IAFSqueeze(batch_size=1)

        self.merge1 = Merge()
        self.merge2 = Merge()
        self.merge3 = Merge()

    def forward(self, x):
        # -- conv. block 0 --
        con1_out = self.conv1(x)
        conv1_iaf_out = self.conv1_iaf(con1_out)
        pool1_out = self.pool1(conv1_iaf_out)
        pool1a_out = self.pool1a(conv1_iaf_out)
        # -- conv. block 1 --
        conv2_out = self.conv2(pool1_out)
        conv2_iaf_out = self.conv2_iaf(conv2_out)
        # -- conv. block 2 --
        merge1_out = self.merge1(pool1a_out, conv2_iaf_out)
        conv3_out = self.conv3(merge1_out)
        conv3_iaf_out = self.conv3_iaf(conv3_out)
        # -- conv. block 3 --
        conv4_out = self.conv4(conv3_iaf_out)
        conv4_iaf_out = self.conv4_iaf(conv4_out)
        flat_out = self.flat(conv4_iaf_out)
        # -- fc clock 4 --
        fc1_out = self.fc1(flat_out)
        fc1_iaf_out = self.fc1_iaf(fc1_out)
        # -- fc clock 5 --
        fc2_out = self.fc2(fc1_iaf_out)
        fc2_iaf_out = self.fc2_iaf(fc2_out)
        # -- fc clock 6 --
        merge2_out = self.merge2(fc1_iaf_out, fc2_iaf_out)
        fc3_out = self.fc3(merge2_out)
        fc3_iaf_out = self.fc3_iaf(fc3_out)
        # -- fc clock 7 --
        merge3_out = self.merge3(fc2_iaf_out, fc3_iaf_out)
        fc4_out = self.fc4(merge3_out)
        fc4_iaf_out = self.fc4_iaf(fc4_out)

        return fc4_iaf_out
    
class EXAMPLE_4(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False)
        self.conv1_iaf = IAFSqueeze(batch_size=1)
        self.pool1 = nn.AvgPool2d(3,3)
        self.pool1a = nn.AvgPool2d(4,4)

        self.conv2 = nn.Conv2d(10, 10, 4, 1, bias=False)
        self.conv2_iaf = IAFSqueeze(batch_size=1)

        self.conv3 = nn.Conv2d(10, 10, 2, 1, bias=False)
        self.conv3_iaf = IAFSqueeze(batch_size=1)

        self.conv4 = nn.Conv2d(10, 1, 2, 1, bias=False)
        self.conv4_iaf = IAFSqueeze(batch_size=1)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(36, 100, bias=False)
        self.fc1_iaf = IAFSqueeze(batch_size=1)
        
        self.fc2 = nn.Linear(100, 100, bias=False)
        self.fc2_iaf = IAFSqueeze(batch_size=1)

        self.fc3 = nn.Linear(100, 10, bias=False)
        self.fc3_iaf = IAFSqueeze(batch_size=1)

        self.merge1 = Merge()
        self.merge2 = Merge()

    def forward(self, x):
        # -- conv. block 0 --
        con1_out = self.conv1(x)
        conv1_iaf_out = self.conv1_iaf(con1_out)
        pool1_out = self.pool1(conv1_iaf_out)
        pool1a_out = self.pool1a(conv1_iaf_out)
        # -- conv. block 1 --
        conv2_out = self.conv2(pool1_out)
        conv2_iaf_out = self.conv2_iaf(conv2_out)
        # -- conv. block 2 --
        merge1_out = self.merge1(pool1a_out, conv2_iaf_out)
        conv3_out = self.conv3(merge1_out)
        conv3_iaf_out = self.conv3_iaf(conv3_out)
        # -- conv. block 3 --
        conv4_out = self.conv4(conv3_iaf_out)
        conv4_iaf_out = self.conv4_iaf(conv4_out)
        flat_out = self.flat(conv4_iaf_out)
        # -- fc clock 4 --
        fc1_out = self.fc1(flat_out)
        fc1_iaf_out = self.fc1_iaf(fc1_out)
        # -- fc clock 5 --
        fc2_out = self.fc2(fc1_iaf_out)
        fc2_iaf_out = self.fc2_iaf(fc2_out)
        # -- fc clock 6 --
        merge2_out = self.merge2(fc1_iaf_out, fc2_iaf_out)
        fc3_out = self.fc3(merge2_out)
        fc3_iaf_out = self.fc3_iaf(fc3_out)

        return fc3_iaf_out