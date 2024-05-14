import torch
import torch.nn as nn
from sinabs.layers import Merge, IAFSqueeze, SumPool2d

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
    
class EXAMPLE_6(nn.Module):
    """ This is the 'two networks with merging outputs' example in https://github.com/synsense/sinabs/issues/181 . """
    def __init__(self) -> None:
        super().__init__()

        self.conv_A = nn.Conv2d(2, 4, 2, 1, bias=False)
        self.iaf_A = IAFSqueeze(batch_size=1)

        self.conv_B = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_B = IAFSqueeze(batch_size=1)
        self.pool_B = SumPool2d(2,2)

        self.conv_C = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_C = IAFSqueeze(batch_size=1)
        self.pool_C = SumPool2d(2,2)

        self.conv_D = nn.Conv2d(2, 4, 2, 1, bias=False)
        self.iaf_D = IAFSqueeze(batch_size=1)

        self.conv_E = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_E = IAFSqueeze(batch_size=1)
        self.pool_E = SumPool2d(2,2)

        self.conv_F = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_F = IAFSqueeze(batch_size=1)
        self.pool_F = SumPool2d(2,2)

        self.flat_brach1 = nn.Flatten()
        self.flat_brach2 = nn.Flatten()
        self.merge = Merge()

        self.fc1 = nn.Linear(196, 200, bias=False)
        self.iaf1_fc = IAFSqueeze(batch_size=1)

        self.fc2 = nn.Linear(200, 200, bias=False)
        self.iaf2_fc = IAFSqueeze(batch_size=1)

        self.fc3 = nn.Linear(200, 10, bias=False)
        self.iaf3_fc = IAFSqueeze(batch_size=1)

    def forward(self, x):
        # conv 1 - A
        conv_A_out = self.conv_A(x)
        iaf_A_out = self.iaf_A(conv_A_out)
        # conv 2 - B
        conv_B_out = self.conv_B(iaf_A_out)
        iaf_B_out = self.iaf_B(conv_B_out)
        pool_B_out = self.pool_B(iaf_B_out)
        # conv 3 - C
        conv_C_out = self.conv_C(pool_B_out)
        iaf_C_out = self.iaf_C(conv_C_out)
        pool_C_out = self.pool_C(iaf_C_out)

        # ---

        # conv 4 - D
        conv_D_out = self.conv_D(x)
        iaf_D_out = self.iaf_D(conv_D_out)
        # conv 5 - E
        conv_E_out = self.conv_E(iaf_D_out)
        iaf_E_out = self.iaf_E(conv_E_out)
        pool_E_out = self.pool_E(iaf_E_out)
        # conv 6 - F
        conv_F_out = self.conv_F(pool_E_out)
        iaf_F_out = self.iaf_F(conv_F_out)
        pool_F_out = self.pool_F(iaf_F_out)

        # ---

        flat_brach1_out = self.flat_brach1(pool_C_out)
        flat_brach2_out = self.flat_brach2(pool_F_out)
        merge_out = self.merge(flat_brach1_out, flat_brach2_out)

        # FC 7 - G
        fc1_out = self.fc1(merge_out)
        iaf1_fc_out = self.iaf1_fc(fc1_out)
        # FC 8 - H
        fc2_out = self.fc2(iaf1_fc_out)
        iaf2_fc_out = self.iaf2_fc(fc2_out)
        # FC 9 - I
        fc3_out = self.fc3(iaf2_fc_out)
        iaf3_fc_out = self.iaf3_fc(fc3_out)

        return iaf3_fc_out
    
class EXAMPLE_7(nn.Module):
    """ This is the 'a network with a merge and a split' example in https://github.com/synsense/sinabs/issues/181 . """
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 4, 2, 1, bias=False)
        self.iaf1 = IAFSqueeze(batch_size=1)

        self.conv2 = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf2 = IAFSqueeze(batch_size=1)
        self.pool2 = SumPool2d(2,2)
        self.pool2a = SumPool2d(5,5)

        self.conv3 = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf3 = IAFSqueeze(batch_size=1)
        self.pool3 = SumPool2d(2,2)

        self.conv4 = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf4 = IAFSqueeze(batch_size=1)

        self.flat = nn.Flatten()
        self.flat_a = nn.Flatten()

        self.fc1 = nn.Linear(144, 144, bias=False)
        self.iaf1_fc = IAFSqueeze(batch_size=1)

        self.fc2 = nn.Linear(144, 144, bias=False)
        self.iaf2_fc = IAFSqueeze(batch_size=1)

        self.fc3 = nn.Linear(144, 10, bias=False)
        self.iaf3_fc = IAFSqueeze(batch_size=1)

        # -- merges --
        self.merge1 = Merge()

    def forward(self, x):
        # conv 1 - A
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)

        # conv 2 - B
        conv2_out = self.conv2(iaf1_out)
        iaf2_out = self.iaf2(conv2_out)
        pool2_out = self.pool2(iaf2_out)
        pool2a_out = self.pool2a(iaf2_out)

        # conv 3 - C
        conv3_out = self.conv3(pool2_out)
        iaf3_out = self.iaf3(conv3_out)
        pool3_out = self.pool3(iaf3_out)

        # conv 4 - D
        conv4_out = self.conv4(pool3_out)
        iaf4_out = self.iaf4(conv4_out)
        flat_out = self.flat(iaf4_out)
        
        # fc 1 - E
        flat_a_out = self.flat_a(pool2a_out)
        fc1_out = self.fc1(flat_a_out)
        iaf1_fc_out = self.iaf1_fc(fc1_out)

        # fc 2 - F
        fc2_out = self.fc2(iaf1_fc_out)
        iaf2_fc_out = self.iaf2_fc(fc2_out)

        # fc 2 - G
        merge1_out = self.merge1(flat_out, iaf2_fc_out)
        fc3_out = self.fc3(merge1_out)
        iaf3_fc_out = self.iaf3_fc(fc3_out)

        return iaf3_fc_out
    
class EXAMPLE_8(nn.Module):
    """ This is the 'a complex network structure' example in https://github.com/synsense/sinabs/issues/181 . """
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 8, 2, 1, bias=False)
        self.iaf1 = IAFSqueeze(batch_size=1)

        self.conv2 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf2 = IAFSqueeze(batch_size=1)
        self.pool2 = SumPool2d(2,2)

        self.conv3 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf3 = IAFSqueeze(batch_size=1)
        self.pool3 = SumPool2d(2,2)
        self.pool3a = SumPool2d(6,6)

        self.conv4 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf4 = IAFSqueeze(batch_size=1)
        self.pool4 = SumPool2d(3,3)

        self.flat = nn.Flatten()
        self.flat_a = nn.Flatten()

        self.fc1 = nn.Linear(200, 200, bias=False)
        self.iaf1_fc = IAFSqueeze(batch_size=1)

        self.fc2 = nn.Linear(200, 10, bias=False)
        self.iaf2_fc = IAFSqueeze(batch_size=1)

        # -- merges --
        self.merge1 = Merge()
        self.merge2 = Merge()

    def forward(self, x):
        # conv 1 - A
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)

        # conv 2 - B
        conv2_out = self.conv2(iaf1_out)
        iaf2_out = self.iaf2(conv2_out)
        pool2_out = self.pool2(iaf2_out)

        # conv 3 - C
        conv3_out = self.conv3(iaf1_out)
        iaf3_out = self.iaf3(conv3_out)
        pool3_out = self.pool3(iaf3_out)
        pool3a_out = self.pool3a(iaf3_out)

        # conv 4 - D
        merge1_out = self.merge1(pool2_out, pool3_out)
        conv4_out = self.conv4(merge1_out)
        iaf4_out = self.iaf4(conv4_out)
        pool4_out = self.pool4(iaf4_out)
        flat_out = self.flat(pool4_out)
        
        # fc 1 - E
        flat_a_out = self.flat_a(pool3a_out)
        fc1_out = self.fc1(flat_a_out)
        iaf1_fc_out = self.iaf1_fc(fc1_out)

        # fc 2 - F
        merge2_out = self.merge2(iaf1_fc_out, flat_out)
        fc2_out = self.fc2(merge2_out)
        iaf2_fc_out = self.iaf2_fc(fc2_out)

        return iaf2_fc_out