from copy import deepcopy


def test_iaf_deepcopy():
    from sinabs.layers import IAF, IAFSqueeze
    lyr = IAF()
    lyr_copy = deepcopy(lyr)

    lyr_sqz = IAFSqueeze(batch_size=1)
    lyr_sqz_copy = deepcopy(lyr_sqz)

def test_lif_deepcopy():
    from sinabs.layers import LIF, LIFSqueeze
    lyr = LIF()
    lyr_copy = deepcopy(lyr)

    lyr_sqz = LIFSqueeze(batch_size=1)
    lyr_sqz_copy = deepcopy(lyr_sqz)

def test_alif_deepcopy():
    raise NotImplementedError

