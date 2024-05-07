from architectures_samples import *

# --- test_NIRtoDynapcnnNetwork_edges_list(snn, edges_list) ---

edges_list_1 = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (6, 5),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (5, 7),
]

edges_list_2 = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (6, 5),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (5, 7),
]

edges_list_3 = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (6, 5),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (11, 13),
    (12, 14),
    (14, 13),
    (15, 16),
    (5, 7),
    (13, 15),
]

edges_list_5 = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (6, 5),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (13, 15),
    (14, 16),
    (16, 15),
    (16, 17),
    (18, 19),
    (19, 17),
    (20, 21),
    (5, 7),
    (15, 18),
    (17, 20),
]

edges_list_4 = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (6, 5),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (13, 15),
    (14, 16),
    (16, 15),
    (17, 18),
    (5, 7),
    (15, 17),
]

args_NIRtoDynapcnnNetwork_edges_list = [
    (EXAMPLE_1(), edges_list_1),
    (EXAMPLE_2(), edges_list_2),
    (EXAMPLE_3(), edges_list_3),
    (EXAMPLE_4(), edges_list_4),
    (EXAMPLE_5(), edges_list_5),
    ]

# --- test_NIRtoDynapcnnNetwork_IO(snn, io_dict) ---

nodes_IO_1 = {
    0: {'in': torch.Size([1, 2, 34, 34]), 'out': torch.Size([1, 10, 33, 33])},
    1: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 33, 33])},
    2: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 11, 11])},
    3: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 8, 8])},
    4: {'in': torch.Size([1, 10, 11, 11]), 'out': torch.Size([1, 10, 8, 8])},
    6: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 10, 8, 8])},
    7: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 1, 7, 7])},
    8: {'in': torch.Size([1, 1, 7, 7]), 'out': torch.Size([1, 1, 7, 7])},
    10: {'in': torch.Size([1, 49]), 'out': torch.Size([1, 500])},
    11: {'in': torch.Size([1, 500]), 'out': torch.Size([1, 500])},
    12: {'in': torch.Size([1, 500]), 'out': torch.Size([1, 10])},
    13: {'in': torch.Size([1, 10]), 'out': torch.Size([1, 10])},
}

nodes_IO_2 = {
    0: {'in': torch.Size([1, 2, 34, 34]), 'out': torch.Size([1, 10, 33, 33])},
    1: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 33, 33])},
    2: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 11, 11])},
    3: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 8, 8])},
    4: {'in': torch.Size([1, 10, 11, 11]), 'out': torch.Size([1, 10, 8, 8])},
    6: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 10, 8, 8])},
    7: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 1, 7, 7])},
    8: {'in': torch.Size([1, 1, 7, 7]), 'out': torch.Size([1, 1, 7, 7])},
    10: {'in': torch.Size([1, 49]), 'out': torch.Size([1, 100])},
    11: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    12: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    13: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    14: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 10])},
    15: {'in': torch.Size([1, 10]), 'out': torch.Size([1, 10])},
}

nodes_IO_3 = {    
    0: {'in': torch.Size([1, 2, 34, 34]), 'out': torch.Size([1, 10, 33, 33])},
    1: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 33, 33])},
    2: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 11, 11])},
    3: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 8, 8])},
    4: {'in': torch.Size([1, 10, 11, 11]), 'out': torch.Size([1, 10, 8, 8])},
    6: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 10, 8, 8])},
    7: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 1, 7, 7])},
    8: {'in': torch.Size([1, 1, 7, 7]), 'out': torch.Size([1, 1, 7, 7])},
    10: {'in': torch.Size([1, 49]), 'out': torch.Size([1, 100])},
    11: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    12: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    14: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    15: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 10])},
    16: {'in': torch.Size([1, 10]), 'out': torch.Size([1, 10])},
}

nodes_IO_4 = {
    0: {'in': torch.Size([1, 2, 34, 34]), 'out': torch.Size([1, 10, 33, 33])},
    1: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 33, 33])},
    2: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 11, 11])},
    3: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 8, 8])},
    4: {'in': torch.Size([1, 10, 11, 11]), 'out': torch.Size([1, 10, 8, 8])},
    6: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 10, 8, 8])},
    7: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 10, 7, 7])},
    8: {'in': torch.Size([1, 10, 7, 7]), 'out': torch.Size([1, 10, 7, 7])},
    9: {'in': torch.Size([1, 10, 7, 7]), 'out': torch.Size([1, 1, 6, 6])},
    10: {'in': torch.Size([1, 1, 6, 6]), 'out': torch.Size([1, 1, 6, 6])},
    12: {'in': torch.Size([1, 36]), 'out': torch.Size([1, 100])},
    13: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    14: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    16: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    17: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 10])},
    18: {'in': torch.Size([1, 10]), 'out': torch.Size([1, 10])},
}

nodes_IO_5 = {    
    0: {'in': torch.Size([1, 2, 34, 34]), 'out': torch.Size([1, 10, 33, 33])},
    1: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 33, 33])},
    2: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 11, 11])},
    3: {'in': torch.Size([1, 10, 33, 33]), 'out': torch.Size([1, 10, 8, 8])},
    4: {'in': torch.Size([1, 10, 11, 11]), 'out': torch.Size([1, 10, 8, 8])},
    6: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 10, 8, 8])},
    7: {'in': torch.Size([1, 10, 8, 8]), 'out': torch.Size([1, 10, 7, 7])},
    8: {'in': torch.Size([1, 10, 7, 7]), 'out': torch.Size([1, 10, 7, 7])},
    9: {'in': torch.Size([1, 10, 7, 7]), 'out': torch.Size([1, 1, 6, 6])},
    10: {'in': torch.Size([1, 1, 6, 6]), 'out': torch.Size([1, 1, 6, 6])},
    12: {'in': torch.Size([1, 36]), 'out': torch.Size([1, 100])},
    13: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    14: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    16: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    18: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    19: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 100])},
    20: {'in': torch.Size([1, 100]), 'out': torch.Size([1, 10])},
    21: {'in': torch.Size([1, 10]), 'out': torch.Size([1, 10])},
}

args_test_NIRtoDynapcnnNetwork_IO = [
    (EXAMPLE_1(), nodes_IO_1),
    (EXAMPLE_2(), nodes_IO_2),
    (EXAMPLE_3(), nodes_IO_3),
    (EXAMPLE_4(), nodes_IO_4),
    (EXAMPLE_5(), nodes_IO_5),
]

# --- test_DynapcnnLyers_edges_list(snn, edges_list) ---

dcnnl_edges_list_1 = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
]

dcnnl_edges_list_2 = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
]

dcnnl_edges_list_3 = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
    (3, 5),
    (4, 5),
]

dcnnl_edges_list_4 = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (4, 6),
    (5, 6),
]

dcnnl_edges_list_5 = [
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (6, 7),
]

args_DynapcnnLyers_edges_list = [
    (EXAMPLE_1(), dcnnl_edges_list_1),
    (EXAMPLE_2(), dcnnl_edges_list_2),
    (EXAMPLE_3(), dcnnl_edges_list_3),
    (EXAMPLE_4(), dcnnl_edges_list_4),
    (EXAMPLE_5(), dcnnl_edges_list_5),
]

# --- test_DynapcnnNetwork_forward_edges(snn, forward_edges_list) ---

forward_edges_list_1 = [
    (0, '0_pool0'),
    (0, '0_pool1'),
    ('0_pool0', 1),
    (('0_pool1', 1), 'merge_0'),
    ('merge_0', 2),
    (2, 3),
    (3, 4),
]

forward_edges_list_2 = [
    (0, '0_pool0'),
    (0, '0_pool1'),
    ('0_pool0', 1),
    (('0_pool1', 1), 'merge_0'),
    ('merge_0', 2),
    (2, 3),
    (3, 4),
    (4, 5),
]

forward_edges_list_3 = [
    (0, '0_pool0'),
    (0, '0_pool1'),
    ('0_pool0', 1),
    (('0_pool1', 1), 'merge_0'),
    ('merge_0', 2),
    (2, 3),
    (3, 4),
    ((3, 4), 'merge_1'),
    ('merge_1', 5),
]

forward_edges_list_4 = [
    (0, '0_pool0'),
    (0, '0_pool1'),
    ('0_pool0', 1),
    (('0_pool1', 1), 'merge_0'),
    ('merge_0', 2),
    (2, 3),
    (3, 4),
    (4, 5),
    ((4, 5), 'merge_1'),
    ('merge_1', 6),
]

forward_edges_list_5 = [
    (0, '0_pool0'),
    (0, '0_pool1'),
    ('0_pool0', 1),
    (('0_pool1', 1), 'merge_0'),
    ('merge_0', 2),
    (2, 3),
    (3, 4),
    (4, 5),
    ((4, 5), 'merge_1'),
    ('merge_1', 6),
    ((5, 6), 'merge_2'),
    ('merge_2', 7),
]

args_DynapcnnNetwork_forward_edges = [
    (EXAMPLE_1(), forward_edges_list_1),
    (EXAMPLE_2(), forward_edges_list_2),
    (EXAMPLE_3(), forward_edges_list_3),
    (EXAMPLE_4(), forward_edges_list_4),
    (EXAMPLE_5(), forward_edges_list_5),
]