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

edges_list_6 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (8, 18),
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 23),
    (23, 24),
]

edges_list_7 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (3, 5),
    (4, 6),
    (5, 7),
    (6, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (7, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (17, 13),
    (18, 19),
    (13, 18),
]

edges_list_8 = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (4, 5),
    (5, 6),
    (3, 7),
    (7, 8),
    (7, 9),
    (8, 6),
    (9, 10),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (10, 16),
    (16, 17),
    (17, 15),
    (18, 19),
    (6, 11),
    (15, 18),
]

args_NIRtoDynapcnnNetwork_edges_list = [
    (EXAMPLE_1(), edges_list_1),
    (EXAMPLE_2(), edges_list_2),
    (EXAMPLE_3(), edges_list_3),
    (EXAMPLE_4(), edges_list_4),
    (EXAMPLE_5(), edges_list_5),
    (EXAMPLE_6(), edges_list_6),
    (EXAMPLE_7(), edges_list_7),
    (EXAMPLE_8(), edges_list_8),
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

nodes_IO_6 = {
    0: {'in': torch.Size([1, 2, 34, 34]), 'out': torch.Size([1, 4, 33, 33])},
    1: {'in': torch.Size([1, 4, 33, 33]), 'out': torch.Size([1, 4, 33, 33])},
    2: {'in': torch.Size([1, 4, 33, 33]), 'out': torch.Size([1, 4, 32, 32])},
    3: {'in': torch.Size([1, 4, 32, 32]), 'out': torch.Size([1, 4, 32, 32])},
    4: {'in': torch.Size([1, 4, 32, 32]), 'out': torch.Size([1, 4, 16, 16])},
    5: {'in': torch.Size([1, 4, 16, 16]), 'out': torch.Size([1, 4, 15, 15])},
    6: {'in': torch.Size([1, 4, 15, 15]), 'out': torch.Size([1, 4, 15, 15])},
    7: {'in': torch.Size([1, 4, 15, 15]), 'out': torch.Size([1, 4, 7, 7])},
    17: {'in':  torch.Size([1, 196]), 'out': torch.Size([1, 200])},
    18: {'in':  torch.Size([1, 200]), 'out': torch.Size([1, 200])},
    8: {'in': torch.Size([1, 2, 34, 34]), 'out': torch.Size([1, 4, 33, 33])},
    9: {'in': torch.Size([1, 4, 33, 33]), 'out': torch.Size([1, 4, 33, 33])},
    10: {'in':  torch.Size([1, 4, 33, 33]), 'out': torch.Size([1, 4, 32, 32])},
    11: {'in':  torch.Size([1, 4, 32, 32]), 'out': torch.Size([1, 4, 32, 32])},
    12: {'in':  torch.Size([1, 4, 32, 32]), 'out': torch.Size([1, 4, 16, 16])},
    13: {'in':  torch.Size([1, 4, 16, 16]), 'out': torch.Size([1, 4, 15, 15])},
    14: {'in':  torch.Size([1, 4, 15, 15]), 'out': torch.Size([1, 4, 15, 15])},
    15: {'in':  torch.Size([1, 4, 15, 15]), 'out': torch.Size([1, 4, 7, 7])},
    19: {'in':  torch.Size([1, 200]), 'out': torch.Size([1, 200])},
    20: {'in':  torch.Size([1, 200]), 'out': torch.Size([1, 200])},
    21: {'in':  torch.Size([1, 200]), 'out': torch.Size([1, 11])},
    22: {'in':  torch.Size([1, 11]), 'out': torch.Size([1, 11])},
}

nodes_IO_7 = {
    0: {'in': torch.Size([1, 2, 34, 34]), 'out': torch.Size([1, 4, 33, 33])},
    1: {'in': torch.Size([1, 4, 33, 33]), 'out': torch.Size([1, 4, 33, 33])},
    2: {'in': torch.Size([1, 4, 33, 33]), 'out': torch.Size([1, 4, 32, 32])},
    3: {'in': torch.Size([1, 4, 32, 32]), 'out': torch.Size([1, 4, 32, 32])},
    4: {'in': torch.Size([1, 4, 32, 32]), 'out': torch.Size([1, 4, 16, 16])},
    5: {'in': torch.Size([1, 4, 32, 32]), 'out': torch.Size([1, 4, 6, 6])},
    6: {'in': torch.Size([1, 4, 16, 16]), 'out': torch.Size([1, 4, 15, 15])},
    7: {'in': torch.Size([1, 4, 15, 15]), 'out': torch.Size([1, 4, 15, 15])},
    8: {'in': torch.Size([1, 4, 15, 15]), 'out': torch.Size([1, 4, 7, 7])},
    12: {'in':  torch.Size([1, 144]), 'out': torch.Size([1, 144])},
    13: {'in':  torch.Size([1, 144]), 'out': torch.Size([1, 144])},
    9: {'in': torch.Size([1, 4, 7, 7]), 'out': torch.Size([1, 4, 6, 6])},
    10: {'in':  torch.Size([1, 4, 6, 6]), 'out': torch.Size([1, 4, 6, 6])},
    16: {'in':  torch.Size([1, 144]), 'out': torch.Size([1, 11])},
    17: {'in':  torch.Size([1, 11]), 'out': torch.Size([1, 11])},
    14: {'in':  torch.Size([1, 144]), 'out': torch.Size([1, 144])},
    15: {'in':  torch.Size([1, 144]), 'out': torch.Size([1, 144])},
}

nodes_IO_8 = {
    0: {'in': torch.Size([1, 2, 34, 34]), 'out': torch.Size([1, 8, 33, 33])},
    1: {'in': torch.Size([1, 8, 33, 33]), 'out': torch.Size([1, 8, 33, 33])},
    2: {'in': torch.Size([1, 8, 33, 33]), 'out': torch.Size([1, 8, 32, 32])},
    4: {'in': torch.Size([1, 8, 32, 32]), 'out': torch.Size([1, 8, 32, 32])},
    5: {'in': torch.Size([1, 8, 32, 32]), 'out': torch.Size([1, 8, 16, 16])},
    3: {'in': torch.Size([1, 8, 33, 33]), 'out': torch.Size([1, 8, 32, 32])},
    7: {'in': torch.Size([1, 8, 32, 32]), 'out': torch.Size([1, 8, 32, 32])},
    8: {'in': torch.Size([1, 8, 32, 32]), 'out': torch.Size([1, 8, 16, 16])},
    9: {'in': torch.Size([1, 8, 32, 32]), 'out': torch.Size([1, 8, 5, 5])},
    10: {'in':  torch.Size([1, 8, 16, 16]), 'out': torch.Size([1, 8, 15, 15])},
    11: {'in':  torch.Size([1, 8, 15, 15]), 'out': torch.Size([1, 8, 15, 15])},
    12: {'in':  torch.Size([1, 8, 15, 15]), 'out': torch.Size([1, 8, 5, 5])},
    14: {'in':  torch.Size([1, 200]), 'out': torch.Size([1, 200])},
    15: {'in':  torch.Size([1, 200]), 'out': torch.Size([1, 200])},
    16: {'in':  torch.Size([1, 200]), 'out': torch.Size([1, 11])},
    17: {'in':  torch.Size([1, 11]), 'out': torch.Size([1, 11])},
}

args_test_NIRtoDynapcnnNetwork_IO = [
    (EXAMPLE_1(), nodes_IO_1),
    (EXAMPLE_2(), nodes_IO_2),
    (EXAMPLE_3(), nodes_IO_3),
    (EXAMPLE_4(), nodes_IO_4),
    (EXAMPLE_5(), nodes_IO_5),
    (EXAMPLE_6(), nodes_IO_6),
    (EXAMPLE_7(), nodes_IO_7),
    (EXAMPLE_8(), nodes_IO_8),
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

dcnnl_edges_list_6 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 3),
    (7, 8),
]

dcnnl_edges_list_7 = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 6),
    (4, 5),
    (6, 5),
]

dcnnl_edges_list_8 = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3),
    (2, 4),
    (3, 5),
    (4, 5),
]

args_DynapcnnLyers_edges_list = [
    (EXAMPLE_1(), dcnnl_edges_list_1),
    (EXAMPLE_2(), dcnnl_edges_list_2),
    (EXAMPLE_3(), dcnnl_edges_list_3),
    (EXAMPLE_4(), dcnnl_edges_list_4),
    (EXAMPLE_5(), dcnnl_edges_list_5),
    (EXAMPLE_6(), dcnnl_edges_list_6),
    (EXAMPLE_7(), dcnnl_edges_list_7),
    (EXAMPLE_8(), dcnnl_edges_list_8),
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

forward_edges_list_6 = [
    (0, 1),
    (1, 2),
    ((2, 6), 'merge_0'),
    ('merge_0', 3),
    (3, 7),
    (4, 5),
    (5, 6),
    (7, 8),
]

forward_edges_list_7 = [
    (1, '1_pool0'),
    (1, '1_pool1'),
    ('1_pool0', 2),
    ('1_pool1', 3),
    (2, 4),
    (3, 6),
    ((4, 6), 'merge_0'),
    ('merge_0', 5),
]

forward_edges_list_8 = [
    (0, 1),
    (2, '2_pool0'),
    (2, '2_pool1'),
    (('2_pool0', 1), 'merge_0'),
    ('merge_0', 3),
    ('2_pool1', 4),
    ((3, 4), 'merge_1'),
    ('merge_1', 5),
]

args_DynapcnnNetwork_forward_edges = [
    (EXAMPLE_1(), forward_edges_list_1),
    (EXAMPLE_2(), forward_edges_list_2),
    (EXAMPLE_3(), forward_edges_list_3),
    (EXAMPLE_4(), forward_edges_list_4),
    (EXAMPLE_5(), forward_edges_list_5),
    (EXAMPLE_6(), forward_edges_list_6),
    (EXAMPLE_7(), forward_edges_list_7),
    (EXAMPLE_8(), forward_edges_list_8),
]