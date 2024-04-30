import os

for w_load in range(3, 9):
    print(w_load)
    os.system(f'python train_script.py {w_load}')