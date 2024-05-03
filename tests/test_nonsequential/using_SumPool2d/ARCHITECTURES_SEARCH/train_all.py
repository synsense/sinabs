import os

for i in range(1, 9):
    os.system(f'python model_training.py ResSCNN{i}')