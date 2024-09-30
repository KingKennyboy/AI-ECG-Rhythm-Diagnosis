import torch

try:
    # Attempt to load the checkpoint file
    checkpoint = torch.load('trainingModel/timesnet_diff_method/checkpoints1/final_checkpoint1.pth', map_location='cpu')
    print("Checkpoint keys:", checkpoint.keys())  # This will show the keys of the saved dictionary
except Exception as e:
    print("Error loading checkpoint:", e)


import os

file_path = 'trainingModel/timesnet_diff_method/checkpoints1/final_checkpoint1.pth'
file_size = os.path.getsize(file_path)
print("File size:", file_size, "bytes")