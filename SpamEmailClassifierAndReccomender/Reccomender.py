import torch
import torch.nn as nn
import torch.functional as F
import pandas as pd
import matplotlib.pyplot as plt

MOVIES_METADATA = pd.read_csv("ArhturkasProjects/SpamEmailClassifierAndReccomender/movies_metadata.csv")
RATINGS = pd.read_csv("ArhturkasProjects/SpamEmailClassifierAndReccomender/ratings_small.csv")

print(MOVIES_METADATA[:1])
print("yay")
print(RATINGS[:1])