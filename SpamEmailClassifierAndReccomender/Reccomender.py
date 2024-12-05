import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

#MOVIES_METADATA = pd.read_csv("ArhturkasProjects/SpamEmailClassifierAndReccomender/movies_metadata.csv")
RATINGS = pd.read_csv("ArhturkasProjects/SpamEmailClassifierAndReccomender/ratings_small.csv")


#print(MOVIES_METADATA["overview"].isnull().sum())
print("yay")
#print(RATINGS[["userId","movieId"]].iloc[2])
# collaborative filtering
num_people_ = RATINGS["userId"].nunique()
num_films_ = RATINGS["movieId"].max()

#hyperparametrs
batch_size = 16
emb_dim = 32
num_epochs = 3
lr =0.001



class RatigsDataset(Dataset):

    def __init__(self, ratings):
        super(RatigsDataset, self).__init__()
        self.xs = ratings[["userId", "movieId"]]
        self.ys = ratings["rating"]

    def __len__(self):
        return len(self.ys)
    
    def __getitem__(self, index):
        x = self.xs.iloc[index]
        y = self.ys.iloc[index]
        return torch.tensor(x, dtype=torch.float32),torch.tensor(y, dtype=torch.float32)


split1 = int(len(RATINGS) * 0.8)
split2 = int(len(RATINGS) * 0.9)

train_dataset = RatigsDataset(RATINGS[:split1])
dev_dataset = RatigsDataset(RATINGS[split1:split2])
val_dataset = RatigsDataset(RATINGS[split2:])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class CollaborativeRec(nn.Module):
    
    def __init__(self, emb_dim, num_people, num_films):
        super(CollaborativeRec, self).__init__()

        self.user_embed = nn.Embedding(num_people+1, emb_dim, dtype=torch.float32)
        self.movie_embed = nn.Embedding(num_films+1, emb_dim, dtype=torch.float32)

        self.fc = nn.Linear(emb_dim * 2, 1, dtype=torch.float32)
    
    def forward(self, x):
        user_id, movie_id = x[:, 0], x[:, 1]
        user_vec = self.user_embed(user_id.int())
        movie_vec = self.movie_embed(movie_id.int())
        conc = torch.cat((user_vec, movie_vec), dim=-1)
        #print(conc.dtype, " c")
        return F.relu(self.fc(conc))

model = CollaborativeRec(emb_dim, num_people_, num_films_)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
criterion = nn.MSELoss()

def train(num_epochs, split):
    model.train()
    loader = train_loader
    if split == "dev":
        loader = dev_loader
    
    for _ in range(num_epochs):
        runn_loss = []
        idx_loss = []
        for batch_id,(xs,ys) in enumerate(loader):

            optimizer.zero_grad()
            y_pred = model(xs)
            loss= criterion(y_pred, ys)
            loss.backward()
            optimizer.step()

            runn_loss.append(loss.item())
            idx_loss.append(batch_id)
            if batch_id % 100 == 0:
                print(f"batch {batch_id}/{len(loader)} loss = {loss} ")

        plt.plot(idx_loss,runn_loss)
        plt.show()


train(num_epochs, "dev")

def eval_performance():
    model.eval()
    num_test = 0
    correct = 0
    for i, (xs,ys) in enumerate(val_loader):
        y_pred = model(xs)
        for j,p in enumerate(y_pred):
            print(p.item(), ys[j].item())
            dif = abs(p.item() - ys[j].item())
            if dif <= 0.5:
                correct+=1
            num_test +=1
    print(correct/num_test, " %")

eval_performance()





        





