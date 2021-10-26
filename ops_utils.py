import pickle
import tempfile

import numpy as np
import torch
from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sacred import Ingredient
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from model import LinearVAE

ops_ingredient = Ingredient("ops")

@ops_ingredient.config
def config():
    ops_timestep = 100

    delay = 0
    pretraining_steps = 5000
    pretraining_times = 1

    batch_size = 128
    clusters = None
    lr = 3e-4
    epochs = 10
    z_features = 10

    kl_weight = 0.0001
    delay_training = False

    human_selected_idx = None # like [0, 0, 0, 0, 1, 1, 1, 1] or None - only used for visualisation

    encoder_in = ["agent"]
    decoder_in = ["obs", "act"] # + "z"
    reconstruct = ["next_obs", "rew"]

class rbDataSet(Dataset):
    @ops_ingredient.capture
    def __init__(self, rb, encoder_in, decoder_in, reconstruct):
        self.rb = rb
        self.data = []
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in encoder_in], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in decoder_in], dim=1))
        self.data.append(torch.cat([torch.from_numpy(self.rb[n]) for n in reconstruct], dim=1))
        
        print([x.shape for x in self.data])
    def __len__(self):
        return self.data[0].shape[0]
    def __getitem__(self, idx):
        return [x[idx, :] for x in self.data]

@ops_ingredient.capture
def compute_clusters(rb, agent_count, batch_size, clusters, lr, epochs, z_features, kl_weight, _log):
    device = "cpu"

    dataset = rbDataSet(rb)
    
    input_size = dataset.data[0].shape[-1]
    extra_decoder_input = dataset.data[1].shape[-1]
    reconstruct_size = dataset.data[2].shape[-1]
    
    model = LinearVAE(z_features, input_size, extra_decoder_input, reconstruct_size)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # criterion = nn.BCELoss(reduction='sum')
    criterion = nn.MSELoss(reduction="sum")
    def final_loss(bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + kl_weight*KLD

    def fit(model, dataloader):
        model.train()
        running_loss = 0.0
        for i, (encoder_in, decoder_in, y) in enumerate(dataloader):
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(encoder_in, decoder_in)
            bce_loss = criterion(reconstruction, y)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(dataloader.dataset)
        return train_loss

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loss = []
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = fit(model, dataloader)
        train_loss.append(train_epoch_loss)

    print(f"Train Loss: {train_epoch_loss:.6f}")
    x = torch.eye(agent_count)

    with torch.no_grad():
        z = model.encode(x)
    z = z.to("cpu")
    z = z[:, :]

    if clusters is None:
        clusters = find_optimal_cluster_number(z)
    _log.info(f"Creating {clusters} clusters.")
    # run k-means from scikit-learn
    kmeans = KMeans(
        n_clusters=clusters, init='k-means++',
        n_init=10
    )
    cluster_ids_x = kmeans.fit_predict(z) # predict labels
    if z_features == 2:
        plot_clusters(kmeans.cluster_centers_, z)
    return torch.from_numpy(cluster_ids_x).long()

@ops_ingredient.capture
def plot_clusters(cluster_centers, z, human_selected_idx, _run):

    if human_selected_idx is None:
        plt.plot(z[:, 0], z[:, 1], 'o')
        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'x')

        for i in range(z.shape[0]):
            plt.annotate(str(i), xy=(z[i, 0], z[i, 1]))

    else:
        colors = 'bgrcmykw'
        for i in range(len(human_selected_idx)):
            plt.plot(z[i, 0], z[i, 1], 'o' + colors[human_selected_idx[i]])

        plt.plot(cluster_centers[:, 0], cluster_centers[:, 1], 'x')
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile, format="png") # File position is at the end of the file.
        _run.add_artifact(tmpfile.name, f"cluster.png")
    # plt.savefig("cluster.png")

def find_optimal_cluster_number(X):

    range_n_clusters = list(range(2, X.shape[0]))
    scores = {}

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        scores[n_clusters] = davies_bouldin_score(X, cluster_labels)

    max_key = min(scores, key=lambda k: scores[k])
    return max_key
