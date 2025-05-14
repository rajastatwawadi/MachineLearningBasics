import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



class VAE(nn.Module):
    def __init__(self, original_dim, intermediate_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
        )

        self.fc_mean = nn.Linear(intermediate_dim, latent_dim)
        self.fc_logvar = nn.Linear(intermediate_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x.view(x.size(0), -1)) #Flatten Input
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar




def loss_function(recon_x, x, mean, logvar):
    Recostruction = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') #Flatten target

    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return Recostruction + KLD




original_dim = 784  # For MNIST (28x28)
intermediate_dim = 392
latent_dim = 16

batch_size = 128
epochs = 20
learning_rate = 1e-3




if __name__ == "__main__":  

    plt.ion()

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    # Initialize the VAE model and optimizer
    vae = VAE(original_dim, intermediate_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    print(vae)


    # Training loop
    train_losses=[]
    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        for data, _ in train_loader:
            optimizer.zero_grad()
            recon_batch, mean, logvar = vae(data)
            loss = loss_function(recon_batch, data, mean, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)


        print(f'Epoch: {epoch+1}, Train Loss: {train_loss / len(train_loader.dataset)}')



    # Plot training loss

    plt.plot(range(1, epochs + 1), train_losses,)
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.show()



    vae.eval()
    device = next(vae.parameters()).device

    # Store images per digit: 0 to 9
    grouped_originals = {i: [] for i in range(10)}
    grouped_reconstructions = {i: [] for i in range(10)}
    max_per_digit = 20

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            recon_batch, _, _ = vae(data)

            for i in range(len(data)):
                label = targets[i].item()
                if len(grouped_originals[label]) < max_per_digit:
                    orig = data[i].view(28, 28).cpu().numpy()
                    recon = recon_batch[i].view(28, 28).cpu().numpy()

                    grouped_originals[label].append(orig)
                    grouped_reconstructions[label].append(recon)

            # Break early if we've filled all classes
            if all(len(grouped_originals[d]) == max_per_digit for d in range(10)):
                break



    # Plot original and reconstructed images
    plt.figure(figsize=(20, 10))

    for digit in range(10):
        for i in range(20):

            plt.subplot(10, 20, digit * 20 + i + 1)
            plt.imshow(grouped_originals[digit][i], cmap='gray')
            plt.axis('off')

    plt.suptitle("Original Images", fontsize=18)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 10))

    for digit in range(10):
        for i in range(20):

            plt.subplot(10, 20, digit * 20 + i + 1)
            plt.imshow(grouped_reconstructions[digit][i], cmap='gray')
            plt.axis('off')

    plt.suptitle("Reconsturcted Images", fontsize=18)
    plt.tight_layout()

    plt.ioff()
    plt.show()
