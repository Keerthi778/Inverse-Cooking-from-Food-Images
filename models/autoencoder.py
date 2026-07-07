import torch.nn as nn


class FoodAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(FoodAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 1024), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(1024, latent_dim)
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 128 * 28 * 28), nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (128, 28, 28)),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 2, stride=2), nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder_fc(self.encoder_conv(x))

    def decode(self, z):
        return self.decoder_conv(self.decoder_fc(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class RecipeHead(nn.Module):
    def __init__(self, latent_dim=512, num_ingredients=1488):
        super(RecipeHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(1024, num_ingredients)
        )

    def forward(self, z):
        return self.fc(z)