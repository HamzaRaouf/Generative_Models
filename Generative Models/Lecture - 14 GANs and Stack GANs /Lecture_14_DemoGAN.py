#source: https://www.youtube.com/watch?v=OljTVUVzPpM
'''
Home work
# To Do list:
1. What happen if you use larger networks
2. Use of batch normalization with batchNorm
3. Try different learing rates
4. Change architectues from NN to CNN
'''
# dependices
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torchvision.utils import save_image
from IPython.display import Image, display



class Discriminator(nn.Module):
    def __init__(self, img_dim): # in_features= image_dim = 784 for MNIST
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # To make outputs is between [-1, 1] as normalize make inputs to [-1, 1] so
        )

    def forward(self, x):
        return self.gen(x)


# Hyperparameters: GANs are very sensitive to hyperparamters so try mutple hyperparamters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # andrej karpathy try with 1e-4
z_dim = 64 # may try 128, 256
image_dim = 28 * 28 * 1  # 784
batch_size = 32 # process 32 images at once
num_epochs = 10 # 500

#initilaized discriminator
disc = Discriminator(image_dim).to(device) # weights and bias are intialzed under the hood
gen = Generator(z_dim, image_dim).to(device)

fixed_noise = torch.randn((batch_size, z_dim)).to(device) # use it for testing purpose
#fixed_noise = > same noise vectors are used to see how much the improvment is done after each epoch
#print ("fixed_noise.shape", fixed_noise.shape) # [32,64]

# ToTensor convert the data
#in tensor type (Tensor datatype is required to run data on GPU)
# and intensity values are range [0->255]
# and use of  transform = transforms.Normalize((0.5,), (0.5,))])
#OR transform = transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])
# make the data intensity value of range from [0-1] to [-1 to 1]
# roughly centered around zero and have a similar scale.
#This can make training more stable and might improve the performance of neural networks,
#As value - mean/ std
# For 0: 0-0.5 / 0.5 = -1
# for 1: 1 - 0.5 / 0/5 = 1
# so the range is [-1 to 1]

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

dataset = datasets.FashionMNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr) # may include beta value
opt_gen = optim.Adam(gen.parameters(), lr=lr)

'''
#https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
In the mathematical model of a GAN I described earlier,
the gradient of this had to be ascended,
but PyTorch and most other Machine Learning frameworks usually minimize functions instead.
Since maximizing a function is
equivalent to minimizing it’s negative,
and the BCE-Loss term has a minus sign,
we don’t need to worry about the sign.


Maximizing log D(G(z)) is equivalent to minimizing it’s negative and
since the BCE-Loss definition has a minus sign, we don’t need to take care of the sign.

'''
criterion = nn.BCELoss()



for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader): # use _ for not read label
        # reshape real image to 784 dim
        #print (real.shape) [32,1,28,28]

        real = real.view(-1, 784).to(device) # [32,784]

        #print (real.shape) # [32,784]
        batch_size = real.shape[0]

        ### Train Discriminator: max [log(D(x)) + log(1 - D(G(z)))]
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1) # flatten() and .view(-1) flattens a tensor in PyTorch.
        #print ("disc_real.shape",disc_real.shape) = > [32] =>(rank 1)
        #print ("disc(real).view(-1).shape", disc(real).shape) => [32,1] =>(rank 2)

        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # log(D(x))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) #log(1 - D(G(z))
                                                                       # torch.zeros_like(disc_fake) creat zeros of same dim as that of disc_fake
                                                                       # 32 is batch size, so disc produce 32 outputs
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True) # retain_graph=True,retain computation graph otherwise no graph is there
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output)) #log(D(G(z)) here y = 1 but inreality output is fake
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)# Here -1 for input x specifies that this dimension should be
                                                                #dynamically computed based on the number of input values in x
                save_image(fake, f"gan_images_epoch_{epoch}.png", nrow=5, normalize=True)

                # Display the generated images in Colab
                display(Image(filename=f"gan_images_epoch_{epoch}.png"))
