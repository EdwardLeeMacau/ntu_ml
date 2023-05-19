# Report

1. Choose a variation of autoencoder. Show an image of the model architecture. Then, list an advantage and a disadvantage comparing with vanilla autoencoder. Also, put on the paper link as reference. Eg, denoising autoencoder, variational autoencoder, etc.

    I use the figure from Prof. Lee's presentation (Generation, ML2017) as example. Here is the model architecture of Variational Auto-Encoder (VAE):

    ![](vae-2017-hungyi-lee.png)

    The major difference between VAE and Vanilla autoencoder is that, instead of generating an embedding vector for each instance x, VAE generates the parameter of distribution (usually choose Normal Distribution), then sample the embedding from latent distribution and feed to decoder for reconstruction. This approach aims to support generating synthetic data from latent space using VAE, not only learning a meaningful latent vector. It has some drawbacks, such as the images generated from VAE is usually blurred.

    Reference:
    - https://arxiv.org/pdf/1312.6114.pdf
    - https://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/GAN%20(v3).pdf

2. Train a fully connected autoencoder and adjust at least two different element of the latent representation. Show your model architecture, plot out the original image, the reconstructed images for each adjustment and describe the differences.

    Here is the fully connected autoencoder I use for this experiment. Its encoder and decoder is a 3-layer MLP, the latent vector has 64 dimensions.

    ```Python
    class FCNAutoEncoder(nn.Module):
    def __init__(self):
        super(FCNAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 64 * 64 * 3),
            nn.Tanh()
        )
    ```

    The first image is the origin image, the second image is reconstructed image under default setting. The third one I negated `z[32]`, seems that the skin tone is more fair than reconstructed one. The final one I negated `z[63]`, seems that the smile is slightly different than reconstructed one, smaller mouth, deeper contour for eyes.

    ![](./demo.png)
