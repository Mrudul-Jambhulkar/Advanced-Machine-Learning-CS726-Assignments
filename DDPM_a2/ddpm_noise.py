#References:
# ChatGpt
#https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch
#https://github.com/TeaPearce/Conditional_Diffusion_MNIST/tree/main
#https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=5d751df2
import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
from torch import nn
import argparse
import torch.nn.functional as F
import utils
import dataset
import os
import matplotlib.pyplot as plt
import numpy as np

class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):
        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == 'cosine':
            self.init_cosine_schedule(**kwargs)
        elif type == 'sigmoid':
            self.init_sigmoid_schedule(**kwargs)
        elif type == 'quadratic':
            self.init_quadratic_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")

    def init_linear_schedule(self, **kwargs):
        """
        Precompute whatever quantities are required for training and sampling
        """
        self.beta_start = kwargs['beta_start']
        self.beta_end = kwargs['beta_end']
        
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, dtype=torch.float32)
        
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def init_cosine_schedule(self, **kwargs):
        """
        Precompute quantities required for training and sampling using cosine beta schedule
        """
        self.betas = self.cosine_beta_schedule(self.num_timesteps, kwargs['s'])
        
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def init_quadratic_schedule(self, **kwargs):
        """
        Precompute quantities required for training and sampling using quadratic beta schedule
        """
        self.beta_start = kwargs['beta_start']
        self.beta_end = kwargs['beta_end']

        # Create a quadratic progression from beta_start to beta_end
        t = torch.linspace(0, 1, self.num_timesteps)  # Linear progression from 0 to 1
        self.betas = self.beta_start + (self.beta_end - self.beta_start) * (t ** 2)
        
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def init_sigmoid_schedule(self, **kwargs):
        """
        Precompute quantities required for training and sampling using sigmoid beta schedule
        """
        self.beta_start = kwargs['beta_start']
        self.beta_end = kwargs['beta_end']

        # Inline sigmoid beta schedule computation
        betas = torch.linspace(-6, 6, self.num_timesteps)
        self.betas = torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        
        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
    def add_noise(self, original, noise, t):
        r"""
        Forward method for diffusion
        :param original: Input data tensor of shape [batch_size, n_dim]
        :param noise: Random Noise Tensor (from normal dist) of shape [batch_size, n_dim]
        :param t: Timestep tensor of shape [batch_size]
        :return:
        """
        original_shape = original.shape  # [batch_size, n_dim]
        batch_size = original_shape[0]
        
        # Get the corresponding alpha values for each batch sample
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device).gather(0, t).view(batch_size, 1)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device).gather(0, t).view(batch_size, 1)

        xt = sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise
        return xt
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        r"""
        Use the noise prediction by model to get xt-1 using xt and the noise predicted.
        
        :param xt: Current timestep sample [batch_size, n_dim]
        :param noise_pred: Model noise prediction [batch_size, n_dim]
        :param t: Current timestep tensor [batch_size]
        :return:
        """
        batch_size, n_dim = xt.shape
        
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(xt.device).gather(0, t).view(batch_size, 1)
        alpha_cum_prod = self.alpha_cum_prod.to(xt.device).gather(0, t).view(batch_size, 1)
        alphas = self.alphas.to(xt.device).gather(0, t).view(batch_size, 1)
        betas = self.betas.to(xt.device).gather(0, t).view(batch_size, 1)

        x0 = ((xt - (sqrt_one_minus_alpha_cum_prod * noise_pred)) / torch.sqrt(alpha_cum_prod))
        x0 = torch.clamp(x0, -1., 1.)

        mean = xt - ((betas * noise_pred) / sqrt_one_minus_alpha_cum_prod)
        mean = mean / torch.sqrt(alphas)

        if t.min() == 0:
            return mean, x0
        else:
            prev_t = t - 1
            variance = ((1 - self.alpha_cum_prod.to(xt.device).gather(0, prev_t).view(batch_size, 1)) /
                        (1.0 - alpha_cum_prod)) * betas
            sigma = torch.sqrt(variance)

            z = torch.randn_like(xt)
            return mean + sigma * z, x0
    
    def __len__(self):
        return self.num_timesteps

class DDPM(nn.Module):
    def __init__(self, n_dim=3, n_steps=200):
        """
        Noise prediction network for the DDPM

        Args:
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well
        """
        super().__init__()
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.embed_dim = 16
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.model = nn.Sequential(
            nn.Linear(n_dim + self.embed_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, n_dim)
        )
    
    def forward(self, x, t):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]

        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        t = t.unsqueeze(-1).float()
        t_embedding = self.time_embed(t)
        x_t_concat = torch.cat([x, t_embedding], dim=-1)
        noise_pred = self.model(x_t_concat)
        return noise_pred

class ConditionalDDPM(nn.Module):
    def __init__(self, n_classes=2, n_dim=3, n_steps=200):
        """
        Class dependent noise prediction network for the DDPM

        Args:
            n_classes: number of classes in the dataset
            n_dim: int, the dimensionality of the data
            n_steps: int, the number of steps in the diffusion process
        We have separate learnable modules for `time_embed` and `model`. `time_embed` can be learned or a fixed function as well
        """
        self.time_embed = None
        self.model = None
        self.class_embed = None

    def forward(self, x, t, y):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
            t: torch.Tensor, the timestep tensor [batch_size]
            y: torch.Tensor, the class label tensor [batch_size]
        Returns:
            torch.Tensor, the predicted noise tensor [batch_size, n_dim]
        """
        pass

class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        pass

    def __call__(self, x):
        pass

    def predict(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted class tensor [batch_size]
        """
        pass

    def predict_proba(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted probabilities for each class [batch_size, n_classes]
        """
        pass

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    """
    Train the model and save the model and necessary plots

    Args:
        model: DDPM, model to train
        noise_scheduler: NoiseScheduler, scheduler for the noise
        dataloader: torch.utils.data.DataLoader, dataloader for the dataset
        optimizer: torch.optim.Optimizer, optimizer to use
        epochs: int, number of epochs to train the model
        run_name: str, path to save the model
    """
    device = next(model.parameters()).device
    criterion = torch.nn.MSELoss()
    best_loss = float('inf')

    if not os.path.exists(run_name):
        os.makedirs(run_name)

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

            noise = torch.randn_like(x).to(device)
            t = torch.randint(0, noise_scheduler.num_timesteps, (x.shape[0],), device=device)

            noisy_x = noise_scheduler.add_noise(x, noise, t)

            optimizer.zero_grad()
            noise_pred = model(noisy_x, t)

            loss = criterion(noise_pred, noise)
            epoch_losses.append(loss)

            loss.backward()
            optimizer.step()

        mean_loss = torch.mean(torch.tensor(epoch_losses, device=device)).item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), os.path.join(run_name, "model.pth"))
            print(f"Saved best model with loss: {best_loss:.4f}")

    print("Training complete.")

@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    """
    Sample from the model
    
    Args:
        model: DDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        return_intermediate: bool
    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]

    If `return_intermediate` is `False`,
            torch.Tensor, samples from the model [n_samples, n_dim]
    Else
        the function returns all the intermediate steps in the diffusion process as well 
        Return: [[n_samples, n_dim]] x n_steps
    """
    device = next(model.parameters()).device
    n_steps = noise_scheduler.num_timesteps
    n_dim = model.n_dim

    x_t = torch.randn(n_samples, n_dim).to(device)
    
    intermediate_samples = [x_t.clone()] if return_intermediate else None

    for t in reversed(range(n_steps)):
        t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)

        noise_pred = model(x_t, t_tensor)

        x_t, _ = noise_scheduler.sample_prev_timestep(x_t, noise_pred, t_tensor)

        if return_intermediate:
            intermediate_samples.append(x_t.clone())

    # Clamp the final samples to a reasonable range
    x_t = torch.clamp(x_t, -1., 1.)

    return intermediate_samples if return_intermediate else x_t

def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    """
    Sample from the conditional model
    
    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        guidance_scale: float
        class_label: int

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass

def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    """
    Sample from the SVDD model

    Args:
        model: ConditionalDDPM
        n_samples: int
        noise_scheduler: NoiseScheduler
        reward_scale: float
        reward_fn: callable, takes in a batch of samples torch.Tensor:[n_samples, n_dim] and returns torch.Tensor[n_samples]

    Returns:
        torch.Tensor, samples from the model [n_samples, n_dim]
    """
    pass

def evaluate_samples(real_data, generated_samples, run_name, seed, n_samples, batch_size=1000):
    """Evaluate generated samples using NLL and EMD metrics with batched processing"""
    device = real_data.device
    # Convert NumPy arrays back to PyTorch tensors for get_nll
    real_tensor = torch.tensor(real_data.cpu().numpy(), dtype=torch.float32)
    generated_tensor = torch.tensor(generated_samples.cpu().numpy(), dtype=torch.float32)
    
    # Compute NLL
    print("Computing NLL...")
    nll_value = utils.get_nll(real_tensor, generated_tensor, temperature=1e-1)
    print(f"Negative Log-Likelihood (NLL): {nll_value:.4f}")
    with open(os.path.join(run_name, f'nll_{seed}_{n_samples}.txt'), 'w') as f:
        f.write(f"NLL: {nll_value:.4f}\n")
    
    # Compute EMD with batched processing to reduce memory usage
    print("Computing EMD with batched processing...")
    real_data_np = real_data.cpu().numpy()
    generated_np = generated_samples.cpu().numpy()
    n_real = real_data_np.shape[0]
    n_fake = generated_np.shape[0]
    
    emd_value = 0.0
    # n_batches = max(1, (n_real * n_fake) // (batch_size * batch_size))
    # batch_size_real = max(1, n_real // n_batches)
    # batch_size_fake = max(1, n_fake // n_batches)

    # for i in tqdm(range(0, n_real, batch_size_real), desc="EMD Batches"):
    #     for j in range(0, n_fake, batch_size_fake):
    #         real_batch = real_data_np[i:i + batch_size_real]
    #         fake_batch = generated_np[j:j + batch_size_fake]
    #         if real_batch.size == 0 or fake_batch.size == 0:
    #             continue
    #         emd_batch = utils.get_emd(real_batch, fake_batch)
    #         emd_value += emd_batch * (real_batch.shape[0] * fake_batch.shape[0]) / (n_real * n_fake)

    # print(f"Earth Mover's Distance (EMD): {emd_value:.4f}")
    # with open(os.path.join(run_name, f'emd_{seed}_{n_samples}.txt'), 'w') as f:
    #     f.write(f"EMD: {emd_value:.4f}\n")
    
    return nll_value, emd_value

def visualize_samples(real_data, real_labels, generated_samples, dataset_name, run_name, seed, n_samples):
    """Visualize samples for real and generated data for the specified dataset"""
    # Create a figure with two subplots (real and generated) side by side
    fig = plt.figure(figsize=(10, 5))  # Match the aspect ratio of the manycircles plot
    
    # Convert data to NumPy for plotting
    real_data_np = real_data.cpu().numpy()
    real_labels_np = real_labels.cpu().numpy()
    generated_np = generated_samples.cpu().numpy()
    
    # Print range of generated samples for debugging
    print(f"Generated {dataset_name} - Min: {generated_np.min(axis=0)}, Max: {generated_np.max(axis=0)}")
    
    # Clamp the generated samples to the range [-1, 1] to match the real data
    generated_np = np.clip(generated_np, -1., 1.)

    # Plot real data (left subplot)
    if real_data_np.shape[1] == 2:
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(real_data_np[:, 0], real_data_np[:, 1], c=real_labels_np, cmap='viridis', s=5)
        ax1.set_title(f"Real Data ({dataset_name.capitalize()})")
        ax1.set_xlabel("Feature 1")
        ax1.set_ylabel("Feature 2")
    elif real_data_np.shape[1] == 3:
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(real_data_np[:, 0], real_data_np[:, 1], real_data_np[:, 2], c=real_labels_np, cmap='viridis', s=5)
        ax1.set_title(f"Real Data ({dataset_name.capitalize()})")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

    # Plot generated samples (right subplot)
    if generated_np.shape[1] == 2:
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(generated_np[:, 0], generated_np[:, 1], c='purple', s=5)
        ax2.set_title(f"Generated Samples ({dataset_name.capitalize()})")
        ax2.set_xlabel("Feature 1")
        ax2.set_ylabel("Feature 2")
    elif generated_np.shape[1] == 3:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(generated_np[:, 0], generated_np[:, 1], generated_np[:, 2], c='purple', s=5)
        ax2.set_title(f"Generated Samples ({dataset_name.capitalize()})")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(os.path.join(run_name, f'samples_plot_{seed}_{n_samples}.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--n_steps", type=int, default=None)
    parser.add_argument("--lbeta", type=float, default=None)
    parser.add_argument("--ubeta", type=float, default=None)
    parser.add_argument("--type", choices=['linear', 'cosine', 'sigmoid', 'quadratic'], default='linear')
    parser.add_argument("--s", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=None)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.s is not None:
        run_name = f'exps/ddpm_{args.type}_{args.n_dim}_{args.n_steps}_{args.s}_{args.dataset}'
    else:
        run_name = f'exps/ddpm_{args.type}_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}'
    os.makedirs(run_name, exist_ok=True)

    # Set default values if parameters are not provided
    beta_start = args.lbeta if args.lbeta is not None else 0.0001
    beta_end = args.ubeta if args.ubeta is not None else 0.02
    s = args.s if args.s is not None else 0.008
    
    model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps,
                                    type = args.type, 
                                    beta_start=beta_start, 
                                    beta_end=beta_end,
                                    s=s)
    model = model.to(device)

    if args.mode == 'train':
        epochs = args.epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_X, data_y), 
            batch_size=args.batch_size, 
            shuffle=True
        )
        train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        print("Generating samples...")
        samples = sample(model, args.n_samples, noise_scheduler)
        print(f"Generated samples shape: {samples.shape}")
        
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
        print(f"Saved generated samples to {run_name}/samples_{args.seed}_{args.n_samples}.pth")
        
        # Load real data for comparison
        print("Loading real dataset...")
        real_data, real_labels = dataset.load_dataset(args.dataset)
        real_data = real_data.to(device)
        real_labels = real_labels.to(device)
        print(f"Real data shape: {real_data.shape}")
        
        # Evaluate samples with batched EMD
        evaluate_samples(real_data, samples, run_name, args.seed, args.n_samples, batch_size=1000)
        
        # Visualize samples
        visualize_samples(real_data, real_labels, samples, args.dataset, run_name, args.seed, args.n_samples)
        print(f"Plots saved to {run_name}/samples_plot_{args.seed}_{args.n_samples}.png")
    else:
        raise ValueError(f"Invalid mode {args.mode}")





# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
#     parser.add_argument("--n_steps", type=int, default=None)
#     parser.add_argument("--lbeta", type=float, default=None)
#     parser.add_argument("--ubeta", type=float, default=None)
#     parser.add_argument("--epochs", type=int, default=None)
#     parser.add_argument("--n_samples", type=int, default=None)
#     parser.add_argument("--batch_size", type=int, default=None)
#     parser.add_argument("--lr", type=float, default=None)
#     parser.add_argument("--dataset", type=str, default = None)
#     parser.add_argument("--seed", type=int, default = 42)
#     parser.add_argument("--n_dim", type=int, default = None)

#     args = parser.parse_args()
#     utils.seed_everything(args.seed)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}' # can include more hyperparams
#     os.makedirs(run_name, exist_ok=True)

#     model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
#     noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
#     model = model.to(device)

#     if args.mode == 'train':
#         epochs = args.epochs
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#         data_X, data_y = dataset.load_dataset(args.dataset)
#         # can split the data into train and test -- for evaluation later
#         data_X = data_X.to(device)
#         data_y = data_y.to(device)
#         dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y), batch_size=args.batch_size, shuffle=True)
#         train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)

#     elif args.mode == 'sample':
#         model.load_state_dict(torch.load(f'{run_name}/model.pth'))
#         samples = sample(model, args.n_samples, noise_scheduler)
#         torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
#     else:
#         raise ValueError(f"Invalid mode {args.mode}")