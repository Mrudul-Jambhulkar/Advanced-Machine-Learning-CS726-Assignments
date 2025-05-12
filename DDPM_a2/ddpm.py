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
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented")

    def init_linear_schedule(self, beta_start, beta_end):
        """
        Precompute whatever quantities are required for training and sampling
        """
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        
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
        super().__init__()
        self.n_classes = n_classes
        self.n_dim = n_dim
        self.n_steps = n_steps
        self.embed_dim = 16

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Class embedding
        self.class_embed = nn.Embedding(n_classes, self.embed_dim)

        # Model
        self.model = nn.Sequential(
            nn.Linear(n_dim + 2 * self.embed_dim, 128),  # Concatenate x, time_embed, and class_embed
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, n_dim)
        )

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).float()  # [batch_size] → [batch_size, 1]
        t_embedding = self.time_embed(t)  # [batch_size, 1] → [batch_size, embed_dim]

        y_embedding = self.class_embed(y)  # [batch_size] → [batch_size, embed_dim]

        # Concatenate x, time_embed, and class_embed
        x_t_y_concat = torch.cat([x, t_embedding, y_embedding], dim=-1)  # [batch_size, n_dim + 2 * embed_dim]

        noise_pred = self.model(x_t_y_concat)  # [batch_size, n_dim + 2 * embed_dim] → [batch_size, n_dim]
        return noise_pred

class ClassifierDDPM():
    """
    ClassifierDDPM implements a classification algorithm using the DDPM model
    """
    def __init__(self, model: ConditionalDDPM, noise_scheduler: NoiseScheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.n_classes = model.n_classes

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted class tensor [batch_size]
        """
        # Compute probabilities for each class
        probs = self.predict_proba(x)

        # Return the class with the highest probability
        return torch.argmax(probs, dim=-1)

    def predict_proba(self, x):
        """
        Args:
            x: torch.Tensor, the input data tensor [batch_size, n_dim]
        Returns:
            torch.Tensor, the predicted probabilites for each class  [batch_size, n_classes]
        """
        device = next(self.model.parameters()).device
        batch_size = x.shape[0]

        # Initialize logits for each class
        logits = torch.zeros((batch_size, self.n_classes), device=device)

        # Iterate over each class and compute the likelihood
        for class_label in range(self.n_classes):
            # Create a tensor of class labels
            y = torch.full((batch_size,), class_label, dtype=torch.long, device=device)

            # Compute the noise prediction for the given class
            t = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=device)
            noise_pred = self.model(x, t, y)

            # Compute the likelihood (MSE between predicted noise and actual noise)
            noise = torch.randn_like(x).to(device)
            noisy_x = self.noise_scheduler.add_noise(x, noise, t)
            mse_loss = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=-1)

            # Use negative MSE as logits (higher likelihood → higher probability)
            logits[:, class_label] = -mse_loss

        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=-1)
        return probs
    
def trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, device, puncond=0.1):
    """
    Train the conditional DDPM model with joint training of conditional and unconditional models.

    Args:
        model: ConditionalDDPM, the model to train
        noise_scheduler: NoiseScheduler, the noise scheduler
        dataloader: DataLoader, the training data loader
        optimizer: torch.optim.Optimizer, the optimizer
        epochs: int, the number of epochs
        device: str, the device to use (e.g., 'cuda' or 'cpu')
        puncond: float, probability of using unconditional model (default: 0.1)
    """
    if not os.path.exists(run_name):
        os.makedirs(run_name)  # Ensure directory exists before saving the model
    
    best_loss = float('inf')  # Track the best loss
    model.train()
    
    for epoch in range(epochs):
        epoch_losses = []
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = x.to(device)
            y = y.to(device)

            # Randomly set y to null token (0) with probability puncond
            mask = torch.rand(y.shape[0]) < puncond
            y[mask] = 0  # Null token for unconditional model

            # Sample timestep t
            t = torch.randint(0, noise_scheduler.num_timesteps, (x.shape[0],), device=device)

            # Corrupt data with noise
            noise = torch.randn_like(x)
            noisy_x = noise_scheduler.add_noise(x, noise, t)

            # Optimize the denoising model
            optimizer.zero_grad()
            noise_pred = model(noisy_x, t, y)
            loss = F.mse_loss(noise_pred, noise)
            epoch_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        # Compute mean epoch loss
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {mean_loss:.4f}")
        
        # Save the best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), os.path.join(run_name, "model.pth"))
            print(f"Saved best model with loss: {best_loss:.4f}")
    
    print("Training complete.")

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

def sampleConditional(model, n_samples, noise_scheduler, guidance_scale, class_label):
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
    device = next(model.parameters()).device

    # Start with pure Gaussian noise
    x_t = torch.randn(n_samples, model.n_dim).to(device)

    # Create a tensor of class labels
    y = torch.full((n_samples,), class_label, dtype=torch.long).to(device)

    # Iterate over the timesteps in reverse order
    for t in reversed(range(noise_scheduler.num_timesteps)):
        # Create a tensor for the current timestep
        t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)

        # Predict noise for the conditional and unconditional cases
        noise_pred_cond = model(x_t, t_tensor, y)  # Conditional prediction
        noise_pred_uncond = model(x_t, t_tensor, torch.zeros_like(y))  # Unconditional prediction (null token)

        # Combine predictions using the guidance scale
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Reverse the noise using the noise scheduler
        x_t, _ = noise_scheduler.sample_prev_timestep(x_t, noise_pred, t_tensor)

    return x_t

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
    device = next(model.parameters()).device

    # Start with pure Gaussian noise
    x_t = torch.randn(n_samples, model.n_dim).to(device)

    # Create a tensor of class labels
    y = torch.full((n_samples,), class_label, dtype=torch.long).to(device)

    # Iterate over the timesteps in reverse order
    for t in reversed(range(noise_scheduler.num_timesteps)):
        # Create a tensor for the current timestep
        t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)

        # Predict noise for the conditional and unconditional cases
        noise_pred_cond = model(x_t, t_tensor, y)  # Conditional prediction
        noise_pred_uncond = model(x_t, t_tensor, torch.zeros_like(y))  # Unconditional prediction (null token)

        # Combine predictions using the guidance scale
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Reverse the noise using the noise scheduler
        x_t, _ = noise_scheduler.sample_prev_timestep(x_t, noise_pred, t_tensor)

    return x_t

def evaluate_samples(real_data, generated_samples, run_name, seed, n_samples, batch_size=1000):
    """Evaluate generated samples using NLL and EMD metrics with batched processing"""
    device = real_data.device
    # Convert NumPy arrays back to PyTorch tensors for get_nll
    real_tensor = torch.tensor(real_data.cpu().detach().numpy(), dtype=torch.float32)
    generated_tensor = torch.tensor(generated_samples.cpu().detach().numpy(), dtype=torch.float32)
    
    # Compute NLL
    print("Computing NLL...")
    nll_value = utils.get_nll(real_tensor, generated_tensor, temperature=1e-1)
    print(f"Negative Log-Likelihood (NLL): {nll_value:.4f}")
    with open(os.path.join(run_name, f'nll_{seed}_{n_samples}.txt'), 'w') as f:
        f.write(f"NLL: {nll_value:.4f}\n")
    
    # Compute EMD with batched processing to reduce memory usage
    print("Computing EMD with batched processing...")
    real_data_np = real_data.cpu().detach().numpy()
    generated_np = generated_samples.cpu().detach().numpy()
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
    real_data_np = real_data.cpu().detach().numpy()
    real_labels_np = real_labels.cpu().detach().numpy()
    generated_np = generated_samples.cpu().detach().numpy()
    
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=None)
    parser.add_argument("--model", choices=['ddpm', 'cfg'], default='ddpm')
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--guidance_scale', type=float, default=0.1)
    parser.add_argument('--class_label', type=int, default=0)

    args = parser.parse_args()
    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'ddpm':
        run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}'
    else:
        run_name = f'exps/cfg_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}'
    os.makedirs(run_name, exist_ok=True)

    if args.model == 'ddpm':
        model = DDPM(n_dim=args.n_dim, n_steps=args.n_steps)
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)
    else:
        # Initialize model and scheduler
        model = ConditionalDDPM(n_classes=args.n_class, n_dim=args.n_dim, n_steps=args.n_steps)  # Use the same parameters as during training
        # model = DDPM(n_dim=n_dim, n_steps=n_steps)
        noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, beta_start=args.lbeta, beta_end=args.ubeta)

    # Move model to device
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
        if args.model == 'ddpm':
            train(model, noise_scheduler, dataloader, optimizer, epochs, run_name)
        else:
            trainConditional(model, noise_scheduler, dataloader, optimizer, epochs, device, puncond=0.1)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))

        if args.model == 'ddpm':
        # Generate Samples
            print("Generating samples...")
            samples = sample(model, args.n_samples, noise_scheduler)
            print(f"Generated samples shape: {samples.shape}")
        
        else:
            # Generate samples
            print("Generating samples...")
            guidance_scale = args.guidance_scale
            class_label = args.class_label
            samples = sampleCFG(model, args.n_samples, noise_scheduler, guidance_scale, class_label)
        #     samples = sample(model, n_samples, noise_scheduler)
            print(f"Generated samples shape: {samples.shape}")  # Should be [n_samples, n_dim]
        
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
        print(f"Saved generated samples to {run_name}/samples_{args.seed}_{args.n_samples}.pth")

        # Load real data for comparison
        print("Loading real dataset...")
        real_data, real_labels = dataset.load_dataset(args.dataset)
        real_data = real_data.to(device)
        real_labels = real_labels.to(device)
        print(f"Real data shape: {real_data.shape}")

        if args.model == 'cfg':
            print("Classifying samples...")
            classifier = ClassifierDDPM(model, noise_scheduler)
            predicted_labels_generated = classifier.predict(samples)

            # Predict labels for the full real dataset
            predicted_labels = classifier.predict(real_data)

            # Filter samples where the classifier predicts the class_label
            mask = (predicted_labels == class_label)
            predicted_data_class = real_data[mask]
            predicted_labels_class = predicted_labels[mask]
            true_labels_class = real_labels[mask]

            if predicted_data_class.size(0) == 0:
                print(f"No samples were predicted as class {class_label}. Skipping accuracy calculation.")
            else:
                # Compute accuracy: proportion of these samples where the true label matches class_label
                correct = (true_labels_class == class_label).sum().item()
                accuracy = correct / predicted_data_class.size(0)
                print(f"Classification accuracy for predicted class {class_label}: {accuracy * 100:.2f}% "
                    f"(based on {predicted_data_class.size(0)} samples predicted as class {class_label} "
                    f"with guidance scale {guidance_scale} during generation).")
        
        # Evaluate samples with batched EMD
        evaluate_samples(real_data, samples, run_name, args.seed, args.n_samples, batch_size=1000)
        
        # Visualize samples
        visualize_samples(real_data, real_labels, samples, args.dataset, run_name, args.seed, args.n_samples)
        print(f"Plots saved to {run_name}/samples_plot_{args.seed}_{args.n_samples}.png")
    else:
        raise ValueError(f"Invalid mode {args.mode}")