
import torch
import torch.nn as nn
import numpy as np

from saits.diff_saits import DiffSAITS


# def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
#     return torch.linspace(start, end, timesteps)

# def get_index_from_list(vals, t, x_shape):
#     """ 
#     Returns a specific index t of a passed list of values vals
#     while considering the batch dimension.
#     """
#     batch_size = t.shape[0]
#     out = vals.gather(-1, t.cpu())
#     return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# def forward_diffusion_sample(x_0, t, device="cpu"):
#     """ 
#     Takes an image and a timestep as input and 
#     returns the noisy version of it
#     """
#     noise = torch.randn_like(x_0)
#     sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
#     sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
#         sqrt_one_minus_alphas_cumprod, t, x_0.shape
#     )
#     # mean + variance
#     return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
#     + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# class SinusoidalPositionEmbeddings(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#     def forward(self, time):
#         device = time.device
#         half_dim = self.dim // 2
#         embeddings = math.log(10000) / (half_dim - 1)
#         embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
#         embeddings = time[:, None] * embeddings[None, :]
#         embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
#         return embeddings



class DiffModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.feature_dim = config['n_features']
        self.time_emb_dim = config['time_emb']
        
        self.num_steps = config['n_steps']
        self.diff_steps = config['diff_steps']
        self.betas = self.beta_schedule(config['schedule'], config['beta_start'], config['beta_end'])
        print(f"Betas: {self.betas}")
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)
        self.beta_tildes = self.betas + 0
        for t in range(1, self.diff_steps):
            self.beta_tildes *= (1 - self.alpha_hats[t - 1]) / (
                1 - self.alpha_hats[t])
        self.sigma = torch.sqrt(self.beta_tildes)
        self.alpha_hats_sqrt = (self.alpha_hats ** 0.5).unsqueeze(1).unsqueeze(1)
        self.comp_alpha_hats_sqrt = ((1.0 - self.alpha_hats) ** 0.5).unsqueeze(1).unsqueeze(1)

        self.diff_model = DiffSAITS(d_time=self.num_steps, d_feature=self.feature_dim, n_layers=config['n_layers'], \
                d_model=config['d_model'], d_inner=config['d_inner'], n_head=config['n_head'], d_k=config['d_k'], \
                d_v=config['d_v'], dropout=config['dropout'], diff_steps=self.diff_steps, time_strategy=config['time_strategy'])
        

    def beta_schedule(self, scheduler, start, end):
        if scheduler == "quad":
            betas = torch.linspace(
                start ** 0.5, end ** 0.5, self.diff_steps, requires_grad=False
            ) ** 2
        else:
            betas = torch.linspace(
                start, end, self.diff_steps, requires_grad=False
            )
        return betas

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def forward_diffusion_sample(self, observed_data, t):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(observed_data)
        sqrt_alpha_hats_t = self.alpha_hats_sqrt[t]
        sqrt_comp_alpha_hats_t = self.comp_alpha_hats_sqrt[t]
        # mean + variance
        return sqrt_alpha_hats_t * observed_data + sqrt_comp_alpha_hats_t * noise, noise

    def process_data(self, data):
        """ Assemble the input data into a dictionary.

        Parameters
        ----------
        data : list
            A list containing data fetched from Dataset by Dataload.

        Returns
        -------
        inputs : dict
            A dictionary with data assembled.
        """
        indices, X_intact, X, missing_mask, indicating_mask = data

        inputs = {
            'X': X,
            'X_intact': X_intact,
            'observed_mask': missing_mask,
            'indicating_mask': indicating_mask
        }

        return inputs

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = cond_mask * observed_data
        noisy_target = (1 - cond_mask) * noisy_data
        total_input = cond_obs + noisy_target # torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def calculate_mse(self, prediction, target, mask):
        num_eval = mask.sum()
        return ((target - prediction) ** 2).sum() / (num_eval if num_eval > 0 else 1)

    def calc_loss(self, observed_data, cond_mask, observed_mask):
        B, K , L = observed_data.shape
        t = torch.randint(0, self.diff_steps, [B])
        noise = torch.randn_like(observed_data)
        # print(f"observed: {observed_data.shape}, curr_alpha: {curr_alpha.shape}, noise: {noise.shape}, self.comp_alphas_sqrt: {self.comp_alphas_sqrt[t].shape}")
        noise_data = self.alpha_hats_sqrt[t] * observed_data + self.comp_alpha_hats_sqrt[t] * noise
        # print(f"noisy data: {noise_data.shape}")
        diff_input = self.set_input_to_diffmodel(noise_data, observed_data, cond_mask)
        diff_inputs = {'X': diff_input, 'missing_mask': observed_mask}
        predicted_mean, X_finals = self.diff_model(diff_inputs, t)

        target_mask = observed_mask - cond_mask
        imputation_loss = self.calculate_mse(predicted_mean, observed_data, target_mask)
        reconstruction_loss  = 0
        for X_tilde in X_finals:
            reconstruction_loss += self.calculate_mse(X_tilde, observed_data, cond_mask)
        reconstruction_loss /= len(X_finals)
        loss = imputation_loss + reconstruction_loss
        # residual = (noise - predicted_mean) * target_mask
        # num_eval = target_mask.sum()
        # loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        print(f"loss: {loss}\npredicted: {predicted_mean}")
        return loss

    def evaluate(self, data, n_samples):
        _, X_art, X_intact, observed_mask, indicating_mask = data

        with torch.no_grad():
            cond_mask = indicating_mask
            target_mask = observed_mask - cond_mask

            # side_info = self.get_side_info(X_intact, cond_mask)
            samples = self.impute(X_intact, cond_mask, observed_mask, n_samples)

            # for i in range(len(cut_length)):  # to avoid double evaluation
            #     target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, X_intact, target_mask, observed_mask #, observed_tp

    def impute(self, observed_data, cond_mask, observerd_mask, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L)#.to(self.device)
        n_steps = self.diff_steps
        for i in range(n_samples):

            current_sample = torch.randn_like(observed_data)

            for t in range(n_steps - 1, -1, -1):
                cond_obs = cond_mask * observed_data
                noisy_target = (1 - cond_mask) * current_sample
                diff_input = cond_obs + noisy_target # torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                diff_inputs = {'X': diff_input, 'missing_mask': observerd_mask}
                ts = (torch.ones(B) * t).long()
                predicted_mean, _ = self.diff_model(diff_inputs, ts)
                print(f"Sample {i} T = {t}:\nalphas: {self.alphas[t]}\nalphas_hat: {self.alpha_hats[t]}\npredicted noise: {predicted_mean}")
                # coeff1 = 1 / torch.sqrt(self.alphas[t])
                # coeff2 = self.betas[t] / torch.sqrt(1.0 - self.alpha_hats[t])
                # print(f"coeff1: {coeff1}\n\ncoeff2: {coeff2}")
                # current_sample = coeff1 * (current_sample - coeff2 * predicted)
                current_sample = predicted_mean
                print(f"Pre-variance sample: {current_sample}")

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        ((1.0 - self.alpha_hats[t - 1]) / (1.0 - self.alpha_hats[t])) * self.betas[t]
                    ) ** 0.5
                    # sigma = self.sigma[t]
                    print(f"Sigma: {sigma}")
                    print(f"Noise: {noise}")
                    current_sample += sigma * noise
                print(f"Curr: \n{current_sample}")
            current_sample = cond_mask * observed_data + (1 - cond_mask) * current_sample
            print(f"Current Sample {i}:\n{current_sample}")
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, data, is_train=True):
        _, X_art, X_intact, observed_mask, indicating_mask = data
        cond_mask = self.get_randmask(indicating_mask)
        return self.calc_loss(X_art, cond_mask, indicating_mask)


