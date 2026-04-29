import os
import torch
import numpy as np
import diffuser.utils as utils
from torch import nn
import torch.nn.functional as F
from envs.quad_2d import Quad2DEnv
from envs.pointmass import PointMassEnv
from diffuser.models.mlp import MLP

# 全局变量声明
obs_dim = None
action_dim = None
dataset_id_observations = None
dataset_id_observations_next = None
dataset_id_actions = None

class EnsembleMLP(nn.Module):
    def __init__(self, n_models, input_size, output_size):
        super().__init__()
        self.models = nn.ModuleList([
            MLP(input_size, output_size)
            for _ in range(n_models)
        ])
        
    def forward(self, x):
        return torch.stack([model(x) for model in self.models], dim=1)

def train_ensemble(models, observations, observations_next, actions, save_dir, n_models=5, steps=100000, batch_size=128, lr=1e-3):
    targets = actions  

    if os.path.exists(save_dir):
        for file in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, file))
    else:
        os.makedirs(save_dir)
    dataset = torch.utils.data.TensorDataset(observations, observations_next, targets)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    for m in range(n_models):
        model = models.models[m]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        best_test_loss = float('inf')
        epochs = int(steps / len(train_loader))
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for current_batch, next_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = model(current_batch, next_batch)
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for current_batch, next_batch, y_batch in test_loader:
                    pred = model(current_batch, next_batch)
                    loss = F.mse_loss(pred, y_batch)
                    test_loss += loss.item()
            
            print(f'Model {m} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Test Loss: {test_loss/len(test_loader):.4f}')
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_path = os.path.join(save_dir, f'model_{m}.pt')
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model {m} to {save_path}')
    pass

def ensemble_predict(models, obs, obs_next, return_variance=False):
    with torch.no_grad():
        inputs = torch.cat([obs, obs_next], dim=-1)
        preds = models(inputs)  # [B, M, A]
        variance = torch.var(preds, dim=1, unbiased=False)  # 修正：在模型维度（dim=1）计算方差
        return variance
    pass

if __name__ == '__main__':

    exp = 'quad2d'  # 'pointmass' or 'quad2d'
    class Parser(utils.Parser):
        dataset: str = exp
        config: str = 'config.' + exp
    if exp == 'pointmass':
        env = PointMassEnv(target=None, max_steps=20, epsilon=0.2, reset_target_reached=False, bonus_reward=False, 
                    reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=1000)
    else:
        env = Quad2DEnv(min_rel_thrust=0.75, max_rel_thrust=1.25, 
                    max_rel_thrust_difference=0.01, target=None, max_steps=20,
                    epsilon=0.5, reset_target_reached=False, bonus_reward=False, 
                    reset_out_of_bounds=True, theta_as_sine_cosine=True, num_episodes=1000)
    

    args = Parser().parse_args('diffusion')
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=env,
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        use_actions=True,
        max_path_length=args.max_path_length,
    )
    
    dataset = dataset_config()
    
    obs_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    
    n_episodes = 1000
    observations = []
    observations_next = []
    actions = []
    
    for _ in range(n_episodes):
        termination_idx = np.where(dataset.fields['terminals'][_, :])[0][0]
        observations.append(torch.tensor(dataset.fields['normed_observations'][_, :termination_idx - 1]))
        observations_next.append(torch.tensor(dataset.fields['normed_observations'][_, 1:termination_idx]))
        actions.append(torch.tensor(dataset.fields['normed_actions'][_, :termination_idx - 1]))
    
    dataset_id_observations = torch.cat(observations, dim=0)
    dataset_id_observations_next = torch.cat(observations_next, dim=0)
    dataset_id_actions = torch.cat(actions, dim=0)
    
    n_models = 50
    model = EnsembleMLP(
        n_models=n_models,
        input_size=obs_dim*2,
        output_size=action_dim
    )
    
    save_dir = f'logs/{exp}/ensemble_inverse_dynamics/H{args.horizon}_T{args.n_diffusion_steps}'

    train_ensemble(
        model, 
        dataset_id_observations,
        dataset_id_observations_next,
        dataset_id_actions,
        save_dir=save_dir,
        n_models=n_models,
        steps=2e5
    )

