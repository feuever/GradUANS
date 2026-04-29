from collections import namedtuple
import torch
import time
import einops
import numpy as np
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations values uncertainties')  # 添加uncertainties字段


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs
        self.previous_trajectories = None

    def __call__(self, conditions, batch_size=1, unsafe_bounds_box=None, unsafe_bounds_circle=None, warm_start_steps=None, verbose=True, id_model=None):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)
        if unsafe_bounds_box is not None:
            unsafe_bounds_box = self._format_unsafe_bounds(unsafe_bounds_box)
            conditions.update({'unsafe_bounds_box': unsafe_bounds_box})

        if unsafe_bounds_circle is not None:
            unsafe_bounds_circle = self._format_unsafe_bounds(unsafe_bounds_circle)
            conditions.update({'unsafe_bounds_circle': unsafe_bounds_circle})
        
        conditions.update({'dims': torch.tensor([0 + self.action_dim, 2 + self.action_dim])})

        if (warm_start_steps is not None) and (self.previous_trajectories is not None):
            x_warmstart = torch.cat((self.previous_trajectories[:,1:,:], self.previous_trajectories[:,-1,:].unsqueeze(1)), dim=1)
            conditions.update({'x_warmstart': x_warmstart})
            conditions.update({'n_warmstart_steps': warm_start_steps})

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)


        if hasattr(samples, 'uncertainties'):
            uncertainties = utils.to_np(samples.uncertainties)
        else:
            uncertainties = None

        ## extract observations [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        ## extract action [ batch_size x horizon x action_dim ]
        if self.action_dim > 0:
            actions = trajectories[:, :, :self.action_dim]
            actions = self.normalizer.unnormalize(actions, 'actions')

            ## extract first action
            action = actions[0, 0]
        else:
            actions = None
            if id_model is not None:
                with torch.no_grad():
                    obs = normed_observations[0, 0]
                    next_obs = normed_observations[0, 1]
                    normed_action = id_model(torch.tensor(obs).float(), torch.tensor(next_obs).float()).detach().numpy()
                    action = self.normalizer.unnormalize(normed_action, 'actions')
            else:
                action = None

        trajectories = Trajectories(actions, observations, samples.values, uncertainties)  # 包含不确定度

        self.previous_trajectories = samples.trajectories

        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cpu')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
    
    def _format_unsafe_bounds(self, unsafe_bounds):
        '''
            unsafe_bounds : dict of lists of obs_dim x 2 arrays
                { t: [ [x_min, x_max], [y_min, y_max] ] }
            unsafe_bounds_formatted : dict of (action_dim + obs_dim) x (2 * n_obs) arrays
                { t: [ x_min, x_max, y_min, y_max ] }
        '''
    
        unsafe_bounds_formatted = {}
        for i, _ in unsafe_bounds.items():
            unsafe_bounds_formatted[i] = np.zeros((self.action_dim + self.diffusion_model.observation_dim, 2 * len(unsafe_bounds[i])))
            for n_obs in range(len(unsafe_bounds[i])):
                if self.action_dim > 0:
                    unsafe_bounds_formatted[i][:self.action_dim, 2 * n_obs] = self.normalizer.normalize(unsafe_bounds[i][n_obs][:self.action_dim, 0], 'actions')
                    unsafe_bounds_formatted[i][:self.action_dim, 2 * n_obs + 1] = self.normalizer.normalize(unsafe_bounds[i][n_obs][:self.action_dim, 1], 'actions')
                
                # 获取观测数据并确保形状正确
                obs_data = unsafe_bounds[i][n_obs][self.action_dim:, 0]
                # 如果观测数据维度小于归一器期望的维度，需要填充
                if len(obs_data) < self.diffusion_model.observation_dim:
                    # 创建完整维度的数组并填充
                    full_obs_data = np.zeros(self.diffusion_model.observation_dim)
                    full_obs_data[:len(obs_data)] = obs_data
                    # 对于缺失的维度，使用归一器的最小值（或合理默认值）填充
                    if hasattr(self.normalizer.normalizers['observations'], 'mins'):
                        full_obs_data[len(obs_data):] = self.normalizer.normalizers['observations'].mins[len(obs_data):]
                    obs_data = full_obs_data
                
                unsafe_bounds_formatted[i][self.action_dim:, 2 * n_obs] = self.normalizer.normalize(obs_data, 'observations')
                
                # 同样的处理用于上界
                obs_data_upper = unsafe_bounds[i][n_obs][self.action_dim:, 1]
                if len(obs_data_upper) < self.diffusion_model.observation_dim:
                    full_obs_data_upper = np.zeros(self.diffusion_model.observation_dim)
                    full_obs_data_upper[:len(obs_data_upper)] = obs_data_upper
                    if hasattr(self.normalizer.normalizers['observations'], 'maxs'):
                        full_obs_data_upper[len(obs_data_upper):] = self.normalizer.normalizers['observations'].maxs[len(obs_data_upper):]
                    obs_data_upper = full_obs_data_upper
                unsafe_bounds_formatted[i][self.action_dim:, 2 * n_obs + 1] = self.normalizer.normalize(obs_data_upper, 'observations')
        
        unsafe_bounds_formatted = utils.to_torch(unsafe_bounds_formatted, dtype=torch.float32, device=self.device)
        return unsafe_bounds_formatted