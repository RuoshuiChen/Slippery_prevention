import numpy as np
from mjrl.utils.fc_network import FCNetwork
import torch
from torch.autograd import Variable


class MLP:
    def __init__(self, env_spec,
                 hidden_sizes=(64,64),
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = FCNetwork(self.n, self.m, hidden_sizes)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)  # another trainable parameter -> action noise
        self.trainable_params = list(self.model.parameters()) + [self.log_std]  # only parameters of self.model are trainable.
        # Old Policy network 
        # ------------------------
        self.old_model = FCNetwork(self.n, self.m, hidden_sizes)
        self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()  # share the same weights as self.model
        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value (only apply the lower bound)
            # [-1] -> self.log_std
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling (used to generate noise for the control actions)
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel()) 
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = \
                torch.clamp(self.old_params[-1], self.min_log_std).data

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel() # FC network when using MLP as policy (self.model = FCNetwork(self.n, self.m, hidden_sizes))
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise  # the noise is very important, especially at the early stage -> exploration to the new state-action pairs
        # the noise magnitude depends on the log_std_val. It decides how much we trust our policy for the good actions. This parameter is algo trainable.
        # I think during training, this value should be smaller which means that we are confident that the current policy would result in better results.
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):  # old model and new model 
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        if type(observations) is not torch.Tensor:
            obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        else:
            obs_var = observations
        if type(actions) is not torch.Tensor:
            act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        else:
            act_var = actions
        mean = model(obs_var)  # re-run the policy to get the actions wrt to each traj at each time step, should be very similar to the input actions?  (No!, noise-free computation!) 
        # we are considering the act_var as the label data for a supervised learning, seems to be
        # this value would be timed with advantage value: if the noisy action(input:actions) leads to the good results with higher rewards, the advantage value is larger
        # which would make the policy learn to generate the similar outputs as this set of noisy actions (exploration).
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):  # i believe this function was used in the first version,
        # then they changed to likelihood_ratio function even for the batch_reinforce (vpg) training.
        # in Pieter's paper, for vpg training, they used only log_likelihood.
        mean, LL = self.mean_LL(observations, actions, model, log_std)  
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)  # mean -> exact policy outputs (without noise), LL -> some kind of error
        return [LL, mean, self.old_log_std] 

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0] # error of the noisy actions and noise-free actions, the old and new policy are the same 
        # -> LR = tensor([1., 1., 1.,  ..., 1., 1., 1.], grad_fn=<ExpBackward0>)
        LL_new = new_dist_info[0]   
        LR = torch.exp(LL_new - LL_old)  # LL_new and LR have a very significant difference
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):   # same as in Pieter's code
        old_log_std = old_dist_info[2]  # old_log_std
        new_log_std = new_dist_info[2]  # log_std
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]  # actions generated by old policy, these are noise-free policy outputs 
        new_mean = new_dist_info[1]  # actions generated by new policy
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2  # numerator
        # In Pieter's codes:
        # kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        Dr = 2 * new_std ** 2 + 1e-8  # denominator
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)
