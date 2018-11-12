import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN:
    def __init__(self, env_spec,
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None):

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
        self.model = MuNet(self.m)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        # Old Policy network
        # ------------------------
        self.old_model = MuNet(self.m)
        self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

        # map the context correctly
        # ------------------------
        self.cpu() # by default

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
            # clip std at minimum value
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
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

    def cpu(self):
        self.device = 'cpu'
        self.model.cpu()
        self.old_model.cpu()
        self.log_std = self.log_std.cpu()
        self.old_log_std = self.old_log_std.cpu()
        self.obs_var = self.obs_var.cpu()

    def cuda(self):
        self.device = 'cuda'
        self.model.cuda()
        self.old_model.cuda()
        self.log_std = self.log_std.cuda()
        self.old_log_std = self.old_log_std.cuda()
        self.obs_var = self.obs_var.cuda()

    def map_context(self, var):
        if self.device == 'cpu':
            return var.cpu()
        elif self.device == 'cuda':
            return var.cuda()
        else:
            print("unknown device")
            quit()

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = observation.copy()
        o = np.expand_dims(o, axis=0)
        self.obs_var.data = self.map_context(torch.from_numpy(o).permute(0, 3, 1, 2).float())
        if self.device == 'cpu':
            mean = self.model(self.obs_var).data.numpy().ravel()
        elif self.device == 'cuda':
            mean = self.model(self.obs_var).data.cpu().numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        obs_var = self.map_context(obs_var.permute(0, 3, 1, 2))
        act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        act_var = self.map_context(act_var)
        mean = model(obs_var)
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)


class MuNet(nn.Module):
    def __init__(self, act_dim,
                 out_shift = None,
                 out_scale = None):
        super(MuNet, self).__init__()

        self.act_dim = act_dim
        self.set_transformations(out_shift, out_scale)

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.fc1   = nn.Linear(115200, 128)
        self.fc2   = nn.Linear(128, act_dim)

    def set_transformations(self, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(out_shift=out_shift, out_scale=out_scale)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)
        self.out_shift = Variable(self.out_shift, requires_grad=False)
        self.out_scale = Variable(self.out_scale, requires_grad=False)

    def forward(self, x):
        out = x / 255.0
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = self.pool1(out)
        out = F.tanh(self.conv3(out))
        out = out.view(out.size()[0], -1)
        out = F.tanh(self.fc1(out))
        out = self.fc2(out)
        out = out * self.out_scale + self.out_shift
        return out

    def cuda(self):
        super(MuNet, self).cuda()
        self.out_shift = self.out_shift.cuda()
        self.out_scale = self.out_scale.cuda()

    def cpu(self):
        super(MuNet, self).cpu()
        self.out_shift = self.out_shift.cpu()
        self.out_scale = self.out_scale.cpu()