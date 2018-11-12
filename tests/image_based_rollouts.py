from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.policies.gaussian_cnn import CNN as CNNPolicy
import mjrl.envs
from mjrl.samplers.trajectory_sampler import sample_paths
import time as timer
SEED = 500

e = GymEnv('mjrl_swimmer-v0')
policy = MLP(e.spec)

# sample paths with MLP policy but render images

paths = sample_paths(5, policy, e.horizon, env=e, pegasus_seed=123, mode='sample',
                     get_image=True,
                     get_image_args=dict(frame_size=(128, 128),
                                         camera_name=None,
                                         device_id=0),
                     image_based=False
                     )


eval_paths =  sample_paths(5, policy, e.horizon, env=e, pegasus_seed=123, mode='evaluation',
                           get_image=True,
                           get_image_args=dict(frame_size=(128, 128),
                                               camera_name=None,
                                               device_id=0),
                           image_based=False
                           )

# sample paths with CNN policy

policy = CNNPolicy(e.spec)
policy.cuda() # will do forward prop on the GPU (~ 25x faster than CPU) | comment out if no GPU in machine
paths = sample_paths(5, policy, e.horizon, env=e, pegasus_seed=123, mode='sample',
                     get_image=True,
                     get_image_args=dict(frame_size=(128, 128),
                                         camera_name=None,
                                         device_id=0),
                     image_based=True,
                     )