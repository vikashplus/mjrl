from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import mjrl.envs
import time as timer
import os

SEED = 500
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = FILE_DIR+'/test_headless_1'

# -------------------------------
# Ant
e = GymEnv('mjrl_swimmer-v0')
policy = MLP(e.spec, hidden_sizes=(32,32), seed=SEED, init_log_std=-0.25)
baseline = QuadraticBaseline(e.spec)
agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)

print("training policy ..................")
ts = timer.time()
train_agent(job_name=EXP_DIR,
            agent=agent,
            seed=SEED,
            niter=5,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=1,
            sample_mode='trajectories',
            num_traj=10,
            save_freq=1,
            evaluation_rollouts=None)
print("time taken = %f" % (timer.time()-ts))
print("testing offscreen rendering")
e.env.env.visualize_policy_offscreen(policy, e.horizon,
                                     num_episodes=10,
                                     frame_size=(128,128),
                                     mode='exploration',
                                     save_loc=EXP_DIR+'/',
                                     filename=e.env_id+'_test_',
                                     camera_name=None, # use default camera
                                     device_id=0,
                                     )