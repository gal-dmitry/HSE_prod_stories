import os
import sys
import shutil
from gym import spaces

import csv
import datetime
from os.path import join

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")
from mapgen import Dungeon, ModifiedDungeon
from utils import update_writer_training, update_writer_trajectory, seed_everything



def create_config(ENVIRONMENT, SEED):
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["log_level"] = "INFO"
    config["framework"] = "torch"
    config["env"] = f"{ENVIRONMENT}"
    config["seed"] = SEED
    config["env_config"] = {
        "width": 20,
        "height": 20,
        "max_rooms": 3,
        "min_room_xy": 5,
        "max_room_xy": 10,
        "observation_size": 11,
        "vision_radius": 5
    }

    config["model"] = {
        "conv_filters": [
            [16, (3, 3), 2],
            [32, (3, 3), 2],
            [32, (3, 3), 1],
        ],
        "post_fcnet_hiddens": [32],
        "post_fcnet_activation": "relu",
        "vf_share_layers": False,
    }


    config["rollout_fragment_length"] = 100
    config["entropy_coeff"] = 0.1
    config["lambda"] = 0.95
    config["vf_loss_coeff"] = 1.0
    
    return config
    
    

def ray_launch(ENVIRONMENT, CHECKPOINT_DIR, RAY_RESULTS_DIR, GIF_DIR):    
    
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env(f"{ENVIRONMENT}", lambda config: ENVIRONMENT(**config))
    
    for directory in [CHECKPOINT_DIR, RAY_RESULTS_DIR, GIF_DIR]:
        shutil.rmtree(directory, ignore_errors=True, onerror=None)
        if not os.path.exists(directory):
            os.makedirs(directory)

    
    
def train(agent, writer_training, timestamp, N_ITER, ENVIRONMENT, CHECKPOINT_DIR, GIF_DIR):

    for n in range(N_ITER):
        result = agent.train()
        writer_training = update_writer_training(writer_training, result, n)
    
        # save and sample trajectory
        if (n + 1) % 5 == 0:
            
            file_name = agent.save(CHECKPOINT_DIR)
            print(f"saved at {file_name}")
            
            env = ENVIRONMENT(20, 20, 3, min_room_xy=5, max_room_xy=10, vision_radius=5)
            seed_everything(SEED, env=env)
            obs = env.reset()
            
            frames = []
            writer_trajectory = SummaryWriter(log_dir=f"tf_trajectory/{timestamp}/{str.zfill(str(n+1), 3)}")
            
            for _ in range(500):
                action = agent.compute_single_action(obs)

                frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
                frames.append(frame)

                obs, reward, done, info = env.step(action)
                writer_trajectory = update_writer_trajectory(writer_trajectory, info, reward, _)                        
                if done:
                    break

            out_path = join(GIF_DIR, f"{str.zfill(str(n+1), 3)}.gif")
            frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0, duration=1000/60)
                     
            
    
if __name__ == "__main__":
    
    SEED = 666
    N_ITER = 500
    ENVIRONMENT = Dungeon
    
    CHECKPOINT_DIR = "tmp/ppo/dungeon"
    RAY_RESULTS_DIR = "./save/ray_results"
    GIF_DIR = "./save/gifs"

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M')
    writer_training = SummaryWriter(log_dir=f"tf_training/{timestamp}")
    
    ray_launch(ENVIRONMENT, CHECKPOINT_DIR, RAY_RESULTS_DIR, GIF_DIR)
    config = create_config(ENVIRONMENT, SEED)
    agent = ppo.PPOTrainer(config)
    train(agent, writer_training, timestamp, N_ITER, ENVIRONMENT, CHECKPOINT_DIR, GIF_DIR)
    
    
    