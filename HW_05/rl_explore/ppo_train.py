import os
import sys
import shutil
from gym import spaces
    
import csv
import datetime
import argparse
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

    
def define_args(): 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--n_iter', type=int, default=500)
    parser.add_argument('--env', type=str, default="Dungeon") 
    parser.add_argument('--agent_ckpt_dir', type=str, default="./tmp/ppo/dungeon")
    parser.add_argument('--ray_result_dir', type=str, default="./save/train/ray_results")
    parser.add_argument('--train_gif_dir', type=str, default="./save/train/gifs")    
    args = parser.parse_args()
    
    if args.env == "Dungeon":
        args.env = Dungeon
    elif args.env == "ModifiedDungeon":
        args.env = ModifiedDungeon
    else:
        raise ValuError("unknown environment")
        
    return args


def create_config(args):
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["log_level"] = "INFO"
    config["framework"] = "torch"
    config["env"] = f"{args.env}"
    config["seed"] = args.seed
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
      

def ray_launch(args):
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    tune.register_env(f"{args.env}", lambda config: args.env(**config))
    
    
def dir_check(args):
    for directory in [args.agent_ckpt_dir, args.ray_result_dir, args.train_gif_dir]:
        shutil.rmtree(directory, ignore_errors=True, onerror=None)
        if not os.path.exists(directory):
            os.makedirs(directory)

    
    
def train(agent, writer_training, timestamp, args):

    for n in range(args.n_iter):
        result = agent.train()
        writer_training = update_writer_training(writer_training, result, n)
    
        # save and sample trajectory
        if (n + 1) % 5 == 0:
            
            file_name = agent.save(args.agent_ckpt_dir)
            print(f"saved at {file_name}")
            
            env = args.env(20, 20, 3, min_room_xy=5, max_room_xy=10, vision_radius=5)
            seed_everything(args.seed, env=env)
            obs = env.reset()
            
            frames = []
            writer_trajectory = SummaryWriter(log_dir=f"tf_trajectory/{timestamp}/{str.zfill(str(n+1), 3)}")
            
            for _ in range(500):
                action = agent.compute_single_action(obs)

                frame = \
                Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
                frames.append(frame)

                obs, reward, done, info = env.step(action)
                writer_trajectory = update_writer_trajectory(writer_trajectory, info, reward, _)                        
                if done:
                    break

            out_path = join(args.train_gif_dir, f"{str.zfill(str(n+1), 3)}.gif")
            frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0, duration=1000/60)

    
    
if __name__ == "__main__":
    
    args = define_args()
    ray_launch(args)
    dir_check(args)
    config = create_config(args)
    
    agent = ppo.PPOTrainer(config)
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M')
    writer_training = SummaryWriter(log_dir=f"tf_training/{timestamp}")
    
    train(agent, writer_training, timestamp, args)
    
    
    