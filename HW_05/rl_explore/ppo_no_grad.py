import os
from os.path import join
import sys
import datetime
import argparse

import ray
import ray.rllib.agents.ppo as ppo
from PIL import Image
from ray import tune

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")
from mapgen import Dungeon, ModifiedDungeon
from ppo_train import create_config, ray_launch

    
def define_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument("--traj_count", type=int, default=5)
    parser.add_argument('--env', type=str, default="Dungeon")
    parser.add_argument("--load_path", type=str, default="./tmp/ppo/dungeon/checkpoint_000005-checkpoint-5")
    parser.add_argument("--no_grad_gif_dir", type=str, default="./save/no_grad/gifs")
    args = parser.parse_args()    
    
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M')
    args.no_grad_gif_dir += f"/{timestamp}"
     
    if args.env == "Dungeon":
        args.env = Dungeon
    elif args.env == "ModifiedDungeon":
        args.env = ModifiedDungeon
    else:
        raise ValuError("unknown environment")
        
    return args

    
def dir_check(args):
    save_dir = args.no_grad_gif_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

def eval(agent, args):
    env = args.env(20, 20, 3, min_room_xy=5, max_room_xy=10, vision_radius=5)
    
    for n in range(args.traj_count):
        print(f"trajectory No {n+1}")
        obs = env.reset()
        frames = []
        
        # make gif
        for _ in range(500):
            action = agent.compute_single_action(obs)
            frame = \
            Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
            frames.append(frame)

            obs, _, done, _ = env.step(action)
            if done:
                break

        out_path = join(args.no_grad_gif_dir, f"traj_{str.zfill(str(n+1), 3)}.gif")
        frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0, duration=1000/60)

    print("Done!")

    
if __name__ == "__main__":
    
    args = define_args()
    ray_launch(args)
    dir_check(args)

    config = create_config(args)
    agent = ppo.PPOTrainer(config)
    agent.restore(args.load_path)
    
    eval(agent, args)
    
    
    
    
    