from gym import spaces
from mapgen import Dungeon


class ModifiedDungeon(Dungeon):
    """Use this class to change the behavior of the original env (e.g. remove the trajectory from observation, like here)"""
    def __init__(self,
        width=20,
        height=20,
        max_rooms=3,
        min_room_xy=5,
        max_room_xy=12,
        max_steps: int = 2000,
        gamma=0.99,
        observation_size: int = 11,
        vision_radius = 5
    ):
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size=observation_size,
            vision_radius = vision_radius,
            max_steps = max_steps
        )
        
        # because we remove trajectory and leave only cell types (UNK, FREE, OCCUPIED)
        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 4])
        self.action_space = spaces.Discrete(3)
        self.gamma = gamma
        
        
    def step(self):

        observation, reward, done, info = super().step()
        
#         info = {
#             "step": self._step,
#             "total_cells": self._map._visible_cells,
#             "total_explored": self._map._total_explored,
#             "new_explored": explored,
#             "avg_explored_per_step": self._map._total_explored / self._step,
#             "moved": moved
          
        step = info["step"]
        collide = not info["moved"]
        full_completion = info["total_cells"] == info["total_explored"]
        
        if collide:
            reward -= 0.1  
        
        if full_completion:
            reward += 1
        
        reward = reward * self.gamma**step
        
        return observation, reward , done, info
    
    
    
    
    
    
    
    