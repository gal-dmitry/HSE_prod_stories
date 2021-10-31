from mapgen import Dungeon


class ModifiedDungeon(Dungeon):
    """Use this class to change the behavior of the original env (e.g. remove the trajectory from observation, like here)"""
    def __init__(self,
        width=20,
        height=20,
        max_rooms=3,
        min_room_xy=5,
        max_room_xy=12,
        max_steps: int = 2000
    ):
        observation_size = 11
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size = 11,
            vision_radius = 5,
            max_steps = max_steps
        )

        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 3]) # because we remove trajectory and leave only cell types (UNK, FREE, OCCUPIED)
        self.action_space = spaces.Discrete(3)

    def step(self):
        observation, reward , done, info = super().step()
        observation = observation[:, :, :-1] # remove trajectory
        return observation, reward , done, info
    
    
    