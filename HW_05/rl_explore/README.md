### Training 

- [`ppo_train.py`](ppo_train.py) - код тренировки
- [`./tmp/ppo/dungeon`](./tmp/ppo/dungeon) - checkpoints
- [`./save/train/ray_results`](./save/train/ray_results) - ray logging
- [`./save/train/gifs`](./save/train/gifs) - gif-результаты
- [`./tf_training`](./tf_training) - tensorflow: логгирование тренировки
- [`./tf_trajectory`](./tf_trajectory) - tensorflow: логгирование траектории

Пример запуска:

```
python ppo_train.py \
--seed 666 \
--n_iter 500 \
--env Dungeon \
--agent_ckpt_dir ./tmp/ppo/dungeon \
--train_gif_dir ./save/train/gifs \
--ray_result_dir ./save/train/ray_results
```


### Evaluation 

- [`ppo_no_grad.py`](ppo_no_grad.py) - код evaluation
- [`./save/no_grad/gifs`](./save/no_grad/gifs) - gif-результаты 

Пример запуска:

```
python ppo_no_grad.py \
--seed 666 \
--traj_count 5 \
--env Dungeon \
--load_path ./tmp/ppo/dungeon/checkpoint_000020/checkpoint-20 \
--no_grad_gif_dir ./save/no_grad/gif
```


### Environment


#### files

- [`./mapgen/mapgen/agent.py`](./mapgen/mapgen/agent.py) - агент
- [`./mapgen/mapgen/env.py`](./mapgen/mapgen/env.py) - стандартное окружение
- [`./mapgen/mapgen/env_modified.py`](./mapgen/mapgen/env_modified.py) - новое окружение
- [`./mapgen/mapgen/dungeon.py`](./mapgen/mapgen/dungeon.py) - генератор среды
- [`./mapgen/mapgen/map.py`](./mapgen/mapgen/map.py) - карта


#### Некоторые поля, важные для формирования награды

- `self._max_steps` - максимально возможное кол-во шагов
- `self.vision_radius` - радиус видимости
- `self._step` - номер текущего шага
- `self._map._visible_cells` - кол-во клеток на карте
- `self._map._total_explored` - кол-во клеток, которые агент уже просмотрел
- `success = self._total_explored == self._visible_cells` - просмотрены все возможные клетки
- `explored` - кол-во новых исследованных на текущем шаге клеток
- `moved` - совершил ли агент движение на текущем шаге



