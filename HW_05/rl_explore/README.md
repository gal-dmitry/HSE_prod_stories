### Files

- [`ppo_example.py`](ppo_example.py) - код тренировки
- [`./tmp/ppo/dungeon`](./tmp/ppo/dungeon) - checkpoints
- [`./save/ray_results`](./save/ray_results) - ray logging
- [`./save/gifs`](./save/gifs) - gif-результаты
- [`./tf_training`](./tf_training) - tensorflow: логгирование тренировки
- [`./tf_trajectory`](./tf_trajectory) - tensorflow: логгирование траектории
- [`./mapgen/mapgen/agent.py`](./mapgen/mapgen/agent.py) - агент
- [`./mapgen/mapgen/env.py`](./mapgen/mapgen/env.py) - стандартное окружение
- [`./mapgen/mapgen/env_modified.py`](./mapgen/mapgen/env_modified.py) - новое окружение
- [`./mapgen/mapgen/dungeon.py`](./mapgen/mapgen/dungeon.py) - генератор среды
- [`./mapgen/mapgen/map.py`](./mapgen/mapgen/map.py) - карта



### Environment

#### Некоторые поля environment, важные для формирования награды

- `self._max_steps` - максимально возможное кол-во шагов
- `self.vision_radius` - радиус видимости
- `self._step` - номер текущего шага
- `self._map._visible_cells` - кол-во клеток на карте
- `self._map._total_explored` - кол-во клеток, которые агент уже просмотрел
- `success = self._total_explored == self._visible_cells` - просмотрены все возможные клетки
- `explored` - кол-во новых исследованных на текущем шаге клеток
- `moved` - совершил ли агент движение на текущем шаге



