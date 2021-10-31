### Files

- [код тренировки](ppo_example.py)
- [checkpoints](./tmp/ppo/dungeon)
- [ray logging](./save/ray_results)
- [gif-результаты](./save/gifs)
- [tensorflow: логгирование тренировки](./tf_training)
- [tensorflow: логгирование траектории](./tf_trajectory)
- [агент](./mapgen/mapgen/agent.py)
- [стандартное окружение](./mapgen/mapgen/env.py)
- [новое окружение](./mapgen/mapgen/env_modified.py)
- [генератор среды](./mapgen/mapgen/dungeon.py)
- [карта](./mapgen/mapgen/map.py)



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



