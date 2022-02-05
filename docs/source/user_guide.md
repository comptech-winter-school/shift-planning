


## Описание операций

Порядок действий для установки и запуска Продукта:

### main
Для установки необходимых библиотек необходимо выполнить:
```shell
  pip install -r requirements.txt
```

Для получения прогноза спроса, числа курьеров и распределения по сменам на следующую неделю необходимо подключить файл ```main.py```. После этого запустить функцию:
```python
  main() -> (pd.DataFrame, pd.DataFrame)
```

### Stage 1.
Для получения распределения заказов на следующую неделю неделю необходимо подключить файл ```stage1_main.py```. 
После этого запустить функцию:
```python
  get_next_week(path: str) -> pd.DataFrame
```
### Stage 2.
Для получения оптимального распределения курьеров на следующую неделю неделю необходимо подключить файл ```stage2_main.py```. 
После этого запустить функцию:
```python
  get_optimal_partners(path_orders: str, path_partners_delays: str, pred_orders: pd.DataFrame) -> pd.DataFrame
```
### Stage 3.
Для получения оптимального распределения курьеров по сменам необходимо подключить файл ```stage3_main.py```. 
После этого запустить функцию:
```python
  get_partners_distribution(pred_partners: pd.DataFrame) -> pd.DataFrame
```
