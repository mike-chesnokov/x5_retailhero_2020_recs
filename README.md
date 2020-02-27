# Решение задачи X5 Retailhero 2020 №2 (рекомендательная система)
- 9 место
- check NMAP: 0,1467
- public NMAP: 0,1286
- private NMAP: 0,143019 

## Описание задачи
https://retailhero.ai/c/recommender_system/overview

Участникам необходимо разработать сервис, 
который сможет отвечать на запросы с предсказаниями будущих покупок клиента
и при этом держать высокую нагрузку. 
По информации о клиенте и его истории покупок необходимо 
построить ранжированный список товаров, 
которые клиент наиболее вероятно купит в следующей покупке. 

## Описание решения

- 1 уровень: implicit.nearest_neighbours.TFIDFRecommender 


## Локальный запуск решения

- Скачиваем docker image __chesnokovmike/python-ds:retailhero2__ 
([Dockerfile](Dockerfile) для сборки образа на основе __geffy/ds-base:retailhero__)

```text
docker pull chesnokovmike/python-ds:retailhero2
```
- Переходим в папку решения и запускаем сервер
```text
cd solution
docker run \
    -v `pwd`:/workspace \ 
    -v `realpath ../solution`:/workspace/solution \ 
    -w /workspace \
    -p 8000:8000  \  
    chesnokovmike/python-ds:retailhero2 \    
    gunicorn --bind 0.0.0.0:8000 server:app
``` 
- В другом терминале выполняем проверку на check файле
```text
python run_queries.py http://localhost:8000/recommend data/check_queries.tsv
```
- Получаем check score = __0.146726__