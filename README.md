# Решение задачи X5 Retailhero 2020 №2 (рекомендательная система)
- 9 место
- check nmap: 0,1467
- public nmap: 0,1286
- private nmap: 0,143019 

## Описание задачи
https://retailhero.ai/c/recommender_system/overview

Участникам необходимо разработать сервис, 
который сможет отвечать на запросы с предсказаниями будущих покупок клиента
и при этом держать высокую нагрузку. 
По информации о клиенте и его истории покупок необходимо 
построить ранжированный список товаров, 
которые клиент наиболее вероятно купит в следующей покупке. 

## Описание решения


## Локальный запуск решения

- Скачиваем docker image __chesnokovmike/python-ds:retailhero2__

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
- в другом терминале выполняем проверку на check файле
```text
python run_queries.py http://localhost:8000/recommend data/check_queries.tsv
```
- Получаем check score = __0.146726__