import sqlite3
import pandas as pd
import numpy as np
import datetime
import itertools
import json
import matplotlib.pyplot as plt


"""
Plots
"""
def get_annotation(x, y, annotation):
    dct = {}
    for i, label in enumerate(annotation):
        key = (x[i], y[i])
        if key not in dct.keys():
            dct[key] = []
        dct[key].append(label)
        
    return dct


def one_session(x, y, avg_price=None, annotation=None):

    _x, _y = x[1:-1], y[1:-1]
    plt.step(x, y, where='post')
    plt.scatter(_x, _y, marker='x', c='red', alpha=0.3)
    if avg_price is not None:
        plt.plot(x, np.ones_like(y)*avg_price, '--', c='grey', alpha=0.3)

    if annotation is not None:
        dct = get_annotation(x, y, annotation)
        for coords, label in dct.items():
            label = str(label)[1:-1]
            plt.annotate(label, coords)


def show_session(xs, ys, avg_prices=None, annotations=None, title='Session', figsize=12):
    
    fig = plt.figure(figsize=(figsize, figsize))

    if avg_prices is None:
        avg_prices = [None] * len(xs)
    
    if annotations is None:
        annotations = [None] * len(xs)
        
    for x, y, avg_price, annotation in zip(xs, ys, avg_prices, annotations):       
        one_session(x, y, avg_price=avg_price, annotation=annotation)

    if title is not None:
        plt.title(title)
    plt.xlabel('time')
    plt.ylabel('price')
    plt.show()       
        
        
def global_trend(df, title=None, figsize=12):
    
    _df = add_xs_ys(df)
    coords = []
    
    for ax, val in zip([[], []], [_df.xxs, _df.yys]):
        lst = list(val)
        for sublst in lst:
            for x in sublst:
                ax.append(x)
        coords.append(ax)
        
    xs, ys = coords
    xs, ys = [xs], [ys] 
    show_session(xs, ys, title=title, figsize=figsize)
    

def global_trend_gaps(df, title=None, figsize=12):
    
    _df = add_xs_ys(df)
    xs, ys, avg_prices, annotations = [], [], [], []
    step = 0
    
    for i, session in _df.iterrows():
        from_start = np.array(session.from_start) + step
#         print(from_start)
        xs.append(from_start)
        ys.append(session.yys)
        avg_prices.append(session.avg_price)
        annotations.append(session.deal_id)
        step = from_start[-1]
#         title = f"session: {session.date}"
        
    show_session(xs, ys, avg_prices=avg_prices, title=title, figsize=figsize)
    
        
def all_session(df, figsize=12):
    
    _df = add_xs_ys(df)
    for i, session in _df.iterrows():
        xs = [session.xxs]
        ys = [session.yys]
        avg_prices = [session.avg_price]
        annotations = [session.deal_id]
        title = f"session: {session.date}"
        
        show_session(xs, ys, avg_prices=avg_prices, title=title, figsize=figsize)
        
        
# def all_session_one_pic(df, figsize=12):
#     xs, ys, avg_prices, annotations = [], [], [], []
#     for i, session in df.iterrows():
#         xs.append(session.from_start)
#         ys.append(session.price)
#         avg_prices.append(session.avg_price)
#         annotations.append(session.deal_id)
# #         title = f"session: {session.date}"
        
#     show_session(xs, ys, avg_prices=avg_prices, figsize=figsize)
    
    
def show_session_dates(df, dates, figsize=12):
    
    for date in dates: 
        _df = df[df.date == date]
        all_session(_df, figsize=figsize)
        
        
"""
Tables
"""
def tables_union(chart_data, trading_session):
    
    new_df = chart_data.copy()

    new_df['date'] = np.nan
    new_df['trading_type'] = np.nan
    new_df['platform_id'] = np.nan

    for _, row in trading_session.iterrows():
        new_df.loc[new_df.session_id == row.id, 'date'] = row.date
        new_df.loc[new_df.session_id == row.id, 'trading_type'] = row.trading_type
        new_df.loc[new_df.session_id == row.id, 'platform_id'] = row.platform_id

    new_df['space'] = ' '
    new_df['timestamp'] = new_df.date + new_df.space + new_df.time
    new_df.timestamp = pd.to_datetime(new_df.timestamp)
    new_df.index = new_df.timestamp

    new_df = new_df.drop(columns = ['space', 'time', 'date', 'timestamp'])   
    # new_df = new_df.drop(columns = ['space', 'time', 'date'])  

    return new_df


def separate_types(new_df):

    plat1_day =   new_df[(new_df['platform_id'] == 1) & (new_df['trading_type'] == 'daily')]
    plat1_month = new_df[(new_df['platform_id'] == 1) & (new_df['trading_type'] == 'monthly')]
    plat2_day =   new_df[(new_df['platform_id'] == 2) & (new_df['trading_type'] == 'daily')]
    plat2_month = new_df[(new_df['platform_id'] == 2) & (new_df['trading_type'] == 'monthly')]

    # Проверим, что никакая из сессий не принадлежит одновременно хотя бы 2м типам
    lst = [plat1_day, plat1_month, plat2_day, plat2_month]
    for pair in itertools.combinations(lst, 2):
        type1, type2 = pair
        assert len(set(type1.session_id.tolist()) & set(type2.session_id.tolist())) == 0
    print("Никакая из сессий не принадлежит одновременно хотя бы 2м типам")
    
    # Отсортируем значения по времени 
    sorted_lst = [df.sort_values(by=['timestamp']) for df in lst]
    
    return sorted_lst


def groupby_session_id(df):
    
    # Отсортируем по времени
    _df = df.copy()
    _df = _df.sort_values(by=['timestamp'])
    _df = _df.reset_index()
    
    data = {}
    for column in _df.columns:
        data[f"{column}"] = _df.groupby(['session_id'])[f"{column}"].apply(np.array)
        
    new_df = pd.DataFrame(data)  
    for column in ['session_id', 'trading_type', 'platform_id']:
        new_df[f'{column}'] = new_df[f'{column}'].apply(np.unique).apply(np.squeeze) 
            
    # check time consistency
    for i, row in new_df.iterrows():
        assert (row.timestamp == sorted(row.timestamp)).all()
            
    # check session durance less than 1 day
    first_deal = new_df.timestamp.apply(np.min)    
    last_deal = new_df.timestamp.apply(np.max)
    delta = last_deal - first_deal 
    assert (delta < pd.Timedelta("1 days")).all()
    
    # начало и конец сессии
    trading_type = new_df.trading_type.unique()[0]
    typical_full_delta = pd.Timedelta('61T') if trading_type == 'monthly' else pd.Timedelta('31T')
    
    if trading_type == 'monthly':
        start = first_deal.dt.floor(freq='H')
#         end = start + pd.Timedelta('61T')
#         end = start + typical_full_delta                                                                             
    else:
        start = first_deal.dt.floor(freq='30T')
#         end = start + pd.Timedelta('31T')
#         end = start + typical_full_delta

    end = start + typical_full_delta                                                                          
    end[end < last_deal] = last_deal.dt.ceil(freq='H') + pd.Timedelta('1T')
    full_delta = end - start  
        
    new_df['date'] = first_deal.dt.date
    
    new_df['first_deal'] = first_deal.dt.time
    new_df['last_deal'] = last_deal.dt.time
    new_df['delta'] = delta
    
    new_df['start'] = start
    new_df['end'] = end
    new_df['full_delta'] = full_delta
        
    # typical
    new_df['session_type'] = 'typical'
    new_df.loc[new_df.full_delta > typical_full_delta, 'session_type'] = 'untypical'    
        
    # sort, indexing    
    new_df = new_df.sort_values(by=['date'])
    new_df = new_df.drop(columns=['session_id'])
    new_df = new_df.reset_index()
    
#     # временные отрезки
#     new_df['only_time'] = new_df.timestamp.apply(lambda x: [pd.Timestamp(i).time() for i in x])
    
    # посчитаем средневзвешенную цену(взвешиваем по объему):
    new_df['deal_count'] = new_df.deal_id.apply(len)
    new_df['total_size'] = new_df.lot_size.apply(np.sum)
    new_df['avg_price'] = new_df.price * new_df.lot_size
    new_df.avg_price = new_df.avg_price.apply(np.sum) / new_df.total_size
    
    # добавим стартовую цену и вынесем отдельно последнюю цену
    new_df['last_price'] = pd.Series(price[-1] for price in new_df.price)    
    new_df['start_price'] = new_df.avg_price.copy()
    new_df.start_price = new_df.start_price.shift()
#     new_df.loc[~new_df.start_price.notna(), 'start_price'] = 0
    
#     # кол-во (60 секунд) от начала торговли
#     new_df['from_start'] = [new_df.timestamp[i] - new_df.start[i].to_numpy() for i in range(len(new_df))]
#     new_df.from_start = new_df.from_start / (60 * 10**9)
#     new_df.from_start = [new_df.from_start[i].tolist() for i in range(len(new_df))]
    
    return new_df


def add_xs_ys(df):
    _df = df.copy()
    _df['xxs'] = [np.concatenate(([start.to_numpy()], timeseries, [end.to_numpy()])) \
                  for timeseries, start, end in zip(_df.timestamp, _df.start, _df.end)]
    
    _df['yys'] = [np.concatenate(([start_price], price, [last_price])) \
                  for price, start_price, last_price in zip(_df.price, _df.start_price, _df.last_price)]

    # кол-во минут от начала торговли
    _df['from_start'] = [row.xxs - row.start.to_numpy() for _, row in _df.iterrows()]
    _df.from_start = _df.from_start / (60 * 10**9)
    _df.from_start = [row.from_start.tolist() for _, row in _df.iterrows()]
    
    return _df


"""
Sesson time
"""
def stat_print(correct_deal, incorrect_deal):
    
    print('Correct_deals')
    print(f"      date |                    | cnt | total size")
    for row in correct_deal:
        print(f"{row.date} | {row.first_deal}  {row.last_deal} | {f'{row.deal_count:3}'} | {f'{row.total_size:4}'}")
    
    print()
    print('Incorrect_deals')
    print(f"      date |                    | cnt | total size")
    for row in incorrect_deal:
        print(f"{row.date} | {row.first_deal}  {row.last_deal} | {f'{row.deal_count:3}'} | {f'{row.total_size:4}'}")    


def separate_deals(df, limits):

    correct_deal = []
    incorrect_deal = []

    for i, row in df.sort_values(by=['last_deal']).iterrows():
        correct = False
        for (start, finish) in limits:
            if row.first_deal >= start and row.last_deal <= finish:
                correct_deal.append(row)
                correct = True
                break
        if not correct:
            incorrect_deal.append(row)

    stat_print(correct_deal, incorrect_deal)