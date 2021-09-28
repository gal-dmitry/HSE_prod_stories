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


def one_session(x, y, start_point, end_point, avg_price=None, annotation=None):
    
    start_price, start_time = start_point
    end_price, end_time = end_point
    _x = np.concatenate(([start_time], x, [end_time]))
    _y = np.concatenate(([start_price], y, [end_price]))
    
    plt.step(_x, _y, where='post')
    plt.scatter(x, y, marker='x', c='red', alpha=0.3)
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
    
    
def trend_session(df, title=None, figsize=12):
    
    coords = []
    
    for ax, val in zip([[], []], [df.timestamp, df.price]):
        lst = list(val)
        for sublst in lst:
            for x in sublst:
                ax.append(x)
        coords.append(ax)
        
    xs, ys = coords
    xs, ys = [xs], [ys] 
    show_session(xs, ys, title=title, figsize=figsize)
    
    
def all_session(df, figsize=12):
    for i, session in df.iterrows():
        xs = [session.timestamp]
        ys = [session.price]
        avg_prices = [session.avg_price]
        annotations = [session.deal_id]
        title = f"session: {session.date}"
        
        show_session(xs, ys, avg_prices=avg_prices, title=title, figsize=figsize)
        
        
def all_session_one_pic(df, figsize=12):
    xs, ys, avg_prices, annotations = [], [], [], []
    for i, session in df.iterrows():
        xs.append(session.from_start)
        ys.append(session.price)
        avg_prices.append(session.avg_price)
        annotations.append(session.deal_id)
#         title = f"session: {session.date}"
        
    show_session(xs, ys, avg_prices=avg_prices, figsize=figsize)
    

def all_session_one_pic_step(df, figsize=12):
    xs, ys, avg_prices, annotations = [], [], [], []
    step = 0
    for i, session in df.iterrows():
        from_start = np.array(session.from_start) + step
#         print(from_start)
        xs.append(from_start)
        ys.append(session.price)
        avg_prices.append(session.avg_price)
        annotations.append(session.deal_id)
        step = from_start[-1]
#         title = f"session: {session.date}"
        
    show_session(xs, ys, avg_prices=avg_prices, figsize=figsize)
    

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


def groupby_session_id(_df):
    
    # Отсортируем по времени
    df = _df.copy()
    df = df.sort_values(by=['timestamp'])
    df = df.reset_index()
    
    data = {}
    for column in df.columns:
        data[f"{column}"] = df.groupby(['session_id'])[f"{column}"].apply(np.array)
        
    new_df = pd.DataFrame(data)  
    for column in ['session_id', 'trading_type', 'platform_id']:
        new_df[f'{column}'] = new_df[f'{column}'].apply(np.unique).apply(np.squeeze) 
            
    # check time consistency
    for i, row in new_df.iterrows():
        assert (row.timestamp == sorted(row.timestamp)).all()
            
    # check data
    first_deal = new_df.timestamp.apply(np.min)    
    last_deal = new_df.timestamp.apply(np.max)
    delta = last_deal - first_deal 
    assert (delta < pd.Timedelta("1 days")).all()
    start = first_deal.dt.floor(freq='H')
    
    new_df['date'] = first_deal.dt.date
    new_df['first_deal'] = first_deal.dt.time
    new_df['last_deal'] = last_deal.dt.time
    new_df['delta'] = delta
    new_df['start'] = start
    
    new_df = new_df.sort_values(by=['date'])
    new_df = new_df.drop(columns=['session_id'])
    new_df = new_df.reset_index()
    
    # временные отрезки
    new_df['only_time'] = new_df.timestamp.apply(lambda x: [pd.Timestamp(i).time() for i in x])
    
    # посчитаем средневзвешенную цену(взвешиваем по объему):
    new_df['deal_count'] = new_df.deal_id.apply(len)
    new_df['total_size'] = new_df.lot_size.apply(np.sum)
    new_df['avg_price'] = new_df.price * new_df.lot_size
    new_df.avg_price = new_df.avg_price.apply(np.sum) / new_df.total_size
    
    # добавим стартовую цену
    new_df['start_price'] = new_df.avg_price.copy()
    new_df.start_price = new_df.start_price.shift()
#     new_df.loc[~new_df.start_price.notna(), 'start_price'] = 0
    
    # кол-во секунд от начала торговли
    new_df['from_start'] = [new_df.timestamp[i] - new_df.start[i].to_numpy() for i in range(len(new_df))]
    new_df.from_start = new_df.from_start / (10**9)
    new_df.from_start = [new_df.from_start[i].tolist() for i in range(len(new_df))]
    
    return new_df


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