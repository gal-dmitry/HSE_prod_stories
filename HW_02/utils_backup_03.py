import sqlite3
import datetime
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn.preprocessing import normalize


"""
Plot utils
"""
def get_annotation(x, y, annotation):
    dct = {}
    for i, label in enumerate(annotation):
        key = (x[i], y[i])
        if key not in dct.keys():
            dct[key] = []
        dct[key].append(label)
        
    return dct


def one_session(x, y, avg_price=None, annotation=None, show_deals=True):

    plt.step(x, y, where='post', label='real price')
    if show_deals:
        _x, _y = x[1:-1], y[1:-1]
        plt.scatter(_x, _y, marker='x', c='red', alpha=0.3, label='deals')
    if avg_price is not None:
        plt.plot(x, np.ones_like(y)*avg_price, '--', c='grey', alpha=0.3, label='average price')
    if annotation is not None:
        dct = get_annotation(x, y, annotation)
        for coords, label in dct.items():
            label = str(label)[1:-1]
            plt.annotate(label, coords)


def show_session(xs, ys, avg_prices=None, annotations=None, show_deals=True, title='[ADD NAME]', figsize=12, legend=True):
    
    fig = plt.figure(figsize=(figsize, figsize))

    if avg_prices is None:
        avg_prices = [None] * len(xs)
    
    if annotations is None:
        annotations = [None] * len(xs)
        
    for x, y, avg_price, annotation in zip(xs, ys, avg_prices, annotations):       
        one_session(x, y, avg_price=avg_price, annotation=annotation, show_deals=show_deals)
    
    plt.xlabel('time')
    plt.ylabel('price')
    plt.xticks(rotation = 45)
    if title is not None:
        plt.title(title)
    if legend:
        plt.legend()
    plt.show()       
 
        
"""
History plots
"""
def global_trend(df, show_deals=True, figsize=12):
    
    title = f"[Type: {np.unique(df.trading_type)[0]}] [Platform: {int(np.unique(df.platform_id)[0])}]"
    
#     _df = add_xs_ys(df)
    coords = []
    
    for ax, val in zip([[], []], [_df.xxs, _df.yys]):
        lst = list(val)
        for sublst in lst:
            for x in sublst:
                ax.append(x)
        coords.append(ax)
        
    xs, ys = coords
    xs, ys = [xs], [ys] 
    show_session(xs, ys, title=title, show_deals=show_deals, figsize=figsize)
    

def global_trend_gaps(df, show_deals=False, figsize=12):
    
    title = f"[Type: {np.unique(df.trading_type)[0]}] [Platform: {int(np.unique(df.platform_id)[0])}]"
    
#     _df = add_xs_ys(df)
    xs, ys, avg_prices, annotations = [], [], [], []
    step = 0
    
    for i, session in _df.iterrows():
        from_start = np.array(session.from_start) + step
        xs.append(from_start)
        ys.append(session.yys)
        avg_prices.append(session.avg_price)
        annotations.append(session.deal_id)
        step = from_start[-1]
        
    show_session(xs, ys, avg_prices=avg_prices, show_deals=show_deals, title=title, figsize=figsize, legend=False)
    
 
"""
Separate sessions
"""
def _all_sessions(df, show_deals=True, figsize=12):
    
#     _df = add_xs_ys(df)
    for i, session in _df.iterrows():
        xs = [session.xxs]
        ys = [session.yys]
        avg_prices = [session.avg_price]
        annotations = [session.deal_id]
        title = f"session: {session.date}"
        
        show_session(xs, ys, avg_prices=avg_prices, show_deals=show_deals, title=title, figsize=figsize)
            
    
def sessions_by_date(df, dates, show_deals=True, figsize=12):
    
    for date in dates: 
        _df = df[df.date == date]
        _all_sessions(_df, show_deals=show_deals, figsize=figsize)
               
        
"""
Binarized plots
"""
def _all_binarized_sessions(df, normalize=False, show_deals=False, figsize=12, legend=False, title=None):
    
    values = df.avg_min_price if not normalize else df.norm_avg_min_price
    for y in values:
        xs = [list(range(y.shape[0]))]
        ys = [y]
        show_session(xs, ys, show_deals=show_deals, title=title, figsize=figsize, legend=legend)
    

def binarized_sessions_by_date(df, dates, normalize=False, show_deals=False, figsize=12, legend=False):
    
    for date in dates:
        title = f"date: {date}"
        _df = df[df.date == date]
        _all_binarized_sessions(_df, normalize=normalize, show_deals=show_deals, figsize=figsize, legend=legend, title=title)
        
        
def one_pic_all_binarized_session(df, normalize=False, show_deals=False, figsize=12, legend=False):
    
    xs, ys = [], []
    values = df.avg_min_price if not normalize else df.norm_avg_min_price
    for y in values:
        xs.append(list(range(y.shape[0])))
        ys.append(y)
    show_session(xs, ys, show_deals=show_deals, figsize=figsize, legend=legend, title='All sessions')
    
        
"""
Dataframe utils
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
    
    # session start, session end
    trading_type = new_df.trading_type.unique()[0]
    typical_full_delta = pd.Timedelta('61T') if trading_type == 'monthly' else pd.Timedelta('31T')   
    start = first_deal.dt.floor(freq='H') if trading_type == 'monthly' else first_deal.dt.floor(freq='30T')
    end = start + typical_full_delta                                                                          
    end[end < last_deal] = last_deal.dt.ceil(freq='H') + pd.Timedelta('1T')
    full_delta = end - start  
        
    # add features    
    new_df['date'] = first_deal.dt.date
    new_df['first_deal'] = first_deal.dt.time
    new_df['last_deal'] = last_deal.dt.time
    new_df['delta'] = delta  
    new_df['start'] = start
    new_df['end'] = end
    new_df['full_delta'] = full_delta
    new_df['session_type'] = 'typical'
    new_df.loc[new_df.full_delta > typical_full_delta, 'session_type'] = 'untypical'    
        
    # sort, indexing    
    new_df = new_df.sort_values(by=['date'])
    new_df = new_df.drop(columns=['session_id'])
    new_df = new_df.reset_index()
        
    # посчитаем средневзвешенную цену(взвешиваем по объему):
    new_df['deal_count'] = new_df.deal_id.apply(len)
    new_df['total_size'] = new_df.lot_size.apply(np.sum)
    new_df['avg_price'] = new_df.price * new_df.lot_size
    new_df.avg_price = new_df.avg_price.apply(np.sum) / new_df.total_size
    
    # добавим стартовую цену и вынесем отдельно последнюю цену
    new_df['last_price'] = pd.Series(price[-1] for price in new_df.price)    
    new_df['start_price'] = new_df.avg_price.copy()
    new_df.start_price = new_df.start_price.shift()
    very_beginning_price = deepcopy(new_df.price[0][0])
    new_df.loc[0, 'start_price'] = very_beginning_price #start price for 1st session
#     new_df.loc[~new_df.start_price.notna(), 'start_price'] = 
    
#     временные отрезки
#     new_df['only_time'] = new_df.timestamp.apply(lambda x: [pd.Timestamp(i).time() for i in x])

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


def normalize_vector(x):
    return normalize(x[:,np.newaxis], axis=0).ravel()


def add_avg_min_price(df):
    
    _df = df.copy()
    series = []
    norm_series = []
    
    for _, row in _df.iterrows():

        min_start = 0
        min_end = row.from_start[-1]
        avg_min_price = np.zeros(min_end)
        
        row.from_start = np.array(row.from_start[1:-1])        
        if row.from_start[0] != 0:       
            avg_min_price[0] = row.start_price
            min_start += 1

        for i in range(min_start, min_end):
            idx = np.where(row.from_start == i)[0]
            if idx.shape[0] == 0:
                avg_min_price[i] = avg_min_price[i - 1]
            else:
                avg_min_price[i] = np.sum(row.price[idx] * row.lot_size[idx]) / np.sum(row.lot_size[idx])

        series.append(avg_min_price)
        norm_series.append(normalize_vector(avg_min_price))  
        
    _df['avg_min_price'] = series
    _df['norm_avg_min_price'] = norm_series
    
    return _df


"""
Print utils
"""
def stat_print(typical_deal, untypical_deal, only_untypical=True):
    
    if not only_untypical:
        print('Typical_sessions:')
        print()
        print(f"      date |                    | cnt | total size")
        for row in typical_deal:
            print(f"{row.date} | {row.first_deal}  {row.last_deal} | {f'{row.deal_count:3}'} | {f'{row.total_size:4}'}")
        print()
        print()
        
    print('Untypical_sessions:')
    print()
    print(f"      date |                    | cnt | total size")
    for row in untypical_deal:
        print(f"{row.date} | {row.first_deal}  {row.last_deal} | {f'{row.deal_count:3}'} | {f'{row.total_size:4}'}")    
    print()
    print()

    
def separate_sessions(df):

    typical_deal = []
    untypical_deal = []

    for i, row in df.sort_values(by=['last_deal']).iterrows():        
        if row.session_type == 'typical':
            typical_deal.append(row)
        else:
            untypical_deal.append(row)
    stat_print(typical_deal, untypical_deal)