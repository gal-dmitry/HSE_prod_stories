import os
import pandas as pd
from math import sqrt


def read_in(name='in'):
    
    
    dt = pd.read_csv(f"{name}.txt", names=['x', 'y'], sep=" ")
    assert not dt.isnull().values.any()
    
    dt = dt.sort_values(by=['x', 'y'])
    dt['y_rank'] = dt.y.rank(ascending=False)
    
    return dt


def write_out(new_dt, name='out'):
    new_dt.to_csv(f"{name}.txt", sep=' ', header=False, index=False)
    

def get_R1_R2(dt, p):
    
    R1 = dt[:p].y_rank.sum()
    R2 = dt[-p:].y_rank.sum()
    
    return R1, R2


def get_err(N, p):    
    return round((N + 0.5) * sqrt(p/6))


def get_measure_conjugacy(diff, N, p):
    measure = diff / (p * (N - p))
    rounded = format(measure, '.2f')
    return rounded


def get_all_stat(dt):
    
    N = dt.shape[0]
    assert N >= 9
    p = round(N / 3)    
    R1, R2 = get_R1_R2(dt, p)
    
    diff = round(R1 - R2)
    std_err = get_err(N, p)    
    measure_conjugacy = get_measure_conjugacy(diff, N, p)
    
    all_stat = {'diff': [diff],
                'std_err': [std_err],
                'measure_conjugacy': [measure_conjugacy]}
    
    return all_stat


def main():
    
    file_path = f"{os.getcwd()}/in.txt"
    if not os.path.isfile(file_path):
        print("Файл in.txt не существует")
        return 
    
    try:
        dt = read_in()
    except:
        print("Неправильный формат входных данных")
        return
    
    try:    
        all_stat = get_all_stat(dt)
    except:
        print("Для этого метода N должно быть равно по меньшей мере 9")
        return
    
    new_dt = pd.DataFrame(data=all_stat)
    write_out(new_dt)
    print("входной файл: int.txt")
    print("результаты: out.txt")
    
    
if __name__ == '__main__':
    main()
