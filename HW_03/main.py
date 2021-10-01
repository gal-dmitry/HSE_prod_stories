import pandas as pd


def read_in(name='in'):
    
    dt = pd.read_csv(f"{name}.txt", names=['x', 'y'], sep=" ")
    dt = dt.sort_values(by=['x', 'y'])
    dt['y_rank'] = dt.y.rank(ascending=False)
    
    return dt


def write_out(new_dt, name='out'):
    new_dt.to_csv(f"{name}.txt", sep=' ', header=False, index=False)
    

def get_R1_R2(dt, p):
    
    R1 = dt[:p].y_rank.sum()
    R2 = dt[-p:].y_rank.sum()
    
    return R1, R2


def check_norm(diff):
    'change'
    return 1


def get_measure_conjugacy(diff, N, p):
    measure = diff / (p * (N - p))
    rounded = format(measure, '.2f')
    return rounded


def get_all_stat(dt):
    
    N = dt.shape[0]
    assert N >= 9, 'Для этого метода N должно быть равно по меньшей мере 9'
    
    p = round(N / 3)    
    R1, R2 = get_R1_R2(dt, p)
    diff = R1 - R2
    
    std_err = check_norm(diff)    
    measure_conjugacy = get_measure_conjugacy(diff, N, p)
    
    all_stat = {'diff': [diff],
                'std_err': [std_err],
                'measure_conjugacy': [measure_conjugacy]}
    
    return all_stat


def main():
    
    dt = read_in()
    all_stat = get_all_stat(dt)
    new_dt = pd.DataFrame(data=all_stat)
    write_out(new_dt)
    
    
if __name__ == '__main__':
    main()
