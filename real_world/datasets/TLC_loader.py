# standard library dependencies
import os
import pickle
from typing import List

# external dependencies
import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

def load_TLC_temporal_edgelist(fname:str, day_range:List[int] , num_nodes:int=266) -> List[nx.Graph]:
    df = pd.read_csv(fname, sep=',', header=None, index_col=None, skiprows=1)
    groups = df.groupby(df[0])
    G_times = []
    days_found = []
    for date in groups.groups.keys():
        date_df = df.loc[groups.groups[date],:]
        t = date_df.iloc[:,0]
        assert len(np.unique(t)) == 1
        if np.unique(t) not in day_range: 
            continue
        else:
            days_found.append(date)
            u = date_df.iloc[:,1].astype(int)
            v = date_df.iloc[:,2].astype(int)
            assert max(u) <= num_nodes
            assert max(v) <= num_nodes
            w = date_df.iloc[:,3].astype(int)
            
            coo_adj = coo_matrix(
                (
                    w,
                    (u,v)
                ), shape=(num_nodes, num_nodes)
            )
            G = nx.from_numpy_matrix(coo_adj.todense(), create_using=nx.DiGraph)

            G_times.append(G)

    assert days_found == day_range
    
    print("# days: " + str(len(G_times)), " ranging from ", min(day_range), " to ", max(day_range))
    return G_times

def load_2016_TLC(file_names: List[str], keep_feb_1st: bool = False) -> List[List[nx.Graph]]:
    """
    """
    num_nodes = 264
    day_range = list(range(0,93)) if keep_feb_1st else list(range(0,92))
    timecourse_dataset = [ 
        load_TLC_temporal_edgelist(fname, day_range , num_nodes=num_nodes) 
        for fname in file_names
    ]
    assert len(set([len(timecourse_data) for timecourse_data in timecourse_dataset])) == 1
    for view in timecourse_dataset:
        for graph in view:
            assert graph.number_of_nodes() == num_nodes
    return timecourse_dataset

def load_TLC(   nviews:int, paths:List[str]=None, 
                day_range:List[int]=None, num_nodes:int=266,
                TLC_data_dir_path:str=None) -> List[List[nx.Graph]]:
    assert nviews in [3,4]
    if TLC_data_dir_path is None:
        TLC_data_dir_path = r"D:\NYC-TLC\2020\cleaned_data"
    if paths is None:
        file_names = [
            'green_tripdata_2020.edgelist.txt',
            'yellow_tripdata_2020.edgelist.txt',
            'fhv_tripdata_2020.edgelist.txt'
        ]
        if nviews == 4: 
            file_names.append('fhvhv_tripdata_2020.edgelist.txt')

        paths = [ 
            os.path.join(
                TLC_data_dir_path,
                file_name
            ) 
            for file_name in file_names
        ]

    if nviews == 3:
        if day_range is None:
            day_range = list(range(0,334))
        else: # nviews == 3, specified range
            assert 0 <= min(day_range) <= max(day_range) <= 333
        
    else: # nviews == 4
        if day_range is None:
            # skip february
            day_range = list(range(28,242))
        else:
            assert 0 <= min(day_range) <= max(day_range) <= 333

    print("ordering of returned timecourse datasets: ", file_names)

    timecourse_dataset = [ 
        load_TLC_temporal_edgelist(fname, day_range , num_nodes=num_nodes) 
        for fname in paths
    ]
    assert len(set([len(timecourse_data) for timecourse_data in timecourse_dataset])) == 1
    for view in timecourse_dataset:
        for graph in view:
            assert graph.number_of_nodes() == num_nodes
    return timecourse_dataset

if __name__ == '__main__':
    load_TLC(3)
    load_TLC(4)