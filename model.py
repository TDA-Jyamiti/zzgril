import numpy as np
import os
from pyfzz import pyfzz
from typing import List
from gudhi import AlphaComplex
from joblib import Parallel, delayed
import itertools
from operator import add
import torch

def intersection_two_lists(list1, list2):
    t1 = map(tuple, list1)
    t2 = map(tuple, list2)
    l = list(map(list, set(t1).intersection(t2)))
    l.sort(key=len)
    return l
        
def set_subtraction_two_lists(list1, list2):
    assert len(list1) >= len(list2)
    t1 = map(tuple, list1)
    t2 = map(tuple, list2)
    l = list(map(list, set(t1).difference(t2)))
    l.sort(key=len)
    return l

def set_subtraction_two_lists_reverse(list1, list2):
    assert len(list1) >= len(list2)
    t1 = map(tuple, list1)
    t2 = map(tuple, list2)
    l = list(map(list, set(t1).difference(t2)))
    l.sort(key=len, reverse=True)
    return l

def union_two_lists(list1, list2):
    t1 = map(tuple, list1)
    t2 = map(tuple, list2)
    l = list(map(list, set(t1).union(t2)))
    l.sort(key=len)
    return l

def get_list_of_tuples_from_list_of_lists(l: list):
    return list(map(tuple, l))
    

class zzMultipersGraph:
    def __init__(self, num_center_pts: int,
                 num_graphs_in_seq: int, 
                 num_vertices: int) -> None:
        self.num_center_pts = num_center_pts
        self.num_graphs_in_seq = num_graphs_in_seq
        self.num_divisions_for_filtration = (2 * self.num_graphs_in_seq) - 1
        self.center_pts = self.sample_center_pts()
        self.gen_rank_val_at_d = np.ones((2, self.num_center_pts, self.num_divisions_for_filtration)) * -1
        self.ranks_dmax = np.zeros((2, 2, self.num_center_pts))
        self.num_vertices = num_vertices
    
    def sample_center_pts(self):
        np.random.seed(0)
        center_pts = np.random.randint(2, self.num_divisions_for_filtration - 2, 
                                       size=(self.num_center_pts, 2))
        return center_pts
    
    def get_filt(self, seq_of_graphs: list, edge_weights: list):
        filt_dict = {}
        vertices = [[i] for i in range(self.num_vertices)]
        vertex_filt_vals = [0 for i in range(self.num_vertices)]
        vertex_filt = np.array(list(zip(vertices, vertex_filt_vals)), dtype=object)
        for graph_idx in range(len(seq_of_graphs)):
            edges = seq_of_graphs[graph_idx]
            # edges = [[edge[0], edge[1]] for edge in edges]
            edge_filt_vals = edge_weights[graph_idx]
            edge_filt = np.array(list(zip(edges, edge_filt_vals)), dtype=object)
            filt_dict[graph_idx] = np.concatenate((vertex_filt, edge_filt))
        return filt_dict

    def get_simplices_birth_times_vertical(self, alpha_filt_dict: dict, edge_weights: list):
        simplices_birth_times = {}
        simplices = {}
        simplex_birth_times_pairs = {}
        simplices_grouped_by_birth_times = {}
        for i in range(len(alpha_filt_dict)):
            simplices_birth_times[i] = np.ceil(alpha_filt_dict[i][:,1] * (1/np.max(np.array(edge_weights[i]))) * (self.num_divisions_for_filtration - 1))
            sorted_idcs = np.argsort(simplices_birth_times[i])
            simplices_birth_times[i] = simplices_birth_times[i][sorted_idcs]
            simplices[i] = alpha_filt_dict[i][:,0]
            simplices[i] = simplices[i][sorted_idcs]
            temp1 = map(tuple, simplices[i])
            simplex_birth_times_pairs[i] = dict(zip(temp1, simplices_birth_times[i]))
            temp = {val: [c for _, c in g] for val, g in itertools.groupby(zip(simplices_birth_times[i], simplices[i]), key = lambda x: x[0])}
            simplices_grouped_by_birth_times[i] = temp
            for levels in range(self.num_divisions_for_filtration):
                if levels not in simplices_grouped_by_birth_times[i]:
                    simplices_grouped_by_birth_times[i][levels] = []
        return simplices_birth_times, simplices, simplex_birth_times_pairs, simplices_grouped_by_birth_times
    
    def get_pts_along_bdry_of_worm(self, center_pt_idx:int, width: int):
        # We implement l = 2 case. Need to implement others?
        pts_along_bdry, direction = [], []
        bdry_start_x = self.center_pts[center_pt_idx][0]
        bdry_start_y = self.center_pts[center_pt_idx][1] - (2 * width)
        pts_along_bdry.append((bdry_start_x, bdry_start_y))
        
        # lower staircase
        flag = True # True for vertical, False for horizontal
        for i in range(4 * width):
            if flag:
                pts_along_bdry.append((pts_along_bdry[-1][0], pts_along_bdry[-1][1] + 1))
                direction.append('u')
                flag = False
            else:
                pts_along_bdry.append((pts_along_bdry[-1][0] - 1, pts_along_bdry[-1][1]))
                direction.append('l')
                flag = True
                
        # top-left inverted L shape
        for i in range(2 * width):
            pts_along_bdry.append((pts_along_bdry[-1][0], pts_along_bdry[-1][1] + 1))
            direction.append('u')
        
        for i in range(2 * width):
            pts_along_bdry.append((pts_along_bdry[-1][0] + 1, pts_along_bdry[-1][1]))
            direction.append('r')
            
        # upper staircase
        flag = True # True for vertical, False for horizontal
        for i in range(4 * width):
            if flag:
                pts_along_bdry.append((pts_along_bdry[-1][0], pts_along_bdry[-1][1] - 1))
                direction.append('d')
                flag = False
            else:
                pts_along_bdry.append((pts_along_bdry[-1][0] + 1, pts_along_bdry[-1][1]))
                direction.append('r')
                flag = True
        
        # bottom-right inverted L shape
        for i in range(2 * width):
            pts_along_bdry.append((pts_along_bdry[-1][0], pts_along_bdry[-1][1] - 1))
            direction.append('d')
            
        for i in range(2 * width - 1):
            pts_along_bdry.append((pts_along_bdry[-1][0] - 1, pts_along_bdry[-1][1]))
            direction.append('l')
        
        direction = direction[:-1]
        assert len(pts_along_bdry) == (16 * width)
        
        pts_along_bdry = np.array(pts_along_bdry)
        pts_along_bdry[pts_along_bdry < 0] = 0
        pts_along_bdry[pts_along_bdry >= self.num_divisions_for_filtration] = self.num_divisions_for_filtration - 1
        
        return pts_along_bdry, direction
    
    def get_simplices_added_on_zigzag_arrows(self, simplices: dict, simplex_birth_times_pairs: dict, simplices_grouped_by_birth_time: dict):
        simplices_added_on_right_arrows = {i: {j : [] for j in range(len(simplices_grouped_by_birth_time[i]))} for i in range(len(simplices_grouped_by_birth_time))}
        simplices_added_on_left_arrows = {i : {j : [] for j in range(len(simplices_grouped_by_birth_time[i]))} for i in range(len(simplices_grouped_by_birth_time))}
        simplices_deleted_on_up_arrows_union = {i : {j : [] for j in range(len(simplices_grouped_by_birth_time[i]))} for i in range(len(simplices_grouped_by_birth_time) - 1)}
        simplices_added_on_up_arrows_union = {i : {j : [] for j in range(len(simplices_grouped_by_birth_time[i]))} for i in range(len(simplices_grouped_by_birth_time) - 1)}
        
        for i in range(len(simplices_grouped_by_birth_time)):
            if i == 0:
                for curr_birth_time in simplices_grouped_by_birth_time[i]:
                    for simplex in simplices_grouped_by_birth_time[i][curr_birth_time]:
                        flag = False
                        try:
                            next_birth_time = simplex_birth_times_pairs[i+1][tuple(simplex)]
                        except KeyError:
                            flag = True
                            next_birth_time = self.num_divisions_for_filtration - 1
                        if next_birth_time < curr_birth_time:
                            for k in range(next_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_deleted_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][next_birth_time].append(simplex)
                            for k in range(next_birth_time, curr_birth_time):
                                simplices_added_on_right_arrows[i][k].append(simplex)
                        elif next_birth_time > curr_birth_time:
                            for k in range(curr_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][curr_birth_time].append(simplex)
                            for k in range(curr_birth_time, next_birth_time):
                                simplices_added_on_left_arrows[i+1][k].append(simplex)
                            if flag:
                                simplices_added_on_left_arrows[i+1][next_birth_time].append(simplex)
                        else:
                            for k in range(curr_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][curr_birth_time].append(simplex)
                            if flag:
                                simplices_added_on_left_arrows[i+1][curr_birth_time].append(simplex)
                                
            elif i == len(simplices_grouped_by_birth_time) - 1:
                for curr_birth_time in simplices_grouped_by_birth_time[i]:
                    for simplex in simplices_grouped_by_birth_time[i][curr_birth_time]:
                        flag = False
                        try:
                            prev_birth_time = simplex_birth_times_pairs[i-1][tuple(simplex)]
                        except KeyError:
                            flag = True
                            prev_birth_time = self.num_divisions_for_filtration - 1
                        if prev_birth_time < curr_birth_time:
                            for k in range(prev_birth_time, curr_birth_time):
                                if simplex not in simplices_added_on_left_arrows[i][k]:
                                    simplices_added_on_left_arrows[i][k].append(simplex)
                        elif prev_birth_time > curr_birth_time:
                            for k in range(curr_birth_time, prev_birth_time):
                                if simplex not in simplices_added_on_right_arrows[i-1][k]:
                                    simplices_added_on_right_arrows[i-1][k].append(simplex)
                            if flag:
                                if simplex not in simplices_added_on_right_arrows[i-1][prev_birth_time]:
                                    simplices_added_on_right_arrows[i-1][prev_birth_time].append(simplex)
                        else:
                            if flag:
                                simplices_added_on_right_arrows[i-1][curr_birth_time].append(simplex)           
            else:
                for curr_birth_time in simplices_grouped_by_birth_time[i]:
                    for simplex in simplices_grouped_by_birth_time[i][curr_birth_time]:
                        flag = False
                        try:
                            next_birth_time = simplex_birth_times_pairs[i+1][tuple(simplex)]
                        except KeyError:
                            flag = True
                            next_birth_time = self.num_divisions_for_filtration - 1
                        if next_birth_time < curr_birth_time:
                            for k in range(next_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][next_birth_time].append(simplex)
                            for k in range(next_birth_time, curr_birth_time):
                                simplices_added_on_right_arrows[i][k].append(simplex)
                        elif next_birth_time > curr_birth_time:
                            for k in range(curr_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][curr_birth_time].append(simplex)
                            for k in range(curr_birth_time, next_birth_time):
                                simplices_added_on_left_arrows[i+1][k].append(simplex)
                            if flag:
                                simplices_added_on_left_arrows[i+1][next_birth_time].append(simplex)
                        else:
                            for k in range(curr_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][curr_birth_time].append(simplex)
                            if flag:
                                if simplex not in simplices_added_on_right_arrows[i-1][curr_birth_time]:
                                    simplices_added_on_left_arrows[i+1][curr_birth_time].append(simplex)
                                
                        flag = False
                        try:
                            prev_birth_time = simplex_birth_times_pairs[i-1][tuple(simplex)]
                        except KeyError:
                            flag = True
                            prev_birth_time = self.num_divisions_for_filtration - 1
                        if prev_birth_time < curr_birth_time:
                            for k in range(prev_birth_time, curr_birth_time):
                                if simplex not in simplices_added_on_left_arrows[i][k]:
                                    simplices_added_on_left_arrows[i][k].append(simplex)
                        elif prev_birth_time > curr_birth_time:
                            for k in range(curr_birth_time, prev_birth_time):
                                if simplex not in simplices_added_on_right_arrows[i-1][k]:
                                    simplices_added_on_right_arrows[i-1][k].append(simplex)
                            if flag:
                                if simplex not in simplices_added_on_right_arrows[i-1][prev_birth_time]:
                                    simplices_added_on_right_arrows[i-1][prev_birth_time].append(simplex)
                        else:
                            if flag:
                                if simplex not in simplices_added_on_right_arrows[i-1][curr_birth_time]:
                                    simplices_added_on_right_arrows[i-1][curr_birth_time].append(simplex)
                                    
        return simplices_added_on_right_arrows, simplices_added_on_left_arrows, simplices_deleted_on_up_arrows_union, simplices_added_on_up_arrows_union
    
    def get_zz_filt_along_bdry_cap_of_worm(self, center_pt_idx: int, width: int,
                                           simplices_added_on_right_arrows: dict, 
                                           simplices_added_on_left_arrows: dict,
                                           simplices_grouped_by_birth_times: dict,
                                           simplices_deleted_on_up_arrows_union: dict,
                                           simplices_added_on_up_arrows_union: dict):
        bdry_pts, directions = self.get_pts_along_bdry_of_worm(center_pt_idx, width)
        zz_filt_along_bdry_cap = []
        full_bar_end_length = 0
        all_added_simplices = []        # For debugging
        all_deleted_simplices = []      # For debugging
        current_simplices_in_filt = []
        start_simplices = []
        
        # manually add start simplices for Tao's code
        curr_x, curr_y = bdry_pts[0][0], bdry_pts[0][1]
        if curr_x % 2 == 0:
            for i in range(curr_y + 1):
                try:
                    start_simplices.extend(simplices_grouped_by_birth_times[curr_x//2][i])
                except KeyError:
                    continue
        else:
            for i in range(curr_y + 1):
                try:
                    start_simplices.extend(simplices_grouped_by_birth_times[(curr_x - 1)//2][i])
                except KeyError:
                    pass
                try:
                    start_simplices.extend(simplices_grouped_by_birth_times[(curr_x + 1)//2][i])
                except KeyError:
                    pass
        
        start_simplices = list(start_simplices)
        start_simplices.sort(key=len)
        current_simplices_in_filt.extend(start_simplices)
        if len(start_simplices) > 0:
            for j in start_simplices:
                zz_filt_along_bdry_cap.append(('i', j))
                full_bar_end_length += 1
            
            for i in range(len(directions)):
                new_simplices, deleted_simplices = [], []
                curr_x, curr_y = bdry_pts[i][0], bdry_pts[i][1]
                next_x, next_y = bdry_pts[i+1][0], bdry_pts[i+1][1]
                
                if curr_x == next_x and curr_y == next_y:
                    continue
                
                if directions[i] == 'u':
                    assert next_x == curr_x
                    if curr_x % 2 == 0:
                        new_simplices = simplices_grouped_by_birth_times[curr_x//2][next_y]
                    else:
                        # new_simplices = simplices_deleted_on_up_arrows_union[(curr_x - 1)//2][next_y]
                        new_simplices = simplices_grouped_by_birth_times[(curr_x - 1)//2][next_y]
                        new_simplices = union_two_lists(new_simplices, simplices_grouped_by_birth_times[(curr_x + 1)//2][next_y])
                        
                        
                elif directions[i] == 'l':
                    assert next_y == curr_y
                    if curr_x % 2 == 0:
                        new_simplices = simplices_added_on_left_arrows[curr_x//2][curr_y]
                        
                    else:
                        deleted_simplices = simplices_added_on_right_arrows[(curr_x - 1)//2][curr_y]
    
                elif directions[i] == 'r':
                    assert next_y == curr_y
                    if curr_x % 2 == 0:
                        new_simplices = simplices_added_on_right_arrows[curr_x//2][curr_y]
                    else:
                        deleted_simplices = simplices_added_on_left_arrows[(curr_x + 1)//2][curr_y]
            
                elif directions[i] == 'd':
                    assert next_x == curr_x
                    if curr_x % 2 == 0:
                        deleted_simplices = simplices_grouped_by_birth_times[curr_x//2][curr_y]
                    else:
                        # deleted_simplices = simplices_grouped_by_birth_times[(curr_x - 1)//2][curr_y]
                        # deleted_simplices = intersection_two_lists(deleted_simplices, simplices_grouped_by_birth_times[(curr_x + 1)//2][curr_y])
                        deleted_simplices = simplices_deleted_on_up_arrows_union[(curr_x - 1)//2][curr_y]

                new_simplices = list(new_simplices)
                deleted_simplices = list(deleted_simplices)
                new_simplices.sort(key=len)
                deleted_simplices.sort(key=len, reverse=True)
                
                for k in new_simplices:
                    zz_filt_along_bdry_cap.append(('i', k))
                    full_bar_end_length += 1
                    current_simplices_in_filt.append(k)
                
                all_added_simplices.append(new_simplices)
                
                
                for k in deleted_simplices:
                    zz_filt_along_bdry_cap.append(('d', k))
                    full_bar_end_length += 1
                    current_simplices_in_filt.remove(k)
                
                all_deleted_simplices.append(deleted_simplices)
                
        return zz_filt_along_bdry_cap, len(start_simplices), full_bar_end_length

    def parse_bars_to_get_full_bars(self, bars: List[tuple], num_simplices_at_idx_zero:int, full_bar_end_length:int):
        num_full_bars_H0 = 0
        num_full_bars_H1 = 0
        for bar in bars:
            if bar[2] == 0:
                if bar[0] <= num_simplices_at_idx_zero and bar[1] >= full_bar_end_length-1:
                    num_full_bars_H0 += 1
            elif bar[2] == 1:
                if bar[0] <= num_simplices_at_idx_zero and bar[1] >= full_bar_end_length-1:
                    num_full_bars_H1 += 1
        return num_full_bars_H0, num_full_bars_H1
    
    def num_full_bars_for_specific_d(self, center_pt_idx: int, d:int, 
                                     simplices_added_on_right_arrows: dict, 
                                     simplices_added_on_left_arrows: dict, 
                                     simplices_grouped_by_birth_times: dict,
                                     simplices_deleted_on_up_arrows_union: dict,
                                     simplices_added_on_up_arrows_union: dict):
        
        filt, num_simplices_at_idx_zero, full_bar_end_length = self.get_zz_filt_along_bdry_cap_of_worm(center_pt_idx, d, 
                                                                                                       simplices_added_on_right_arrows, 
                                                                                                       simplices_added_on_left_arrows,
                                                                                                       simplices_grouped_by_birth_times,
                                                                                                       simplices_deleted_on_up_arrows_union,
                                                                                                       simplices_added_on_up_arrows_union)
        zz = pyfzz()
        if len(filt) > 0:
            bars = zz.compute_zigzag(filt)
            num_full_bars = self.parse_bars_to_get_full_bars(bars, num_simplices_at_idx_zero, full_bar_end_length)
        else:
            num_full_bars = (0, 0)
        return num_full_bars
    
    def compute_max_width_for_given_rank(self, center_pt_idx:int, rank:int, 
                                        simplices_added_on_right_arrows: dict, 
                                        simplices_added_on_left_arrows: dict, 
                                        simplices_grouped_by_birth_times: dict, 
                                        simplices_deleted_on_up_arrows_union: dict, 
                                        simplices_added_on_up_arrows_union: dict, hom_deg:int):
        
        
        if rank == 1:
            d_max = min([abs(self.num_divisions_for_filtration - 1 - self.center_pts[center_pt_idx, 0]), 
                     abs(self.num_divisions_for_filtration - 1 - self.center_pts[center_pt_idx, 1]),
                     abs(self.center_pts[center_pt_idx, 0]),
                        abs(self.center_pts[center_pt_idx, 1])])
        else:
            d_max = self.ranks_dmax[hom_deg, rank-2, center_pt_idx]
        d_min = 1
        d = 1
        ans = 1
        
        while d_min <= d_max:
            d = int((d_min + d_max)/2)
            if self.gen_rank_val_at_d[hom_deg, center_pt_idx, d] == -1:
                num_full_bars = self.num_full_bars_for_specific_d(center_pt_idx, d, 
                                                                simplices_added_on_right_arrows, 
                                                                simplices_added_on_left_arrows,
                                                                simplices_grouped_by_birth_times,
                                                                simplices_deleted_on_up_arrows_union,
                                                                simplices_added_on_up_arrows_union)
                if hom_deg == 0:
                    self.gen_rank_val_at_d[hom_deg, center_pt_idx, d] = num_full_bars[0]
                    self.gen_rank_val_at_d[hom_deg+1, center_pt_idx, d] = num_full_bars[1]
                elif hom_deg == 1:
                    self.gen_rank_val_at_d[hom_deg, center_pt_idx, d] = num_full_bars[1]
                
                num_full_bars_for_hom_deg = self.gen_rank_val_at_d[hom_deg, center_pt_idx, d]
            else:
                num_full_bars_for_hom_deg = self.gen_rank_val_at_d[hom_deg, center_pt_idx, d]
            
            if num_full_bars_for_hom_deg >= rank:
                ans = d
                d_min = d+1
            else:
                d_max = d-1
        if rank == 1 or rank == 2:
            self.ranks_dmax[hom_deg, rank-1, center_pt_idx] = ans      
        return ans
    
    def compute_zz_landscape(self, seq_of_graphs: np.ndarray, edge_weights: np.ndarray, n_jobs: int = 8, ranks: list = [1,2,3]):

        filt_dict = self.get_filt(seq_of_graphs, edge_weights)
        simplices_birth_times_vertical, simplices, simplex_birth_times_pairs, simplices_grouped_by_birth_times = self.get_simplices_birth_times_vertical(filt_dict, edge_weights)
        simplices_added_on_right_arrows, simplices_added_on_left_arrows, simplices_deleted_on_up_arrows_union, simplices_added_on_up_arrows_union = self.get_simplices_added_on_zigzag_arrows(simplices, simplex_birth_times_pairs, simplices_grouped_by_birth_times)
        
        landscape_k_1_H_0 = []
        landscape_k_2_H_0 = []
        landscape_k_3_H_0 = []
        landscape_k_1_H_1 = []
        landscape_k_2_H_1 = []
        landscape_k_3_H_1 = []
        
        for center_pt_idx in range(self.num_center_pts):
            # print("Center point idx", center_pt_idx)
            landscape_k_1_H_0.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[0], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=0))
            landscape_k_2_H_0.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[1], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=0))
            landscape_k_3_H_0.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[2], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=0))
            # print("H0 done")
            landscape_k_1_H_1.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[0], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=1))
            landscape_k_2_H_1.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[1], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=1))
            landscape_k_3_H_1.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[2], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=1))
            
        # Parallel version:
        
        
        # landscape_k_1_H_0 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[0], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=0)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_2_H_0 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[1], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=0)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_3_H_0 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[2], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=0)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_1_H_1 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[0], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=1)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_2_H_1 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[1], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=1)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_3_H_1 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[2], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=1)
        #                                                                     for center_pt_idx in range(self.num_center_pts))    
        
        
        landscape_k_1_H_0 = np.array(landscape_k_1_H_0).reshape(-1, 1)
        landscape_k_2_H_0 = np.array(landscape_k_2_H_0).reshape(-1, 1)
        landscape_k_3_H_0 = np.array(landscape_k_3_H_0).reshape(-1, 1)
        landscape_k_1_H_1 = np.array(landscape_k_1_H_1).reshape(-1, 1)
        landscape_k_2_H_1 = np.array(landscape_k_2_H_1).reshape(-1, 1)
        landscape_k_3_H_1 = np.array(landscape_k_3_H_1).reshape(-1, 1)
        
        landscape_H_0 = np.concatenate((landscape_k_1_H_0, landscape_k_2_H_0, landscape_k_3_H_0), axis=1)
        landscape_H_1 = np.concatenate((landscape_k_1_H_1, landscape_k_2_H_1, landscape_k_3_H_1), axis=1)
        
        return landscape_H_0, landscape_H_1        
    
    def refresh_gril_info(self):
        self.gen_rank_val_at_d = np.ones((2, self.num_center_pts, self.num_divisions_for_filtration)) * -1
        self.ranks_dmax = np.zeros((2, 2, self.num_center_pts))        


class zzMultipersPointCloud:
    def __init__(self, num_center_pts: int,
                 num_point_clouds_in_seq: int,
                 max_alpha_square: float = 5e-5) -> None:
        self.num_center_pts = num_center_pts
        self.num_point_clouds_in_seq = num_point_clouds_in_seq
        self.max_alpha_square = max_alpha_square
        self.num_divisions_for_filtration = (2 * self.num_point_clouds_in_seq) - 1
        self.center_pts = self.sample_center_pts()
        self.gen_rank_val_at_d = np.ones((2, self.num_center_pts, self.num_divisions_for_filtration)) * -1
        self.ranks_dmax = np.zeros((2, 2, self.num_center_pts))
    
    def sample_center_pts(self):
        np.random.seed(0)
        center_pts = np.random.randint(2, self.num_divisions_for_filtration - 2, 
                                       size=(self.num_center_pts, 2))
        return center_pts  

    def get_alpha_filt(self, seq_of_pt_clouds: np.ndarray):
        alpha_filt_dict = {}
        ac = AlphaComplex(points = seq_of_pt_clouds[0].reshape(-1,1))
        stree = ac.create_simplex_tree(max_alpha_square=self.max_alpha_square)
        alpha_filt_dict[0] = np.array(list(stree.get_filtration()), dtype=object)
        for pt_cloud_idx in range(1, seq_of_pt_clouds.shape[0]):
            ac = AlphaComplex(points = seq_of_pt_clouds[pt_cloud_idx].reshape(-1,1))
            stree = ac.create_simplex_tree(max_alpha_square=self.max_alpha_square)
            alpha_filt_dict[pt_cloud_idx] = np.array(list(stree.get_filtration()), dtype=object)
        return alpha_filt_dict

    def get_simplices_birth_times_vertical(self, alpha_filt_dict: dict):
        simplices_birth_times = {}
        simplices = {}
        simplex_birth_times_pairs = {}
        simplices_grouped_by_birth_times = {}
        for i in range(len(alpha_filt_dict)):
            simplices_birth_times[i] = np.ceil(alpha_filt_dict[i][:,1] * (1/self.max_alpha_square) * (self.num_divisions_for_filtration - 1))
            simplices[i] = alpha_filt_dict[i][:,0]
            temp1 = map(tuple, simplices[i])
            simplex_birth_times_pairs[i] = dict(zip(temp1, simplices_birth_times[i]))
            temp = {val: [c for _, c in g] for val, g in itertools.groupby(zip(simplices_birth_times[i], simplices[i]), key=lambda x: x[0])}
            simplices_grouped_by_birth_times[i] = temp
            for levels in range(self.num_divisions_for_filtration):
                if levels not in simplices_grouped_by_birth_times[i]:
                    simplices_grouped_by_birth_times[i][levels] = []
        return simplices_birth_times, simplices, simplex_birth_times_pairs, simplices_grouped_by_birth_times
    
    def get_pts_along_bdry_of_worm(self, center_pt_idx:int, width: int):
        # We implement l = 2 case. Need to implement others?
        pts_along_bdry, direction = [], []
        bdry_start_x = self.center_pts[center_pt_idx][0]
        bdry_start_y = self.center_pts[center_pt_idx][1] - (2 * width)
        pts_along_bdry.append((bdry_start_x, bdry_start_y))
        
        # lower staircase
        flag = True # True for vertical, False for horizontal
        for i in range(4 * width):
            if flag:
                pts_along_bdry.append((pts_along_bdry[-1][0], pts_along_bdry[-1][1] + 1))
                direction.append('u')
                flag = False
            else:
                pts_along_bdry.append((pts_along_bdry[-1][0] - 1, pts_along_bdry[-1][1]))
                direction.append('l')
                flag = True
                
        # top-left inverted L shape
        for i in range(2 * width):
            pts_along_bdry.append((pts_along_bdry[-1][0], pts_along_bdry[-1][1] + 1))
            direction.append('u')
        
        for i in range(2 * width):
            pts_along_bdry.append((pts_along_bdry[-1][0] + 1, pts_along_bdry[-1][1]))
            direction.append('r')
            
        # upper staircase
        flag = True # True for vertical, False for horizontal
        for i in range(4 * width):
            if flag:
                pts_along_bdry.append((pts_along_bdry[-1][0], pts_along_bdry[-1][1] - 1))
                direction.append('d')
                flag = False
            else:
                pts_along_bdry.append((pts_along_bdry[-1][0] + 1, pts_along_bdry[-1][1]))
                direction.append('r')
                flag = True
        
        # bottom-right inverted L shape
        for i in range(2 * width):
            pts_along_bdry.append((pts_along_bdry[-1][0], pts_along_bdry[-1][1] - 1))
            direction.append('d')
            
        for i in range(2 * width - 1):
            pts_along_bdry.append((pts_along_bdry[-1][0] - 1, pts_along_bdry[-1][1]))
            direction.append('l')
        
        direction = direction[:-1]
        assert len(pts_along_bdry) == (16 * width)
        
        pts_along_bdry = np.array(pts_along_bdry)
        pts_along_bdry[pts_along_bdry < 0] = 0
        pts_along_bdry[pts_along_bdry >= self.num_divisions_for_filtration] = self.num_divisions_for_filtration - 1
        
        return pts_along_bdry, direction
    
    def get_simplices_added_on_zigzag_arrows(self, simplices: dict, simplex_birth_times_pairs: dict, simplices_grouped_by_birth_time: dict):
        simplices_added_on_right_arrows = {i: {j : [] for j in range(len(simplices_grouped_by_birth_time[i]))} for i in range(len(simplices_grouped_by_birth_time))}
        simplices_added_on_left_arrows = {i : {j : [] for j in range(len(simplices_grouped_by_birth_time[i]))} for i in range(len(simplices_grouped_by_birth_time))}
        simplices_deleted_on_up_arrows_union = {i : {j : [] for j in range(len(simplices_grouped_by_birth_time[i]))} for i in range(len(simplices_grouped_by_birth_time) - 1)}
        simplices_added_on_up_arrows_union = {i : {j : [] for j in range(len(simplices_grouped_by_birth_time[i]))} for i in range(len(simplices_grouped_by_birth_time) - 1)}
        
        for i in range(len(simplices_grouped_by_birth_time)):
            if i == 0:
                for curr_birth_time in simplices_grouped_by_birth_time[i]:
                    for simplex in simplices_grouped_by_birth_time[i][curr_birth_time]:
                        flag = False
                        try:
                            next_birth_time = simplex_birth_times_pairs[i+1][tuple(simplex)]
                        except KeyError:
                            flag = True
                            next_birth_time = self.num_divisions_for_filtration - 1
                        if next_birth_time < curr_birth_time:
                            for k in range(next_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_deleted_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][next_birth_time].append(simplex)
                            for k in range(next_birth_time, curr_birth_time):
                                simplices_added_on_right_arrows[i][k].append(simplex)
                        elif next_birth_time > curr_birth_time:
                            for k in range(curr_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][curr_birth_time].append(simplex)
                            for k in range(curr_birth_time, next_birth_time):
                                simplices_added_on_left_arrows[i+1][k].append(simplex)
                            if flag:
                                simplices_added_on_left_arrows[i+1][next_birth_time].append(simplex)
                        else:
                            for k in range(curr_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][curr_birth_time].append(simplex)
                            if flag:
                                simplices_added_on_left_arrows[i+1][curr_birth_time].append(simplex)
                                
            elif i == len(simplices_grouped_by_birth_time) - 1:
                for curr_birth_time in simplices_grouped_by_birth_time[i]:
                    for simplex in simplices_grouped_by_birth_time[i][curr_birth_time]:
                        flag = False
                        try:
                            prev_birth_time = simplex_birth_times_pairs[i-1][tuple(simplex)]
                        except KeyError:
                            flag = True
                            prev_birth_time = self.num_divisions_for_filtration - 1
                        if prev_birth_time < curr_birth_time:
                            for k in range(prev_birth_time, curr_birth_time):
                                if simplex not in simplices_added_on_left_arrows[i][k]:
                                    simplices_added_on_left_arrows[i][k].append(simplex)
                        elif prev_birth_time > curr_birth_time:
                            for k in range(curr_birth_time, prev_birth_time):
                                if simplex not in simplices_added_on_right_arrows[i-1][k]:
                                    simplices_added_on_right_arrows[i-1][k].append(simplex)
                            if flag:
                                if simplex not in simplices_added_on_right_arrows[i-1][prev_birth_time]:
                                    simplices_added_on_right_arrows[i-1][prev_birth_time].append(simplex)
                        else:
                            if flag:
                                simplices_added_on_right_arrows[i-1][curr_birth_time].append(simplex)           
            else:
                for curr_birth_time in simplices_grouped_by_birth_time[i]:
                    for simplex in simplices_grouped_by_birth_time[i][curr_birth_time]:
                        flag = False
                        try:
                            next_birth_time = simplex_birth_times_pairs[i+1][tuple(simplex)]
                        except KeyError:
                            flag = True
                            next_birth_time = self.num_divisions_for_filtration - 1
                        if next_birth_time < curr_birth_time:
                            for k in range(next_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][next_birth_time].append(simplex)
                            for k in range(next_birth_time, curr_birth_time):
                                simplices_added_on_right_arrows[i][k].append(simplex)
                        elif next_birth_time > curr_birth_time:
                            for k in range(curr_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][curr_birth_time].append(simplex)
                            for k in range(curr_birth_time, next_birth_time):
                                simplices_added_on_left_arrows[i+1][k].append(simplex)
                            if flag:
                                simplices_added_on_left_arrows[i+1][next_birth_time].append(simplex)
                        else:
                            for k in range(curr_birth_time, self.num_divisions_for_filtration - 1):
                                simplices_added_on_up_arrows_union[i][k].append(simplex)
                            simplices_deleted_on_up_arrows_union[i][curr_birth_time].append(simplex)
                            if flag:
                                if simplex not in simplices_added_on_right_arrows[i-1][curr_birth_time]:
                                    simplices_added_on_left_arrows[i+1][curr_birth_time].append(simplex)
                                
                        flag = False
                        try:
                            prev_birth_time = simplex_birth_times_pairs[i-1][tuple(simplex)]
                        except KeyError:
                            flag = True
                            prev_birth_time = self.num_divisions_for_filtration - 1
                        if prev_birth_time < curr_birth_time:
                            for k in range(prev_birth_time, curr_birth_time):
                                if simplex not in simplices_added_on_left_arrows[i][k]:
                                    simplices_added_on_left_arrows[i][k].append(simplex)
                        elif prev_birth_time > curr_birth_time:
                            for k in range(curr_birth_time, prev_birth_time):
                                if simplex not in simplices_added_on_right_arrows[i-1][k]:
                                    simplices_added_on_right_arrows[i-1][k].append(simplex)
                            if flag:
                                if simplex not in simplices_added_on_right_arrows[i-1][prev_birth_time]:
                                    simplices_added_on_right_arrows[i-1][prev_birth_time].append(simplex)
                        else:
                            if flag:
                                if simplex not in simplices_added_on_right_arrows[i-1][curr_birth_time]:
                                    simplices_added_on_right_arrows[i-1][curr_birth_time].append(simplex)
                                    
        return simplices_added_on_right_arrows, simplices_added_on_left_arrows, simplices_deleted_on_up_arrows_union, simplices_added_on_up_arrows_union
    
    def get_zz_filt_along_bdry_cap_of_worm(self, center_pt_idx: int, width: int,
                                           simplices_added_on_right_arrows: dict, 
                                           simplices_added_on_left_arrows: dict,
                                           simplices_grouped_by_birth_times: dict,
                                           simplices_deleted_on_up_arrows_union: dict,
                                           simplices_added_on_up_arrows_union: dict):
        bdry_pts, directions = self.get_pts_along_bdry_of_worm(center_pt_idx, width)
        zz_filt_along_bdry_cap = []
        full_bar_end_length = 0
        all_added_simplices = []        # For debugging
        all_deleted_simplices = []      # For debugging
        current_simplices_in_filt = []
        start_simplices = []
        
        # manually add start simplices for Tao's code
        curr_x, curr_y = bdry_pts[0][0], bdry_pts[0][1]
        if curr_x % 2 == 0:
            for i in range(curr_y + 1):
                start_simplices.extend(simplices_grouped_by_birth_times[curr_x//2][i])
        else:
            for i in range(curr_y + 1):
                start_simplices.extend(simplices_grouped_by_birth_times[(curr_x - 1)//2][i])
                start_simplices.extend(simplices_grouped_by_birth_times[(curr_x + 1)//2][i])
        
        start_simplices = list(start_simplices)
        start_simplices.sort(key=len)
        current_simplices_in_filt.extend(start_simplices)
        if len(start_simplices) > 0:
            for j in start_simplices:
                zz_filt_along_bdry_cap.append(('i', j))
                full_bar_end_length += 1
            
            for i in range(len(directions)):
                new_simplices, deleted_simplices = [], []
                curr_x, curr_y = bdry_pts[i][0], bdry_pts[i][1]
                next_x, next_y = bdry_pts[i+1][0], bdry_pts[i+1][1]
                
                if curr_x == next_x and curr_y == next_y:
                    continue
                
                if directions[i] == 'u':
                    assert next_x == curr_x
                    if curr_x % 2 == 0:
                        new_simplices = simplices_grouped_by_birth_times[curr_x//2][next_y]
                    else:
                        # new_simplices = simplices_deleted_on_up_arrows_union[(curr_x - 1)//2][next_y]
                        new_simplices = simplices_grouped_by_birth_times[(curr_x - 1)//2][next_y]
                        new_simplices = union_two_lists(new_simplices, simplices_grouped_by_birth_times[(curr_x + 1)//2][next_y])
                        
                        
                elif directions[i] == 'l':
                    assert next_y == curr_y
                    if curr_x % 2 == 0:
                        new_simplices = simplices_added_on_left_arrows[curr_x//2][curr_y]
                        
                    else:
                        deleted_simplices = simplices_added_on_right_arrows[(curr_x - 1)//2][curr_y]
    
                elif directions[i] == 'r':
                    assert next_y == curr_y
                    if curr_x % 2 == 0:
                        new_simplices = simplices_added_on_right_arrows[curr_x//2][curr_y]
                    else:
                        deleted_simplices = simplices_added_on_left_arrows[(curr_x + 1)//2][curr_y]
            
                elif directions[i] == 'd':
                    assert next_x == curr_x
                    if curr_x % 2 == 0:
                        deleted_simplices = simplices_grouped_by_birth_times[curr_x//2][curr_y]
                    else:
                        # deleted_simplices = simplices_grouped_by_birth_times[(curr_x - 1)//2][curr_y]
                        # deleted_simplices = intersection_two_lists(deleted_simplices, simplices_grouped_by_birth_times[(curr_x + 1)//2][curr_y])
                        deleted_simplices = simplices_deleted_on_up_arrows_union[(curr_x - 1)//2][curr_y]

                new_simplices = list(new_simplices)
                deleted_simplices = list(deleted_simplices)
                new_simplices.sort(key=len)
                deleted_simplices.sort(key=len, reverse=True)
                
                for k in new_simplices:
                    zz_filt_along_bdry_cap.append(('i', k))
                    full_bar_end_length += 1
                    current_simplices_in_filt.append(k)
                
                all_added_simplices.append(new_simplices)
                
                
                for k in deleted_simplices:
                    zz_filt_along_bdry_cap.append(('d', k))
                    full_bar_end_length += 1
                    current_simplices_in_filt.remove(k)
                
                all_deleted_simplices.append(deleted_simplices)
                
        return zz_filt_along_bdry_cap, len(start_simplices), full_bar_end_length

    def parse_bars_to_get_full_bars(self, bars: List[tuple], num_simplices_at_idx_zero:int, full_bar_end_length:int):
        num_full_bars_H0 = 0
        num_full_bars_H1 = 0
        for bar in bars:
            if bar[2] == 0:
                if bar[0] <= num_simplices_at_idx_zero and bar[1] >= full_bar_end_length-1:
                    num_full_bars_H0 += 1
            elif bar[2] == 1:
                if bar[0] <= num_simplices_at_idx_zero and bar[1] >= full_bar_end_length-1:
                    num_full_bars_H1 += 1
        return num_full_bars_H0, num_full_bars_H1
    
    def num_full_bars_for_specific_d(self, center_pt_idx: int, d:int, 
                                     simplices_added_on_right_arrows: dict, 
                                     simplices_added_on_left_arrows: dict, 
                                     simplices_grouped_by_birth_times: dict,
                                     simplices_deleted_on_up_arrows_union: dict,
                                     simplices_added_on_up_arrows_union: dict):
        
        filt, num_simplices_at_idx_zero, full_bar_end_length = self.get_zz_filt_along_bdry_cap_of_worm(center_pt_idx, d, 
                                                                                                       simplices_added_on_right_arrows, 
                                                                                                       simplices_added_on_left_arrows,
                                                                                                       simplices_grouped_by_birth_times,
                                                                                                       simplices_deleted_on_up_arrows_union,
                                                                                                       simplices_added_on_up_arrows_union)
        zz = pyfzz()
        if len(filt) > 0:
            bars = zz.compute_zigzag(filt)
            num_full_bars = self.parse_bars_to_get_full_bars(bars, num_simplices_at_idx_zero, full_bar_end_length)
        else:
            num_full_bars = (0, 0)
        return num_full_bars
    
    def compute_max_width_for_given_rank(self, center_pt_idx:int, rank:int, 
                                        simplices_added_on_right_arrows: dict, 
                                        simplices_added_on_left_arrows: dict, 
                                        simplices_grouped_by_birth_times: dict, 
                                        simplices_deleted_on_up_arrows_union: dict, 
                                        simplices_added_on_up_arrows_union: dict, hom_deg:int):
        
        
        if rank == 1:
            d_max = min([abs(self.num_divisions_for_filtration - 1 - self.center_pts[center_pt_idx, 0]), 
                     abs(self.num_divisions_for_filtration - 1 - self.center_pts[center_pt_idx, 1]),
                     abs(self.center_pts[center_pt_idx, 0]),
                        abs(self.center_pts[center_pt_idx, 1])])
        else:
            d_max = self.ranks_dmax[hom_deg, rank-2, center_pt_idx]
        d_min = 1
        d = 1
        ans = 1
        
        while d_min <= d_max:
            d = int((d_min + d_max)/2)
            if self.gen_rank_val_at_d[hom_deg, center_pt_idx, d] == -1:
                num_full_bars = self.num_full_bars_for_specific_d(center_pt_idx, d, 
                                                                simplices_added_on_right_arrows, 
                                                                simplices_added_on_left_arrows,
                                                                simplices_grouped_by_birth_times,
                                                                simplices_deleted_on_up_arrows_union,
                                                                simplices_added_on_up_arrows_union)
                if hom_deg == 0:
                    self.gen_rank_val_at_d[hom_deg, center_pt_idx, d] = num_full_bars[0]
                    self.gen_rank_val_at_d[hom_deg+1, center_pt_idx, d] = num_full_bars[1]
                elif hom_deg == 1:
                    self.gen_rank_val_at_d[hom_deg, center_pt_idx, d] = num_full_bars[1]
                
                num_full_bars_for_hom_deg = self.gen_rank_val_at_d[hom_deg, center_pt_idx, d]
            else:
                num_full_bars_for_hom_deg = self.gen_rank_val_at_d[hom_deg, center_pt_idx, d]
            
            if num_full_bars_for_hom_deg >= rank:
                ans = d
                d_min = d+1
            else:
                d_max = d-1
        if rank == 1 or rank == 2:
            self.ranks_dmax[hom_deg, rank-1, center_pt_idx] = ans      
        return ans
    
    def compute_zz_landscape(self, seq_of_pt_clouds: np.ndarray, n_jobs: int = 16, ranks: list = [1,2,3]):

        alpha_filt_dict = self.get_alpha_filt(seq_of_pt_clouds)
        simplices_birth_times_vertical, simplices, simplex_birth_times_pairs, simplices_grouped_by_birth_times = self.get_simplices_birth_times_vertical(alpha_filt_dict)
        simplices_added_on_right_arrows, simplices_added_on_left_arrows, simplices_deleted_on_up_arrows_union, simplices_added_on_up_arrows_union = self.get_simplices_added_on_zigzag_arrows(simplices, simplex_birth_times_pairs, simplices_grouped_by_birth_times)
        
        landscape_k_1_H_0 = []
        landscape_k_2_H_0 = []
        landscape_k_3_H_0 = []
        landscape_k_1_H_1 = []
        landscape_k_2_H_1 = []
        landscape_k_3_H_1 = []
        
        for center_pt_idx in range(self.num_center_pts):
            # print("Center point idx", center_pt_idx)
            landscape_k_1_H_0.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[0], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=0))
            landscape_k_2_H_0.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[1], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=0))
            landscape_k_3_H_0.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[2], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=0))
            # print("H0 done")
            landscape_k_1_H_1.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[0], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=1))
            landscape_k_2_H_1.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[1], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=1))
            landscape_k_3_H_1.append(self.compute_max_width_for_given_rank(center_pt_idx, ranks[2], simplices_added_on_right_arrows, 
                                                                           simplices_added_on_left_arrows,
                                                                           simplices_grouped_by_birth_times, 
                                                                           simplices_deleted_on_up_arrows_union, 
                                                                           simplices_added_on_up_arrows_union, hom_deg=1))
            
        # Parallel version:
        
        
        # landscape_k_1_H_0 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[0], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=0)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_2_H_0 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[1], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=0)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_3_H_0 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[2], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=0)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_1_H_1 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[0], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=1)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_2_H_1 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[1], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=1)
        #                                                                     for center_pt_idx in range(self.num_center_pts))
        
        # landscape_k_3_H_1 = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.compute_max_width_for_given_rank)(center_pt_idx, ranks[2], simplices_added_on_right_arrows, 
        #                                                                    simplices_added_on_left_arrows,
        #                                                                    simplices_grouped_by_birth_times, 
        #                                                                    simplices_deleted_on_up_arrows_union, 
        #                                                                    simplices_added_on_up_arrows_union, hom_deg=1)
        #                                                                     for center_pt_idx in range(self.num_center_pts))    
        
        
        landscape_k_1_H_0 = np.array(landscape_k_1_H_0).reshape(-1, 1)
        landscape_k_2_H_0 = np.array(landscape_k_2_H_0).reshape(-1, 1)
        landscape_k_3_H_0 = np.array(landscape_k_3_H_0).reshape(-1, 1)
        landscape_k_1_H_1 = np.array(landscape_k_1_H_1).reshape(-1, 1)
        landscape_k_2_H_1 = np.array(landscape_k_2_H_1).reshape(-1, 1)
        landscape_k_3_H_1 = np.array(landscape_k_3_H_1).reshape(-1, 1)
        
        landscape_H_0 = np.concatenate((landscape_k_1_H_0, landscape_k_2_H_0, landscape_k_3_H_0), axis=1)
        landscape_H_1 = np.concatenate((landscape_k_1_H_1, landscape_k_2_H_1, landscape_k_3_H_1), axis=1)
        
        return landscape_H_0, landscape_H_1        
    
    def refresh_gril_info(self):
        self.gen_rank_val_at_d = np.ones((2, self.num_center_pts, self.num_divisions_for_filtration)) * -1
        self.ranks_dmax = np.zeros((2, 2, self.num_center_pts))        

