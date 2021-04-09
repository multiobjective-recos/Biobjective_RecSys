#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
import json
import copy
import random
from time import time
import pickle
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import multiprocessing
import operator
import sklearn.preprocessing as pp
import datetime as dt
from collections import deque
from decorators import memoized
import sys


# In[ ]:


class SimpleQueue(object):
    def __init__(self):
        self.buffer = deque()

    def push(self, value):
        self.buffer.appendleft(value)

    def pop(self):
        return self.buffer.pop()

    def __len__(self):
        return len(self.buffer)


# In[ ]:


class Node(object):
    def __init__(self, level, selected_items, cost, weight, bound):
        self.level = level
        self.selected_items = selected_items
        self.cost = cost
        self.weight = weight
        self.bound = bound


# In[ ]:


class Knapsack(object):
    
    def solver_BB(number, capacity, weight_cost):
    """Branch and bounds method for solving knapsack problem
    http://faculty.cns.uni.edu/~east/teaching/153/branch_bound/knapsack/overview_algorithm.html
    :param number: number of existing items
    :param capacity: the capacity of knapsack
    :param weight_cost: list of tuples like: [(weight, cost), (weight, cost), ...]
    :return: tuple like: (best cost, best combination list(contains 1 and 0))
    """
    priority_queue = SimpleQueue()

    #sort items in non-increasing order by benefit/cost
    ratios = [(index, item[1] / float(item[0])) for index, item in enumerate(weight_cost)]
    ratios = sorted(ratios, key=lambda x: x[1], reverse=True)

    best_so_far = Node(0, [], 0.0, 0.0, 0.0)
    a_node = Node(0, [], 0.0, 0.0, calculate_bound(best_so_far, number, capacity, weight_cost, ratios))
    priority_queue.push(a_node)

    while len(priority_queue) > 0:
        curr_node = priority_queue.pop()
        if curr_node.bound > best_so_far.cost:
            curr_node_index = ratios[curr_node.level][0]
            next_item_cost = weight_cost[curr_node_index][1]
            next_item_weight = weight_cost[curr_node_index][0]
            next_added = Node(
                curr_node.level + 1,
                curr_node.selected_items + [curr_node_index],
                curr_node.cost + next_item_cost,
                curr_node.weight + next_item_weight,
                curr_node.bound
            )

            if next_added.weight <= capacity:
                if next_added.cost > best_so_far.cost:
                    best_so_far = next_added

                if next_added.bound > best_so_far.cost:
                    priority_queue.push(next_added)

            next_not_added = Node(curr_node.level + 1, curr_node.selected_items, curr_node.cost,
                                  curr_node.weight, curr_node.bound)
            next_not_added.bound = calculate_bound(next_not_added, number, capacity, weight_cost, ratios)
            if next_not_added.bound > best_so_far.cost:
                priority_queue.push(next_not_added)

    best_combination = [0] * number
    for wc in best_so_far.selected_items:
        best_combination[wc] = 1
    return int(best_so_far.cost), best_combination


    def calculate_bound(node, number, capacity, weight_cost, ratios):
        if node.weight >= capacity:
            return 0
        else:
            upper_bound = node.cost
            total_weight = node.weight
            current_level = node.level

            while current_level < number:
                current_index = ratios[current_level][0]

                if total_weight + weight_cost[current_index][0] > capacity:
                    cost = weight_cost[current_index][1]
                    weight = weight_cost[current_index][0]
                    upper_bound += (capacity - total_weight) * cost/weight
                    break

                upper_bound += weight_cost[current_index][1]
                total_weight += weight_cost[current_index][0]
                current_level += 1

            return upper_bound
    
    def solver_DP(capacity,weight_cost):
    items = len(weight_cost)
    values = []
    weights = []
    values.append(0)
    weights.append(0)
    
    for i in range(1, items+1):
    
        values.append(weight_cost[i-1][1])
        weights.append(weight_cost[i-1][0])

    #Use dynamic programming 

    KN=[]

    for i in range(0, capacity+1):
        KN.append([])
        for j in range(0, items+1):
            KN[i].append([])
            if(i==0 or j==0):
                KN[i][j]=0;
            else:
                if( (i<weights[j])or ( KN[i][j-1]>(KN[i-weights[j]][j-1]+values[j]) ) ):
                    KN[i][j]=KN[i][j-1]
                else:
                    KN[i][j]=(KN[i-weights[j]][j-1]+values[j])
            

    value = KN[capacity][items]
    
    #Do Backtracking
    taken = []
    
    ptr=[capacity,items]
    while (ptr[1]>0):
        if( (KN[ptr[0]][ptr[1]]-KN[ptr[0]][ptr[1]-1]) == 0):
            taken.append(0)
            ptr[1]-=1
        else:
            taken.append(1)
            ptr[0]-=weights[ptr[1]]
            ptr[1]-=1
          
    return value, taken

    def solver_DP2(number, capacity, weight_cost):
    """
    Solve the knapsack problem by finding the most valuable subsequence of `weight_cost` subject that weighs 
    no more than `capacity`.
    Top-down solution from: http://codereview.stackexchange.com/questions/20569/dynamic-programming-solution-to-knapsack-problem
    :param weight_cost: is a sequence of pairs (weight, cost)
    :param capacity: is a non-negative integer
    :return: a pair whose first element is the sum of costs in the best combination,
    and whose second element is the combination.
    """

    # Return the value of the most valuable subsequence of the first i
    # elements in items whose weights sum to no more than j.
    @memoized
    def bestvalue(i, j):
        if i == 0:
            return 0
        weight, cost = weight_cost[i - 1]
        if weight > j:
            return bestvalue(i - 1, j)
        else:
            # maximizing the cost
            return max(bestvalue(i - 1, j), bestvalue(i - 1, j - weight) + cost)

    j = capacity
    result = [0] * number
    for i in range(len(weight_cost), 0, -1):
        if bestvalue(i, j) != bestvalue(i - 1, j):
            result[i - 1] = 1
            j -= weight_cost[i - 1][0]
            print(j)
    return bestvalue(len(weight_cost), capacity), result

    def ratio_greedy(number, capacity, weight_cost):
    """
    Greedy 1/0 ratio method for solving knapsack problem
    :param number: number of existing items
    :param capacity: the capacity of knapsack
    :param weight_cost: list of tuples like: [(weight, cost), (weight, cost), ...]
    :return: tuple like: (best cost, best combination list(contains 1 and 0))
    """
    ratios = [(index, item[1] / float(item[0])) for index, item in enumerate(weight_cost)]
    ratios = sorted(ratios, key=lambda x: x[1], reverse=True)
    best_combination = [0] * number
    best_cost = 0
    weight = 0
    for index, ratio in ratios:
        if weight_cost[index][0] + weight <= capacity:
            weight += weight_cost[index][0]
            best_cost += weight_cost[index][1]
            best_combination[index] = 1
    return best_cost, best_combination
    
    
    def margin(user,item):
    a = user_station[user].get('COCO', 0)
    b = user_station[user].get('CODO', 0)
    c= user_station[user].get('DODO', 0)

    p1 =  net_prices[item].get('COCO',0)
    p2 =  net_prices[item].get('CODO',0)
    p3 =  net_prices[item].get('DODO',0)
    margin=0
    if item_cat[item] == 'BOUTIQUE':
        margin = (a*0.5*p1) + (b*0.09 *p2)
    elif item_cat[item] == 'RESTAURATION':
        margin = (a*0.7*p1)
    elif item_cat[item] == 'LAVAGE':
        margin = (a*0.8*p1) + (b*0.61 *p2) + (c *0.36 * p3  )
    elif item_cat[item] == 'CAFE':
        margin = (a*0.8*p1) + (b*0.3 *p2) + (c *0.3 * p3  )
    else:
        margin = 0.1
    return margin

    def compute_utilities(user, customer_list, items_list, sim_matrix):
        user_util = profile_similarities (user, customer_list, items_list,sim_matrix).toarray().reshape(-1)
        dict_utilities = {}
        for i,j in enumerate(user_util):
            dict_utilities [items_list[i]]= j
        return dict_utilities

    def compute_margins(user, items_list):
        margin_utilities = {}

        for i in items_list:
            margin_utilities[i]= margin(user,i)
        return margin_utilities

    def num_interval(product, dict1, error, epsilon):
    temp = int(math.log(dict1[product],2.0)/(error*math.log((1+epsilon),2.0)))+1
    return temp

    

def product_dominance(product1, product2, dict1, dict2,dict3,error,epsilon):
    
    if dict1[product1]*(1+epsilon) >= dict1[product2] and dict2[product1]*(1+epsilon) >= dict2[product2] and dict3[product1]<= dict3[product2]:
        return True
    else:
        return False
       
    
def set_dominance(set1, set2, dict1, dict2,dict3,error,epsilon):
    uti1 = 0
    uti2 = 0
    for m in set1:
        #print('ilyé',p)
        uti1 = uti1 + dict1[m]
    for m in set2:
        uti2 = uti2 + dict1[m]

    mar1= 0
    mar2= 0

    for m in set1:
        mar1 = mar1 + dict2[m]
    for m in set2:
        mar2 = mar2 + dict2[m]
    
    sp1 = 0
    sp2 = 0
    
    for m in set1:
        sp1 = sp1 + dict3[m]
    for m in set2:
        sp2 = sp2 + dict3[m]


    if uti1*(1+epsilon) >= uti2 and mar1*(1+epsilon) >= mar2 and sp1 <= sp2:
        return True
    else:
        return False
    
    def bi_objective(input,dict_utilities,dict_margins,dict_price,error,epsilon,B):
    
    set_C = list()
    #set_T = list()
    s = set()
    set_C = [[0,0,0,s]]
    #uti = 0
    #marge = 0
    #weight = 0
    for item in items:
        print('item =', item)
        set_T = list()
        for i in range(len(set_C)):
            if (set_C[i][2] + dict_price[item] <= B):
                #weight = dict_price[item] + set_C[i][2]
                #marge = dict_margins[item] + set_C[i][1]
                #uti = dict_utilities[item] + set_C[i][0]
                s = set_C[0][3].copy()
                s.add(item)
                print('s=',s)
                set_T.append([dict_utilities[item]+set_C[i][0], dict_margins[item]+set_C[i][1], dict_price[item]+set_C[i][2],s])
        
        print('setT=',set_T)
        for i in range(len(set_T)):
            dominated = False
            dominates = False 
            j = 0
            
            while(j<len(set_C) and not(dominated) and not(dominates)):
                if(set_dominance(set_C[j][3],set_T[i][3],dict_utilities,dict_margins,dict_price,error,epsilon)):
                    dominated = True
                    set_T.pop(j)
                elif(set_dominance(set_T[i][3],set_C[j][3],dict_utilities,dict_margins,dict_price,error,epsilon)):
                    dominates = True
                    set_C.pop(j)
                j = j + 1
                
            if(not(dominated)):
                while(j<len(set_C)):
                    if(set_dominance(set_T[i][3],set_C[j][3],dict_utilities,dict_margins,dict_price,error,epsilon)):
                        set_C.pop(j)
                    j = j + 1
                
        set_C = set_C + set_T
    
    return set_C
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




