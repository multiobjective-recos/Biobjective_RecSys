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
from memoize.wrapper import memoize
import asyncio
import sys
import Knapsack
import Pareto


# In[ ]:


kp = Knapsack() 
pareto = Pareto()

print('1 : Budget recos \ 2 : Business recos \ 3 : Promos')
nm = int(input())

if nm == 1:
    print('BB: Branch and Bound \ DP : Dynamic Programmi \ Greedy: Greedy_IBCF  ? ')
    method = str(input())
elif nm == 2:
    print('Pareto \ FPTAS ? ')
    method = str(input())
else:
    method = 'Promos'


# In[ ]:





# In[ ]:


with open('user_station','rb') as handle:
    user_station = pickle.load(handle)

with open('net_prices' , 'rb') as handle:
    net_prices = pickle.load(handle)
    
with open('item_cat', 'rb') as handle:
    item_cat = pickle.load(handle)


# In[ ]:


data = pd.read_csv('full_dataset.csv', sep = ',', encoding= 'iso-8859-1')


# In[ ]:


data['TRANSACTION_DATE'] = pd.to_datetime(data['TRANSACTION_DATE'], format='%Y-%m-%d %H:%M:%S')
data['PRICE'] = data['PRICE'].apply(lambda x: float(x.replace(',','.')))

price = data.groupby(['ARTICLE_ID'])['PRICE'].mean().reset_index() #Mean of product in the same type
data = data[['TRANSACTION_DATE','CUST_ID','ARTICLE_ID']]
data = data.merge(price, on=['ARTICLE_ID'])

data_t = data[data['TRANSACTION_DATE'] >= dt.datetime(2019,6,1,0,0,0)]
data = data[data['TRANSACTION_DATE'] < dt.datetime(2019,6,1,0,0,0)]

data = data[['CUST_ID','ARTICLE_ID','PRICE']]
data_t = data_t[['CUST_ID','ARTICLE_ID','PRICE']]

data = data[data['ARTICLE_ID'].isin(items)]
data = data[data['CUST_ID'].isin(user_station.keys())]
samples = data.groupby(['CUST_ID','ARTICLE_ID']).size().reset_index(name="FREQUENCY") #FREQUENCY
data = data.merge(samples, on=['CUST_ID','ARTICLE_ID'])
data = data.drop_duplicates()


# In[ ]:


users = sorted(list(data.CUST_ID.unique()))
rating = list(np.sort(data.FREQUENCY))

rows = data.CUST_ID.astype(pd.api.types.CategoricalDtype(categories = users)).cat.codes    # Get the associated row indices
cols = data.ARTICLE_ID.astype(pd.api.types.CategoricalDtype(categories = items)).cat.codes    # Get the associated row indices

# Create the purchase matrix as a sparce matrix, each row is a customer and each column is a product
rating_matrix = sparse.csc_matrix((rating, (rows, cols)), shape = (len(users), len(items)  )  )

# transform the matrix into a binary matrix, 1 for purchased and 0 for not purchased.
rating_matrix[rating_matrix >= 1] =1


# In[ ]:


data_t = data_t.drop_duplicates()
test_set = data_t[['CUST_ID', 'ARTICLE_ID']]

keys = list(data[['CUST_ID', 'ARTICLE_ID']].columns.values)
i1 = test_set.set_index(keys).index
i2 = data[['CUST_ID', 'ARTICLE_ID']].set_index(keys).index
test_set = test_set[~i1.isin(i2)]

test_set = test_set[test_set['CUST_ID'].isin(set(users))]  # Keep only users appearing in the training set
test_set = test_set[test_set.groupby(['CUST_ID'])['CUST_ID'].transform('count').ge(10)] # Only users with at least 5 or 10 purchases

# Create a dictionnary where keys are customers and values are set of purchased products in the test set
test_dict = test_set.groupby(['CUST_ID']).ARTICLE_ID.agg(set).to_dict()


# In[ ]:


def cosine_similarities(mat):
    col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
    return col_normed_mat.T * col_normed_mat


# In[ ]:


similarity_matrix= cosine_similarities(rating_matrix)


# In[ ]:


# indices of Items purchased by a customer

def get_items_purchased_user(customer_id, customer_list, items_list, purchase_matrix):
    cust_ind = np.where(customer_list == customer_id)[0][0]  # return the index row of the customer
    purchased_ind = purchase_matrix [cust_ind,:].nonzero()[1]  #get column indices of purchased items
    purchased_liste = []
    for i in purchased_ind:
        purchased_liste.append(items_list[i])
    return purchased_liste

# indices of users that purchased an item
def get_users_purchased_item (item_id, customer_list, items_list, purchase_matrix):
    item_ind = np.where(items_list==item_id)[0][0] # return the index column of the item
    users_ind = purchase_matrix[:,item_ind].nonzero()[0] #get row indices of users purchased the item

    return users_ind

def item_similarities(item_id, items_list, sim_matrix):

    item_ind = np.where(items_list==item_id)[0][0]
    item_sim_vector = sim_matrix[item_ind,:]
    return item_sim_vector

def profile_similarities (user, customer_list, items_list,sim_matrix  ):
    purchased = get_items_purchased_user(user, customer_list, items_list, rating_matrix)
    liste_sim = []
    for p in purchased:
        liste_sim.append(item_similarities(p, items, sim_matrix))
    reco_vector =liste_sim[0]
    for l in liste_sim[1:]:
        reco_vector= reco_vector + l
    return reco_vector/len(liste_sim)

def compute_utilities(user, customer_list, items_list, sim_matrix):
    user_util = profile_similarities (user, customer_list, items_list,sim_matrix).toarray().reshape(-1)
    dict_utilities = {}
    for i,j in enumerate(user_util):
        dict_utilities [items_list[i]]= j
    return dict_utilities

def price(user,item):
    a = user_station[user].get('COCO', 0)
    b = user_station[user].get('CODO', 0)
    c= user_station[user].get('DODO', 0)

    p1 =  net_prices[item].get('COCO',0)
    p2 =  net_prices[item].get('CODO',0)
    p3 =  net_prices[item].get('DODO',0)
    price = (a*p1) + (b*p2) + (c*p3)
    if price != 0:
        return price
    else :
        return 1

def compute_prices(user, items_list):
    item_prices = {}

    for i in items_list:
        item_prices[i]= price(user,i)
    return item_prices

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
        margin = 0
    return margin


def get_avg_margin(user,ensemble_produits):
    avg_m =0
    for i in ensemble_produits:
        avg_m = avg_m + margin(user,i)
    return avg_m / len(ensemble_produits)


# In[ ]:





# In[ ]:


def worker(key,p,k,capacity,results_avg,recall,f1_score,selected_null,marginList):
    weight_cost = []
    prix = compute_prices(np.int64(key),items)
    util = compute_utilities(np.int64(key), users, items, similarity_matrix)
    for i in items:
        weight_cost.append([prix[i],util[i]])
    purchased = p
    
    best_cost, best_combination = kp.method(len(weight_cost),capacity,weight_cost)
    
    selected_items = []
    
    ind_best = np.argsort(best_combination)[::-1]
    i=0
    n=10
    for ind in ind_best:
        if i >= n:
            break
        else:
            if best_combination[ind] == 1:
                selected_items.append(items[ind])
                i+=1
    '''
    for i in range(len(best_combination)):
        if best_combination[i] == 1:
            selected_items.append(items[i])
    '''        
    
    if (len(selected_items)!= 0):
        precision = len(set(selected_items).intersection(purchased))/len(selected_items)
        marge = get_avg_margin(np.int64(key),set(selected_items))

        reca = len(set(selected_items).intersection(purchased))/len(purchased)
        if precision+reca ==0:
            f1=0
        else:
            f1 = (2*precision*reca)/(precision+reca)
    
        results_avg.append(precision)
        recall.append(reca)
        f1_score.append(f1)
        marginList.append(marge)
        print('AVG Margin : ', marge)
        print('k = ', len(selected_items))
        print('AVG Precision : ', precision,'recall : ', reca, 'f1 score : ',f1)
        print (selected_items)
        
    else:
        print('selected==0')
        selected_null.append(1)

    print('number user : ', k/num)


# In[ ]:


def worker(key,p,k,capacity,results_avg,recall,f1_score,selected_null,marginList):
    weight_cost = []
    prix = compute_prices(np.int64(key),items)
    util = compute_utilities(np.int64(key), users, items, similarity_matrix)
    #cp = 0
    for i in items:
        weight_cost.append([int(10*prix[i]),util[i]])
        
    purchased = p
    if method == 'BB':
        best_cost, best_combination = kp.solver_BB(capacity, weight_cost)
    elif method == 'DP':
        best_cost, best_combination = kp.solver_DP(capacity, weight_cost)
    else:
        best_cost, best_combination = kp.greedy_ratio(capacity, weight_cost)
    
    selected_items = []
    '''
    ind_best = np.argsort(best_combination)[::-1]
    i=0
    n=10
    for ind in ind_best:
        if i >= n:
            break
        else:
            if best_combination[ind] == 1:
                selected_items.append(items[ind])
                i+=1         
    '''     
    
    for i in range(len(best_combination)):
        if best_combination[i] == 1:
            selected_items.append(items[i])    
    
    if (len(selected_items)!= 0):
        precision = len(set(selected_items).intersection(purchased))/10
        marge = get_avg_margin(np.int64(key),set(selected_items))

        reca = len(set(selected_items).intersection(purchased))/len(purchased)
        if precision+reca ==0:
            f1=0
        else:
            f1 = (2*precision*reca)/(precision+reca)
    
        results_avg.append(precision)
        recall.append(reca)
        f1_score.append(f1)
        marginList.append(marge)
        #print('AVG Margin : ', marge)
        #print('k = ', len(selected_items))
        print('AVG Precision : ', precision,'recall : ', reca, 'f1 score : ',f1,'AVG Margin : ', marge)


# In[ ]:


def solver_fptas(key,p,k,capacity,results_avg,ep)
 
    dict_utilities = compute_utilities( np.int64(key)  , users, items, similarity_matrix)
    items_l = list(net_prices.keys())
    dict_margins= compute_margins( np.int64(1)  , items_l)
    y=[]
    for item in (dict_utilities.keys()):
        if item in dict_margins.keys():
            y.append(item)
            v.append(dict_utilities[item])
            p.append(dict_margins[item])
            w.append(1)

    # Create model
    m = GEKKO()
    # Variables
    x = m.Array(m.Var,len(y),lb=0,ub=1,integer=True)
    # Objective
    m.Obj(-sum(v[i]*x[i] for i in range(len(y))))        
    m.Obj(-sum(p[i]*x[i] for i in range(len(y))))
    # Constraint
    limit = 2
    print(len(y))
    m.Equation(sum([w[i]*x[i] for i in range(len(y))]) <= limit)

    # Optimize with APOPT
    m.options.SOLVER = 1
    m.solve()

    # Print the value of the variables at the optimum
    for i in range(items):
        print("%s = %f" % (y[i], x[i].value[0]))
    # Print the value of the objective
    print("Objective = %f" % (m.options.objfcnval))
        


# In[ ]:


def worker_bi(key, p, B, results_avg, results_uti, results_mar,results_avg1, results_uti1, results_mar1,error,epsilon):
    
    print ('Veuillez choisir la valeur de epsilon : ')
        epsilon = float(input())
    
    dict_utilities = compute_utilities( np.int64(key)  , users, items, similarity_matrix)

    dict_margins= compute_margins( np.int64(key)  , items)

    purchased = p
    ordered_dict = {i: dict_utilities[i] + dict_margins[i] for i in dict_utilities.keys() }        
    input_list= sorted(ordered_dict.items(), key=operator.itemgetter(1), reverse=True)
    dominance_dict= dict()
    dominance_dict[input_list[0][0]] = set()


    for i in range(1,len(input_list)):
        dominance_dict[input_list[i][0]] = set()
        for j in range(i-1, 0, -1):
            if product_dominance(input_list[j][0], input_list[i][0], dict_utilities, dict_margins,dict_price,error,epsilon):
                dominance_dict[input_list[i][0]].add(input_list[j][0])
                if len(dominance_dict[input_list[i][0]]) >= k:
                    break

    input_dict_pruned = {i:v  for i, v in dominance_dict.items() if len(v) < k }
    input_list_pruned =  sorted(input_dict_pruned.items(), key=operator.itemgetter(1), reverse=True)
    input_list_pruned = [tup[0]  for tup in input_list_pruned]

    if (len(input_list_pruned)<50):
        if method == 'FPTAS':
            TOTO = solver_fptas(input_list_pruned ,dict_utilities ,dict_margins, dict_price, error, epsilon,B)
        else:
            TOTO = pareto.Pareto(k, input_list_pruned ,dict_utilities ,dict_margins,input_dict_pruned)

        precisions =0
        max_uti = 0
        max_marge=0
        margins = 0
        maximum_marge=0
        min =19999
        max =0
        if len(TOTO) > max:
            max = len(TOTO)
        if len(TOTO) < min:                
            min = len(TOTO)
            
        for ens in TOTO:
            prec = len(ens.intersection(purchased))/k
            marge = get_avg_margin(key,ens)
            if max_uti <= prec :
                max_uti = prec
                maximum_marge = marge
            if max_marge < marge :
                max_marge = marge
                max_precision = prec
            precisions = precisions + prec
            margins = margins + marge
        avg_precision = precisions/len(TOTO)
        avg_margins = margins/len(TOTO)
        print('AVG Precision : ', avg_precision, 'AVG Margin :', avg_margins)
        print('MAX utility: ', 'precision :' , max_uti, 'marge :',maximum_marge )
        print('MAX margin: ', 'precision :' , max_precision, 'marge :',max_marge)
        results_avg.append(avg_precision)
        results_uti.append(max_uti)
        results_mar.append(max_precision)
        results_avg1.append(avg_margins)
        results_uti1.append(maximum_marge)
        results_mar1.append(max_marge)    


# In[ ]:


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    results_avg = manager.list()
    recall = manager.list()
    f1_score = manager.list()
    selected_null = manager.list()
    marginList = manager.list()

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    
    t_start = time()
    
    if nm == 1: 
        for i,(key,p) in enumerate(test_dict.items()):
            capacity = get_capacity(key,users,items)
            pool.apply_async(worker,args=(key,p,i,capacity,results_avg,recall,f1_score,selected_null,marginList))
        pool.close()
        pool.join()
    else:
        for i,(key,p) in enumerate(test_dict.items()):
            pool.apply_async(worker,args=(key,p,i,capacity,results_avg,recall,f1_score,selected_null,marginList))
        pool.close()
        pool.join()
        
    t_finished = time()
    
    print('time =',t_finished - t_start ) 
    print('AVG precision =',  sum(results_avg)/len(results_avg))
    print('recall =',  sum(recall)/len(recall))
    print('AVG f1 =',  sum(f1_score)/len(f1_score))
    print('Marge =',  sum(marginList)/len(marginList))


# In[ ]:




