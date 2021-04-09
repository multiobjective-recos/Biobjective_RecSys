#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
import pickle
import time
import multiprocessing
import operator
import sklearn.preprocessing as pp


class Pareto(object):
    

    def cosine_similarities(mat):
        col_normed_mat = pp.normalize(mat.tocsc(), axis=0)
        return col_normed_mat.T * col_normed_mat


    def jaccard_similarities(mat):
        cols_sum = mat.getnnz(axis=0)
        ab = mat.T * mat

        # for rows
        aa = np.repeat(cols_sum, ab.getnnz(axis=0))
        # for columns
        bb = cols_sum[ab.indices]

        similarities = ab.copy()
        similarities.data /= (aa + bb - ab.data)

        return similarities


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

    def top_N_recommendations(user,n, customer_list, items_list, sim_matrix):
        purchased = get_items_purchased_user(user, customer_list, items_list, rating_matrix)
        ensemble_produits = set(purchased)
        pref = profile_similarities (user, customer_list, items_list,sim_matrix).toarray().reshape(-1)
        items_ind = np.argsort(pref)[::-1]
        rec_list= []
        i=0
        for ind in items_ind:
            if i >= n:
                break
            else:

                item_id = items_list[ind]
                if item_id in ensemble_produits:
                    continue
                else:
                    rec_list.append(item_id)
                    i=i+1

        return rec_list


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

    def get_margin_recommendations(user, n , customer_list, items_list):

        user_ind = np.where(customer_list == user)[0][0]  # return the index row of the customer
        user_margin = np.zeros(len(items_list))

        for i in range(len(items_list)):
            user_margin[i] = margin(user,items[i])

        items_ind = np.argsort(user_margin)[::-1]
        rec_list= []
        for ind in items_ind[:n]:
            item_id = items_list[ind]
            rec_list.append(item_id)

        return rec_list

    def scalarization(user, n , customer_list, items_list, purchase_matrix, Confidence_matrix, alpha):
        user_ind = np.where(customer_list == user)[0][0]  # return the index row of the customer
        user_pref = (purchase_matrix[user_ind,:].dot(Confidence_matrix)).toarray().reshape(-1)
        user_pref = user_pref/user_pref.max()
        user_margin = np.zeros(len(items_list))

        for i in range(len(items_list)):
            user_margin[i] = margin(user,items[i])
        user_margin = user_margin/user_margin.max()

        scalarized = alpha* user_pref + (1-alpha)* user_margin
        items_ind = np.argsort(scalarized)[::-1]
        rec_list= []
        for ind in items_ind[:n]:
            item_id = items_list[ind]
            rec_list.append(item_id)

        return rec_list

    def scalarization(user, n , customer_list, items_list, sim_matrix, alpha):
        purchased = get_items_purchased_user(user, customer_list, items_list, rating_matrix)
        ensemble_produits = set(purchased)
        user_pref = profile_similarities (user, customer_list, items_list,sim_matrix).toarray().reshape(-1)
        user_pref = user_pref/user_pref.max()
        user_margin = np.zeros(len(items_list))

        for i in range(len(items_list)):
            user_margin[i] = margin(user,items[i])
        user_margin = user_margin/user_margin.max()

        scalarized = alpha* user_pref + (1-alpha)* user_margin
    
        items_ind = np.argsort(scalarized)[::-1]
        rec_list= []
        for ind in items_ind[:n]:
            item_id = items_list[ind]
            if item_id in ensemble_produits:
                continue
            else:
                rec_list.append(item_id)

        return rec_list

    def print_prices(liste_items):
        return [net_prices[i] for i in liste_items]

    def get_avg_margin(user,ensemble_produits):
        avg_m =0
        for i in ensemble_produits:
            avg_m = avg_m + margin(user,i)
        return avg_m / len(ensemble_produits)

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

    def product_dominance(product1, product2, dict1, dict2):

        if dict1[product1] > dict1[product2] and dict2[product1]> dict2[product2] :
            return True
        else:
            return False
    
    def set_dominance(set1, set2, dict1, dict2):
        uti1 = 0
        uti2 = 0
        for p in set1:
            uti1 = uti1 + dict1[p]
        for p in set2:
            uti2 = uti2 + dict1[p]
        mar1= 0
        mar2= 0

        for p in set1:
            mar1 = mar1 + dict2[p]
        for p in set2:
            mar2 = mar2 + dict2[p]

        if uti1 > uti2 and mar1>mar2:
            return True
        else:
            return False

    def Pareto(k,input,dict_utilities ,dict_margins,input_dict_pruned):
        cop = input.copy()
        cop1 = input.copy()
        if k == 1 :
            S2= set()
            S2.add(frozenset( {cop[-1] }))
        
        else:
            S2 = set()

            product = cop[-1]
            cop.remove(cop[-1])
            Pk1 = Pareto(k-1,cop,dict_utilities ,dict_margins,input_dict_pruned)
            
            for ensemble in Pk1 :
                if input_dict_pruned[product].issubset(ensemble):
                
                    Candidate = ensemble.union({product})
                    S2.add(Candidate)

        if k<len(cop1):
            cop1.remove(cop1[-1])
            S1 = Pareto(k,cop1,dict_utilities ,dict_margins,input_dict_pruned)
            print('S1=', S1)
        else:
            S1 = set()

        candidates = S2.union(S1)
        P=set()
        for i in candidates:
            toto= 0
            for j in candidates:
                if  set_dominance(j ,i, dict_utilities ,dict_margins):
                    toto= 1
                    break
            
            if toto==0:
                P.add(i)  
        return P




    def worker_bi(key, p, k,results_avg, results_uti, results_mar,results_avg1, results_uti1, results_mar1,results_recall,results_f):

        dict_utilities = compute_utilities( np.int64(key)  , users, items, similarity_matrix)
        dict_margins= compute_margins( np.int64(key)  , items_l)
        print(" dict_margins = " ,len(dict_margins))

        purchased = p

        ordered_dict = dict()
        for i in dict_utilities.keys():
            if (i in dict_margins.keys()):
                ordered_dict[i] = dict_utilities[i] + dict_margins[i]  	

        input_list= sorted(ordered_dict.items(), key=operator.itemgetter(1), reverse=True)
        dominance_dict= dict()
        dominance_dict[input_list[0][0]] = set()


        for i in range(1,len(input_list)):
            dominance_dict[input_list[i][0]] = set()
            for j in range(i-1, 0, -1):

                if product_dominance(input_list[j][0], input_list[i][0], dict_utilities, dict_margins):
                    dominance_dict[input_list[i][0]].add(input_list[j][0])
                    if len(dominance_dict[input_list[i][0]]) >= k:
                        break

        input_dict_pruned = {i:v  for i, v in dominance_dict.items() if len(v) < k }
        input_list_pruned =  sorted(input_dict_pruned.items(), key=operator.itemgetter(1), reverse=True)
        input_list_pruned = [tup[0]  for tup in input_list_pruned]

        print (len(input_list_pruned))


        if (len(input_list_pruned)<50):
            TOTO = Pareto(k, input_list_pruned ,dict_utilities ,dict_margins,input_dict_pruned

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
                reca = len(ens.intersection(purchased))/len(purchased)

                marge = get_avg_margin(key,ens)


                if max_uti <= prec :
                    max_uti = prec
                    maximum_marge = marge
                if max_marge < marge :
                    max_marge = marge
                    max_precision = prec
                precisions = precisions + prec
                margins = margins + marge

            if(prec+reca == 0):
                f=0
            else:
                f=2*prec*reca/prec+reca

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
            results_recall.append(reca)
            results_f.append(f)

    def worker1( key, p, results ):
        liste= get_margin_recommendations(np.int64(key),  5 , users, items)

        purchased = p
        prec = len(set(liste).intersection(purchased))/5
        print(prec)
        results.append(prec)

    def worker( key, p, results ):
        liste= get_recommendations(np.int64(key),  5, users, items, rating_matrix, Association_matrix)

        purchased = p
        prec = len(set(liste).intersection(purchased))/5
        print(prec)
        results.append(prec)

