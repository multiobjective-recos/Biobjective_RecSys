#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import random


# In[2]:


class MUD(object):
    #SAMPLE IS AN ARRAY [CUST - ITEM - RATINGS]
    def __init__(self, samples, num_users, num_items, CFK, CFlr, CFbeta, CFiterations,     K, lr, beta, iterations, proba_dist,item_price, negative_samples,bought):
        """
        CFlr: learning rate for CF
        CFK: latent dimension in CF
        CFbeta: the regularizer of CF
        CFiterations: the number of iterations of CF
        lr: learning rate
        K: latent dimension
        beta: regularizer
        iterations: the number of iterations
        item_path: the path for rating distribution
        sample_path: the path for related items
        item_price_path: the path for the price of items
        """
        self.samples = samples
        self.num_users = num_users
        self.num_items = num_items
        self.CFK = CFK
        self.CFlr = CFlr
        self.CFbeta = CFbeta
        self.CFiterations = CFiterations
        self.K = K
        self.lr = lr
        self.beta = beta
        self.iterations = iterations
        self.item_result = proba_dist
        self.item_price = item_price
        self.negative_samples = negative_samples
        self.all_bought = bought

    def train(self, users, items):
        
        """
        initialize the parameters
        """
        ### Initialize user and item latent feature matrice of alpha
        self.Pa = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Qa = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        
        ### Initialize the biases of alpha
        self.ba_u = np.zeros(self.num_users)
        self.ba_i = np.zeros(self.num_items)
        self.ba = 0
        
        ### Initialize user and item latent feature matrice of rating
        self.Pr = np.random.normal(scale=1./self.CFK, size=(self.num_users, self.K))
        self.Qr = np.random.normal(scale=1./self.CFK, size=(self.num_items, self.K))
        
        ### Initialize the biases of rating
        self.br_u = np.zeros(self.num_users)
        self.br_i = np.zeros(self.num_items)
        rating = [i[3] for i in self.samples] 
        self.br = np.mean(rating)

        self.corres_items = dict()
        self.corres_users = dict()
        n = 0

        for s in users:
            self.corres_users[s] = n
            n = n+1
        
        n = 0
        for s in items:
            self.corres_items[s] = n
            n = n+1
        
        n = 0
        print("===========================")
        mse = self.mse()
        print("Iteration: %d ; error = %.4f" % (n, mse))
        n += 1
        
        while(n<=self.CFiterations):
            print("===========================")
            self.cf()
            mse = self.mse()
            print("Iteration: %d ; error = %.4f" % (n, mse))
            n += 1
        n = 1
        
        while(n<=self.iterations):
            print("===========================")
            self.risk()
            self.sgd(items)
            print("Iteration: %d" % n)
            n += 1

    def cf(self):
        """
        get predicted rating
        """
        for i,j,typ,q in self.samples:
            rij = self.get_rating(i,j,typ)
            commonTerm_r1 = - 2 * (q - rij)

            s_j = self.corres_items[str(j)+'_'+typ]
            s_i = self.corres_users[i]

            self.br_u[s_i] -= self.CFlr * (commonTerm_r1 + 2 * self.CFbeta * self.br_u[s_i])
            self.br_i[s_j] -= self.CFlr * (commonTerm_r1 + 2* self.CFbeta * self.br_i[s_j])
            self.Pr[s_i, :] -= self.CFlr * (commonTerm_r1 * self.Qr[s_j, :] + 2 * self.CFbeta * self.Pr[s_i,:])
            self.Qr[s_j, :] -= self.CFlr * (commonTerm_r1 * self.Pr[s_i, :] + 2 * self.CFbeta * self.Qr[s_j,:])
        
    def risk(self):
        """
        risk neutral
        """
        for i, j, typ, q in self.samples:
            aij = self.get_aij(i,j,typ)

            s_j = self.corres_items[str(j)+'_'+typ]
            s_i = self.corres_users[i]

            j = str(j)+'_'+typ

            commonTerm_a1 = 2 * (aij * self.u_bar(self.item_result[j]) * math.log(2) -             aij * self.u_r_bar(self.item_result[j]) * math.log(2) )              * math.log(2) * (self.u_bar(self.item_result[j]) - self.u_r_bar(self.item_result[j]))
            self.ba_u[s_i] -= self.lr * (commonTerm_a1 + 2 * self.beta * self.ba_u[s_i])
            self.ba_i[s_j] -= self.lr * (commonTerm_a1 + 2 * self.beta * self.ba_i[s_j])
            self.ba -= self.lr * (commonTerm_a1 + 2 * self.beta * self.ba)
            self.Pa[s_i, :] -= self.lr * (commonTerm_a1 * self.Qa[s_j, :] + 2 * self.beta * self.Pa[s_i,:])
            self.Qa[s_j, :] -= self.lr * (commonTerm_a1 * self.Pa[s_i, :] + 2 * self.beta * self.Qa[s_j,:])  
            
    def sgd(self, items):
        """
        maximize mud
        """
        all_items = items

        for i, j, typ, q in self.samples:
            negative_samples_j = self.negative_samples[str(j)+'_'+typ]
            negative_samples_j = negative_samples_j[0]
            bought = self.all_bought[i]
            for k in bought:
                if k in negative_samples_j:
                    negative_samples_j.remove(k) 
            negative_samples_j.append(str(j)+'_'+typ)
            if len(negative_samples_j) < 2:
                not_bought = list(set(all_items) - set(bought))
                negative_samples_j.append(random.sample(not_bought,1)[0])
            
            aij = self.get_aij(i,j,typ)
            rij = self.get_rating(i,j,typ)
            sigij = 2/(1+math.exp(-rij)) - 1
            sigijd = 2 * math.exp(-rij) / np.square(1 + math.exp(-rij))
            sigpij = 1/(1+math.exp( -self.item_price[ str(j)+'_'+typ ][0] ))
            
            commonTerm_a = sigij / sigpij
            temp_0 = 0
            temp_1 = 0
            temp_2 = 0
            aik_total = []
            rik_total = []

            for k in negative_samples_j:
                k = k.split('_')
                k[0] = int(k[0])

                aik = self.get_aij(i,k[0],k[1])
                aik_total.append(aik)
                rik = self.get_rating(i,k[0],k[1])
                rik_total.append(rik)
                sigik = 2/(1+math.exp(-rik)) - 1
                sigpik = 1/(1+math.exp( -self.item_price[ str(k[0])+'_'+k[1] ][0] ))
                temp_0 += math.exp(aik * sigik / sigpik)
                temp_1 += math.exp(aik * sigik / sigpik) * sigik / sigpik
                
            commonTerm_ai = commonTerm_a - temp_1/temp_0
            commonTerm_aj = commonTerm_a - commonTerm_a * math.exp(aij * commonTerm_a)/temp_0
            
            s_j = self.corres_items[str(j)+'_'+typ]
            s_i = self.corres_users[i]

            self.ba_u[s_i] += self.lr * (commonTerm_ai - 2 * self.beta * self.ba_u[s_i])
            self.ba_i[s_j] += self.lr * (commonTerm_aj - 2 * self.beta * self.ba_i[s_j])
            self.ba += self.lr * (commonTerm_ai - 2 * self.beta * self.ba)
            self.Pa[s_i, :] += self.lr * (commonTerm_ai * self.Qa[s_j, :] - 2 * self.beta * self.Pa[s_i,:])
            self.Qa[s_j, :] += self.lr * (commonTerm_aj * self.Pa[s_i, :] - 2 * self.beta * self.Qa[s_j,:])
    
    def u_r_bar (self, row):
        """
        calculate the utility of expected rating
        """
        row = row[0]
        mean_val = 0
        for ele in row:
            mean_val += ele[0] * ele[1]       
        val = 2/(1+math.exp(-mean_val))-1
        return val
    
    def u_bar(self,row):
        """
        calculate the expectation of the utility
        """
        row = row[0]
        val = 0
        for ele in row:
            val += ele[1]*(2/(1+math.exp(-ele[0]))-1)
        return val
      
    def mse(self):
        error = 0
        for s in self.samples:
            rij = self.get_rating(s[0],s[1],s[2])
            error += pow(rij - s[3], 2)
        return np.sqrt(error/len(self.samples))
        
    def get_aij(self,i,j,typ):
        s_j = self.corres_items[str(j)+'_'+typ]
        s_i = self.corres_users[i]

        aij = self.ba + self.ba_u[s_i] + self.ba_i[s_j] + self.Pa[s_i, :].dot(self.Qa[s_j, :].T)
        return aij
    
    def get_rating(self,i,j,typ):
        s_j = self.corres_items[str(j)+'_'+typ]
        s_i = self.corres_users[i]

        rij = self.br + self.br_u[s_i] + self.br_i[s_j] + self.Pr[s_i, :].dot(self.Qr[s_j, :].T)
        return rij
    
    
    def get_top_n(self, user, list_items_station, typ, pr, num):
        test_score = dict()

        #i = self.corres_users[user]

        for item in list_items_station:
            #j = self.corres_items[ str(item)+'_'+typ ]

            rij = self.get_rating(user,item,typ)
            aij = self.get_aij(user,item,typ)
            sigij = 2 /(1+math.exp(-rij)) - 1
            sigpij = 1/(1+math.exp(-self.item_price[ str(item)+'_'+typ ] [0] ))
            sij = aij * sigij /sigpij
            test_score[str(item)+'_'+typ] = sij
        
        test_score = sorted(test_score.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)

        bought = self.all_bought[user]
        k = 0
        rec = []
        for (s,j) in test_score:
            if k >= num:
                break
            else:
                if s in bought:
                    continue
                else:
                    if float(self.item_price[s][0]) <= pr:
                        rec.append(s)
                        k = k+1
        return rec


# In[ ]:





# In[ ]:





# In[ ]:




