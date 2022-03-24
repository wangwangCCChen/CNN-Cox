# -*- coding: utf-8 -*-
"""
__author__ = 'Sungsoo Park'
"""


import tensorflow as tf
from keras import metrics, optimizers, applications, callbacks
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import numpy as np
import pandas as pd
from wx_hyperparam import WxHyperParameter
import pickle
from numpy import array,argmax
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold,KFold
import collections
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from tensorflow.keras.optimizers import SGD, RMSprop,Adam
from tensorflow.keras.models import Sequential, Model,Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Input,concatenate,Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from keras.utils import to_categorical
from sklearn.utils import shuffle
import operator
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, concatenate, Dropout, Activation
from keras import optimizers, applications, callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from lifelines.utils import concordance_index
from keras.callbacks import LearningRateScheduler
from abc import ABCMeta, abstractmethod
from sklearn.metrics import roc_auc_score
def get_risk_group(x_trn, c_trn, s_trn, high_risk_th, low_risk_th):
    hg = []
    lg = []
    for n,os in enumerate(s_trn):
        if os <= high_risk_th and c_trn[n] == 1:
            hg.append(x_trn[n])
        if os > low_risk_th:
            lg.append(x_trn[n])

    return np.asarray(hg), np.asarray(lg)
def get_train_val(hg, lg, is_categori_y, seed):
    x_all = np.concatenate([hg, lg])
    hg_y = np.ones(len(hg))
    lg_y = np.zeros(len(lg))
    y_all = np.concatenate([hg_y, lg_y])
    if is_categori_y:            
        y_all = to_categorical(y_all, num_classes=2)
    x_all, y_all = shuffle(x_all, y_all, random_state=seed)
    x_trn, x_dev , s_trn, s_dev=train_test_split(x_all, y_all,test_size=0.2,stratify=y_all,random_state=1)

    return x_trn, s_trn , x_dev, s_dev
wx_hyperparam = WxHyperParameter(learning_ratio=0.001)
def NaiveSLPmodel(x_train, y_train, x_val, y_val,cancer_type, hyper_param=wx_hyperparam):
    input_dim = len(x_train[0])
    inputs = Input((input_dim,))
    fc_out = Dense(2,  kernel_initializer='zeros', bias_initializer='zeros', activation='softmax')(inputs)
    model = Model(inputs=inputs, outputs=fc_out)

    #build a optimizer
    sgd = optimizers.SGD(lr=hyper_param.learning_ratio, decay=hyper_param.weight_decay, momentum=hyper_param.momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #call backs
    def step_decay(epoch):
        exp_num = int(epoch/10)+1
        return float(hyper_param.learning_ratio/(10 ** exp_num))

    best_model_path="D:/Machine learning/各种cox和deepsur/deepsur/modelwx/slp_wx_weights_best"+"%s.hdf5" % (cancer_type)
    save_best_model = ModelCheckpoint(best_model_path, monitor="val_loss", verbose=hyper_param.verbose, save_best_only=True, mode='min')
    change_lr = LearningRateScheduler(step_decay)                                

    #run
    history = model.fit(x_train, y_train, validation_data=(x_val,y_val), 
                epochs=hyper_param.epochs, batch_size=hyper_param.batch_size,verbose=0, shuffle=True,callbacks=[save_best_model, change_lr])

    #load best model
    model.load_weights(best_model_path)

    return model

def WxSlp(x_train, y_train, x_val, y_val, test_x, test_y,cancer_type, n_selection=100, hyper_param=wx_hyperparam, num_cls=2):#suppot 2 class classification only now.
    tf_session = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(tf_session)
    
#     K.set_session(sess)

    input_dim = len(x_train[0])

    # make model and do train
    model = NaiveSLPmodel(x_train, y_train, x_val, y_val,cancer_type, hyper_param=hyper_param)

    #load weights
    weights = model.get_weights()

    #cacul WX scores
    num_data = {}
    running_avg={}
    tot_avg={}
    Wt = weights[0].transpose() #all weights of model
    Wb = weights[1].transpose() #all bias of model
    for i in range(num_cls):
        tot_avg[i] = np.zeros(input_dim) # avg of input data for each output class
        num_data[i] = 0.
    for i in range(len(x_train)):
        c = y_train[i].argmax()
        x = x_train[i]
        tot_avg[c] = tot_avg[c] + x
        num_data[c] = num_data[c] + 1
    for i in range(num_cls):
        tot_avg[i] = tot_avg[i] / num_data[i]

    #data input for first class
    wx_00 = tot_avg[0] * Wt[0]# + Wb[0]# first class input avg * first class weight + first class bias
    wx_01 = tot_avg[0] * Wt[1]# + Wb[1]# first class input avg * second class weight + second class bias

    #data input for second class
    wx_10 = tot_avg[1] * Wt[0]# + Wb[0]# second class input avg * first class weight + first class bias
    wx_11 = tot_avg[1] * Wt[1]# + Wb[1]# second class input avg * second class weight + second class bias

    wx_abs = np.zeros(len(wx_00))
    for idx, _ in enumerate(wx_00):
        wx_abs[idx] = np.abs(wx_00[idx] - wx_01[idx]) + np.abs(wx_11[idx] - wx_10[idx])

    selected_idx = np.argsort(wx_abs)[::-1][0:n_selection]
    selected_weights = wx_abs[selected_idx]

    #get evaluation acc from best model
    loss, test_acc = model.evaluate(test_x, test_y,verbose=0)
    tf.compat.v1.keras.backend.clear_session()

#     K.clear_session()

    return selected_idx, selected_weights, test_acc
def DoFeatureSelectionWX(train_x, train_y, val_x, val_y, test_x, test_y,cancer_type, f_list, hp, n_sel = 14, sel_option='top'):
    ITERATION = 10
    feature_num = len(f_list)

    all_weight = np.zeros(feature_num)    
    all_count = np.ones(feature_num)

    accs = []
    for i in range(0, ITERATION):    
        sel_idx, sel_weight, test_acc = WxSlp(train_x, train_y, val_x, val_y, test_x, test_y,cancer_type, n_selection=min(n_sel*100, feature_num), hyper_param=hp)
        accs.append(test_acc)
        for j in range(0,min(n_sel*100, feature_num)):
            all_weight[sel_idx[j]] += sel_weight[j]
            all_count[sel_idx[j]] += 1        

    all_weight = all_weight / all_count
    sort_index = np.argsort(all_weight)[::-1]
    if sel_option == 'top':
        sel_index = sort_index[:n_sel]

    sel_index = np.asarray(sel_index)
    sel_weight =  all_weight[sel_index]
    gene_names = np.asarray(f_list)
    sel_genes = gene_names[sel_index]

    return sel_index, sel_genes, sel_weight, np.mean(accs,axis=0)
def get_wx_sel_idx(x, c, s,cancer_type,high_risk_th, low_risk_th, feature_list, set_feature, sel_feature_num, sel_op, div_ratio = 4):
    high_risk_group, low_risk_group = get_risk_group(x,c,s,high_risk_th,low_risk_th)
    trn_x, trn_y, val_x, val_y = get_train_val(high_risk_group, low_risk_group, is_categori_y=True, seed=1)
    if len(set_feature):
        trn_x = trn_x[:,set_feature]
        val_x = val_x[:,set_feature]
    feature_num = trn_x.shape[1]

    if sel_feature_num == 0:
        hp = WxHyperParameter(epochs=50, learning_ratio=0.01, batch_size = int(len(trn_x)/4), verbose=False)
        sel_gene_num = int(max(sel_feature_num, feature_num/div_ratio))
    else:
        hp = WxHyperParameter(epochs=50, learning_ratio=0.001, batch_size = int(len(trn_x)/4), verbose=False)
        sel_gene_num = sel_feature_num
    sel_idx, sel_genes, sel_weight, test_auc = DoFeatureSelectionWX(trn_x, trn_y, val_x, val_y, val_x, val_y,cancer_type, feature_list, hp, 
                                                n_sel=sel_gene_num, sel_option=sel_op)

    return sel_idx
def feature_selection( x, c, s, names, sel_f_num,cancer_type): 
    if cancer_type == 'BLCA':
        step1_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,730,730,names,[], 0, 'top', div_ratio = 2)
        step2_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,365,1095,step1_sel_idx,step1_sel_idx, 0, 'top', div_ratio = 2)
        sel_f_num_write = len(step2_sel_idx)
        step3_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,200,1825,step2_sel_idx,step1_sel_idx[step2_sel_idx], sel_f_num_write, 'top', div_ratio = 2)
        final_sel_idx = step1_sel_idx[step2_sel_idx[step3_sel_idx]]
    if cancer_type == 'BRCA':
        step1_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,2555,2555,names,[], 0, 'top', div_ratio = 2)
        step2_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,2190,2920,step1_sel_idx,step1_sel_idx, 0, 'top', div_ratio = 2)
        sel_f_num_write = len(step2_sel_idx)
        step3_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,1095,3285,step2_sel_idx,step1_sel_idx[step2_sel_idx], sel_f_num_write, 'top', div_ratio = 2)
        final_sel_idx = step1_sel_idx[step2_sel_idx[step3_sel_idx]]
    if cancer_type == 'LUAD':
        step1_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,1095,1095,names,[], 0, 'top', div_ratio = 2)
        step2_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,730,1460,step1_sel_idx,step1_sel_idx, 0, 'top', div_ratio = 2)
        sel_f_num_write = len(step2_sel_idx)
        step3_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,365,1825,step2_sel_idx,step1_sel_idx[step2_sel_idx], sel_f_num_write, 'top', div_ratio = 2)
        final_sel_idx = step1_sel_idx[step2_sel_idx[step3_sel_idx]]
    if cancer_type == 'HNSC':
        step1_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,912,912,names,[], 0, 'top', div_ratio = 2)
        step2_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,548,1278,step1_sel_idx,step1_sel_idx, 0, 'top', div_ratio = 2)
        sel_f_num_write = len(step2_sel_idx)
        step3_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,365,1460,step2_sel_idx,step1_sel_idx[step2_sel_idx], sel_f_num_write, 'top', div_ratio = 2)
        final_sel_idx = step1_sel_idx[step2_sel_idx[step3_sel_idx]]
    if cancer_type == 'KIRC':
        step1_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,1825,1825,names,[], 0, 'top', div_ratio = 2)
        step2_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,1095,2190,step1_sel_idx,step1_sel_idx, 0, 'top', div_ratio = 2)
        sel_f_num_write = len(step2_sel_idx)
        step3_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,365,2920,step2_sel_idx,step1_sel_idx[step2_sel_idx], sel_f_num_write, 'top', div_ratio = 2)
        final_sel_idx = step1_sel_idx[step2_sel_idx[step3_sel_idx]]        
    if cancer_type == 'LGG':
        step1_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,1460,1460,names,[], 0, 'top', div_ratio = 2)
        step2_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,730,2190,step1_sel_idx,step1_sel_idx, 0, 'top', div_ratio = 2)
        sel_f_num_write = len(step2_sel_idx)
        step3_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,548,2555,step2_sel_idx,step1_sel_idx[step2_sel_idx], sel_f_num_write, 'top', div_ratio = 2)
        final_sel_idx = step1_sel_idx[step2_sel_idx[step3_sel_idx]]        
    if cancer_type == 'LUSC':
        step1_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,1095,1095,names,[], 0, 'top', div_ratio = 2)
        step2_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,730,1460,step1_sel_idx,step1_sel_idx, 0, 'top', div_ratio = 2)
        sel_f_num_write = len(step2_sel_idx)
        step3_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,365,1825,step2_sel_idx,step1_sel_idx[step2_sel_idx], sel_f_num_write, 'top', div_ratio = 2)
        final_sel_idx = step1_sel_idx[step2_sel_idx[step3_sel_idx]]       
    if cancer_type == 'SKCM':
        step1_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,1825,1825,names,[], 0, 'top', div_ratio = 2)
        step2_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,1460,2190,step1_sel_idx,step1_sel_idx, 0, 'top', div_ratio = 2)
        sel_f_num_write = len(step2_sel_idx)
        step3_sel_idx = get_wx_sel_idx(x, c, s,cancer_type,1095,2555,step2_sel_idx,step1_sel_idx[step2_sel_idx], sel_f_num_write, 'top', div_ratio = 2)
        final_sel_idx = step1_sel_idx[step2_sel_idx[step3_sel_idx]]
       
    final_sel_idx = final_sel_idx[:sel_f_num]

    return final_sel_idx


def preprocess( x, c, s, names, n_sel,cancer_type):
    sel100_idx = feature_selection(x, c, s, names,  n_sel,cancer_type)
    x_new = x[:,sel100_idx]
    return x_new ,sel100_idx
def get_data(X,E,Y,trn_index_list,tst_index_list, cvi):

    trn_index = trn_index_list[cvi]
    tst_index = tst_index_list[cvi]

    return X[trn_index], E[trn_index], Y[trn_index],X[tst_index], E[tst_index], Y[tst_index]
def preprocess_eval(sel100_idx, x):
    x_new = x[:,sel100_idx]
    return x_new



def merge500(sel_500_idx):
    
    list01 = list(set(sel_500_idx[0]).union(set(sel_500_idx[1])))
    list012 = list(set(list01).union(set(sel_500_idx[2])))
    list0123 = list(set(list012).union(set(sel_500_idx[3])))
    list01234 = list(set(list0123).union(set(sel_500_idx[4])))
    sel_merge_idx=np.array(list01234)
    common_index = np.array([i for i in sel_500_idx[0] if i in sel_500_idx[1]
                       if i in sel_500_idx[2] if i in sel_500_idx[3] if i in sel_500_idx[4]])

    return sel_merge_idx,common_index

def merge200(sel_merge_idx,sel_500_idx,n_feature):
    
    all_merge = np.zeros(len(sel_merge_idx))    
    all_merge_count = np.zeros(len(sel_merge_idx))
    for z in range(0,len(sel_merge_idx)):
        for i in range(0,5):
            for j in range(0,200):
                if sel_merge_idx[z] == sel_500_idx[i][j]:
                    all_merge[z] += j
                    all_merge_count[z] += 1   
                else:
                    pass
    all_merge_ave=all_merge/all_merge_count
    sort_merge_index = np.argsort(all_merge_ave)[0:n_feature]
    end_merge_idx=sel_merge_idx[sort_merge_index]
    return end_merge_idx 


def merge100(sel_merge_idx,sel_500_idx,n_feature):
    
    all_merge = np.zeros(len(sel_merge_idx))    
    all_merge_count = np.zeros(len(sel_merge_idx))
    for z in range(0,len(sel_merge_idx)):
        for i in range(0,5):
            for j in range(0,100):
                if sel_merge_idx[z] == sel_500_idx[i][j]:
                    all_merge[z] += j
                    all_merge_count[z] += 1   
                else:
                    pass
    all_merge_ave=all_merge/all_merge_count
    sort_merge_index = np.argsort(all_merge_ave)[0:n_feature]
    end_merge_idx=sel_merge_idx[sort_merge_index]
    return end_merge_idx 


def merge1500(sel_merge_idx,sel_500_idx,n_feature):
    
    all_merge = np.zeros(len(sel_merge_idx))    
    all_merge_count = np.zeros(len(sel_merge_idx))
    for z in range(0,len(sel_merge_idx)):
        for i in range(0,5):
            for j in range(0,1500):
                if sel_merge_idx[z] == sel_500_idx[i][j]:
                    all_merge[z] += j
                    all_merge_count[z] += 1   
                else:
                    pass
    all_merge_ave=all_merge/all_merge_count
    sort_merge_index = np.argsort(all_merge_ave)[0:n_feature]
    end_merge_idx=sel_merge_idx[sort_merge_index]
    return end_merge_idx  

def merge1601(sel_merge_idx,sel_500_idx,n_feature):
    
    all_merge = np.zeros(len(sel_merge_idx))    
    all_merge_count = np.zeros(len(sel_merge_idx))
    for z in range(0,len(sel_merge_idx)):
        for i in range(0,5):
            for j in range(0,1601):
                if sel_merge_idx[z] == sel_500_idx[i][j]:
                    all_merge[z] += j
                    all_merge_count[z] += 1   
                else:
                    pass
    all_merge_ave=all_merge/all_merge_count
    sort_merge_index = np.argsort(all_merge_ave)[0:n_feature]
    end_merge_idx=sel_merge_idx[sort_merge_index]
    return end_merge_idx  

def merge81(sel_merge_idx,sel_500_idx,n_feature):
    
    all_merge = np.zeros(len(sel_merge_idx))    
    all_merge_count = np.zeros(len(sel_merge_idx))
    for z in range(0,len(sel_merge_idx)):
        for i in range(0,5):
            for j in range(0,81):
                if sel_merge_idx[z] == sel_500_idx[i][j]:
                    all_merge[z] += j
                    all_merge_count[z] += 1   
                else:
                    pass
    all_merge_ave=all_merge/all_merge_count
    sort_merge_index = np.argsort(all_merge_ave)[0:n_feature]
    end_merge_idx=sel_merge_idx[sort_merge_index]
    return end_merge_idx 
def merge49(sel_merge_idx,sel_500_idx,n_feature):
    
    all_merge = np.zeros(len(sel_merge_idx))    
    all_merge_count = np.zeros(len(sel_merge_idx))
    for z in range(0,len(sel_merge_idx)):
        for i in range(0,5):
            for j in range(0,49):
                if sel_merge_idx[z] == sel_500_idx[i][j]:
                    all_merge[z] += j
                    all_merge_count[z] += 1   
                else:
                    pass
    all_merge_ave=all_merge/all_merge_count
    sort_merge_index = np.argsort(all_merge_ave)[0:n_feature]
    end_merge_idx=sel_merge_idx[sort_merge_index]
    return end_merge_idx 
def merge25(sel_merge_idx,sel_500_idx,n_feature):
    
    all_merge = np.zeros(len(sel_merge_idx))    
    all_merge_count = np.zeros(len(sel_merge_idx))
    for z in range(0,len(sel_merge_idx)):
        for i in range(0,5):
            for j in range(0,25):
                if sel_merge_idx[z] == sel_500_idx[i][j]:
                    all_merge[z] += j
                    all_merge_count[z] += 1   
                else:
                    pass
    all_merge_ave=all_merge/all_merge_count
    sort_merge_index = np.argsort(all_merge_ave)[0:n_feature]
    end_merge_idx=sel_merge_idx[sort_merge_index]
    return end_merge_idx 

def merge9(sel_merge_idx,sel_500_idx,n_feature):
    
    all_merge = np.zeros(len(sel_merge_idx))    
    all_merge_count = np.zeros(len(sel_merge_idx))
    for z in range(0,len(sel_merge_idx)):
        for i in range(0,5):
            for j in range(0,9):
                if sel_merge_idx[z] == sel_500_idx[i][j]:
                    all_merge[z] += j
                    all_merge_count[z] += 1   
                else:
                    pass
    all_merge_ave=all_merge/all_merge_count
    sort_merge_index = np.argsort(all_merge_ave)[0:n_feature]
    end_merge_idx=sel_merge_idx[sort_merge_index]
    return end_merge_idx     
    
    
    
    

def data(path):
    df=pd.read_csv(path)
    drop_elements=['sample','_PATIENT','age_at_initial_pathologic_diagnosis','gender','race','ajcc_pathologic_tumor_stage','clinical_stage',
                   'histological_type','histological_grade','initial_pathologic_dx_year','menopause_status','birth_days_to',
                   'vital_status','tumor_status','last_contact_days_to','death_days_to','cause_of_death','new_tumor_event_type',
                   'new_tumor_event_site','new_tumor_event_site_other','new_tumor_event_dx_days_to','treatment_outcome_first_course',
                   'margin_status','residual_tumor','DSS','DSS.time','DFI','DFI.time','PFI','PFI.time','Redaction',
                   'cancer type abbreviation'
                  ]
    x = df.drop(drop_elements, axis=1)
    ZSJ=x.dropna(axis=0,subset = ["OS", "OS.time"])
    E=np.array(ZSJ["OS"])
    Y=np.array(ZSJ["OS.time"])
    X=np.array(ZSJ)
    X=X.astype('float64')
    X=X[:,:-2]
    scaler=StandardScaler().fit(X)
    X=scaler.transform(X)
    elements=["OS", "OS.time"]
    T=ZSJ.drop(elements, axis=1)
    index=T.columns
    return X,E,Y,index
