import tensorflow as tf
import numpy as np
import pandas as pd

#loss
def nll(E, NUM_E):
    def loss(y_true, y_pred):
        hazard_ratio = tf.squeeze(tf.exp(y_pred))
        log_risk = tf.math.log(tf.math.cumsum(hazard_ratio))
        uncensored_likelihood = tf.subtract(tf.squeeze(y_pred),log_risk)
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood = -tf.reduce_sum(censored_likelihood) / NUM_E
        return neg_likelihood

    return loss



#cancer_types=['BLCA','HNSC','KIRC','LGG','LUAD','LUSC','SKCM']
#numbers=[422,543,596,514,551,539,454]
def avgcindex(Cindex,cancer_types,numbers):
    cisum=[]
    for i in range(7):
        cancer_name = cancer_types[i]
        number = numbers[i]
        print(cancer_name,np.mean(Cindex[i]))
        cisum.append(np.mean(Cindex[i])*number)
        
    avgci= np.array(cisum).sum()/np.array(numbers).sum()
    return avgci