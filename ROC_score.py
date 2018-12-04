#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:50:16 2018

@author: semvijverberg
"""
import numpy
import random

def ROC_score(predictions, observed, thr_event, lag, n_boot, thr_pred='default'):
    
#    predictions = crosscorr_mcK
#    observed = mcKts
#    thr_event = hotdaythreshold
    
   # calculate ROC scores
    observed = numpy.copy(observed)
    # Standardize predictor time series
#    predictions = predictions - numpy.mean(predictions)
    # P_index = numpy.copy(AIR_rain_index)	
    # Test ROC-score			
    
    TP_rate = numpy.ones((11))
    FP_rate =  numpy.ones((11))
    TP_rate[10] = 0
    FP_rate[10] = 0
    AUC_new = numpy.zeros((n_boot))
    
    #print(fixed_event_threshold) 
    events = numpy.where(observed > thr_event)[0][:]  
    not_events = numpy.where(observed <= thr_event)[0][:]     
    for p in numpy.linspace(1, 9, 9, dtype=int):	
        if str(thr_pred) == 'default':
            p_pred = numpy.percentile(predictions, p*10)
        else:
            p_pred = thr_pred.sel(percentile=p).values[0]
        positives_pred = numpy.where(predictions > p_pred)[0][:]
        negatives_pred = numpy.where(predictions <= p_pred)[0][:]

						
        True_pos = [a for a in positives_pred if a in events]
        False_neg = [a for a in negatives_pred if a in events]
        
        False_pos = [a for a in positives_pred if a  in not_events]
        True_neg = [a for a in negatives_pred if a  in not_events]
        
        True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
        False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
        
        FP_rate[p] = False_pos_rate
        TP_rate[p] = True_pos_rate
        
     
    ROC_score = numpy.abs(numpy.trapz(TP_rate, x=FP_rate ))
    # shuffled ROc
    
    ROC_bootstrap = 0
    for j in range(n_boot):
        
        # shuffle observations / events
        old_index = range(0,len(observed),1)
                
        sample_index = random.sample(old_index, len(old_index))
        #print(sample_index)
        new_observed = observed[sample_index]    
        # _____________________________________________________________________________
        # calculate new AUC score and store it
        # _____________________________________________________________________________
        #
    
        new_observed = numpy.copy(new_observed)
        # P_index = numpy.copy(MT_rain_index)	
        # Test AUC-score			
        TP_rate = numpy.ones((11))
        FP_rate =  numpy.ones((11))
        TP_rate[10] = 0
        FP_rate[10] = 0

        events = numpy.where(new_observed > thr_event)[0][:]  
        not_events = numpy.where(new_observed <= thr_event)[0][:]     
        for p in numpy.linspace(1, 9, 9, dtype=int):	
            if str(thr_pred) == 'default':
                p_pred = numpy.percentile(predictions, p*10)
            else:
                p_pred = thr_pred.sel(percentile=p).values[0]
            
            p_pred = numpy.percentile(predictions, p*10)
            positives_pred = numpy.where(predictions > p_pred)[0][:]
            negatives_pred = numpy.where(predictions <= p_pred)[0][:]
    
    						
            True_pos = [a for a in positives_pred if a in events]
            False_neg = [a for a in negatives_pred if a in events]
            
            False_pos = [a for a in positives_pred if a  in not_events]
            True_neg = [a for a in negatives_pred if a  in not_events]
            
            True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
            False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
            
            FP_rate[p] = False_pos_rate
            TP_rate[p] = True_pos_rate
            
            #check
            if len(True_pos+False_neg) != len(events) :
                print("check 136")
            elif len(True_neg+False_pos) != len(not_events) :
                print("check 138")
           
            True_pos_rate = len(True_pos)/(float(len(True_pos)) + float(len(False_neg)))
            False_pos_rate = len(False_pos)/(float(len(False_pos)) + float(len(True_neg)))
            
            FP_rate[p] = False_pos_rate
            TP_rate[p] = True_pos_rate
        
        AUC_score  = numpy.abs(numpy.trapz(TP_rate, FP_rate))
        AUC_new[j] = AUC_score
        AUC_new    = numpy.sort(AUC_new[:])[::-1]
        pval       = (numpy.asarray(numpy.where(AUC_new > ROC_score)).size)/ n_boot
        ROC_bootstrap = AUC_new 
  
    return ROC_score, ROC_bootstrap