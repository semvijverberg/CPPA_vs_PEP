#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:50:16 2018

@author: semvijverberg
"""
import numpy
import random
import os
import numpy as np
import pandas as pd
import func_mcK
import xarray as xr

def ROC_score_wrapper(test, trian, ds_mcK, ds_Sem, ex):
        
    # =============================================================================
    # calc ROC scores
    # =============================================================================
    ROC_Sem  = np.zeros(len(ex['lags']))
    ROC_mcK  = np.zeros(len(ex['lags']))
    ROC_boot = np.zeros(len(ex['lags']))
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        dates_test = func_mcK.to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
                                           test['Prec'].time[0].dt.hour)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
        var_test_mcK = func_mcK.find_region(test['Prec'], region='PEPrectangle')[0]
    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
    
        var_test_mcK = var_test_mcK.sel(time=dates_min_lag)
        var_test_reg = test['Prec'].sel(time=dates_min_lag)        
    
        crosscorr_mcK = func_mcK.cross_correlation_patterns(var_test_mcK, 
                                                            ds_mcK['pattern'].sel(lag=lag))
        crosscorr_Sem = func_mcK.cross_correlation_patterns(var_test_reg, 
                                                            ds_Sem['pattern'].sel(lag=lag))
        
        
        if ex['method'] == 'iter':
            if ex['n'] == 0:
                ex['test_ts_mcK'][idx] = crosscorr_mcK.values 
                ex['test_ts_Sem'][idx] = crosscorr_Sem.values
                ex['test_RV'][idx]  = test['RV'].values
    #                ex['test_RV_Sem'][idx]  = test['RV'].values
            else:
    #                update_ROCS = ex['test_ts_mcK'][idx].append(list(crosscorr_mcK.values))
                ex['test_ts_mcK'][idx] = np.concatenate( [ex['test_ts_mcK'][idx], crosscorr_mcK.values] )
                ex['test_ts_Sem'][idx] = np.concatenate( [ex['test_ts_Sem'][idx], crosscorr_Sem.values] )
                ex['test_RV'][idx] = np.concatenate( [ex['test_RV'][idx], test['RV'].values] )
    #                ex['test_RV_Sem'][idx] = np.concatenate( [ex['test_RV_Sem'][idx], var_test_reg.values] )
    #            print(len(ex['test_ts_Sem']))
        
            if  ex['n'] == ex['n_conv']-1:
                n_boot = 5
                ROC_mcK[idx], ROC_boot = ROC_score(ex['test_ts_mcK'][idx], ex['test_RV'][idx],
                                      ex['hotdaythres'], lag, n_boot, 'default')
                ROC_Sem[idx] = ROC_score(ex['test_ts_Sem'][idx], ex['test_RV'][idx],
                                      ex['hotdaythres'], lag, 0, 'default')[0]
                print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
                '\t ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
                  lag, ROC_mcK[idx], ROC_Sem[idx], 2*np.std(ROC_boot)))
            
                
        elif ex['method'] == 'random':        
                               
            # check detection of precursor:
            Prec_threshold_mcK = ds_mcK['perc'].sel(percentile=60 /10).values[0]
            Prec_threshold_Sem = ds_Sem['perc'].sel(percentile=60 /10).values[0]
            
            # =============================================================================
            # Determine events in time series
            # =============================================================================
            # check if there are any detections
            Prec_det_mcK = (func_mcK.Ev_timeseries(crosscorr_mcK, 
                                           Prec_threshold_mcK).size > ex['min_detection'])
            Prec_det_Sem = (func_mcK.Ev_timeseries(crosscorr_Sem, 
                                           Prec_threshold_Sem).size > ex['min_detection'])
            
    #        # plot the detections
    #        func_mcK.plot_events_validation(crosscorr_Sem, crosscorr_mcK, RV_ts_test, Prec_threshold_Sem, 
    #                                        Prec_threshold_mcK, ex['hotdaythres'], test_years[0])
    
    
            if Prec_det_mcK == True:
                n_boot = 1
                ROC_mcK[idx], ROC_boot = ROC_score(crosscorr_mcK, test['RV'],
                                      ex['hotdaythres'], lag, n_boot, ds_mcK['perc'])
    
    
            else:
                print('Not enough predictions detected, neglecting this predictions')
                ROC_mcK[idx] = ROC_boot = 0.5
    
    
            
            if Prec_det_Sem == True:
                n_boot = 0
                ROC_Sem[idx] = ROC_score(crosscorr_Sem, test['RV'],
                                      ex['hotdaythres'], lag, n_boot, ds_Sem['perc'])[0]
    #            ROC_std = 2 * np.std([ROC_boot_Sem, ROC_boot_mcK])
                
    #                Sem_ROCS.append(commun_comp.sel(lag=lag))
            else:
                print('Not enough predictions detected, neglecting this predictions')
                ROC_Sem = ROC_boot = 0.5
                                  
            
            print('\n*** ROC score for {} lag {} ***\n\nMck {:.2f} \t Sem {:.2f} '
                '\t ±{:.2f} 2*std random events\n\n'.format(ex['region'], 
                  lag, ROC_mcK[idx], ROC_Sem[idx], 2*np.std(ROC_boot)))
        
        # store output:
        ds_mcK['score'] = xr.DataArray(data=ROC_mcK, coords=[ex['lags']], 
                          dims=['lag'], name='score_diff_lags',
                          attrs={'units':'-'})
        ds_Sem['score'] = xr.DataArray(data=ROC_Sem, coords=[ex['lags']], 
                          dims=['lag'], name='score_diff_lags',
                          attrs={'units':'-'})
    
    ex['score_per_run'].append([ex['test_years'], len(test['events']), ds_mcK, ds_Sem, ROC_boot])
    return ex['score_per_run']

def ROC_score(predictions, observed, thr_event, lag, n_boot, thr_pred='default'):
    
#    predictions = ex['test_ts_mcK'][idx]
#    observed = RV_ts
#    thr_event = ex['hotdaythres']
    
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