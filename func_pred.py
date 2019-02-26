#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:13:39 2019

@author: semvijverberg
"""
import os
import xarray as xr
import numpy as np
import pandas as pd
import func_CPPA





def spatial_cov(RV_ts, ex):
    
    ex['test_ts_mcK'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
    ex['test_yrs'] = np.zeros( len(ex['lags']) , dtype=list)
    
    if ex['use_ts_logit'] == False:
        ex['test_ts_Sem'] = np.zeros( len(ex['lags']) , dtype=list)

    
#    # Event time series 
#    events = func_CPPA.Ev_timeseries(RV_ts, ex['hotdaythres'], ex).time
#    dates = pd.to_datetime(RV_ts.time.values)
#    event_idx = [list(dates.values).index(E) for E in events.values]
#    binary_events = np.zeros(dates.size)   
#    binary_events[event_idx] = 1
#    mask_events = np.array(binary_events, dtype=bool)
    
    
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        
        test =ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        print('test year(s) {}, with {} events.'.format(ex['test_year'],
                                 test['events'].size))
        
        
        
        for lag_idx, lag in enumerate(ex['lags']):
            # load in timeseries
            csv_train_test_data = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
            path = os.path.join(ex['output_ts_folder'], csv_train_test_data)
            data = pd.read_csv(path)
            
            # only training dataset
            all_yrs = list(pd.to_datetime(data.date.values))
            test_yrs = [all_yrs.index(d) for d in all_yrs if d.year in ex['test_year']]
            
            idx = lag_idx
            if ex['use_ts_logit'] == False:
                # spatial covariance CPPA
                spat_cov_lag_i = data['spatcov_CPPA'][test_yrs]
                
            
                if ex['n'] == 0:
                    ex['test_ts_Sem'][idx] = spat_cov_lag_i.values
        #                ex['test_RV_Sem'][idx]  = test['RV'].values
                else:
                    ex['test_ts_Sem'][idx] = np.concatenate( [ex['test_ts_Sem'][idx], spat_cov_lag_i.values] ) 
            
            # spatial covariance PEP
            spat_cov_lag_i = data['spatcov_PEP'][test_yrs]

            if ex['n'] == 0:
                ex['test_ts_mcK'][idx] = spat_cov_lag_i.values
                ex['test_RV'][idx]     = test['RV'].values
                ex['test_yrs'][idx]    = test['RV'].time
    #                ex['test_RV_Sem'][idx]  = test['RV'].values
            else:
                ex['test_ts_mcK'][idx] = np.concatenate( [ex['test_ts_mcK'][idx], spat_cov_lag_i.values] )
                ex['test_RV'][idx]     = np.concatenate( [ex['test_RV'][idx], test['RV'].values] )  
                ex['test_yrs'][idx]    = np.concatenate( [ex['test_yrs'][idx], test['RV'].time] )  

    return ex


def logit_fit_new(l_ds_CPPA, RV_ts, ex):
    #%%
    if (
        ex['leave_n_out'] == True and ex['method'] == 'iter'
        or ex['ROC_leave_n_out'] or ex['method'][:6] == 'random'
        ):
        ex['test_ts_mcK'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_ts_Sem'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_RV'] = np.zeros( len(ex['lags']) , dtype=list)
        ex['test_yrs'] = np.zeros( len(ex['lags']) , dtype=list)
    
    # Event time series 
    events = func_CPPA.Ev_timeseries(RV_ts, ex['hotdaythres'], ex).time
    dates = pd.to_datetime(RV_ts.time.values)
    event_idx = [list(dates.values).index(E) for E in events.values]
    binary_events = np.zeros(dates.size)   
    binary_events[event_idx] = 1
    mask_events = np.array(binary_events, dtype=bool)
    
    
    
    for n in range(len(ex['train_test_list'])):
        ex['n'] = n
        
        test =ex['train_test_list'][n][1]
        ex['test_year'] = list(set(test['RV'].time.dt.year.values))
        print('test year(s) {}, with {} events.'.format(ex['test_year'],
                                 test['events'].size))
        
        ds_Sem = l_ds_CPPA[n]
        Composite = ds_Sem['pattern_CPPA']
        lats = Composite.latitude.values
        lons = Composite.longitude.values
        # copy pattern info to update
        ds_Sem['pat_num_logit'] = ds_Sem['pat_num_CPPA'].copy()
        ds_Sem['pattern_logit'] = ds_Sem['pattern_CPPA'].copy()
        
        logit_model = []
        ex['ts_train_std'] = [] #np.zeros( (len(ex['lags'])), dtype=list )
        
        
        for lag_idx, lag in enumerate(ex['lags']):
            # load in timeseries
            csv_train_test_data = 'testyr{}_{}.csv'.format(ex['test_year'], lag)
            path = os.path.join(ex['output_ts_folder'], csv_train_test_data)
            data = pd.read_csv(path)
            regions_for_ts = [int(r[:-2]) for r in data.columns[3:].values]
            ts_regions_lag_i = data.iloc[:,3:].values
            
            # only training dataset
            all_yrs = list(pd.to_datetime(data.date.values))
            train_yrs = [all_yrs.index(d) for d in all_yrs if d.year not in ex['test_year']]
            ts_regions_train = ts_regions_lag_i[train_yrs,:]
            mask_events_train = mask_events[train_yrs]
            sign_ts_regions = np.sign(np.mean(ts_regions_train[mask_events_train],axis=0))
            binary_train = binary_events[train_yrs]
            
            # Perform training
            odds, regions_kept, combs_kept, logitmodel = train_weights_LogReg(
                    ts_regions_train, sign_ts_regions, regions_for_ts, binary_train, ex)
            
            
            # update regions that were kicked out
            Regions_lag_i = ds_Sem['pat_num_CPPA'][lag_idx].squeeze().values
            Composite_lag = Composite[lag_idx]
            
            upd_regions = np.zeros(Regions_lag_i.shape)
            for reg in regions_kept:
                upd_regions[Regions_lag_i == reg] =  reg
        
            # create map of precursor regions
            npmap = np.ma.reshape(upd_regions, (len(lats), len(lons)))
            mask_strongest = (npmap!=0) 
            npmap[mask_strongest==False] = 0
            xrnpmap = Composite_lag.copy()
            xrnpmap.values = npmap
            
            # update the mask for the composite mean
            mask = (('latitude', 'longitude'), mask_strongest)
            Composite_lag.coords['mask'] = mask
            xrnpmap.coords['mask'] = mask
            xrnpmap = xrnpmap.where(xrnpmap.mask==True)
            Composite_lag = Composite_lag.where(xrnpmap.mask==True)
            
        #    plt.figure()
        #    xrnpmap.plot.contourf(cmap=plt.cm.tab10)
                
            # calculating test year
            test_yrs = [all_yrs.index(d) for d in all_yrs if d.year in ex['test_year']]
            ts_regions_test = ts_regions_lag_i[test_yrs,:]

            ts_regions_lag_i = ts_regions_test[:,:] * sign_ts_regions[None,:]
            # use only regions which were not kicked out by logit valid
            idx_regions_kept = [regions_for_ts.index(r) for r in regions_kept]
            ts_regions_lag_i = ts_regions_lag_i[:,idx_regions_kept]
            # normalize time series (as done in the training)        
            X_n = ts_regions_lag_i / ex['ts_train_std'][lag_idx]
            
            ts_pred = logitmodel.predict(X_n)
            
            idx = lag_idx
            if (
                ex['leave_n_out'] == True and ex['method'] == 'iter'
                or ex['ROC_leave_n_out'] or ex['method'][:6] == 'random'
                ):
                if ex['n'] == 0:
#                    ex['test_ts_mcK'][idx] = crosscorr_mcK.values 
                    ex['test_ts_Sem'][idx] = ts_pred
                    ex['test_RV'][idx]     = test['RV'].values
                    ex['test_yrs'][idx]    = test['RV'].time
        #                ex['test_RV_Sem'][idx]  = test['RV'].values
                else:
        #                update_ROCS = ex['test_ts_mcK'][idx].append(list(crosscorr_mcK.values))
    #                ex['test_ts_mcK'][idx] = np.concatenate( [ex['test_ts_mcK'][idx], crosscorr_mcK.values] )
                    ex['test_ts_Sem'][idx] = np.concatenate( [ex['test_ts_Sem'][idx], ts_pred] )
                    ex['test_RV'][idx]     = np.concatenate( [ex['test_RV'][idx], test['RV'].values] )  
                    ex['test_yrs'][idx]    = np.concatenate( [ex['test_yrs'][idx], test['RV'].time] )  
            
#            pattern_p2[idx] = norm_mean * ds_Sem['weights'].sel(lag=lag) 
#            pattern_num_p2[lag_idx] = xrnpmap
            logit_model.append(logitmodel)
            
            ds_Sem['pattern_logit'][lag_idx] = Composite_lag.values
            ds_Sem['pat_num_logit'][lag_idx] = xrnpmap.values
            
    # append logit models    
    ds_Sem.attrs['logitmodel'] = logit_model
    # info of logit fit is now appended to ds_Sem
    l_ds_CPPA[n] = ds_Sem

    #%%
    return l_ds_CPPA, ex




def train_weights_LogReg(ts_regions_train, sign_ts_regions, regions_for_ts, binary_train, ex):
    #%%
#    from sklearn.linear_model import LogisticRegression
#    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import itertools
    # I want to the importance of each regions as is shown in the composite plot
    # A region will increase the probability of the events when it the coef_ is 
    # higher then 1. This means that if the ts shows high values, we have a higher 
    # probability of the events. To ease interpretability I will multiple each ts
    # with the sign of the region, to that each params should be above one (higher
    # coef_, higher probability of events)      
    
    
    ex['std_of_initial_ts'] = np.std(ts_regions_train, axis=0) #!!
    X_init = ts_regions_train / np.std(ts_regions_train, axis=0) #!!
    y = binary_train
    

#    # Balance Dataset
#    pos = np.where(binary_train == 1)[0]
#    neg = np.where(binary_train == 0)[0]
#   
##    # Create 75/25 NEG/POS Selection, Balancing Data Imbalance and Data Amount
#    idx = np.random.permutation(np.concatenate([pos, np.random.choice(neg, 3*len(pos), False)]))
##    
#    y = y[idx]
#    X_init = X_init[idx,:]
    
    n_feat = len(np.unique(regions_for_ts))
    n_samples = len(X_init[:,0])
    
    regions          = regions_for_ts
    track_super_sign = regions_for_ts
    track_r_kept     = regions_for_ts
    signs = sign_ts_regions

    init_vs_final_bic = []
    if ex['logit_valid'] == True:
        # =============================================================================
        # Step 1: Preselection 
        # =============================================================================
        # first kick out regions which do not show any relation to the event 
        # i.e. low p-value
        X = X_init
        all_regions_significant = True
        i = 0
         #np.zeros( (len(ex['lags'])) , dtype=list)
        while all_regions_significant:
            X_train = X * signs[None,:]
            y_train = y
    
            logit_model=sm.Logit(y_train,X_train)
            result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
            if i == 0:
    #            print('initial bic value {}'.format(result.bic))
                init_vs_final_bic.append(result.bic)
            if result.mle_retvals['converged'] == False:
                print('logistic regression did not converge, taking odds of prev'
                      'iteration or - if not present -, all odds (wghts) are equal')
                try:
                    # capture odds array of previous iteration
                    odds = odds
                except NameError as error:
                    # if odds in not defined, then it has never converged and there
                    # are no odds available, set all odds (wghts) to 1
                    odds = np.ones(len(n_feat))
                    
            elif result.mle_retvals['converged'] == True:
                odds = np.exp(result.params)
        #        print('statsmodel logit Odds of event happening conditioned on X (p / (1+p) = exp(params) \n{}\n'.format(odds))
    #            print(result.summary2())
                p_vals = result.pvalues
                mask_really_insig = (p_vals >= ex['pval_logit_first'])
                regions_not_sign = np.where(mask_really_insig)[0]
                        
                # update regions accounted for in fit
                X = np.delete(X, regions_not_sign, axis=1)
                track_r_kept = np.delete(track_r_kept, regions_not_sign)
                
                signs  = np.delete(signs, regions_not_sign, axis=0)
            
                                            
                if len(regions_not_sign) == 0:
                    all_regions_significant = False
            i += 1
    #    print(result.summary2())
    
        # regions to be futher investigated before throwing out
        mask_not_super_sig = (p_vals >= ex['pval_logit_final'])
        idx_not_super_sig = (np.where(mask_not_super_sig)[0])
        track_super_sign = np.delete(track_r_kept, idx_not_super_sig)
        regs_for_interac = [i for i in track_r_kept if i not in track_super_sign]
 
        # =============================================================================
        # Step 2: Check 'intermediate regions' should be kicked out
        # =============================================================================
        # Perform logistic regression on combination of two with the regions that were
        # 'not super sign' (pval < ex['pval_logit_final'])
        
        if len(regs_for_interac) != 0:
            # check if regions deleted do not have an interacting component
            combi = list(itertools.product([0, 1], repeat=n_feat))[1:]
            comb_int = [c for c in combi if np.sum(c)==2]
            weak_p = []
            for c in comb_int:
                idx_f = np.where(np.array(c) == 1)[0]
                # get number of region
                regs = [regions[i] for i in idx_f]
                for r in regs_for_interac:
                    if r in regs:   
        #                print(True)
                        weak_p.append(c) 
            
            # remove duplicates in weak_p
            def remove_duplicates(seq):
                seen = set()
                seen_add = seen.add
                return [x for x in seq if not (x in seen or seen_add(x))]
            
            weak_p = remove_duplicates(weak_p)
            
            X_train_cl = X_init
            X_train_cl = X_train_cl * sign_ts_regions[None,:]
            # add interaction time series
            combregions = []
            ts_regions_interaction = np.zeros( (n_samples, len(weak_p)) )
            for comb in weak_p:
                i = weak_p.index(comb)
                idx_f = np.where(np.array(comb) == 1)[0]
                two_ts = X_train_cl[:,idx_f]
                ts_regions_interaction[:,i] = two_ts[:,0] * two_ts[:,1]
                corresponding_regions = [regions[i] for i in idx_f]
                combregions.append(corresponding_regions)
            
            X_inter = ts_regions_interaction / np.std(ts_regions_interaction, axis=0)
            y_train = y
        
            logit_model=sm.Logit(y_train,X_inter)
            result_inter = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
            p_vals = result_inter.pvalues 
            odds_interac   = np.exp(result_inter.params)
            mask_cregions_keeping = (p_vals <= ex['pval_logit_final']) #* (odds > 1.)
            cregions_sign_idx = list(np.where(mask_cregions_keeping)[0])
            cregions_sign = np.array(combregions)[cregions_sign_idx]
            result_inter.summary2()
            if len(cregions_sign_idx)==0.:
                # delete regions that are not kept after testing if they got good enough
                # p_vals
                idx_not_keeping = [list(track_r_kept).index(i) for i in regs_for_interac]
                track_r_kept    = np.delete(track_r_kept, idx_not_keeping)
            else:
                # =============================================================================
                # test if combination of regions is truly giving a better p_val then when it 
                # is taken as a single region
                # =============================================================================
                
                keep_duetointer = []
                comb_sign_r = np.unique(cregions_sign)
                comb_sign_ind = [i for i in comb_sign_r if i not in regs_for_interac]
                comb_sign_ind_idx = [comb_sign_ind.index(r) for r in comb_sign_ind if r in regions]
                X_invol_inter = X_init
                X_invol_inter = X_invol_inter * sign_ts_regions[None,:]
                X_invol_inter = X_invol_inter[:,comb_sign_ind_idx]
                
                ts_r_int_sign = ts_regions_interaction[:,cregions_sign_idx]
                
                X_sing_inter = np.concatenate((X_invol_inter, ts_r_int_sign), axis=1)
                
                logit_model=sm.Logit(y_train,X_sing_inter)
                result_int_test = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
                odds_sin_inter = np.exp(result_int_test.params)
                p_vals = result_int_test.pvalues 
                n_sign_single = len(comb_sign_ind_idx)
                p_vals_single = p_vals[:n_sign_single]
                odds_single   = odds_sin_inter[:n_sign_single]
                odds_inter    = odds_sin_inter[n_sign_single:]
                p_vals_inter  = p_vals[n_sign_single:]
                for i in range(len(p_vals_inter)):
                    p_val_inter = p_vals_inter[i]
                    # interacting with
                    comb_r = [t for t in cregions_sign[i]]
                    # check p_value of perhaps two regions which were tested for interaction
                    if all(elem in regs_for_interac  for elem in comb_r): 
                        p_val = p_vals_inter[i]
                        if p_val < ex['pval_logit_final'] and odds_inter[i] > 1.0:
    #                        print('p_val inter two weak regions : {}'.format(p_val))
                            keep_duetointer.append(comb_r)
                            # the regions are kept into track_r_kept
                            
                        
                    else:
                        # a 'single' region investigated for interaction with region
                        # which showed not a very significant p_val
                        reg_investigated = [i for i in comb_r if i in regs_for_interac]
                        reg_individual = [i for i in comb_r if i not in regs_for_interac]
                        idx_sign_single = comb_sign_ind.index(reg_individual)
                        p_val_inter  = p_vals_inter[i]
                        p_val_single = p_vals_single[idx_sign_single]
                        odds_int     = odds_inter[i]
                        odds_sin     = odds_single[idx_sign_single]
                        if p_val_inter < p_val_single and odds_int > odds_sin:
    #                        print('p_val_inter : {}\np_val_single {}'.format(
    #                                p_val_inter, p_val_single))
                            if p_val_inter < ex['pval_logit_final']:
                                keep_duetointer.append(reg_investigated)
                            
                keep_duetointer = [i for i in keep_duetointer]
                
                # delete regions that are not kept after testing if they got good enough
                # p_vals
                flat_keeping = list(set([item for sublist in keep_duetointer for item in sublist]))
                idx_not_keeping = [list(track_r_kept).index(r) for r in reg_investigated if r not in flat_keeping]
                track_r_kept    = np.delete(track_r_kept, idx_not_keeping)
            
#                # if it adds likelihood when reg_investiged interacts with reg_single, 
#                # then I keep both reg_investiged and reg_single in as single predictors
#                # if two reg_investigated became significant together, then I will keep 
#                # them in as an 'interacting time series' reg1 * reg2 = predictor.
#                if (regs_for_interac in keep_duetointer) and len(regs_for_interac)==2:
#                    add_pred_inter = [i for i in keep_duetointer if all(np.equal(i, regs_for_interac))][0]
#                    idx_pred_inter = [i for i in range(n_feat) if regions[i] in add_pred_inter]
#                    ts_inter_kept   = ts_regions_train[:,idx_pred_inter[0]] * ts_regions_train[:,idx_pred_inter[1]]
#                    X_inter_kept = ts_inter_kept[:,None] / np.std(ts_inter_kept[:,None], axis = 0)
#    #            
#    #                X_final = np.concatenate( (X_train, X_inter_kept), axis=1)
#    #            else:
#    #                X_final = X_train
    # =============================================================================
    # Step 3 Perform log regression on (individual) ex['ts_train_std']that past step 1 & 2.
    # =============================================================================
    # all regions
    X = X_init
    # select only region in track_regions_kept:
    r_kept_idx = [regions.index(r) for r in regions if r in track_r_kept]
    sign_r_kept = sign_ts_regions[r_kept_idx]

    ex['ts_train_std'].append(np.std(ts_regions_train[:,r_kept_idx], axis=0))
    X_final = X[:,r_kept_idx] * sign_r_kept[None,:]
    logit_model=sm.Logit(y,X_final)
    result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
    odds   = np.exp(result.params)
    p_vals = result.pvalues 
    #        print(result.summary2())
            
#            odds_new = odds_new[:len(track_r_kept)]
    logitmodel = result
#    print('final bic value {}'.format(logitmodel.bic))
    init_vs_final_bic.append(logitmodel.bic)
        
#    results = pd.DataFrame(columns=['aic', 'aicc', 'bic', 'p_vals'])
#    for comb in combinations:
##        idx = combinations.index(comb)
#        idx_f = np.where(np.array(comb) == 1)[0]
#        X_train = X[:,idx_f] * signs[None,idx_f]
#        y_train = y
#        
#        logit_model=sm.Logit(y_train,X_train)
#        result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
#        aicc = result.aic + (2*n_feat**2 + 2*n_feat) / (n_samples - n_feat -1)
#        pd_out = pd.Series( [ result.aic, aicc, result.bic, result.pvalues ], 
#                           index=results.columns )
#        results = results.append(pd_out, ignore_index=True)
#    # select model with lowest AIC
#    min_aic = results[results.aic == results.aic.min()].aic.values
#    for idx in range(len(results)):
#        rel_likely = np.exp(min_aic-)
    
    #%%        
    return odds, track_r_kept, sign_r_kept, logitmodel



