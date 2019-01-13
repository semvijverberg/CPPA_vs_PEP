#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:29:47 2018

@author: semvijverberg
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
from netCDF4 import num2date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
from shapely.geometry.polygon import LinearRing
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime
import scipy 



# =============================================================================
# =============================================================================
# Wrapper functions
# =============================================================================
# =============================================================================

def mcKmean(train, ex):

    Prec_train_mcK = find_region(train['Prec'], region=ex['regionmcK'])[0]
    dates_train = to_datesmcK(train['RV'].time, train['RV'].time.dt.hour[0], 
                                           train['Prec'].time[0].dt.hour)
#        time = Prec_train_mcK.time
    lats = Prec_train_mcK.latitude
    lons = Prec_train_mcK.longitude
    pthresholds = np.linspace(1, 9, 9, dtype=int)
    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='McK_Comp_diff_lags',
                          attrs={'units':'Kelvin'})
    array = np.zeros( (len(ex['lags']), len(dates_train)) )
    pattern_ts = xr.DataArray(data=array, coords=[ex['lags'], dates_train], 
                          dims=['lag','time'], name='McK_mean_ts_diff_lags',
                          attrs={'units':'Kelvin'})
    
    array = np.zeros( (len(ex['lags']), len(pthresholds)) )
    pattern_p = xr.DataArray(data=array, coords=[ex['lags'], pthresholds], 
                          dims=['lag','percentile'], name='McK_mean_p_diff_lags')
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        event_train = Ev_timeseries(train['RV'], ex['hotdaythres']).time
        event_train = to_datesmcK(event_train, event_train.dt.hour[0], Prec_train_mcK.time[0].dt.hour)
        events_min_lag = event_train - pd.Timedelta(int(lag), unit='d')
        dates_train_min_lag = dates_train - pd.Timedelta(int(lag), unit='d')
        
        ### exlude leap days ###
        # no leap days in dates_train_min_lag
        noleapdays = ((dates_train_min_lag.dt.month == 2) & (dates_train_min_lag.dt.day == 29))==False
        # only keep noleapdays
        dates_train_min_lag = dates_train_min_lag[noleapdays].dropna(dim='time', how='all')
        # also kick out the corresponding events
        dates_train = dates_train[noleapdays].dropna(dim='time', how='all')
        
       # no leap days in events_min_lag
        noleapdays = ((events_min_lag.dt.month == 2) & (events_min_lag.dt.day == 29))==False
        # only keep noleapdays
        events_min_lag = events_min_lag[noleapdays].dropna(dim='time', how='all') 
        # also kick out the corresponding events
        event_train = event_train[noleapdays].dropna(dim='time', how='all')   


        pattern_atlag = Prec_train_mcK.sel(time=events_min_lag).mean(dim='time')
        pattern[idx] = pattern_atlag 
        ts_3d = Prec_train_mcK.sel(time=dates_train_min_lag)
        
        
        crosscorr = cross_correlation_patterns(ts_3d, pattern_atlag)
        crosscorr['time'] = pattern_ts.time
        pattern_ts[idx] = crosscorr
        # Percentile values based on training dataset
        p_pred = []
        for p in pthresholds:	
            p_pred.append(np.percentile(crosscorr.values, p*10))
        pattern_p[idx] = p_pred
    ds_mcK = xr.Dataset( {'pattern' : pattern, 'ts' : pattern_ts, 'perc' : pattern_p} )
    return ds_mcK


def train_test_wrapper(RV_ts, Prec_reg, ex):
    #%%
    if ex['leave_n_out'] == True and ex['method'] == 'random':
        train, test, ex['test_years'] = rand_traintest(RV_ts, Prec_reg, 
                                          ex)
        
        foldername = 'leave_{}_out_{}_{}_tf{}_{}'.format(ex['leave_n_years_out'],
                            ex['startyear'], ex['endyear'], ex['tfreq'],
                            ex['lags'])
        ex['exp_folder'] = os.path.join(ex['figpathbase'],foldername)
    if ex['leave_n_out']==True and ex['method']=='iter' and ex['ROC_leave_n_out']==False:
        train, test, ex['test_years'] = rand_traintest(RV_ts, Prec_reg, 
                                          ex)
        now = datetime.datetime.now()
        ex['exp_folder'] = '{}_{}_{}_tf{}_lags{}_{}_{}deg_{}'.format(ex['method'],
                          ex['startyear'], ex['endyear'],
                          ex['tfreq'], ex['lags'], ex['mcKthres'], ex['grid_res'],
                          now.strftime("%Y-%m-%d"))
    elif ex['leave_n_out'] == False:
        train = dict( { 'Prec'  : Prec_reg,
                        'RV'    : RV_ts,
                        'events': Ev_timeseries(RV_ts, ex['hotdaythres'])})
        test = train.copy()

        foldername = 'hindcast_{}_{}_tf{}_{}'.format(ex['startyear'],
                             ex['endyear'], ex['tfreq'], ex['lags'])
        ex['exp_folder'] = os.path.join(ex['figpathbase'],foldername)
        ex['test_years'] = 'all_years'
        print('Training on all years')
    #%%
    
    return train, test, ex
        
def find_precursor(RV_ts, Prec_reg, train, test, ex):  
    #%%
# =============================================================================
#  Mean over 230 hot days
# =============================================================================

    ds_mcK = mcKmean(train, ex)  
    
# =============================================================================
# Extracting feature to build spatial map
# ============================================================================= 
   
    ds_Sem = extract_precursor(train, test, ex)   

    
    if ex['wghts_accross_lags'] == True:
        ds_Sem['pattern'] = filter_autocorrelation(ds_Sem, ex)

    
    return ds_mcK, ds_Sem, ex


def extract_precursor(train, test, ex):
    #%%
    lats = train['Prec'].latitude
    lons = train['Prec'].longitude
    pthresholds = np.linspace(1, 9, 9, dtype=int)
    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='communities_composite',
                          attrs={'units':'Kelvin'})

    
    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern_num = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='communities_numbered', 
                          attrs={'units':'Precursor regions'})

    array = np.zeros( (len(ex['lags']), len(lats), len(lons)) )
    pattern_num_init = xr.DataArray(data=array, coords=[ex['lags'], lats, lons], 
                          dims=['lag','latitude','longitude'], name='commun_numb_init', 
                          attrs={'units':'Regions'})
     
    logit_model = []
    
    ex['ts_train_std'] = []
    
    combs_reg_kept = np.zeros( len(ex['lags']), dtype=list)
    

    array = np.zeros( (len(ex['lags']), len(pthresholds)) )
    pattern_p = xr.DataArray(data=array, coords=[ex['lags'], pthresholds], 
                          dims=['lag','percentile'], name='Sem_mean_p_diff_lags')

    
    
    Actors_ts_GPH = [[] for i in ex['lags']] #!
    
    x = 0
    for lag in ex['lags']:
#            i = ex['lags'].index(lag)
        idx = ex['lags'].index(lag)
        event_train = Ev_timeseries(train['RV'], ex['hotdaythres']).time
        event_train = to_datesmcK(event_train, event_train.dt.hour[0], 
                                           train['Prec'].time[0].dt.hour)
        events_min_lag = event_train - pd.Timedelta(int(lag), unit='d')
        
        dates_train = to_datesmcK(train['RV'].time, train['RV'].time.dt.hour[0], 
                                           train['Prec'].time[0].dt.hour)
        dates_train_min_lag = dates_train - pd.Timedelta(int(lag), unit='d')
        
        ### exlude leap days ###
        # no leap days in dates_train_min_lag
        noleapdays = ((dates_train_min_lag.dt.month == 2) & (dates_train_min_lag.dt.day == 29))==False
        # only keep noleapdays
        dates_train_min_lag = dates_train_min_lag[noleapdays].dropna(dim='time', how='all')
        # also kick out the corresponding events
        dates_train = dates_train[noleapdays].dropna(dim='time', how='all')
        
       # no leap days in events_min_lag
        noleapdays = ((events_min_lag.dt.month == 2) & (events_min_lag.dt.day == 29))==False
        # only keep noleapdays
        events_min_lag = events_min_lag[noleapdays].dropna(dim='time', how='all') 
        # also kick out the corresponding events
        event_train = event_train[noleapdays].dropna(dim='time', how='all')   

        
        
        event_idx = [list(dates_train.values).index(E) for E in event_train.values]
        binary_events = np.zeros(dates_train.size)    
        binary_events[event_idx] = 1
        ts_3d = train['Prec'].sel(time=dates_train_min_lag)
        #%%
        # extract communities
        pattern_atlag, comm_numb, commun_init, combs_kept, logitmodel_lag_i = extract_commun(
                                                events_min_lag, ts_3d, binary_events, ex)  
        
        pattern[idx] = pattern_atlag
        pattern_num[idx]  = comm_numb
        pattern_num_init[idx] = commun_init
        combs_reg_kept[idx] = combs_kept
        logit_model.append( logitmodel_lag_i )
        
        # not using pattern covariance, but ts_from_logit
        crosscorr_Sem = cross_correlation_patterns(ts_3d, pattern_atlag)
#        crosscorr_Sem.values = ts_regions_lag_i
#        crosscorr_Sem['time'] = pattern_ts.time
#        pattern_ts[idx] = crosscorr_Sem
        # Percentile values based on training dataset
        p_pred = []
        for p in pthresholds:	
            p_pred.append(np.percentile(crosscorr_Sem.values, p*10))
        pattern_p[idx] = p_pred
    ds_Sem = xr.Dataset( {'pattern' : pattern, 'pattern_num' : pattern_num, 
                          'pattern_num_init' : pattern_num_init,
                          'logitmodel' : logit_model, 'combs_kept' : combs_reg_kept,
                          'perc' : pattern_p} )
        
    
    return ds_Sem


# =============================================================================
# =============================================================================
# Core functions
# =============================================================================
# =============================================================================

def extract_commun(events_min_lag, ts_3d, binary_events, ex):
    x=0
#    T, pval, mask_sig = Welchs_t_test(sample, full, alpha=0.01)
#    threshold = np.reshape( mask_sig, (mask_sig.size) )
#    mask_threshold = threshold 
#    plt.figure()
#    plt.imshow(mask_sig)
    # divide train set into train-feature and train-weights part:
#%% 
    lats = ts_3d.latitude
    lons = ts_3d.longitude
    full_years  = list(ts_3d.time.dt.year.values)
    comp_years = list(events_min_lag.time.dt.year.values)
    
    all_yrs_set = list(set(ts_3d.time.dt.year.values))
    
    randyrs_trainfeat = np.random.choice(all_yrs_set, int(len(all_yrs_set)/2), replace=False)
    randyrs_trainwgts = [yr for yr in all_yrs_set if yr not in randyrs_trainfeat]

    yrs_trainfeat =  [i for i in range(len(comp_years)) if comp_years[i] in randyrs_trainfeat]    
    yrs_trainwghts = [i for i in range(len(full_years)) if full_years[i] in randyrs_trainwgts]



    # composite taken only over train feature part
    if ex['splittrainfeat'] == True:
        composite = ts_3d.sel(time=events_min_lag)
        mean = composite.isel(time=yrs_trainfeat).mean(dim='time')
        # ts_3d only for train-weights part
        ts_3d_trainwghts = ts_3d.isel(time=yrs_trainwghts)
        bin_event_trainwghts = binary_events[yrs_trainwghts]
    
    # ts_3d only for total time series
    else:
        ts_3d_trainwghts = ts_3d
        bin_event_trainwghts = binary_events
        # iterative leave two years out

        iter_regions = np.zeros( (len(all_yrs_set), ts_3d[0].size))
        
        # chunks of two years
        n_ch = 2
        chunks = [all_yrs_set[n_ch*i:n_ch*(i+1)] for i in range(int(len(all_yrs_set)/n_ch))]
        
        for yr in chunks:
            idx = [all_yrs_set.index(yr[0]), all_yrs_set.index(yr[1])]
            yrs_trainfeat =  [i for i in all_yrs_set if i not in yr] 
            # exclude year in ts_3d 
#            one_out_idx_ts = [i for i in range(len(full_years) ) if full_years[i] in yrs_trainfeat]
#            ts_3d_trainwghts = ts_3d.isel(time=one_out_idx_ts)
            
            # exclude year in event time series
            one_out_idx_ev = [i for i in range(len(comp_years) ) if comp_years[i] in yrs_trainfeat]
            event_one_out = events_min_lag.isel( time = one_out_idx_ev)
            
            comp_n_out = ts_3d_trainwghts.sel(time=event_one_out).mean(dim='time')
            
            std_lag = ts_3d_trainwghts.std(dim='time')
            StoN = abs(comp_n_out/std_lag)
            StoN = StoN / np.mean(StoN)
#            StoN_wghts = comp_n_out*StoN
            mean = comp_n_out/std_lag
            threshold = mean.quantile(ex['perc_map']/100).values
            nparray = np.reshape(np.nan_to_num(mean.values), (mean.size))
#            threshold = np.percentile(nparray, ex['perc_map'])
            mask_threshold = np.array(abs(nparray) < ( threshold ), dtype=int)
            Corr_Coeff = np.ma.MaskedArray(nparray, mask=mask_threshold)
            Regions_lag_i = define_regions_and_rank_new(Corr_Coeff, lats.values, lons.values)
            iter_regions[idx] = Regions_lag_i
#            iter_mean[idx] = Regions_lag_i
    
    
#    if ex['wghts_std_anom'] == True: 
#        # get a feeling for the variation within the composite, if anomalies are 
#        # persistent they are assigned with a heavier weight
#        std_lag =  ts_3d_trainwghts.std(dim='time')
#    #    std_lag.plot()
#        # smoothen field
#        
#    #    std_lag.where(std_lag.values < 0.1*std_lag.std().values)
#    #    std_lag.plot()
#    #    std_lag = rolling_mean_xr(std_lag, 3)
#        
#        StoN = abs(mean/std_lag)
#        StoN = StoN / np.mean(StoN)
#    #    StoN = StoN.where(StoN.values < 10*StoN.median().values)
#    #    StoN.plot()
#        StoN_wghts = mean*StoN
#        mean = StoN_wghts
    
#    nparray = np.reshape(np.nan_to_num(iter_mean.values), (len(all_yrs_set), iter_mean[0].size))
#    threshold = np.percentile(np.mean(nparray,axis=0), ex['perc_map'])
##    mask_threshold = np.zeros(len(all_yrs_set))
#    mask_threshold = np.array(abs(nparray) > ( threshold ), dtype=int)
#    mask_sum    = np.sum(mask_threshold, axis=0)    
##    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(mask_sum, (lats.size, lons.size))) ; plt.colorbar()
#    mask_sum    = mask_sum > int( 0.99* len(all_yrs_set) )
#    mask_sum = np.array(abs(mask_sum-1), dtype=bool)    
#    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(mask_sum, (lats.size, lons.size))) ; plt.colorbar()
    #%%
    mask_reg_all_1 = (iter_regions != 0.)
    reg_all_1 = iter_regions.copy()
    reg_all_1[mask_reg_all_1] = 1
    mask_final = ( np.sum(reg_all_1, axis=0) < int(ex['comp_perc'] * len(all_yrs_set)))
#    plt.figure(figsize=(10,15)) ; plt.imshow(np.reshape(np.array(mask_final,dtype=int), (lats.size, lons.size))) ; plt.colorbar()
    normal_composite = ts_3d.sel(time=events_min_lag).mean(dim='time')
    nparray_comp = np.reshape(np.nan_to_num(normal_composite.values), (normal_composite.size))
    Corr_Coeff = np.ma.MaskedArray(nparray_comp, mask=mask_final)
    lat_grid = mean.latitude.values
    lon_grid = mean.longitude.values
#        if Corr_Coeff.ndim == 1:
#            lag_steps = 1
#            n_rows = 1
#        else:
#            lag_steps = Corr_Coeff.shape[1]
#            n_rows = Corr_Coeff.shape[1]

    # retrieve regions sorted in order of 'strength'
    # strength is defined as an area weighted values in the composite
    Regions_lag_i = define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid, 10)
    
    # Regions are iteratively counted starting from first lag (R0) to last lag R(-1)
    # adapt numbering of different communities/Regions to account for 
    # multiple variables/lags
    if Regions_lag_i.max()> 0:
        n_regions_lag_i = int(Regions_lag_i.max()) 	
        A_r = np.reshape(Regions_lag_i, (lat_grid.size, lon_grid.size))
        A_r + x
    x = A_r.max() 

    # if there are less regions that are desired, the n_strongest is lowered
    if n_regions_lag_i <= ex['n_strongest']:
        ex['upd_n_strongest'] = n_regions_lag_i
   
    # regions investigated to create ts timeseries
    regions_for_ts = list(np.arange(1, ex['upd_n_strongest']+1))
    

    ts_regions_lag_i, sign_ts_regions = spatial_mean_regions(Regions_lag_i, 
                                         regions_for_ts, ts_3d_trainwghts, Corr_Coeff)[:2]

    # reshape to latlon grid
    npmap = np.reshape(Regions_lag_i, (lat_grid.size, lon_grid.size))
    mask_strongest = (npmap!=0) 
    npmap[mask_strongest==False] = 0
    xrnpmap_init = normal_composite.copy()
    xrnpmap_init.values = npmap
    
    mask = (('latitude', 'longitude'), mask_strongest)
    normal_composite.coords['mask'] = mask
    xrnpmap_init.coords['mask'] = mask
    xrnpmap_init = xrnpmap_init.where(xrnpmap_init.mask==True)
#    plt.figure()
#    xrnpmap_init.plot.contourf(cmap=plt.cm.tab10)   
#%%
    # get wgths and only regions that contributed to probability    
    if ex['new_model_sel'] == False:
        odds, regions_kept, combs_kept, logitmodel = train_weights_LogReg(
                ts_regions_lag_i, sign_ts_regions, bin_event_trainwghts, ex)

    if ex['new_model_sel'] == True:
        odds, regions_kept, combs_kept, logitmodel = NEW_train_weights_LogReg(
                ts_regions_lag_i, sign_ts_regions, bin_event_trainwghts, ex)

    upd_regions = np.zeros(Regions_lag_i.shape)
    for i in range(len(regions_kept)):
        reg = regions_kept[i]
#        print(reg)
        upd_regions[Regions_lag_i == reg] =  i+1

    
    npmap = np.ma.reshape(upd_regions, (len(lat_grid), len(lon_grid)))
    mask_strongest = (npmap!=0) 
    npmap[mask_strongest==False] = 0
    xrnpmap = normal_composite.copy()
    xrnpmap.values = npmap
    
    mask = (('latitude', 'longitude'), mask_strongest)
    normal_composite.coords['mask'] = mask
    xrnpmap.coords['mask'] = mask
    xrnpmap = xrnpmap.where(xrnpmap.mask==True)
#    plt.figure()
#    xrnpmap.plot.contourf(cmap=plt.cm.tab10)
    
    # normal mean of extracted regions
    norm_mean = normal_composite.where(normal_composite.mask==True)
    
    
    
    # standardize coefficients
#    coeff_features = (coeff_features - np.mean(coeff_features)) / np.std(coeff_features)
    features = list(np.arange(xrnpmap.min(), xrnpmap.max() + 1 ) )
    weights = npmap.copy()
    for f in features:
        idx_f = features.index(f)
        mask_single_feature = (npmap==f)
        weight = round(odds[int(idx_f)], 2) 
        np.place(arr=weights, mask=mask_single_feature, vals=weight)
        weights[mask_single_feature] = weight
#            weights = weights/weights.max()
    
    weighted_mean = norm_mean * abs(weights)
#    plt.figure() 
#    weighted_mean.plot.contourf()
    #%%
    return weighted_mean, xrnpmap, xrnpmap_init, combs_kept, logitmodel



def train_weights_LogReg(ts_regions_lag_i, sign_ts_regions, binary_events, ex):
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
    n_feat = len(ts_regions_lag_i[0])
    n_samples = len(ts_regions_lag_i[:,0])

    regions          = np.arange(1, n_feat + 1)   
    track_super_sign = np.arange(1, n_feat + 1)   
    track_r_kept     = np.arange(1, n_feat + 1)
    signs = sign_ts_regions
       
    
#    regs_for_interac = np.arange(1, n_feat + 1)
    ex['std_of_initial_ts'] = np.std(ts_regions_lag_i, axis=0) #!!
    X_init = ts_regions_lag_i / np.std(ts_regions_lag_i, axis=0) #!!
    y = binary_events
#    X_train, X_test, y_train, y_test = train_test_split(
#                                   X[:,:], y, test_size=0.33)
    # =============================================================================
    # Step 1: Preselection 
    # =============================================================================
    # first kick out regions which do not show any relation to the event 
    # i.e. low p-value
    X = X_init
    all_regions_significant = True
    i = 0
    init_vs_final_bic = [] #np.zeros( (len(ex['lags'])) , dtype=list)
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
#%% 
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
            # add 1 to get number of region
            idx_f += 1
            for i in regs_for_interac:
                if i in list(idx_f):   
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
            combregions.append(idx_f+1)
        
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
            comb_sign_r = set(cregions_sign.flatten())
            comb_sign_ind = [i for i in comb_sign_r if i not in regs_for_interac]
            comb_sign_ind_idx = [i-1 for i in comb_sign_ind if i in regions]
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
            idx_not_keeping = [i for i in reg_investigated if i not in flat_keeping]
            track_r_kept    = np.delete(track_r_kept, idx_not_keeping)
        
            # if it adds likelihood when reg_investiged interacts with reg_single, 
            # then I keep both reg_investiged and reg_single in as single predictors
            # if two reg_investigated became significant together, then I will keep 
            # them in as an 'interacting time series' reg1 * reg2 = predictor.
            if (regs_for_interac in keep_duetointer) and len(regs_for_interac)==2:
                add_pred_inter = [i for i in keep_duetointer if all(np.equal(i, regs_for_interac))][0]
                idx_pred_inter = [i for i in range(n_feat) if regions[i] in add_pred_inter]
                ts_inter_kept   = ts_regions_lag_i[:,idx_pred_inter[0]] * ts_regions_lag_i[:,idx_pred_inter[1]]
                X_inter_kept = ts_inter_kept[:,None] / np.std(ts_inter_kept[:,None], axis = 0)
#            
#                X_final = np.concatenate( (X_train, X_inter_kept), axis=1)
#            else:
#                X_final = X_train
    # =============================================================================
    # Step 3 Perform log regression on (individual) regions that past step 1 & 2.
    # =============================================================================
    # all regions
    X = X_init
    # select only region in track_regions_kept:
    r_kept_idx = [i-1 for i in regions if i in track_r_kept]
    sign_r_kept = sign_ts_regions[None,r_kept_idx]
    ts_r_kept   = ts_regions_lag_i[None,r_kept_idx]
    ex['ts_train_std'].append(np.std(ts_regions_lag_i[:,r_kept_idx], axis=0))
    X_final = X[:,r_kept_idx] * sign_r_kept
    logit_model=sm.Logit(y_train,X_final)
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

def spatial_mean_regions(Regions_lag_i, regions_for_ts, ts_3d, mean):
    #%%
    n_time   = ts_3d.time.size
    lat_grid = ts_3d.latitude
    lon_grid = ts_3d.longitude
    regions_for_ts = list(regions_for_ts)
    
    actbox = np.reshape(ts_3d.values, (n_time, 
                  lat_grid.size*lon_grid.size))  
    
    # get lonlat array of area for taking spatial means 
    lons_gph, lats_gph = np.meshgrid(lon_grid, lat_grid)
    cos_box = np.cos(np.deg2rad(lats_gph))
    cos_box_array = np.repeat(cos_box[None,:], actbox.shape[0], 0)
    cos_box_array = np.reshape(cos_box_array, (cos_box_array.shape[0], -1))
    

    # this array will be the time series for each feature
    ts_regions_lag_i = np.zeros((actbox.shape[0], len(regions_for_ts)))
    
    # track sign of eacht region
    sign_ts_regions = np.zeros( len(regions_for_ts) )
    
    # std regions
    std_regions     = np.zeros( (len(regions_for_ts)) )
    
    if len(Regions_lag_i.shape) == 1.:
        Regions = Regions_lag_i
    elif len(Regions_lag_i.shape) == 2.:
        Regions = np.reshape(Regions_lag_i, (Regions_lag_i.size))
    # calculate area-weighted mean over features
    for r in regions_for_ts:
        idx = regions_for_ts.index(r)
        # start with empty lonlat array
        B = np.zeros(Regions.shape)
        # Mask everything except region of interest
        B[Regions == r] = 1	
        # Calculates how values inside region vary over time
        ts_regions_lag_i[:,idx] = np.mean(actbox[:, B == 1] * cos_box_array[:, B == 1], axis =1)
        # get sign of region
        sign_ts_regions[idx] = np.sign(np.mean(mean[B==1]))
#    print(sign_ts_regions)
        
    std_regions = np.std(ts_regions_lag_i, axis=0)
    #%%
    return ts_regions_lag_i, sign_ts_regions, std_regions


def timeseries_for_test(ds_Sem, test, ex):
    #%%
    time = test['RV'].time
    
    array = np.zeros( (len(ex['lags']), len(time)) )
    ts_predic = xr.DataArray(data=array, coords=[ex['lags'], time.values], 
                          dims=['lag','time'], name='ts_predict_logit',
                          attrs={'units':'Kelvin'})
    ds_Sem['ts_prediction'] = ts_predic

    
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        dates_test = to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
                                           test['Prec'].time[0].dt.hour)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
        var_test_mcK = find_region(test['Prec'], region=ex['regionmcK'])[0]
    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
    
        var_test_mcK = var_test_mcK.sel(time=dates_min_lag)
        var_test_reg = test['Prec'].sel(time=dates_min_lag) 
        mean = ds_Sem['pattern'].sel(lag=lag)
        mean = np.reshape(mean.values, (mean.size))

        xrpattern_lag_i = ds_Sem['pattern_num'].sel(lag=lag)
        regions_for_ts = np.arange(xrpattern_lag_i.min(), xrpattern_lag_i.max()+1)
        
        ts_regions_lag_i, sign_ts_regions = spatial_mean_regions(
                        xrpattern_lag_i.values, regions_for_ts, 
                        var_test_reg, mean)[:2]
        ts_regions_lag_i = ts_regions_lag_i[:,:] * sign_ts_regions[None,:]
        # normalize time series (as done in the training)        
        X_n = ts_regions_lag_i / ex['ts_train_std'][idx]
        
        logit_model_lag_i = ds_Sem['logitmodel'].values[idx]
        ts_pred = logit_model_lag_i.predict(X_n)
        ts_predic[idx] = ts_pred
    if ex['n'] != 0:
        # make sure time axis align, otherwise it gives nans
        ds_Sem['ts_prediction']['time'].values = time
    ds_Sem['ts_prediction'] = ts_predic
#    print(ds_Sem['ts_prediction'])
    
    #%%
    return ds_Sem


def NEW_train_weights_LogReg(ts_regions_lag_i, sign_ts_regions, binary_events, ex):
    #%%
#    from sklearn.linear_model import LogisticRegression
#    from sklearn.model_selection import train_test_split
    import statsmodels.api as sm
    import itertools
    n_feat = len(ts_regions_lag_i[0])
    n_samples = len(ts_regions_lag_i[:,0])

    regions          = np.arange(1, n_feat + 1)   
    track_super_sign = np.arange(1, n_feat + 1)   
    track_r_kept     = np.arange(1, n_feat + 1)
    signs = sign_ts_regions
    
    assert n_feat != 0, 'no features found'

    all_comb = list(itertools.product([0, 1], repeat=n_feat))[1:]
    combs = [i for i in all_comb if (np.sum(i) == 2) or np.sum(i) == 1]
    
    
    X_n = ts_regions_lag_i / np.std(ts_regions_lag_i, axis=0) #!!!
    X_n = X_n[:,:] * sign_ts_regions[None,:]
    y_train = binary_events
    
    
    X = np.zeros( (n_samples, len(combs)) )
    for c in combs:
        idx = combs.index(c)
        ind_r = np.where(np.array(c) == 1)[0]
        X[:,idx] = np.product(X_n[:,ind_r], axis=1)
     
# =============================================================================
#    # Forward selection
# =============================================================================
    # First parameter
    combs_kept = []
    final_bics = [9E8]
    bic = []
    # first comb that add likelihood
    first_par_decr = True
    while first_par_decr:
        odds_list  = []
        for m in range(X.shape[1]):
            logit_model = sm.Logit(y_train,X[:,m])
            result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
            bic.append(result.bic)
            odds_list.append( np.exp(result.params) )
        min_bic = np.argmin(bic)
#        min_bic = np.argmax(odds_list)
        # if first par does not decrease the likelihood, then I keep it
        # otherwise, I am just picking a random area from my regions, that has 
        # likely nothing to do with the event
        logit_model = sm.Logit(y_train,X[:,min_bic])
        
        
        result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
#        if np.exp(result.params) > 1.0:
        X_for = X[:,min_bic,None]
        combs_kept.append(combs[min_bic])
        val_bic = bic[min_bic]
        X = np.delete(X, min_bic, axis=1)
        first_par_decr = False
#        else:
#            # Delete from combs
#            X = np.delete(X, min_bic, axis=1)
        
    # incrementally add parameters
    final_bics.append(val_bic)
    diff = []
    forward_not_converged = True
      
    while forward_not_converged:
        # calculate bic when adding an extra parameter
        pd_res = pd.DataFrame(columns=['bic', 'odds'])
        bic = []
        odds_list = []
        for m in range(X.shape[1]):
            X_ = np.concatenate( (X_for, X[:,m,None]), axis=1)
            logit_model = sm.Logit( y_train, X_ )
            result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
            odds_added = np.exp(result.params[-1]) 
            pd_res = pd_res.append( {'bic':result.bic, 'odds':odds_added} , ignore_index=True )
        
#        sorted_odds = pd_res.sort_values(by=['odds'], ascending=False)
#        # take top 25 % of odds
#        idx = max(1, int(np.percentile(range(X.shape[1]), 30)) )
#        min_bic = sorted_odds[:idx]['bic'].idxmin()
    
        # just take param with min_bic
        min_bic = pd_res['bic'].idxmin()
        
        val_bic = pd_res.iloc[min_bic]['bic']
#        print(val_bic)
        final_bics.append(val_bic)
            # relative difference vs. initial bic
        diff_m = (final_bics[-1] - final_bics[-2])/final_bics[1] 
        threshold_bic = 1E-4 # used to be -1%, i.e. 0.01
#        if diff_m < threshold_bic and np.exp(result.params[-1]) > 1.:
        threshold_bic = -0.01
        if diff_m < threshold_bic:
            # add parameter to model only if they substantially decreased bic
            X_for = np.concatenate( (X_for, X[:,min_bic,None] ), axis=1)
            combs_kept.append(combs[min_bic])
        # delete parameter from list
        X = np.delete(X, min_bic, axis=1)
        #reloop, is new bic value really lower then the old one?
        diff.append( (final_bics[-1] - final_bics[-2])/final_bics[1] )
        # terminate if condition is not met 4 times in a row
        test_n_extra = 5
        if len(diff) > test_n_extra:
            # difference of last 5 attempts
            # Decrease in bic should be bigger than 1% of the initial bic value (only first par)
            diffs = [i for i in diff[-test_n_extra:] if i > threshold_bic]
            
            if len(diffs) == test_n_extra:
                forward_not_converged = False
        # if all attempts are made, then do not delete the last variable from X
        if X.shape[1] == 1:
            forward_not_converged = False

        
    final_forward = sm.Logit( y_train, X_for )
    result = final_forward.fit(disp=0, method='newton', tol=1E-8, retall=True)
#    print(result.summary2())
            
    
    
#    plt.plot(diff[1:])
   #%%
# =============================================================================
#   # Backward selection
# =============================================================================
    backward_not_converged = True
    while backward_not_converged and X_for.shape[1] > 1:
        bicb = []
        for d in range(X_for.shape[1]):
            X_one_d = np.delete(X_for, d, axis=1)
            logit_model = sm.Logit( y_train, X_one_d )
            result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
            bicb.append(result.bic)
        min_bicb = np.argmin(bicb)
        val_bicb = bicb[min_bicb]
        diff = (val_bicb - val_bic) / val_bic
        # even if val_bicb is not lower, but within 1% of larger model,
        # then the model with df-1 is kept.
        if abs(diff) < threshold_bic:
            X_for = np.delete(X_for, min_bicb, axis=1)
            combs_kept = [c for c in combs_kept if combs_kept.index(c) != min_bicb]
        else:
            backward_not_converged = False
   
         
# =============================================================================
#   # Final model
# =============================================================================
    final_model = sm.Logit( y_train, X_for )
    result = final_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
    odds = np.exp(result.params)
    
    # quantify strength of each region
    region_str = np.array(combs_kept, dtype=float)
    for c in combs_kept:
        idx = combs_kept.index(c)
        ind_r = np.where(np.array(c) == 1)[0]
        region_str[idx][ind_r] = float(odds[idx])
        
    pd_str = pd.DataFrame(columns=['B_coeff'], index= track_r_kept)
    for i in range(len(np.swapaxes(region_str, 1,0))):
        reg_c = np.swapaxes(region_str, 1,0)[i]
        idx_r = np.where(reg_c != 0)[0]
        if len(idx_r) == 0:
            pass
        else:
            pd_str.loc[i+1, 'B_coeff'] = np.product( reg_c[idx_r] )
    pd_str = pd_str.dropna()
    
    
    B_coeff = pd_str['B_coeff'].values.squeeze().flatten()
    
    track_r_kept = pd_str['B_coeff'].index.values
    final_model = result
    # update combs_kept, region numbers are changed further up to 1,2,3, etc..
    del_zeros = np.sum(combs_kept,axis=0) >= 1
    combs_kept_new = [tuple(np.array(c)[del_zeros]) for c in combs_kept ]
    # store std of regions that were kept
    
    #%%
    return B_coeff, track_r_kept, combs_kept_new, final_model



def NEW_timeseries_for_test(ds_Sem, test, ex):
    #%%
    time = test['RV'].time
    
    array = np.zeros( (len(ex['lags']), len(time)) )
    ts_predic = xr.DataArray(data=array, coords=[ex['lags'], time], 
                          dims=['lag','time'], name='ts_predict_logit',
                          attrs={'units':'Kelvin'})
    
    for lag in ex['lags']:
        idx = ex['lags'].index(lag)
        dates_test = to_datesmcK(test['RV'].time, test['RV'].time.dt.hour[0], 
                                           test['Prec'].time[0].dt.hour)
        # select antecedant SST pattern to summer days:
        dates_min_lag = dates_test - pd.Timedelta(int(lag), unit='d')
#        var_test_mcK = find_region(test['Prec'], region=ex['regionmcK'])[0]
#    #    full_timeserie_regmck = var_test_mcK.sel(time=dates_min_lag)
#    
#        var_test_mcK = var_test_mcK.sel(time=dates_min_lag)
        var_test_reg = test['Prec'].sel(time=dates_min_lag) 
        mean = ds_Sem['pattern'].sel(lag=lag)
        mean = np.reshape(mean.values, (mean.size))

        xrpattern_lag_i = ds_Sem['pattern_num'].sel(lag=lag)
        regions_for_ts = np.arange(xrpattern_lag_i.min(), xrpattern_lag_i.max()+1)
        
        ts_regions_lag_i, sign_ts_regions = spatial_mean_regions(
                        xrpattern_lag_i.values, regions_for_ts, 
                        var_test_reg, mean)
        # make all ts positive
        ts_regions_lag_i = ts_regions_lag_i[:,:] * sign_ts_regions[None,:]
        # normalize time series (as done in the training)        
        X_n = ts_regions_lag_i / np.std(ts_regions_lag_i, axis=0)
        
        combs_kept_lag_i  = ds_Sem['combs_kept'].values[idx] 
        ts_new = np.zeros( (time.size, len(combs_kept_lag_i)) )
        for c in combs_kept_lag_i:
            i = combs_kept_lag_i.index(c)
            idx_r = np.where(np.array(c) == 1)[0]
#            idx_r = ind_r - 1
            ts_new[:,i] = np.product(X_n[:,idx_r], axis=1)
        

        logit_model_lag_i = ds_Sem['logitmodel'].values[idx]
        
        ts_pred = logit_model_lag_i.predict(ts_new)
#        ts_prediction = (ts_pred-np.mean(ts_pred))/ np.std(ts_pred)
        ts_predic[idx] = ts_pred
    
    ds_Sem['ts_prediction'] = ts_predic
    #%%
    return ds_Sem



        
#def train_weights_LogReg(ts_regions_lag_i, sign_ts_regions, binary_events, ex):
#    #%%
#    from sklearn.linear_model import LogisticRegression
#    from sklearn.model_selection import train_test_split
#    # I want to the importance of each regions as is shown in the composite plot
#    # A region will increase the probability of the events when it the coef_ is 
#    # higher then 1. This means that if the ts shows high values, we have a higher 
#    # probability of the events. To ease interpretability I will multiple each ts
#    # with the sign of the region, to that each params should be above one (higher
#    # coef_, higher probability of events)
#    n_feat = len(ts_regions_lag_i[0])
#    track_r_kept = np.arange(1, n_feat + 1)
#    signs = sign_ts_regions
#    X = ts_regions_lag_i / np.std(ts_regions_lag_i, axis=0) #!!!
##    X = ts_regions_lag_i
#    y = binary_events
#    
#    all_regions_significant = True
#    while all_regions_significant:
#        X_train = X * signs[None,:]
#        y_train = y
##        X_train = ts_regions_lag_i
##        y_train = binary_events
#    #    X_train, X_test, y_train, y_test = train_test_split(
##                                   X[:,:], y, test_size=0.33)
#        
##        Log_reg = LogisticRegression(random_state=5, penalty = 'l2', solver='saga',
##                           tol = 1E-9, multi_class='ovr', max_iter=8000, fit_intercept=False)
##        model = Log_reg.fit(X_train, y_train)
##        odds_SK = np.exp(model.coef_)
##        print('SK logit score {}'.format(model.score(X_train,y_train)))
#        import statsmodels.api as sm
#        logit_model=sm.Logit(y_train,X_train)
#        result = logit_model.fit(disp=0, method='newton', tol=1E-8, retall=True)
#        if result.mle_retvals['converged'] == False:
#            print('logistic regression did not converge, taking odds of prev'
#                  'iteration or - if not present -, all odds (wghts) are equal')
#            try:
#                # capture odds array of previous iteration
#                odds = odds
#            except NameError as error:
#                # if odds in not defined, then it has never converged and there
#                # are no odds available, set all odds (wghts) to 1
#                odds = np.ones(len(n_feat))
#                
#        elif result.mle_retvals['converged'] == True:
#            odds = np.exp(result.params)
#    #        print('statsmodel logit Odds of event happening conditioned on X (p / (1+p) = exp(params) \n{}\n'.format(odds))
##            print(result.summary2())
#            p_vals = result.pvalues
#            regions_not_sign = np.where(p_vals >= ex['p_value_logit'])[0]
#
#
#
#            # update regions accounted for in fit
#            X = np.delete(X, regions_not_sign, axis=1)
#            track_r_kept = np.delete(track_r_kept, regions_not_sign)
#            signs  = np.delete(signs, regions_not_sign, axis=0)
#            if len(regions_not_sign) == 0:
#                all_regions_significant = False
#
#    #%%        
#    return odds, track_r_kept


def rand_traintest(RV_ts, Prec_reg, ex):
    all_years = np.arange(ex['startyear'], ex['endyear']+1)
    
    # conditions failed initally assumed True
    a_conditions_failed = True
    while a_conditions_failed == True:
        
        a_conditions_failed = False
        # Divide into random sampled 25 year for train & rest for test
    #        n_years_sampled = int((ex['endyear'] - ex['startyear']+1)*0.66)
        if ex['method'] == 'random':
            rand_test_years = np.random.choice(all_years, ex['leave_n_years_out'], replace=False)
        elif ex['method'] == 'iter':
            ex['leave_n_years_out'] = 1
            rand_test_years = [all_years[ex['n']]]
            
    
            
        # test duplicates
        a_conditions_failed = (len(set(rand_test_years)) != ex['leave_n_years_out'])
        # Update random years to be selected as test years:
    #        initial_years = [yr for yr in initial_years if yr not in random_test_years]
        rand_train_years = [yr for yr in all_years if yr not in rand_test_years]
        
    #            datesRV = pd.to_datetime(RV_ts.time.values)
    #            matchdatesRV = to_datesmcK(datesRV, datesRV[0].hour, Prec_reg.time[0].dt.hour)
    #            RV_dates = list(matchdatesRV.time.dt.year.values)
        full_years  = list(Prec_reg.time.dt.year.values)
        RV_years  = list(RV_ts.time.dt.year.values)
        
    #            RV_dates_train_idx = [i for i in range(len(RV_dates)) if RV_dates[i] in rand_train_years]
        Prec_train_idx = [i for i in range(len(full_years)) if full_years[i] in rand_train_years]
        RV_train_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_train_years]
        
    #            RV_dates_test_idx = [i for i in range(len(RV_dates)) if RV_dates[i] in rand_test_years]
        Prec_test_idx = [i for i in range(len(full_years)) if full_years[i] in rand_test_years]
        RV_test_idx = [i for i in range(len(RV_years)) if RV_years[i] in rand_test_years]
        
        
    #            dates_train = matchdatesRV.isel(time=RV_dates_train_idx)
        Prec_train = Prec_reg.isel(time=Prec_train_idx)
        RV_train = RV_ts.isel(time=RV_train_idx)
        
    #    if len(RV_dates_test_idx) 
    #            dates_test = matchdatesRV.isel(time=RV_dates_test_idx)
        Prec_test = Prec_reg.isel(time=Prec_test_idx)
        RV_test = RV_ts.isel(time=RV_test_idx)
        
        event_train = Ev_timeseries(RV_train, ex['hotdaythres']).time
        event_test = Ev_timeseries(RV_test, ex['hotdaythres']).time
        
        test_years = [yr for yr in list(set(RV_years)) if yr in rand_test_years]
        
        ave_events_pyr = (len(event_train) + len(event_test))/len(all_years)
        exp_events     = int(ave_events_pyr) * len(rand_test_years)
        tolerance      = 0.2 * exp_events
        diff           = abs(len(event_test) - exp_events)
        
        print('test year is {}, with {} events'.format(test_years, len(event_test)))
        if diff > tolerance and ex['method']=='random': 
            print('not a representative sample, drawing new sample')
            a_conditions_failed = True
                   
        
        train = dict( {    'Prec'  : Prec_train,
                           'RV'    : RV_train,
                           'events' : event_train})
        test = dict( {     'Prec'  : Prec_test,
                           'RV'    : RV_test,
                           'events' : event_test})
    return train, test, test_years

def filter_autocorrelation(ds_Sem, ex):
    n_lags = len(ex['lags'])
    n_lats = ds_Sem['pattern'].latitude.size
    n_lons = ds_Sem['pattern'].longitude.size
    ex['n_steps'] = len(ex['lags'])
    weights = np.zeros( (n_lags, n_lats, n_lons) )
    xrweights = ds_Sem['pattern'].copy()
    xrweights.values = weights
    for lag in ex['lags']:
        data = np.nan_to_num(ds_Sem['pattern'].sel(lag=lag).values)
        mask = np.ma.masked_array(data, dtype=bool)
        idx = ex['lags'].index(lag)
        weights[idx] = mask
        xrweights[idx].values = mask
    weights = np.sum(weights, axis=0)
    return weights * ds_Sem['pattern']


def Welchs_t_test(sample, full, alpha):
    np.warnings.filterwarnings('ignore')
    mask = (sample[0] == 0.).values
#    mask = np.reshape(mask, (mask.size))
    n_space = full.latitude.size*full.longitude.size
    npfull = np.reshape(full.values, (full.time.size, n_space))
    npsample = np.reshape(sample.values, (sample.time.size, n_space))
    
#    npsample = npsample[np.broadcast_to(mask==False, npsample.shape)] 
#    npsample = np.reshape(npsample, (sample.time.size, 
#                                     int(npsample.size/sample.time.size) ))
#    npfull   = npfull[np.broadcast_to(mask==False, npfull.shape)] 
#    npfull = np.reshape(npfull, (full.time.size, 
#                                     int(npfull.size/full.time.size) ))
       
    T, pval = scipy.stats.ttest_ind(npsample, npfull, axis=0, 
                                equal_var=False, nan_policy='omit')
    pval = np.reshape(pval, (full.latitude.size, full.longitude.size))
    T = np.reshape(T, (full.latitude.size, full.longitude.size))
    mask_sig = (pval > alpha) 
    mask_sig[mask] = True
    return T, pval, mask_sig

def merge_neighbors(lsts):
  sets = [set(lst) for lst in lsts if lst]
  merged = 1
  while merged:
    merged = 0
    results = []
    while sets:
      common, rest = sets[0], sets[1:]
      sets = []
      for x in rest:
        if x.isdisjoint(common):
          sets.append(x)
        else:
          merged = 1
          common |= x
      results.append(common)
    sets = results
  return sets


def define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid, minsize='mean'):
    '''
	takes Corr Coeffs and defines regions by strength

	return A: the matrix whichs entries correspond to region. 1 = strongest, 2 = second strongest...
    '''
#    print('extracting features ...\n')

	
	# initialize arrays:
	# A final return array 
    A = np.ma.copy(Corr_Coeff)
#    A = np.ma.zeros(Corr_Coeff.shape)
	#========================================
	# STEP 1: mask nodes which were never significantly correlatated to index (= count=0)
	#========================================
	
	#========================================
	# STEP 2: define neighbors for everey node which passed Step 1
	#========================================

    indices_not_masked = np.where(A.mask==False)[0].tolist()

    lo = lon_grid.shape[0]
    la = lat_grid.shape[0]
	
	# create list of potential neighbors:
    N_pot=[[] for i in range(A.shape[0])]

	#=====================
	# Criteria 1: must bei geographical neighbors:
	#=====================
    for i in indices_not_masked:
        n = []	

        col_i= i%lo
        row_i = i//lo

		# knoten links oben
        if i==0:	
            n= n+[lo-1, i+1, lo ]

		# knoten rechts oben	
        elif i== lo-1:
            n= n+[i-1, 0, i+lo]

		# knoten links unten
        elif i==(la-1)*lo:
            n= n+ [i+lo-1, i+1, i-lo]

		# knoten rechts unten
        elif i == la*lo-1:
            n= n+ [i-1, i-lo+1, i-lo]

		# erste zeile
        elif i<lo:
            n= n+[i-1, i+1, i+lo]
	
		# letzte zeile:
        elif i>la*lo-1:
            n= n+[i-1, i+1, i-lo]
	
		# erste spalte
        elif col_i==0:
            n= n+[i+lo-1, i+1, i-lo, i+lo]
	
		# letzt spalte
        elif col_i ==lo-1:
            n= n+[i-1, i-lo+1, i-lo, i+lo]
	
		# nichts davon
        else:
            n = n+[i-1, i+1, i-lo, i+lo]
	
	#=====================
	# Criteria 2: must be all at least once be significanlty correlated 
	#=====================	
        m =[]
        for j in n:
            if j in indices_not_masked:
                m = m+[j]
		
		# now m contains the potential neighbors of gridpoint i

	
	#=====================	
	# Criteria 3: sign must be the same for each step 
	#=====================				
        l=[]
	
        cc_i = A.data[i]
        cc_i_sign = np.sign(cc_i)
		
	
        for k in m:
            cc_k = A.data[k]
            cc_k_sign = np.sign(cc_k)
		

            if cc_i_sign *cc_k_sign == 1:
                l = l +[k]

            else:
                l = l
			
            if len(l)==0:
                l =[]
                A.mask[i]=True	
			
            else: l = l +[i]	
		
		
            N_pot[i]=N_pot[i]+ l	



	#========================================	
	# STEP 3: merge overlapping set of neighbors
	#========================================
    Regions = merge_neighbors(N_pot)
	
	#========================================
	# STEP 4: assign a value to each region
	#========================================
	

	# 2) combine 1A+1B 
    B = np.abs(A)
	
	# 3) calculate the area size of each region	
	
    Area =  [[] for i in range(len(Regions))]
	
    for i in range(len(Regions)):
        indices = np.array(list(Regions[i]))
        indices_lat_position = indices//lo
        lat_nodes = lat_grid[indices_lat_position[:]]
        cos_nodes = np.cos(np.deg2rad(lat_nodes))		
		
        area_i = [np.sum(cos_nodes)]
        Area[i]= Area[i]+area_i
	
	#---------------------------------------
	# OPTIONAL: Exclude regions which only consist of less than n nodes
	# 3a)
	#---------------------------------------	
	
    # keep only regions which are larger then the mean size of the regions
    if minsize == 'mean':
        n_nodes = int(np.mean([len(r) for r in Regions]))
    else:
        n_nodes = minsize
    
    R=[]
    Ar=[]
    for i in range(len(Regions)):
        if len(Regions[i])>=n_nodes:
            R.append(Regions[i])
            Ar.append(Area[i])
	
    Regions = R
    Area = Ar	
	
	
	
	# 4) calcualte region value:
	
    C = np.zeros(len(Regions))
	
    Area = np.array(Area)
    for i in range(len(Regions)):
        C[i]=Area[i]*np.mean(B[list(Regions[i])])


	
	
	# mask out those nodes which didnot fullfill the neighborhood criterias
    A.mask[A==0] = True	
		
		
	#========================================
	# STEP 5: rank regions by region value
	#========================================
	
	# rank indices of Regions starting with strongest:
    sorted_region_strength = np.argsort(C)[::-1]
	
	# give ranking number
	# 1 = strongest..
	# 2 = second strongest
    
    # create clean array
    Regions_lag_i = np.zeros(A.data.shape)
    for i in range(len(Regions)):
        j = list(sorted_region_strength)[i]
        Regions_lag_i[list(Regions[j])]=i+1
    
    Regions_lag_i = np.array(Regions_lag_i, dtype=int)
    return Regions_lag_i



def read_T95(T95name, ex):
    filepath = os.path.join(ex['path_pp'], T95name)
    data = pd.read_csv(filepath)
    datelist = []
    values = []
    for r in data.values:
        year = int(r[0][:4])
        month = int(r[0][5:7])
        day = int(r[0][7:11])
        string = '{}-{}-{}'.format(year, month, day)
        values.append(float(r[0][10:]))
        datelist.append( pd.Timestamp(string) )
    
    dates = pd.to_datetime(datelist)
    RVts = xr.DataArray(values, coords=[dates], dims=['time'])
    return RVts, dates

def Ev_timeseries(xarray, threshold):   
    Ev_ts = xarray.where( xarray.values > threshold) 
    Ev_dates = Ev_ts.dropna(how='all', dim='time').time
    return Ev_dates

def timeseries_tofit_bins(xarray, ex):
    datetime = pd.to_datetime(xarray['time'].values)
    one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
    
    seldays_pp = pd.DatetimeIndex(start=one_yr[0], end=one_yr[-1], 
                                freq=(datetime[1] - datetime[0]))
    end_day = one_yr.max() 
    # after time averaging over 'tfreq' number of days, you want that each year 
    # consists of the same day. For this to be true, you need to make sure that
    # the selday_pp period exactly fits in a integer multiple of 'tfreq'
    temporal_freq = np.timedelta64(ex['tfreq'], 'D') 
    fit_steps_yr = (end_day - seldays_pp.min() ) / temporal_freq
    # line below: The +1 = include day 1 in counting
    start_day = (end_day - (temporal_freq * np.round(fit_steps_yr, decimals=0))) \
                + np.timedelta64(1, 'D') 
    
    def make_datestr_2(datetime, start_yr):
        breakyr = datetime.year.max()
        datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
        nyears = (datetime.year[-1] - datetime.year[0])+1
        startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
        endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
        firstyear = startday[:4]
        datesdt = start_yr
        def plusyearnoleap(curr_yr, startday, endday, incr):
            startday = startday.replace(firstyear, str(curr_yr+incr))
            endday = endday.replace(firstyear, str(curr_yr+incr))
            next_yr = pd.DatetimeIndex(start=startday, end=endday, 
                            freq=(datetime[1] - datetime[0]))
            # excluding leap year again
            noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
            next_yr = next_yr[noleapdays].dropna(how='all')
            return next_yr
        
        for yr in range(0,nyears-1):
            curr_yr = yr+datetime.year[0]
            next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
            datesdt = np.append(datesdt, next_yr)
#            print(len(next_yr))
#            nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
#            datesstr = datesstr + nextstr
#            print(nextstr[0])
            
            upd_start_yr = plusyearnoleap(next_yr.year[0], startday, endday, 1)

            if next_yr.year[0] == breakyr:
                break
        datesdt = pd.to_datetime(datesdt)
        return datesdt, upd_start_yr
    
    start_yr = pd.DatetimeIndex(start=start_day, end=end_day, 
                                freq=(datetime[1] - datetime[0]))
    # exluding leap year from cdo select string
    noleapdays = (((start_yr.month==2) & (start_yr.day==29))==False)
    start_yr = start_yr[noleapdays].dropna(how='all')
    datesdt, next_yr = make_datestr_2(datetime, start_yr)
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',
                         8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
    startdatestr = '{} {}'.format(start_day.day, months[start_day.month])
    enddatestr   = '{} {}'.format(end_day.day, months[end_day.month])
    print('adjusted time series to fit bins: \nFrom {} to {}'.format(
                startdatestr, enddatestr))
    adj_array = xarray.sel(time=datesdt)
    return adj_array, datesdt
    

def time_mean_bins(xarray, ex):
    datetime = pd.to_datetime(xarray['time'].values)
    one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
    
    if one_yr.size % ex['tfreq'] != 0:
        possible = []
        for i in np.arange(1,20):
            if 214%i == 0:
                possible.append(i)
        print('Error: stepsize {} does not fit in one year\n '
                         ' supply an integer that fits {}'.format(
                             ex['tfreq'], one_yr.size))   
        print('\n Stepsize that do fit are {}'.format(possible))
        print('\n Will shorten the \'subyear\', so that it the temporal'
              'frequency fits in one year')
        xarray, datetime = timeseries_tofit_bins(xarray, ex)
        one_yr = datetime.where(datetime.year == datetime.year[0]).dropna(how='any')
          
    else:
        pass
    fit_steps_yr = (one_yr.size)  / ex['tfreq']
    bins = list(np.repeat(np.arange(0, fit_steps_yr), ex['tfreq']))
    n_years = (datetime.year[-1] - datetime.year[0]) + 1
    for y in np.arange(1, n_years):
        x = np.repeat(np.arange(0, fit_steps_yr), ex['tfreq'])
        x = x + fit_steps_yr * y
        [bins.append(i) for i in x]
    label_bins = xr.DataArray(bins, [xarray.coords['time'][:]], name='time')
    label_dates = xr.DataArray(xarray.time.values, [xarray.coords['time'][:]], name='time')
    xarray['bins'] = label_bins
    xarray['time_dates'] = label_dates
    xarray = xarray.set_index(time=['bins','time_dates'])
    
    half_step = ex['tfreq']/2.
    newidx = np.arange(half_step, datetime.size, ex['tfreq'], dtype=int)
    newdate = label_dates[newidx]
    

    group_bins = xarray.groupby('bins').mean(dim='time', keep_attrs=True)
    group_bins['bins'] = newdate.values
    dates = pd.to_datetime(newdate.values)
    return group_bins.rename({'bins' : 'time'}), dates

def expand_times_for_lags(datetime, ex):
    expanded_time = []
    for yr in set(datetime.year):
        one_yr = datetime.where(datetime.year == yr).dropna(how='any')
        start_mcK = one_yr[0]
        #start day shifted half a time step
        half_step = ex['tfreq']/2.
#        origshift = np.arange(half_step, datetime.size, ex['tfreq'], dtype=int)
        start_mcK = start_mcK - np.timedelta64(int(half_step+0.49), 'D')
        last_day = '{}{} {}:00:00'.format(yr, ex['senddate'][4:], datetime[0].hour)
        end_mcK   = pd.to_datetime(last_day)
#        adj_year = pd.DatetimeIndex(start=start_mcK, end=end_mcK, 
#                                    freq=(datetime[1] - datetime[0]), 
#                                    closed = None).values
        steps = len(one_yr)
        shift_start = start_mcK - (steps) * np.timedelta64(ex['tfreq'], 'D')
        adj_year = pd.DatetimeIndex(start=shift_start, end=end_mcK, 
                                    freq=pd.Timedelta( '1 days'), 
                                    closed = None).values
        [expanded_time.append(date) for date in adj_year]
    
    return pd.to_datetime(expanded_time)

def make_datestr(dates, ex):
    start_yr = pd.DatetimeIndex(start=ex['sstartdate'], end=ex['senddate'], 
                                freq=(dates[1] - dates[0]))
    breakyr = dates.year.max()
    datesstr = [str(date).split('.', 1)[0] for date in start_yr.values]
    nyears = (dates.year[-1] - dates.year[0])+1
    startday = start_yr[0].strftime('%Y-%m-%dT%H:%M:%S')
    endday = start_yr[-1].strftime('%Y-%m-%dT%H:%M:%S')
    firstyear = startday[:4]
    def plusyearnoleap(curr_yr, startday, endday, incr):
        startday = startday.replace(firstyear, str(curr_yr+incr))
        endday = endday.replace(firstyear, str(curr_yr+incr))
        next_yr = pd.DatetimeIndex(start=startday, end=endday, 
                        freq=(dates[1] - dates[0]))
        # excluding leap year again
        noleapdays = (((next_yr.month==2) & (next_yr.day==29))==False)
        next_yr = next_yr[noleapdays].dropna(how='all')
        return next_yr
    

    for yr in range(0,nyears-1):
        curr_yr = yr+dates.year[0]
        next_yr = plusyearnoleap(curr_yr, startday, endday, 1)
        nextstr = [str(date).split('.', 1)[0] for date in next_yr.values]
        datesstr = datesstr + nextstr

        if next_yr.year[0] == breakyr:
            break
    datesmcK = pd.to_datetime(datesstr)
    return datesmcK

def import_array(filename, ex):
    file_path = os.path.join(ex['path_pp'], filename)        
    ncdf = xr.open_dataset(file_path, decode_cf=True, decode_coords=True, decode_times=False)
    variables = list(ncdf.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    var = [var for var in strvars if var not in ' time time_bnds longitude latitude '][0] 
    marray = np.squeeze(ncdf.to_array(file_path).rename(({file_path: var})))
    numtime = marray['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    dates = pd.to_datetime(dates)
#    print('temporal frequency \'dt\' is: \n{}'.format(dates[1]- dates[0]))
    marray['time'] = dates
    return marray

def save_figure(data, path):
    import os
    import matplotlib.pyplot as plt
#    if 'path' in locals():
#        pass
#    else:
#        path = '/Users/semvijverberg/Downloads'
    if path == 'default':
        path = '/Users/semvijverberg/Downloads'
    else:
        path = path
    import datetime
    today = datetime.datetime.today().strftime("%d-%m-%y_%H'%M")
    if type(data.name) is not type(None):
        name = data.name.replace(' ', '_')
    if 'name' in locals():
        print('input name is: {}'.format(name))
        name = name + '.jpeg'
        pass
    else:
        name = 'fig_' + today + '.jpeg'
    print(('{} to path {}'.format(name, path)))
    plt.savefig(os.path.join(path,name), format='jpeg', dpi=300, bbox_inches='tight')
    
def area_weighted(xarray):
    # Area weighted     
    coslat = np.cos(np.deg2rad(xarray.coords['latitude'].values)).clip(0., 1.)
    area_weights = np.tile(np.sqrt(coslat)[..., np.newaxis],(1,xarray.longitude.size))
    xarray.values = xarray.values * area_weights 
    return xarray
    
def convert_longitude(data):
    import numpy as np
    import xarray as xr
    lon_above = data.longitude[np.where(data.longitude > 180)[0]]
    lon_normal = data.longitude[np.where(data.longitude <= 180)[0]]
    # roll all values to the right for len(lon_above amount of steps)
    data = data.roll(longitude=len(lon_above))
    # adapt longitude values above 180 to negative values
    substract = lambda x, y: (x - y)
    lon_above = xr.apply_ufunc(substract, lon_above, 360)
    if lon_normal[0] == 0.:
        convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')
    else:
        convert_lon = xr.concat([lon_normal, lon_above], dim='longitude')
    data['longitude'] = convert_lon
    return data


def rolling_mean_xr(xarray, win):
    closed = int(win/2)
    flatarray = xarray.values.flatten()
    ext_array = np.insert(flatarray, 0, flatarray[-closed:])
    ext_array = np.insert(ext_array, 0, flatarray[:closed])
    
    df = pd.DataFrame(ext_array)
    std = xarray.where(xarray.values!=0.).std().values
#    scipy.signal.gaussian(win, std)
    rollmean = df.rolling(win, center=True, 
                          win_type='gaussian').mean(std=std).dropna()
    
    # replace values with smoothened values
    new_xarray = xarray.copy()
    new_values = np.reshape(rollmean.squeeze().values, xarray.shape)
    # ensure LSM mask
    mask = np.array((xarray.values!=0.),dtype=int)
    new_xarray.values = (new_values * mask)

    return new_xarray

def to_datesmcK(datesmcK, to_hour, from_hour):
    
    dt_hours = from_hour - to_hour
    matchdaysmcK = datesmcK + pd.Timedelta(int(dt_hours), unit='h')
    return xr.DataArray(matchdaysmcK, dims=['time'])


def find_region(data, region='Mckinnonplot'):
    if region == 'Mckinnonplot':
        west_lon = -240; east_lon = -40; south_lat = -10; north_lat = 80

    elif region ==  'U.S.':
        west_lon = -120; east_lon = -70; south_lat = 20; north_lat = 50
    elif region ==  'U.S.cluster':
        west_lon = -100; east_lon = -70; south_lat = 20; north_lat = 50
    elif region ==  'PEPrectangle':
        west_lon = -215; east_lon = -125; south_lat = 19; north_lat = 50
    elif region ==  'Pacific':
        west_lon = -215; east_lon = -120; south_lat = 19; north_lat = 60
    elif region ==  'Whole':
        west_lon = -360; east_lon = -1; south_lat = -80; north_lat = 80
    elif region ==  'Northern':
        west_lon = -360; east_lon = -1; south_lat = -10; north_lat = 80
    elif region ==  'Southern':
        west_lon = -360; east_lon = -1; south_lat = -80; north_lat = -10

    region_coords = [west_lon, east_lon, south_lat, north_lat]
    import numpy as np
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return int(idx)
#    if data.longitude.values[-1] > 180:
#        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
#        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
#        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)
        
    if west_lon <0 and east_lon > 0:
        # left_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(0, east_lon)))
        # right_of_meridional = np.array(data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360)))
        # all_values = np.concatenate((np.reshape(left_of_meridional, (np.size(left_of_meridional))), np.reshape(right_of_meridional, np.size(right_of_meridional))))
        lon_idx = np.concatenate(( np.arange(find_nearest(data['longitude'], 360 + west_lon), len(data['longitude'])),
                              np.arange(0,find_nearest(data['longitude'], east_lon), 1) ))
        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)
        all_values = data.sel(latitude=slice(north_lat, south_lat), 
                              longitude=(data.longitude > 360 + west_lon) | (data.longitude < east_lon))
    if west_lon < 0 and east_lon < 0:
        all_values = data.sel(latitude=slice(north_lat, south_lat), longitude=slice(360+west_lon, 360+east_lon))
        lon_idx = np.arange(find_nearest(data['longitude'], 360 + west_lon), find_nearest(data['longitude'], 360+east_lon))
        lat_idx = np.arange(find_nearest(data['latitude'],north_lat),find_nearest(data['latitude'],south_lat),1)

    return all_values, region_coords


def cross_correlation_patterns(full_timeserie, pattern):
#    full_timeserie = var_test_mcK
#    pattern = ds_mcK['pattern'].sel(lag=lag)
    mask = np.ma.make_mask(np.isnan(pattern.values)==False)
    
    n_time = full_timeserie.time.size
    n_space = pattern.size
    

#    mask_pattern = np.tile(mask_pattern, (n_time,1))
    # select only gridcells where there is not a nan
    full_ts = np.nan_to_num(np.reshape( full_timeserie.values, (n_time, n_space) ))
    pattern = np.nan_to_num(np.reshape( pattern.values, (n_space) ))

    mask_pattern = np.reshape( mask, (n_space) )
    full_ts = full_ts[:,mask_pattern]
    pattern = pattern[mask_pattern]
    
    crosscorr = np.zeros( (n_time) )
    spatcov   = np.zeros( (n_time) )
    covself   = np.zeros( (n_time) )
    corrself  = np.zeros( (n_time) )
    for t in range(n_time):
        # Corr(X,Y) = cov(X,Y) / ( std(X)*std(Y) )
        # cov(X,Y) = E( (x_i - mu_x) * (y_i - mu_y) )
        crosscorr[t] = np.correlate(full_ts[t], pattern)
        M = np.stack( (full_ts[t], pattern) )
        spatcov[t] = np.cov(M)[0,1] #/ (np.sqrt(np.cov(M)[0,0]) * np.sqrt(np.cov(M)[1,1]))
#        sqrt( Var(X) ) = sigma_x = std(X)
#        spatcov[t] = np.cov(M)[0,1] / (np.std(full_ts[t]) * np.std(pattern))        
        covself[t] = np.mean( (full_ts[t] - np.mean(full_ts[t])) * (pattern - np.mean(pattern)) )
        corrself[t] = covself[t] / (np.std(full_ts[t]) * np.std(pattern))
    dates_test = full_timeserie.time
    corrself = xr.DataArray(corrself, coords=[dates_test.values], dims=['time'])
    
#    # standardize
    corrself -= corrself.mean(dim='time')
    return corrself

# =============================================================================
# =============================================================================
# Plotting functions
# =============================================================================
# =============================================================================
def xarray_plot(data, path='default', name = 'default', saving=False):
    # from plotting import save_figure
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
#    original
    plt.figure()
    if len(data.longitude[np.where(data.longitude > 180)[0]]) != 0:
        if data.longitude.where(data.longitude==0).dropna(dim='longitude', how='all') == 0.:
            print('hoi')   
            data = convert_longitude(data)
    else:
        pass
    if data.ndim != 2:
        print("number of dimension is {}, printing first element of first dimension".format(np.squeeze(data).ndim))
        data = data[0]
    else:
        pass
    if 'mask' in list(data.coords.keys()):
        cen_lon = data.where(data.mask==True, drop=True).longitude.mean()
        data = data.where(data.mask==True, drop=True)
    else:
        cen_lon = data.longitude.mean().values
    proj = ccrs.PlateCarree(central_longitude=cen_lon)
    ax = plt.axes(projection=proj)
    ax.coastlines()
    vmin = np.round(float(data.min())-0.01,decimals=2) 
    vmax = np.round(float(data.max())+0.01,decimals=2) 
    vmin = -max(abs(vmin),vmax) ; vmax = max(abs(vmin),vmax)
    # ax.set_global()
    if 'mask' in list(data.coords.keys()):
        plot = data.copy().where(data.mask==True).plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True,
                             vmin=vmin, vmax=vmax)
    else:
        plot = data.plot.pcolormesh(ax=ax, cmap=plt.cm.RdBu_r,
                             transform=ccrs.PlateCarree(), add_colorbar=True,
                             vmin=vmin, vmax=vmax)
    if saving == True:
        save_figure(data, path=path)
    plt.show()
    
    
def plot_events_validation(pred1, pred2, obs, pt1, pt2, othreshold, test_year=None):
    #%%
#    pred1 = crosscorr_Sem
#    pred2 = crosscorr_mcK
#    obs = RV_ts_test
#    pt1 = Prec_threshold_Sem
#    pt2 = Prec_threshold_mcK
#    othreshold = ex['hotdaythres']
#    test_year = int(crosscorr_Sem.time.dt.year[0])
    
    
        
    
    def predyear(pred, obs):
        if str(type(test_year)) == "<class 'numpy.int64'>" or str(type(test_year)) == "<class 'int'>":
            predyear = pred.where(pred.time.dt.year == test_year).dropna(dim='time', how='any')
            obsyear  = obs.where(obs.time.dt.year == test_year).dropna(dim='time', how='any')
            predyear['time'] = obsyear.time
        elif type(test_year) == type(['list']):
            years_in_obs = list(obs.time.dt.year.values)
            test_years = [i for i in range(len(years_in_obs)) if years_in_obs[i] in test_year]
            # Warning this is wrong #!!!
            predyear = pred.isel(time=test_years)
            obsyear = obs.isel(time=test_years)
        else:
            predyear = pred
            predyear['time'] = obs.time
            obsyear = obs
        return predyear, obsyear
        
    predyear1, obsyear = predyear(pred1, obs)
    predyear2, obsyear = predyear(pred2, obs)

    eventdays = obsyear.where( obsyear.values > othreshold ) 
    eventdays = eventdays.dropna(how='all', dim='time').time
    preddays = predyear1.where(predyear1.values > pt1)
    preddays1 = preddays.dropna(how='all', dim='time').time
    preddays = predyear2.where(predyear2.values > pt2)
    preddays2 = preddays.dropna(how='all', dim='time').time
#    # standardize obsyear
#    othreshold -= obsyear.mean(dim='time').values
#    obsyear    -= obsyear.mean(dim='time')
#    
#    # standardize predyear(s)
#    pthreshold -= predyear.mean(dim='time').values
#    predyear    -= predyear.mean(dim='time')
      
    TP1 = [day for day in preddays1.time.values if day in list(eventdays.values)]
    TP2 = [day for day in preddays2.time.values if day in list(eventdays.values)]
#    pthreshold = ((pthreshold - pred1.mean()) * obsyear.std()/predyear.std()).values
#    predyear = (predyear) * obsyear.std()/predyear.std() 
    plt.figure(figsize = (10,5))
    ax1 = plt.subplot(311)
    ax1.plot(pd.to_datetime(obsyear.time.values), obsyear, label='observed',
             color = 'blue')
    ax1.axhline(y=othreshold, color='blue')
    for days in eventdays.time.values:
        ax1.axvline(x=pd.to_datetime(days), color='blue', alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(312)
    ax2.plot(pd.to_datetime(obsyear.time.values),predyear1, label='Sem pattern ts',
             color='red')
    ax2.axhline(y=pt1, color='red')
    for days in preddays1.time.values:
        ax2.axvline(x=pd.to_datetime(days), color='red', alpha=0.3)
    for days in pd.to_datetime(TP1):
        ax2.axvline(x=pd.to_datetime(days), color='green', alpha=1.)
    ax2.legend()
    # second prediction
    ax3 = plt.subplot(313)
    ax3.plot(pd.to_datetime(obsyear.time.values),predyear2, label='mcK pattern ts',
             color='red')
    ax3.axhline(y=pt2, color='red')
    for days in preddays2.time.values:
        ax3.axvline(x=pd.to_datetime(days), color='red', alpha=0.3)
    for days in pd.to_datetime(TP2):
        ax3.axvline(x=pd.to_datetime(days), color='green', alpha=1.)
    ax3.legend()


def plot_oneyr_events(xarray, threshold, test_year, folder, saving=False):
    testyear = xarray.where(xarray.time.dt.year == test_year).dropna(dim='time', how='any')
    freq = pd.Timedelta(testyear.time.values[1] - testyear.time.values[0])
    plotpaper = xarray.sel(time=pd.DatetimeIndex(start=testyear.time.values[0], 
                                                end=testyear.time.values[-1], 
                                                freq=freq ))
    #plotpaper = mcKtsfull.sel(time=pd.DatetimeIndex(start='2012-06-23', end='2012-08-21', 
    #                                freq=(datesmcK[1] - datesmcK[0])))
    eventdays = plotpaper.where( plotpaper.values > threshold) 
    eventdays = eventdays.dropna(how='all', dim='time').time
    plt.figure()
    plotpaper.plot()
    plt.axhline(y=threshold)
    for days in eventdays.time.values:
        plt.axvline(x=days)
    if saving == True:
        filename = os.path.join(folder, 'ts_{}'.format(test_year))
        plt.savefig(filename+'.png', dpi=300)

def plotting_wrapper(plotarr, filename, ex, kwrgs=None):
#    map_proj = ccrs.Miller(central_longitude=240)  
    folder_name = os.path.join(ex['figpathbase'], ex['exp_folder'])
    if os.path.isdir(folder_name) != True : 
        os.makedirs(folder_name)
    file_name = os.path.join(ex['figpathbase'], filename)

    if kwrgs == None:
        kwrgs = dict( {'title' : plotarr.name, 'clevels' : 'notdefault', 'steps':17,
                        'vmin' : -3*plotarr.std().values, 'vmax' : 3*plotarr.std().values, 
                       'cmap' : plt.cm.RdBu_r, 'column' : 2} )
    else:
        kwrgs = kwrgs
        kwrgs['title'] = plotarr.attrs['title']
    finalfigure(plotarr, file_name, kwrgs)
    

def finalfigure(xrdata, file_name, kwrgs):
    #%%
    map_proj = ccrs.Miller(central_longitude=240)  
    lons = xrdata.longitude.values
    lats = xrdata.latitude.values
    strvars = [' {} '.format(var) for var in list(xrdata.dims)]
    var = [var for var in strvars if var not in ' longitude latitude '][0] 
    var = var.replace(' ', '')
    g = xr.plot.FacetGrid(xrdata, col=var, col_wrap=kwrgs['column'], sharex=True,
                      sharey=True, subplot_kws={'projection': map_proj},
                      aspect= (xrdata.longitude.size) / xrdata.latitude.size, size=3)
    figwidth = g.fig.get_figwidth() ; figheight = g.fig.get_figheight()


    if kwrgs['clevels'] == 'default':
        vmin = np.round(float(xrdata.min())-0.01,decimals=2) ; vmax = np.round(float(xrdata.max())+0.01,decimals=2)
        clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
    else:
        vmin=kwrgs['vmin']
        vmax=kwrgs['vmax']
        clevels = np.linspace(vmin,vmax,kwrgs['steps'])
    cmap = kwrgs['cmap']
    
    n_plots = xrdata[var].size
    for n_ax in np.arange(0,n_plots):
        ax = g.axes.flatten()[n_ax]
#        print(n_ax)
        plotdata = xrdata[n_ax]
        im = plotdata.plot.contourf(ax=ax, cmap=cmap,
                               transform=ccrs.PlateCarree(),
                               subplot_kws={'projection': map_proj},
                               levels=clevels, add_colorbar=False)
        ax.coastlines(color='grey', alpha=0.3)
        
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], ccrs.PlateCarree())
#        lons = [-5.8, -5.8, -5.5, -5.5]
#        lats = [50.27, 50.48, 50.48, 50.27]
        lons_sq = [-215, -215, -125, -125]
        lats_sq = [50, 19, 19, 50]
        ring = LinearRing(list(zip(lons_sq , lats_sq )))
        ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='green')
        if map_proj.proj4_params['proj'] in ['merc', 'Plat']:
            print(True)
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
            gl.xlabels_top = False;
    #        gl.xformatter = LONGITUDE_FORMATTER
            gl.ylabels_right = False;
            gl.xlabels_bottom = False
    #        gl.yformatter = LATITUDE_FORMATTER
        else:
            pass
        
    g.fig.text(0.5, 0.95, kwrgs['title'], fontsize=15, horizontalalignment='center')
    cbar_ax = g.fig.add_axes([0.25, (figheight/25)/n_plots, 
                                  0.5, (figheight/25)/n_plots])
    plt.colorbar(im, cax=cbar_ax, orientation='horizontal', 
                 label=xrdata.attrs['units'], extend='neither')
    g.fig.savefig(file_name ,dpi=250)
    #%%
    return