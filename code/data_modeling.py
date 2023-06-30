#!/usr/bin/env python
# coding: utf-8

# Importing libraries and cleaned data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats


def baseline(df):
    y = df['price']
    X = df['sqft_living']
    model = sm.OLS(y, sm.add_constant(X))
    results = model.fit()
    res_dic = {}
    num_features = len(results.params)
    res_dic['num_features'] = num_features
    res_dic['r2_adj'] = str(round(results.rsquared_adj * 100, 2)) + '%'
    res_dic['f_pvalue'] = round(results.f_pvalue, 3)
    mae = str(int(results.resid.abs().sum() / len(y)))
    res_dic['MAE'] = '$' + mae[:3] + ',' + mae[3:]
    res_dic['large_pvals'] = (results.pvalues.apply(lambda x: round(x, 3))
                              > 0.01).sum()
    res_dic['cond_num'] = round(results.condition_number, 2)
    pd.DataFrame(res_dic, index=[0])
    return results, pd.DataFrame(res_dic, index=[0])


def final_res(df):
    features = ['sqft_living_norm', 'waterfront', 'zipcode']
    y = df['price']
    X = pd.get_dummies(df[features]).drop(columns=[
        'zipcode_98059', 'zipcode_98019', 'zipcode_98045', 'zipcode_98106',
        'zipcode_98108', 'zipcode_98146', 'zipcode_98166', 'zipcode_98014',
        'zipcode_98051', 'zipcode_98070'
    ])
    model = sm.OLS(y, sm.add_constant(X))
    results = model.fit()
    res_dic = {}
    num_features = len(results.params)
    res_dic['num_features'] = num_features
    res_dic['r2_adj'] = str(round(results.rsquared_adj * 100, 2)) + '%'
    res_dic['f_pvalue'] = round(results.f_pvalue, 3)
    mae = str(int(results.resid.abs().sum() / len(y)))
    res_dic['MAE'] = '$' + mae[:3] + ',' + mae[3:]
    res_dic['large_pvals'] = (results.pvalues.apply(lambda x: round(x, 3))
                              > 0.01).sum()
    res_dic['cond_num'] = round(results.condition_number, 2)
    pd.DataFrame(res_dic, index=[0])
    return results, pd.DataFrame(res_dic, index=[0])


def RFE_df(df):
    """
    This function goes runs MLR models and removes the any p-values larger
    than 0.01. If there are none, it removes the feature with the smallest 
    absolute coefficient. It then adds a row and creates a dataframe with 
    relevent metrics. 
    """
    pd.set_option('display.max_rows', None)
    res_df = pd.DataFrame({})
    to_drop = ['zipcode_98059']
    features = [
        'sqft_living_norm', 'bathrooms_norm', 'bedrooms_norm', 'view_norm',
        'sqft_basement_norm', 'greenbelt', 'waterfront', 'zipcode'
    ]
    dropped = None
    num_features = 3
    while num_features > 2:
        y = df['price']
        X = pd.get_dummies(df[features]).drop(columns=to_drop)
        model = sm.OLS(y, sm.add_constant(X))
        results = model.fit()
        res_dic = {}
        num_features = len(results.params)
        res_dic['num_features'] = num_features
        res_dic['r2_adj'] = str(round(results.rsquared_adj * 100, 2)) + '%'
        res_dic['f_pvalue'] = round(results.f_pvalue, 3)
        mae = str(int(results.resid.abs().sum() / len(y)))
        res_dic['MAE'] = '$' + mae[:3] + ',' + mae[3:]
        res_dic['large_pvals'] = (results.pvalues.apply(lambda x: round(x, 3))
                                  > 0.01).sum()
        res_dic['cond_num'] = round(results.condition_number, 2)
        res_dic['dropped'] = dropped
        pvals = results.pvalues.apply(lambda x: round(x, 3))
        large_pvals = list(pvals[pvals > 0.01].index)

        if len(large_pvals) == 0:
            feat = results.params.abs().sort_values().index[0]
            if feat == 'const':
                dropped = results.params.abs().sort_values().index[1]
            else:
                dropped = results.params.abs().sort_values().index[0]
            to_drop.append(dropped)
        elif len(large_pvals) > 1:
            dropped = ""
            for i in large_pvals[:-1]:
                dropped += i + ', '
            dropped += large_pvals[-1]
            for i in large_pvals:
                to_drop.append(i)
        else:
            dropped = large_pvals[0]
            to_drop.append(dropped)
        res_df = res_df.append(pd.DataFrame(res_dic, index=[0]),
                               ignore_index=True)
    return res_df


def corr_price(df):
    feats = [
        'price', 'sqft_living', 'sqft_lot', 'sqft_basement', 'population',
        'density', 'bedrooms', 'bathrooms', 'floors', 'grade', 'view',
        'condition', 'waterfront', 'greenbelt', 'nuisance', 'zipcode',
        'yr_built', 'yr_last_construction'
    ]
    return df[feats].corr().abs()['price'].sort_values(ascending=False)