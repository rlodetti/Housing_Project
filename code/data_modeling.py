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


def pair_viz(df):
    pair = [
        'price', 'sqft_living', 'sqft_lot', 'sqft_basement', 'population',
        'density'
    ]
    sns.pairplot(data=df[pair],
                 kind='reg',
                 plot_kws={'line_kws': {
                     'color': 'red'
                 }})
    plt.show()


def box_viz(df):
    box = [
        'bedrooms', 'bathrooms', 'floors', 'grade', 'view', 'condition',
        'waterfront', 'greenbelt', 'nuisance'
    ]
    fig, ax = plt.subplots(3, 3, figsize=[20, 20])
    for i, j in enumerate(box):
        col = i % 3
        row = i // 3
        axis = ax[row][col]
        sns.boxplot(data=df, y='price', x=j, ax=axis)
    plt.show()


def heat_viz(df):
    heat = [
        'sqft_living', 'grade', 'bathrooms', 'bedrooms', 'view',
        'sqft_basement', 'waterfront', 'greenbelt', 'zipcode'
    ]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(df[heat].corr().abs(), annot=True, ax=ax)
    plt.show()

def sqft_viz(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=df,
                y='price',
                x='sqft_living',
                scatter_kws={'alpha': 0.1},
                line_kws={'color': 'red'},
                ax=ax)
    ax.set(title='Comparing Sale Price to Square Footage',
           xlim=(0, df['sqft_living'].max() - 500),
           ylim=(0, df['price'].max() + 250000),
           ylabel=('Sales Price (in millions)'),
           xlabel=('Square Footage of Living Space'))
    ax.yaxis.set_major_formatter(lambda x, pos: f'${round(x/1000000,1)}')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    plt.show()


def water_viz(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, y='price', x='waterfront', ax=ax)
    ax.set(title='Comparing Sale Price to Proximity to Waterfront',
           ylabel=('Sales Price (in millions)'),
           xlabel=(''),
           xticklabels=(['Not on Waterfront', 'On Waterfront']))
    ax.yaxis.set_major_formatter(lambda x, pos: f'${round(x/1000000,1)}')
    plt.axhline(y=df['price'].mean(),
                ls='--',
                color='purple',
                label='Mean Sale Price')
    ax.legend()
    fig.tight_layout()
    plt.show()


def zip_viz(df):
    final_results, res_df = final_res(df)
    zips = final_results.params[3:]
    zips.describe()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=df.groupby('zipcode').median()['price'], ax=ax, bins=15)
    ax.set(title='Median Sales Price by Zip Code',
           ylabel=('Number of Zip Codes'),
           xlabel=('Sales Price (in millions)'))
    ax.xaxis.set_major_formatter(lambda x, pos: f'${round(x/1000000,1)}')
    plt.yticks(ticks=np.arange(21, step=2, dtype=int))
    plt.axvline(x=df['price'].mean(),
                ls='--',
                color='purple',
                label='Mean Sale Price')
    ax.legend()
    fig.tight_layout()
    plt.show()


def part_viz(df):
    features = ['price', 'sqft_living_norm', 'waterfront', 'zipcode']
    X = pd.get_dummies(df[features]).drop(columns=[
        'zipcode_98059', 'zipcode_98019', 'zipcode_98045', 'zipcode_98106',
        'zipcode_98108', 'zipcode_98146', 'zipcode_98166', 'zipcode_98014',
        'zipcode_98051', 'zipcode_98070'
    ])
    cols = X.columns[2:]
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.plot_partregress(endog=X['price'],
                                 exog_i=X['sqft_living_norm'],
                                 exog_others=X[cols],
                                 obs_labels=False,
                                 alpha=0.1,
                                 ax=ax)
    ax.set(title='Partial Regression Plot for Square Feet',
           ylabel=('Sales Price (in millions)'),
           xlabel=('Square Footage of Living Space'))
    x_ticks = (np.arange(9000, step=1000, dtype=int) -
               df['sqft_living'].mean()) / df['sqft_living'].std()
    plt.xticks(ticks=x_ticks)
    ax.yaxis.set_major_formatter(lambda x, pos: f'${round(x/1000000,1)}')
    ax.xaxis.set_major_formatter(lambda x, pos: int(x * df['sqft_living'].std(
    ) + df['sqft_living'].mean()))

    ax.tick_params(axis='x', rotation=30)

    fig.tight_layout()
    plt.show()
    
    
def zip_coef_viz(df,results):
    zips = results.params[3:]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.histplot(zips, bins=20, ax=ax)
    ax.set(title="Zip Code's Effect on Sale price",
           ylabel=('Frequency of Zip Code'),
           xlabel=('Coefficient (in millions)'))
    ax.xaxis.set_major_formatter(lambda x, pos: f'$ {round(x/1000000,2)}')
    ax.tick_params(axis='x', rotation=30)
    plt.yticks(ticks=np.arange(16, step=2, dtype=int))
    fig.tight_layout()
    plt.show()
    
def resid_vis(df,results):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    plt.scatter(df["price"], results.resid, alpha=0.2)
    plt.axhline(y=0, color="black")
    ax.set_xlabel("price")
    ax.set_ylabel("residuals")
    plt.show();
              
def qq_viz(df,results):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sm.graphics.qqplot(results.resid, dist=stats.norm, line='45', fit=True, ax=ax)
    plt.show();
           