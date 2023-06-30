import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sns.set_theme(context='talk', style='whitegrid', palette='tab10')

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
    fig.tight_layout()
    plt.savefig('./images/box.png')
    plt.show()


def heat_viz(df):
    heat = [
        'sqft_living', 'grade', 'bathrooms', 'bedrooms', 'view',
        'sqft_basement', 'waterfront', 'greenbelt', 'zipcode'
    ]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(round(df[heat].corr().abs(), 2),
                annot=True,
                ax=ax,
                annot_kws={'fontsize': 'small'})
    fig.tight_layout()
    plt.savefig('./images/heat.png')
    plt.show()


def pair_viz(df):
    pair = [
        'price', 'sqft_living', 'sqft_lot', 'sqft_basement', 'population',
        'density'
    ]
    sns.pairplot(data=df[pair],
                 kind='reg',
                 plot_kws={
                     'line_kws': {
                         'color': 'red'
                     },
                     'scatter_kws': {
                         'alpha': 0.1
                     }
                 })
    plt.savefig('./images/pair.png')
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
    plt.savefig('./images/part.png')
    plt.show()


def price_viz(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data=df, x='price', ax=ax, bins='auto')
    ax.set(title='Distribution of Sales Price',
           ylabel=('Count'),
           xlabel=('Sales Price (in millions)'))
    ax.xaxis.set_major_formatter(lambda x, pos: f'${round(x/1000000,1)}')
    plt.axvline(x=df['price'].mean(),
                ls='--',
                color='purple',
                label='Mean Sale Price')
    fig.tight_layout()
    ax.legend()
    plt.savefig('./images/price.png')
    plt.show()


def qq_viz(df, results):
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.graphics.qqplot(results.resid,
                       dist=stats.norm,
                       line='45',
                       fit=True,
                       ax=ax)
    fig.tight_layout()
    plt.savefig('./images/qq.png')
    plt.show()


def resid_vis(df, results):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(df["price"], results.resid, alpha=0.1)
    plt.axhline(y=0, color="black")
    ax.set_xlabel("price")
    ax.set_ylabel("residuals")
    fig.tight_layout()
    plt.savefig('./images/resid.png')
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
    plt.savefig('./images/sqft.png')
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
    plt.savefig('./images/water.png')
    plt.show()


def zip_viz(df, results):
    zips = results.params[3:]
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
    plt.savefig('./images/zip.png')
    plt.show()


def zip_coef_viz(df, results):
    zips = results.params[3:]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.histplot(zips, bins=20, ax=ax, color='tab:green')
    ax.set(title="Zip Code's Effect on Sale price",
           ylabel=('Frequency of Zip Code'),
           xlabel=('Coefficient (in millions)'))
    ax.xaxis.set_major_formatter(lambda x, pos: f'$ {round(x/1000000,2)}')
    ax.tick_params(axis='x', rotation=30)
    plt.yticks(ticks=np.arange(16, step=2, dtype=int))
    ax.patches[0].set_facecolor('tab:red')
    ax.patches[1].set_facecolor('tab:red')
    fig.tight_layout()
    plt.savefig('./images/zip_coef.png')
    plt.show()