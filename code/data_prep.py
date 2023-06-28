#!/usr/bin/env python
# coding: utf-8

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_id(name):
    """
    This function combines two columns to make unique ids for each property,
    to be used later for merging.
    """
    name['Major'] = name['Major'].astype(str).str.zfill(6)
    name['Minor'] = name['Minor'].astype(str).str.zfill(4)
    name['id'] = name['Major'].str.cat(name['Minor'])
    name.drop(columns=['Major', 'Minor'], inplace=True)
    return name


def original_prep(df, ods):
    """
    This function cleans and prepares the original dataset.
    """
    # Importing list of zipcodes from King County
    kings_zips = list(ods['Zip Code'])

    # Extracting the zipcode from the address column.
    df['zipcode'] = df['address'].apply(lambda x: int(x[-20:-15]))
    df = df[df['zipcode'].isin(kings_zips)]

    # Making sure id's are 10 characters long and dropping duplicate id's.
    df['id'] = df['id'].astype(str).str.zfill(10)
    df.drop_duplicates(subset=['id'], inplace=True)

    # Renaming elements in categorical variables to a binary 0 and 1, or a
    # numerical ranking.
    df['greenbelt'] = df['greenbelt'].map({'NO': 0, 'YES': 1})
    df['nuisance'] = df['nuisance'].map({'NO': 0, 'YES': 1})
    df['waterfront'] = df['waterfront'].map({'NO': 0, 'YES': 1})
    df['view'] = df['view'].map({
        'NONE': 0,
        'POOR': 1,
        'FAIR': 2,
        'AVERAGE': 3,
        'GOOD': 4,
        'EXCELLENT': 5
    })
    df['condition'] = df['condition'].map({
        'Poor': 1,
        'Fair': 2,
        'Average': 3,
        'Good': 4,
        'Very Good': 5
    })
    df['grade'] = df['grade'].map({
        '1 Cabin': 1,
        '2 Substandard': 2,
        '3 Poor': 3,
        '4 Low': 4,
        '5 Fair': 5,
        '6 Low Average': 6,
        '7 Average': 7,
        '8 Good': 8,
        '9 Better': 9,
        '10 Very Good': 10,
        '11 Excellent': 11,
        '12 Luxury': 12,
        '13 Mansion': 13
    })

    # Selecting columns to keep and setting the id as the index.
    keep = [
        'id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'floors', 'waterfront', 'greenbelt', 'nuisance', 'view', 'condition',
        'grade', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode'
    ]
    df_clean = df[keep].set_index('id', verify_integrity=True)
    return df_clean


def res_prep(res, ods):
    """
    This function cleans and prepares the residential dataset.
    """
    make_id(res)  # Making 'id' column.

    # Ensuring that all of the the zipcodes are in the proper format and filtering
    # for zipcodes only in King County.
    to_drop = []
    kings_zips = list(ods['Zip Code'])
    for i in res['ZipCode']:
        try:
            int(i[:5])
        except:
            if i in to_drop:
                pass
            else:
                to_drop.append(i)
    res.loc[res['ZipCode'].isin(to_drop), 'ZipCode'] = np.nan
    res.dropna(subset=['ZipCode'], inplace=True)
    res['zipcode'] = res['ZipCode'].apply(lambda x: int(str(x)[:5]))
    res.loc[~res['zipcode'].isin(kings_zips), 'zipcode'] = np.nan
    res.dropna(subset=['ZipCode'], inplace=True)

    # Dropping duplicate id's.
    res.drop_duplicates(subset=['id'], inplace=True, keep=False)

    # Renaming columns to match the original dataset.
    mapping = {
        'Stories': 'floors',
        'BldgGrade': 'grade',
        'SqFtTotLiving': 'sqft_living',
        'Bedrooms': 'bedrooms',
        'BathFullCount': 'bathrooms',
        'YrBuilt': 'yr_built',
        'Condition': 'condition',
        'SqFtTotBasement': 'sqft_basement',
        'YrRenovated': 'yr_renovated'
    }

    # Selecting columns to keep and setting the id as the index.
    keep = [
        'id', 'bathrooms', 'bedrooms', 'condition', 'floors', 'grade',
        'sqft_basement', 'sqft_living', 'yr_built', 'yr_renovated', 'zipcode'
    ]
    res_clean = res.rename(columns=mapping)[keep].set_index(
        'id', verify_integrity=True)
    return res_clean


def parcel_prep(parcel):
    """
    This function cleans and prepares the parcel dataset.
    """

    make_id(parcel)  # Making 'id' column.

    # Filtering data to only include properties labeled as 'Condominium' or 'Residential'.
    parcel = parcel[parcel['PropType'].isin(['K', 'R'])]

    # Combining the information from multiple view columns to create a 'view'
    # column which includes a view rating from 0 to 5.
    for i in [
            'MtRainier', 'Olympics', 'Cascades', 'Territorial',
            'SeattleSkyline', 'PugetSound', 'LakeWashington', 'LakeSammamish',
            'SmallLakeRiverCreek', 'OtherView'
    ]:
        parcel.loc[parcel[i] > 0, i] = 1
        parcel.loc[parcel[i] == 0, i] = 0
    parcel['total_views'] = parcel['MtRainier'] + parcel['Olympics'] + parcel[
        'Cascades'] + parcel['Territorial'] + parcel[
            'SeattleSkyline'] + parcel['PugetSound'] + parcel[
                'LakeWashington'] + parcel['LakeSammamish'] + parcel[
                    'SmallLakeRiverCreek'] + parcel['OtherView']
    parcel['view'] = parcel['total_views'].map({
        0: 0,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 5,
        6: 5,
        7: 5,
        8: 5
    })

    # Renaming elements in categorical variables to a binary 0 and 1
    parcel.loc[parcel['WfntLocation'] > 0, 'waterfront'] = 1
    parcel.loc[parcel['WfntLocation'] == 0, 'waterfront'] = 0
    parcel['nuisance'] = 0
    parcel.loc[parcel['PowerLines'] == 'Y', 'nuisance'] = 1
    parcel.loc[parcel['TrafficNoise'] > 0, 'nuisance'] = 1
    parcel.loc[parcel['AirportNoise'] != 0, 'nuisance'] = 1
    parcel.loc[parcel['OtherNuisances'] == 'Y', 'nuisance'] = 1
    parcel['greenbelt'] = parcel['AdjacentGreenbelt'].map({'N': 0, 'Y': 1})

    # Dropping duplicate id's.
    parcel.drop_duplicates(subset=['id'], inplace=True, keep=False)

    # Renaming column to match the original dataset.
    mapping = {'SqFtLot': 'sqft_lot'}

    # Selecting columns to keep and setting the id as the index.
    keep = ['id', 'greenbelt', 'nuisance', 'sqft_lot', 'view', 'waterfront']
    parcel_clean = parcel.rename(columns=mapping)[keep].set_index(
        'id', verify_integrity=True)
    return parcel_clean


def sales_prep(sales):
    """
    This function cleans and prepares the sales dataset.
    """
    # In this case, the sales data had many id's in an inproper format. This code
    # cleans and filter the ids
    sales['Major'] = sales['Major'].astype(str)
    sales['Minor'] = sales['Minor'].astype(str)
    sales['id'] = sales['Major'].str.cat(sales['Minor'])
    to_drop = []
    for i, j in enumerate(sales['id']):
        try:
            int(j)
        except:
            to_drop.append(j)
    sales.loc[sales['id'].isin(to_drop), 'id'] = np.nan
    sales.dropna(subset=['id'], inplace=True)

    # Converting and extracting the year from the date column.
    sales['date'] = pd.to_datetime(sales['DocumentDate'])
    sales['year'] = sales['date'].apply(lambda x: x.year)

    # Selecting and renaming the columns.
    keep = ['id', 'date', 'price']
    mapping = {'SalePrice': 'price'}

    cond1 = sales['id'] != 0

    # Filter to include condominiums, apartments, residential buildings, and mobile homes.
    cond2 = sales['PrincipalUse'].isin([2, 4, 6, 8])

    cond3 = sales['year'] > 2020
    sales2 = sales[cond1 & cond2 & cond3].rename(columns=mapping)[keep]

    # Only keeping the most recent sale of a property.
    sales3 = sales2.sort_values('date',
                                ascending=False).drop_duplicates(subset=['id'],
                                                                 keep='first')

    sales_clean = sales3.sort_values('id').set_index('id',
                                                     verify_integrity=True)
    return sales_clean


def ods_prep(ods):
    """
    This function cleans and prepares the open datasoft dataset.
    """
    # Selecting and renaming the columns.
    ods_clean = ods[['Zip Code', 'Population', 'Density']]
    ods_clean.rename(columns={
        'Zip Code': 'zipcode',
        'Population': 'population',
        'Density': 'density'
    },
                     inplace=True)
    return ods_clean


def merge_prep(sales_clean, orig_clean, res_clean, parcel_clean, ods_clean):
    """
    Merging, updating, cleaning, and filtering all of the datasets used in this project.
    """
    # Merging all of the above datasets.
    data_clean = sales_clean.join(orig_clean.iloc[:, 1:])
    data_clean.update(res_clean)
    data_clean.update(parcel_clean)
    data_clean = data_clean.reset_index().merge(ods_clean,
                                                how="left",
                                                on='zipcode').set_index(
                                                    'id',
                                                    verify_integrity=True)

    # Filtering and removing all missing values.
    data_clean.loc[data_clean['price'] <= 0, 'price'] = np.nan
    data_clean.loc[data_clean['sqft_living'] <= 0, 'sqft_living'] = np.nan
    data_clean.dropna(subset=['price', 'sqft_living', 'zipcode'], inplace=True)

    # Excluding outliers by selecting for the middle 95% price and sqft_living data.
    data_clean['price_nlog'] = (np.log(data_clean['price']) - np.log(
        data_clean['price']).mean()) / np.log(data_clean['price']).std()
    data_clean.loc[(data_clean['price_nlog'] > 2) |
                   (data_clean['price_nlog'] < -2), 'price_nlog'] = np.nan
    data_clean.dropna(subset=['price_nlog'], inplace=True)

    # Creating a column to calculate the last year construction was done on the property.
    data_clean['yr_last_construction'] = data_clean['yr_built']
    data_clean['yr_last_construction'].update(
        data_clean['yr_renovated'][data_clean['yr_renovated'] != 0])

    data_clean = data_clean[data_clean['bedrooms'].between(1, 6)]
    data_clean = data_clean[data_clean['bathrooms'].between(1, 6)]
    data_clean = data_clean[(data_clean['sqft_lot'] < 43560)]
    data_clean = data_clean[data_clean['sqft_basement'] < 4000]

    # Excluding zipcodes with few data points.
    zip_counts = data_clean['zipcode'].value_counts()
    low_zips = list(zip_counts[zip_counts <= 20].index)
    data_clean.loc[data_clean['zipcode'].isin(low_zips), 'zipcode'] = np.nan

    # Dropping any missing values.
    data_clean.dropna(subset=['price', 'sqft_living', 'zipcode'], inplace=True)

    data_clean = data_clean.astype(int)
    # Creating new columns with all numerical variables normalized.
    nums = [
        'price', 'sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot',
        'sqft_basement', 'population', 'density', 'view', 'grade', 'floors'
    ]

    for i in nums:
        data_clean[i +
                   '_norm'] = (data_clean[i] -
                               data_clean[i].mean()) / (data_clean[i].std())
    data_clean = data_clean[[
        'price', 'sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot',
        'sqft_basement', 'floors', 'grade', 'population', 'density', 'view',
        'waterfront', 'greenbelt', 'nuisance', 'condition', 'yr_built',
        'yr_renovated', 'yr_last_construction', 'zipcode', 'sqft_living_norm',
        'bedrooms_norm', 'bathrooms_norm', 'sqft_lot_norm',
        'sqft_basement_norm', 'floors_norm', 'grade_norm', 'population_norm',
        'density_norm', 'view_norm'
    ]]
    return data_clean
