import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# Loading Dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Removing outliers in 'GrLivArea'
train.drop(train[train['GrLivArea'] > 4000].index, inplace=True)

# Imputing using mode
test.loc[666, 'GarageQual'] = test['GarageQual'].mode()[0]
test.loc[666, 'GarageCond'] = test['GarageCond'].mode()[0]
test.loc[666, 'GarageFinish'] = test['GarageFinish'].mode()[0]
test.loc[666, 'GarageYrBlt'] = np.nanmedian(test['GarageYrBlt'])

test.loc[1116, 'GarageType'] = np.nan

# Encoding categorical variables
le = LabelEncoder()


def LEncode(data, var, fill_na=None):
    if fill_na is not None:
        data[var] = data[var].fillna(fill_na)
    le.fit(data[var])
    data[var] = le.transform(data[var])
    return data


# Combine the data set
alldata = pd.concat([train, test], ignore_index=True)

# Impute 'LotFrontage' by median of 'Neighborhood'
lot_frontage_by_neighborhood = train['LotFrontage'].groupby(
    train['Neighborhood'])

for key, group in lot_frontage_by_neighborhood:
    idx = (alldata['Neighborhood'] == key) & (alldata['LotFrontage'].isnull())
    alldata.loc[idx, 'LotFrontage'] = group.median()

# Imputing numeric variables
alldata["MasVnrArea"] = alldata["MasVnrArea"].fillna(0)
alldata["BsmtFinSF1"] = alldata["BsmtFinSF1"].fillna(0)
alldata["BsmtFinSF2"] = alldata["BsmtFinSF2"].fillna(0)
alldata["BsmtUnfSF"] = alldata["BsmtUnfSF"].fillna(0)
alldata["TotalBsmtSF"] = alldata["TotalBsmtSF"].fillna(0)
alldata["GarageArea"] = alldata["GarageArea"].fillna(0)
alldata["BsmtFullBath"] = alldata["BsmtFullBath"].fillna(0)
alldata["BsmtHalfBath"] = alldata["BsmtHalfBath"].fillna(0)
alldata["GarageCars"] = alldata["GarageCars"].fillna(0)
alldata["GarageYrBlt"] = alldata["GarageYrBlt"].fillna(0.0)
alldata["PoolArea"] = alldata["PoolArea"].fillna(0)

# Converting categorical variables into ordinal values
qual_dict = {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
name = np.array(['ExterQual', 'PoolQC', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond'])


for i in name:
    alldata[i] = alldata[i].map(qual_dict).astype(int)

alldata["BsmtExposure"] = alldata["BsmtExposure"].map(
    {np.nan: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

bsmt_fin_dict = {np.nan: 0, "Unf": 1, "LwQ": 2,
                 "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
alldata["BsmtFinType1"] = alldata["BsmtFinType1"].map(
    bsmt_fin_dict).astype(int)
alldata["BsmtFinType2"] = alldata["BsmtFinType2"].map(
    bsmt_fin_dict).astype(int)
alldata["Functional"] = alldata["Functional"].map(
    {np.nan: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

alldata["GarageFinish"] = alldata["GarageFinish"].map(
    {np.nan: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)
alldata["Fence"] = alldata["Fence"].map(
    {np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)

# Encoding data
alldata["CentralAir"] = (alldata["CentralAir"] == "Y") * 1.0
varst = np.array(['MSSubClass', 'LotConfig', 'Neighborhood', 'Condition1',
                 'BldgType', 'HouseStyle', 'RoofStyle', 'Foundation', 'SaleCondition'])

for x in varst:
    LEncode(alldata, x)

# encode variables and impute missing values
alldata = LEncode(alldata, "Exterior1st", "Other")
alldata = LEncode(alldata, "Exterior2nd", "Other")
alldata = LEncode(alldata, "MasVnrType", "None")
alldata = LEncode(alldata, "SaleType", "Oth")
alldata = LEncode(alldata, "MSZoning", "RL")

# Creating new variable (1 or 0) based on irregular count levels
# The level with highest count is kept as 1 and rest as 0
alldata["IsRegularLotShape"] = (alldata["LotShape"] == "Reg") * 1
alldata["IsLandLevel"] = (alldata["LandContour"] == "Lvl") * 1
alldata["IsLandSlopeGentle"] = (alldata["LandSlope"] == "Gtl") * 1
alldata["IsElectricalSBrkr"] = (alldata["Electrical"] == "SBrkr") * 1
alldata["IsGarageDetached"] = (alldata["GarageType"] == "Detchd") * 1
alldata["IsPavedDrive"] = (alldata["PavedDrive"] == "Y") * 1
alldata["HasShed"] = (alldata["MiscFeature"] == "Shed") * 1
alldata["Remodeled"] = (alldata["YearRemodAdd"] != alldata["YearBuilt"]) * 1

# Did the modeling happen during the sale year?
alldata["RecentRemodel"] = (alldata["YearRemodAdd"] == alldata["YrSold"]) * 1

# Was this house sold in the year it was built?
alldata["VeryNewHouse"] = (alldata["YearBuilt"] == alldata["YrSold"]) * 1
alldata["Has2ndFloor"] = (alldata["2ndFlrSF"] == 0) * 1
alldata["HasMasVnr"] = (alldata["MasVnrArea"] == 0) * 1
alldata["HasWoodDeck"] = (alldata["WoodDeckSF"] == 0) * 1
alldata["HasOpenPorch"] = (alldata["OpenPorchSF"] == 0) * 1
alldata["HasEnclosedPorch"] = (alldata["EnclosedPorch"] == 0) * 1
alldata["Has3SsnPorch"] = (alldata["3SsnPorch"] == 0) * 1
alldata["HasScreenPorch"] = (alldata["ScreenPorch"] == 0) * 1

# Calculating total area using all area columns
area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
             'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea']

alldata["TotalArea"] = alldata[area_cols].sum(axis=1)
alldata["TotalArea1st2nd"] = alldata["1stFlrSF"] + alldata["2ndFlrSF"]
alldata["Age"] = 2010 - alldata["YearBuilt"]
alldata["TimeSinceSold"] = 2010 - alldata["YrSold"]
alldata["SeasonSold"] = alldata["MoSold"].map(
    {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}).astype(int)
alldata["YearsSinceRemodel"] = alldata["YrSold"] - alldata["YearRemodAdd"]

# Setting levels with high count as 1 and the rest as 0
alldata["HighSeason"] = alldata["MoSold"].replace(
    {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
alldata["NewerDwelling"] = alldata["MSSubClass"].replace(
    {20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0, 90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

# Binning 'Neighbourhood' based on median sale prices
neighborhood_map = {"MeadowV": 0, "IDOTRR": 0, "BrDale": 0, "OldTown": 0, "Edwards": 0, "BrkSide": 0, "Sawyer": 1, "Blueste": 1, "SWISU": 1, "NAmes": 1, "NPkVill": 1, "Mitchel": 1,
                    "SawyerW": 2, "Gilbert": 2, "NWAmes": 2, "Blmngtn": 2, "CollgCr": 2, "ClearCr": 2, "Crawfor": 2, "Veenker": 3, "Somerst": 3, "Timber": 3, "StoneBr": 4, "NoRidge": 4, "NridgHt": 4}

alldata['NeighborhoodBin'] = alldata['Neighborhood'].map(neighborhood_map)

# Split the data into train and test
train_new = alldata[alldata['SalePrice'].notnull()]
test_new = alldata[alldata['SalePrice'].isnull()]


# Transform numeric features to remove skewness
numeric_features = [
    f for f in train_new.columns if train_new[f].dtype != object]


skewed = train_new[numeric_features].skew()
skewed = skewed[skewed > 0.75]
skewed = skewed.index

train_new[skewed] = np.log1p(train_new[skewed])
test_new[skewed] = np.log1p(test_new[skewed])
del test_new['SalePrice']

# Standardizing numeric variables
scaler = StandardScaler()
scaler.fit(train_new[numeric_features])
scaled = scaler.transform(train_new[numeric_features])

for i, col in enumerate(numeric_features):
    train_new.loc[:, col] = scaled[:, i]

numeric_features.remove('SalePrice')
scaled = scaler.fit_transform(test_new[numeric_features])

for i, col in enumerate(numeric_features):
    test_new.loc[:, col] = scaled[:, i]


# One-hot encoding for categorical features


def onehot(onehot_df, df, column_name, fill_na):
    onehot_df[column_name] = df[column_name]
    if fill_na is not None:
        onehot_df[column_name].fillna(fill_na, inplace=True)

    dummies = pd.get_dummies(
        onehot_df[column_name], prefix="_"+column_name, dtype=float)
    onehot_df = onehot_df.join(dummies)
    onehot_df = onehot_df.drop([column_name], axis=1)
    return onehot_df


def munge_onehot(df):
    onehot_df = pd.DataFrame(index=df.index)

    onehot_df = onehot(onehot_df, df, "MSSubClass", None)
    onehot_df = onehot(onehot_df, df, "MSZoning", "RL")
    onehot_df = onehot(onehot_df, df, "LotConfig", None)
    onehot_df = onehot(onehot_df, df, "Neighborhood", None)
    onehot_df = onehot(onehot_df, df, "Condition1", None)
    onehot_df = onehot(onehot_df, df, "BldgType", None)
    onehot_df = onehot(onehot_df, df, "HouseStyle", None)
    onehot_df = onehot(onehot_df, df, "RoofStyle", None)
    onehot_df = onehot(onehot_df, df, "Exterior1st", "VinylSd")
    onehot_df = onehot(onehot_df, df, "Exterior2nd", "VinylSd")
    onehot_df = onehot(onehot_df, df, "Foundation", None)
    onehot_df = onehot(onehot_df, df, "SaleType", "WD")
    onehot_df = onehot(onehot_df, df, "SaleCondition", "Normal")

    # Fill in missing MasVnrType for rows that do have a MasVnrArea.
    temp_df = df[["MasVnrType", "MasVnrArea"]].copy()
    idx = (df["MasVnrArea"] != 0) & (
        (df["MasVnrType"] == "None") | (df["MasVnrType"].isnull()))
    temp_df.loc[idx, "MasVnrType"] = "BrkFace"
    onehot_df = onehot(onehot_df, temp_df, "MasVnrType", "None")

    onehot_df = onehot(onehot_df, df, "LotShape", None)
    onehot_df = onehot(onehot_df, df, "LandContour", None)
    onehot_df = onehot(onehot_df, df, "LandSlope", None)
    onehot_df = onehot(onehot_df, df, "Electrical", "SBrkr")
    onehot_df = onehot(onehot_df, df, "GarageType", "None")
    onehot_df = onehot(onehot_df, df, "PavedDrive", None)
    onehot_df = onehot(onehot_df, df, "MiscFeature", "None")
    onehot_df = onehot(onehot_df, df, "Street", None)
    onehot_df = onehot(onehot_df, df, "Alley", "None")
    onehot_df = onehot(onehot_df, df, "Condition2", None)
    onehot_df = onehot(onehot_df, df, "RoofMatl", None)
    onehot_df = onehot(onehot_df, df, "Heating", None)

    # we'll have these as numerical variables too
    onehot_df = onehot(onehot_df, df, "ExterQual", "None")
    onehot_df = onehot(onehot_df, df, "ExterCond", "None")
    onehot_df = onehot(onehot_df, df, "BsmtQual", "None")
    onehot_df = onehot(onehot_df, df, "BsmtCond", "None")
    onehot_df = onehot(onehot_df, df, "HeatingQC", "None")
    onehot_df = onehot(onehot_df, df, "KitchenQual", "TA")
    onehot_df = onehot(onehot_df, df, "FireplaceQu", "None")
    onehot_df = onehot(onehot_df, df, "GarageQual", "None")
    onehot_df = onehot(onehot_df, df, "GarageCond", "None")
    onehot_df = onehot(onehot_df, df, "PoolQC", "None")
    onehot_df = onehot(onehot_df, df, "BsmtExposure", "None")
    onehot_df = onehot(onehot_df, df, "BsmtFinType1", "None")
    onehot_df = onehot(onehot_df, df, "BsmtFinType2", "None")
    onehot_df = onehot(onehot_df, df, "Functional", "Typ")
    onehot_df = onehot(onehot_df, df, "GarageFinish", "None")
    onehot_df = onehot(onehot_df, df, "Fence", "None")
    onehot_df = onehot(onehot_df, df, "MoSold", None)

    # Divide  the years between 1871 and 2010 into slices of 20 years
    year_map = pd.concat(pd.Series(
        "YearBin" + str(i+1), index=range(1871+i*20, 1891+i*20)) for i in range(0, 7))
    yearbin_df = pd.DataFrame(index=df.index)
    yearbin_df["GarageYrBltBin"] = df.GarageYrBlt.map(year_map)
    yearbin_df["GarageYrBltBin"].fillna("NoGarage", inplace=True)
    yearbin_df["YearBuiltBin"] = df.YearBuilt.map(year_map)
    yearbin_df["YearRemodAddBin"] = df.YearRemodAdd.map(year_map)

    onehot_df = onehot(onehot_df, yearbin_df, "GarageYrBltBin", None)
    onehot_df = onehot(onehot_df, yearbin_df, "YearBuiltBin", None)
    onehot_df = onehot(onehot_df, yearbin_df, "YearRemodAddBin", None)
    return onehot_df


# Create one-hot features
onehot_df = munge_onehot(train)

neighborhood_train = pd.DataFrame(index=train_new.shape)
neighborhood_train['NeighborhoodBin'] = train_new['NeighborhoodBin']
neighborhood_test = pd.DataFrame(index=test_new.shape)
neighborhood_test['NeighborhoodBin'] = test_new['NeighborhoodBin']

onehot_df = onehot(onehot_df, neighborhood_train, 'NeighborhoodBin', None)

# Adding one hot features to train
train_new = train_new.join(onehot_df)

# Adding one hot features to test
onehot_df_te = munge_onehot(test)
onehot_df_te = onehot(onehot_df_te, neighborhood_test, "NeighborhoodBin", None)
test_new = test_new.join(onehot_df_te)

print(train_new.shape, test_new.shape)

# Dropping some columns to standardize columns between test and train data
cols = test_new.columns.tolist()
cols_to_keep = [col for col in cols if col in train_new.columns]
test_new = test_new[cols_to_keep]
cols_to_keep += ['SalePrice']
train_new = train_new[cols_to_keep]
print(train_new.shape, test_new.shape)

# Transforming target variable
label_df = pd.DataFrame(index=train_new.index, columns=['SalePrice'])
label_df['SalePrice'] = np.log(train['SalePrice'])
