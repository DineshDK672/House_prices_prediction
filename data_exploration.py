from scipy.stats import norm
from scipy import stats
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (10.0, 8.0)

# Loading Dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Printing data info and null values
print(train.info())
print("\n", train.columns[train.isnull().any()])

# Finding the percentage of null values
miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)

# Visualizing null values percentage
miss = miss.to_frame()
miss.columns = ['count']
miss.index.set_names(['Name'], inplace=True)
print("\n", miss)

# sns.set(style="whitegrid", color_codes=True)
# sns.barplot(x='Name', y='count', data=miss)
# plt.xticks(rotation=90)
# plt.yticks(np.arange(0, 1, 0.1))
# plt.show()

# # Distribution of target variable
# print("The skewness of SalePrice is {}".format(train['SalePrice'].skew()))
# sns.displot(train['SalePrice'], kde=True)
# plt.show()


# # Performing log tansformation to adjust the distribution
# target = np.log(train['SalePrice'])
# print("The skewness of Sale Price after log is ", target.skew())
# sns.displot(target, kde=True)
# plt.show()

# Seperating numeric and categorical columns
numeric = train.select_dtypes(include=np.number)
cat = train.select_dtypes(exclude=np.number)
numeric.drop('Id', axis=1, inplace=True)

# # Correlation of Numeric variables
# corr = numeric.corr()
# sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
# plt.show()

# print("\n", corr['SalePrice'].sort_values(ascending=False)[:15])

# # OverallQual vs SalePrice
# pivot = numeric.pivot_table(
#     index="OverallQual", values='SalePrice', aggfunc='median')
# pivot.sort_index()
# pivot.plot(kind='bar')
# plt.show()

# # GrLivArea vs SalePrice
# sns.jointplot(x=numeric['GrLivArea'], y=numeric["SalePrice"])
# plt.show()

cat_data = [f for f in cat.columns]


def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = cat_data
    pvals = []
    for c in cat_data:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')


cat['SalePrice'] = train['SalePrice'].values
k = anova(cat)
k['disparity'] = np.log(1./k['pval'].values)
sns.barplot(data=k, x='features', y='disparity')
plt.xticks(rotation=90)
plt.show()
