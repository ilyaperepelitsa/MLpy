import os
import tarfile
import pandas as pd
import urllib
import numpy as np
from sklearn import model_selection

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# download the tgz file
fetch_housing_data()

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()
housing.info()

housing.select_dtypes(include = ["object"]).value_counts()
pd.value_counts(housing.select_dtypes(include = ["object"]).values.flatten())


housing.describe()

housing["ocean_proximity"].value_counts()


import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize = (20, 15))
plt.show()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")


#### so the author is super anal about preserving the datasets
# import hashlib
#
# def test_set_check(identifier, test_ratio, hash):
#     return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
#
# def split_train_test_by_id(data, test_ratio, id_column, hash = hashlib.md5):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
#     return data.loc[~in_test_set], data.loc[in_test_set]
#
# housing_with_id = housing.reset_index()
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
len(train_set)
len(test_set)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].describe()
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)
housing["income_cat"].describe()

## now split with sklearn

split = model_selection.StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
split
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#check whether stratified split worked
housing["income_cat"].value_counts() / len(housing)

# now remove the artificial column
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis = 1, inplace = True)


### play with a copy
housing = strat_train_set.copy()

housing.plot(kind = "scatter", x = "longitude", y = "latitude")
plt.show()
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)
plt.show()


housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.4,
    s = housing["population"] / 100, label = "population",
    c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True,)
# plt.legend()
plt.show()


##### EXPLORE CORRELATIONS
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)
corr_matrix



### plot the scatter matrix thing with a few attributes

from pandas.tools.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize = (12, 8))
plt.show()

housing.plot(kind = "scatter", x = "median_income", y = "median_house_value", alpha = 0.1)
plt.show()


### creatig new variables
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"]

### prep before writing transformation functions
housing = strat_train_set.drop("median_house_value", axis = 1)
housing
housing_labels = strat_train_set["median_house_value"].copy()


### NA - if using median or mean - save it to reuse in the model when online

## using sklearn imputer class_
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = "median")

# since median - create a copy without categorical variables
housing_num = housing.drop("ocean_proximity", axis = 1)

imputer.fit(housing_num)


imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)
## putting that NP thing back into pandas df
housing_tr = pd.DataFrame(X, columns = housing_num.columns)



# converting a categorical thing to numeric
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded
print(encoder.classes_)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
housing_cat_1hot

## if a dense np array needed:
housing_cat_1hot.toarray()

#do same in one line
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot
#LabelBinarizer(sparse_output = True) for sparse matrix








#### CUSTOM TRANSFORMER

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
attr_adder
housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs





from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer", Imputer(strategy = "median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler())])


### PUttig together quantitative and categorical variables into one Pipeline

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import DataFrameSelector

from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values



num_attribs = list(housing_num)
num_attribs
cat_attribs = ["ocean_proximity"]
cat_attribs

num_pipeline = Pipeline([
    ("selector", DataFrameSelector(num_attribs)),
    ("imputer", Imputer(strategy = "median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler())])

cat_pipeline = Pipeline([
    ("selector", DataFrameSelector(cat_attribs)),
    ("label_binarizer", LabelBinarizer())])

full_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared.shape






from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_data

some_labels = housing_labels.iloc[:5]
some_labels
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_mse
lin_rmse = np.sqrt(lin_mse)
lin_rmse


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_mse
tree_rmse = np.sqrt(tree_mse)
tree_rmse


### CROSS VALIDATE
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                            scoring = "neg_mean_squared_error", cv = 10)

scores
rmse_scores = np.sqrt(-scores)
rmse_scores


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

display_scores(rmse_scores)


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                            scoring = "neg_mean_squared_error", cv = 10)

lin_rmse_scores = np.sqrt(-lin_scores)
lin_rmse_scores
display_scores(lin_rmse_scores)



from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                            scoring = "neg_mean_squared_error", cv = 10)

forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


### saving models

from sklearn.externals import joblib

joblib.dump(forest_reg, "/Users/ilyaperepelitsa/quant/MLpy/tensor1/some_forest.pkl")

my_model_loaded = joblib.load("/Users/ilyaperepelitsa/quant/MLpy/tensor1/some_forest.pkl")
my_model_loaded



### HYPERPARAMETERS
from sklearn.model_selection import GridSearchCV

param_grid = [
    {"n_estimators" : [3, 10, 30], "max_features" : [2, 4, 6, 8]},
    {"bootstrap" : [False], "n_estimators" : [3, 10], "max_features" : [2, 3, 4]}]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv = 5,
                            scoring = "neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)

# get the best parameters
grid_search.best_params_


### getting it directlu???
grid_search.best_estimator_


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

pd.DataFrame(grid_search.cv_results_)

### RADNOMIZED
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

params_distribs = {
    "n_estimators" : randint(low = 1, high = 200),
    "max_features" : randint(low = 1, high = 8)}

forest_reg = RandomForestRegressor(random_state = 42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions = params_distribs,
                                n_iter = 10, cv = 5, scoring = "neg_mean_squared_error", random_state = 42)

rnd_search.fit(housing_prepared, housing_labels)

cvres2 = rnd_search.cv_results_
for mean_score, params in zip(cvres2["mean_test_score"], cvres2["params"]):
    print(np.sqrt(-mean_score), params)

# feature importance thingy
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


#display them next to the attribute names

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
cat_one_hot_attribs

attributes = num_attribs + extra_attribs + cat_one_hot_attribs
attributes

sorted(zip(feature_importances, attributes), reverse = True)


### TEST set

final_model = grid_search.best_estimator_
final_model

X_test = strat_test_set.drop("median_house_value", axis = 1)
y_test = strat_test_set["median_house_value"]
y_test = y_test
X_test
X_test_prepared = full_pipeline.transform(X_test)
X_test_prepared

final_predictions = final_model.predict(X_test_prepared)
final_predictions
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
