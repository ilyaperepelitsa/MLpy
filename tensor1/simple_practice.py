import os
import tarfile
import pandas as pd
import urllib
import numpy as np
from sklearn import model_selection
from pandas import DataFrame
import matplotlib.pyplot as plt
path = "/Users/ilyaperepelitsa/quant/MLpy/tensor1"
csv_path = os.path.join(path, "HR_comma_sep.csv")
hrdata = pd.read_csv(csv_path)

hrdata.head()
hrdata.info()
## transform
# hrdata["Work_accident"] = hrdata["Work_accident"].astype(object)
# hrdata["Work_accident"] = hrdata["Work_accident"].map({1: "accident", 0: "no_accident"})
# hrdata["left"] = hrdata["left"].astype(object)
# hrdata["left"] = hrdata["left"].map({1: "left", 0: "stayed"})
# hrdata["promotion_last_5years"] = hrdata["promotion_last_5years"].astype(object)
# hrdata["promotion_last_5years"] = hrdata["promotion_last_5years"].map({1: "promoted", 0: "not_promoted"})




### describe
# hrdata["Work_accident"].astype(object).value_counts()
# hrdata["Work_accident"].astype(object).value_counts()
# hrdata["left"].astype(object).value_counts()
# hrdata["left"].astype(object).value_counts()
# hrdata["promotion_last_5years"].astype(object).value_counts()
# hrdata["promotion_last_5years"].astype(object).value_counts()
# hrdata["number_project"].describe()
# hrdata.describe()



def print_categories(df, inc):
    for i in range(DataFrame(df.select_dtypes(include = [inc])).shape[1]):
        if inc in ["object", "categorical", "uint8", "int64"]:
            print(DataFrame(df.select_dtypes(include = [inc])).iloc[:,i].value_counts())
            print("\n\n")
        else:
            print(DataFrame(df.select_dtypes(include = [inc])).iloc[:,i].describe())
            print("\n\n")

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


print_categories(hrdata, "object")
hrdata.dtypes
hrdata.head()


# train_set, test_set = split_train_test(hrdata, 0.2)
# print(len(train_set), "train +", len(test_set), "test")

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(hrdata, test_size = 0.2, random_state = 42)
print(len(train_set), "train +", len(test_set), "test")

split = model_selection.StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
split

for train_index, test_index in split.split(hrdata, hrdata["left"]):
    strat_train_set = hrdata.loc[train_index]
    strat_test_set = hrdata.loc[test_index]

hrdata["left"].value_counts() / len(hrdata)
strat_train_set["left"].value_counts() / len(strat_train_set)
strat_test_set["left"].value_counts() / len(strat_test_set)
len(strat_train_set)
len(strat_test_set)


strat_train_set.plot(kind = "scatter", x = "last_evaluation", y = "satisfaction_level")
plt.show()

corr_matrix = strat_train_set.corr()
corr_matrix["satisfaction_level"].sort_values(ascending = False)
corr_matrix

# from pandas.tools.plotting import scatter_matrix
#
#
# attributes = strat_train_set.columns.values.tolist()
# scatter_matrix(strat_train_set[attributes], figsize = (12, 8))
# plt.show()


### PRINT THE TOTAL NUMBER OF NAS
strat_train_set.isnull().values.ravel().sum()

# strat_train_set.select_dtypes(include = ["float64", "int64"]).columns.values.tolist()
# strat_train_set.select_dtypes(include = ["object", "categorical"]).columns.values.tolist()



def explode(data, *target):
    categories = data.select_dtypes(include = ["object"]).columns.values.tolist()
    data_cat_only = data[categories]
    new_data = DataFrame()
    for category in categories:
        if category not in target:
            # print(i)
            cat_binarized = DataFrame(pd.get_dummies(data_cat_only[category]))

            # print(cat_binarized.columns.values.tolist())
            cat_binarized = cat_binarized.rename(columns = {cat_binarized.columns.values.tolist()[n]: str(category) + "^" + cat_binarized.columns.values.tolist()[n] for n in range(len(cat_binarized.columns.values.tolist()))})
            for new_col in range(cat_binarized.shape[1]):
                new_data[cat_binarized.iloc[:,new_col].name] = cat_binarized.iloc[:,new_col]
        else:
            new_data[category] = data_cat_only[category]

    quantities = data.select_dtypes(include = ["float64", "int64"]).columns.values.tolist()
    data_quant_only = data[quantities]
    for quantity in quantities:
        new_data[quantity] = data_quant_only[quantity]

    return DataFrame(new_data)


#
# explode(strat_train_set, "Work_accident", "left")
# explode(strat_train_set)

def shrink(data):
    exploded_labels = set()
    [exploded_labels.add(i.split("^")[0]) for i in data.columns.values.tolist() if "^" in i]
    data_to_return = DataFrame()
    for label in exploded_labels:
        columns_to_shrink = [ii for ii in data.columns.values.tolist() if ("^" in ii) and (label in ii)]
        data_to_shrink = data[columns_to_shrink]
        data_to_shrink = data_to_shrink.stack()
        data_to_shrink = DataFrame(data_to_shrink[data_to_shrink != 0])
        data_to_shrink.reset_index(level=1, inplace=True)
        extra_name = str(label) + "\^"
        shrunk_column = data_to_shrink.iloc[::,0].str.replace(extra_name, "")
        shrunk_column.name = str(label)
        data_to_return[label] = shrunk_column

    regular_labels = set()
    [regular_labels.add(i) for i in data.columns.values.tolist() if "^" not in i]
    for regular_label in regular_labels:
        data_to_return[regular_label] = data[regular_label]
    return data_to_return

strat_train_set["left"].astype(bool).astype(int).head()
strat_train_set["left"].head()

shrink(explode(shrink(strat_train_set)))


from sklearn.base import BaseEstimator, TransformerMixin





class CatAttributesTransformer(BaseEstimator, TransformerMixin):
    """
    data: a dataframe
    target: a column that the explode will ignore (in cases like skl accepting
                a categorical object column as input)
    __________________________________

    explode: take a categorical column and binarize it with the following format
                Original_column_name^category

    shrink: take the column that was originally exploded and make it of the
                original format

    """

    def __init__(self, data, *target):
        self.data = data
        self.target = target
    def explode(self, *target):
        categories = self.data.select_dtypes(include = ["object"]).columns.values.tolist()
        data_cat_only = self.data[categories]
        new_data = DataFrame()
        for category in categories:
            if category not in self.target:
                # print(i)
                # if data_cat_only[category].dtypes == np.bool:
                #     new_data[category] = data_cat_only[category].astype(int)
                #     continue
                cat_binarized = DataFrame(pd.get_dummies(data_cat_only[category]))
                # print(cat_binarized.columns.values.tolist())
                cat_binarized = cat_binarized.rename(columns = {cat_binarized.columns.values.tolist()[n]: str(category) + "^" + cat_binarized.columns.values.tolist()[n] for n in range(len(cat_binarized.columns.values.tolist()))})
                new_cols = cat_binarized.columns.values.tolist()
                for new_col in new_cols:
                    new_data[new_col] = cat_binarized[new_col]


                # for new_col in range(cat_binarized.shape[1]):
                #     new_data[cat_binarized.iloc[:,new_col].name] = cat_binarized.iloc[:,new_col]
            else:
                new_data[category] = data_cat_only[category]

        quantities = self.data.select_dtypes(include = ["float64", "int64", "uint8"]).columns.values.tolist()
        data_quant_only = self.data[quantities]
        for quantity in quantities:
            new_data[quantity] = data_quant_only[quantity]

        booleans = self.data.select_dtypes(include = ["bool"]).columns.values.tolist()
        data_bool_only = self.data[booleans]
        for boolean in booleans:
            new_data[boolean] = DataFrame(data_bool_only[boolean], dtype = "uint8")
            # new_data[boolean] = data_bool_only[boolean].astype(int)

        return DataFrame(new_data)
    def shrink(self):
        exploded_labels = set()
        [exploded_labels.add(i.split("^")[0]) for i in self.data.columns.values.tolist() if "^" in i]
        data_to_return = DataFrame()
        for label in exploded_labels:
            columns_to_shrink = [ii for ii in data.columns.values.tolist() if ("^" in ii) and (label in ii)]
            data_to_shrink = self.data[columns_to_shrink]
            data_to_shrink = data_to_shrink.stack()
            data_to_shrink = DataFrame(data_to_shrink[data_to_shrink != 0])
            data_to_shrink.reset_index(level=1, inplace=True)
            extra_name = str(label) + "\^"
            shrunk_column = data_to_shrink.iloc[::,0].str.replace(extra_name, "")
            shrunk_column.name = str(label)
            data_to_return[label] = shrunk_column

        regular_labels = set()
        [regular_labels.add(i) for i in self.data.columns.values.tolist() if "^" not in i]
        for regular_label in regular_labels:
            data_to_return[regular_label] = self.data[regular_label]
        return data_to_return



class CatAttributesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ignore = [], target = None):
        self.ignore = ignore
        self.target = target

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        categories = X.select_dtypes(include = ["object"]).columns.values.tolist()
        data_cat_only = X[categories]
        new_data = DataFrame()
        for category in categories:
            if category != self.target:
                if category not in self.ignore:
                    cat_binarized = DataFrame(pd.get_dummies(data_cat_only[category]))
                    cat_binarized = cat_binarized.rename(columns = {cat_binarized.columns.values.tolist()[n]: str(category) + "^" + cat_binarized.columns.values.tolist()[n] for n in range(len(cat_binarized.columns.values.tolist()))})
                    new_cols = cat_binarized.columns.values.tolist()
                    for new_col in new_cols:
                        new_data[new_col] = cat_binarized[new_col]
                else:
                    new_data[category] = data_cat_only[category]

        quantities = X.select_dtypes(include = ["float64", "int64", "uint8"]).columns.values.tolist()
        data_quant_only = X[quantities]
        for quantity in quantities:
            new_data[quantity] = data_quant_only[quantity]

        booleans = X.select_dtypes(include = ["bool"]).columns.values.tolist()
        data_bool_only = X[booleans]
        for boolean in booleans:
            new_data[boolean] = DataFrame(data_bool_only[boolean], dtype = "uint8")
            # new_data[boolean] = data_bool_only[boolean].astype(int)

        return DataFrame(new_data)



class VariableSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variable_type):
        self.variable_type = variable_type

        # self.data = data
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        data_categorical = DataFrame()
        data_numeric = DataFrame()

        for i in X.columns:

            if sorted(X[i].unique().tolist()) == [0, 1]:
                data_categorical[X[i].name] = X[i]

            else:
                data_numeric[X[i].name] = X[i]

        if self.variable_type == "categorical":
            return DataFrame(data_categorical)

        if self.variable_type == "numeric":
            return DataFrame(data_numeric)



attr_adder = CatAttributesTransformer(ignore = ["sales"])
attr_adder = CatAttributesTransformer()
attr_adder
housing_extra_attribs = attr_adder.transform(strat_train_set)
housing_extra_attribs.info()








from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin

strat_train_set.info()

num_pipeline = Pipeline([
    ("cat_transform", CatAttributesTransformer(target = "average_montly_hours")),
    ("num_selector", VariableSelector(variable_type = "numeric")),
    ("imputer", Imputer(strategy = "median")),
    ("std_scaler", StandardScaler())])


cat_pipeline = Pipeline([
    ("cat_transform", CatAttributesTransformer(target = "average_montly_hours")),
    ("cat_selector", VariableSelector(variable_type = "categorical"))])


full_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)])


hr_prepared = full_pipeline.fit_transform(strat_train_set)
hr_prepared
hr_prepared.shape

full_pipeline

full_pipeline.transformer_list[0][1].steps

hr_labels = strat_train_set["average_montly_hours"].copy()



from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(hr_prepared, hr_labels)
lin_reg.coef_
strat_train_set



from sklearn.metrics import mean_squared_error
hr_predictions = lin_reg.predict(hr_prepared)
lin_mse = mean_squared_error(hr_labels, hr_predictions)
lin_mse

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(hr_prepared, hr_labels)

tree_reg.get_params
tree_reg.coef_

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, hr_prepared, hr_labels,
                            scoring = "neg_mean_squared_error", cv = 10)

scores
rmse_scores = np.sqrt(-scores)
rmse_scores


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

display_scores(rmse_scores)



lin_scores = cross_val_score(lin_reg, hr_prepared, hr_labels,
                            scoring = "neg_mean_squared_error", cv = 10)

lin_rmse_scores = np.sqrt(-lin_scores)
lin_rmse_scores
display_scores(lin_rmse_scores)






# np.set_printoptions(suppress=False)
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
