import os
import tarfile
import pandas as pd
import urllib
import numpy as np
from sklearn import model_selection
from pandas import DataFrame
path = "/Users/ilyaperepelitsa/quant/MLpy/tensor1"
csv_path = os.path.join(path, "HR_comma_sep.csv")
hrdata = pd.read_csv(csv_path)

hrdata.head()
hrdata.info()
hrdata["Work_accident"] = hrdata["Work_accident"].astype(object)
hrdata["Work_accident"].astype(object).value_counts()

hrdata["left"] = hrdata["left"].astype(object)
hrdata["left"].astype(object).value_counts()

hrdata["promotion_last_5years"] = hrdata["promotion_last_5years"].astype(object)
hrdata["promotion_last_5years"].astype(object).value_counts()


# hrdata["number_project"].describe()

hrdata.describe()

range(1)

def print_dat_mothafucking_categories(df, inc):
    for i in range(DataFrame(df.select_dtypes(include = [inc])).shape[1]):
        if inc == "object":
            print(DataFrame(df.select_dtypes(include = [inc])).iloc[:,i].value_counts())
            print("\n\n")
        else:
            print(DataFrame(df.select_dtypes(include = [inc])).iloc[:,i].describe())
            print("\n\n")

print_dat_mothafucking_categories(hrdata, "object")




hrdata["salary"].value_counts()
type(hrdata["salary"])
hrdata["salary"]
pd.Categorical(hrdata["salary"], ordered = True).reorder_categories(["low", "medium", "high"], ordered = True)
hrdata["salary"] = pd.Categorical(hrdata["salary"], ordered = True, ["high" > "medium" > "low"])


DataFrame(hrdata.select_dtypes(include = ["object"])).iloc[:,0].value_counts()

pd.value_counts(hrdata.select_dtypes(include = ["object"]).values.flatten())

hrdata.describe()
