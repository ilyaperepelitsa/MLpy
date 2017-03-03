import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    0.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
df

df.isnull().sum()
df.values
df.dropna()
df
df.dropna(axis = 1)
df.dropna(how = "all") # Drop where all equal to Nan
df.dropna(thresh = 4) # At least 4 values in a row need to be not NaN
df.dropna(subset = ["C"]) # Drop rows where NAs appear in specfic columns



##### REPLACE WITH INTERPOLATED values
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values = "NaN", strategy = "mean", axis = 0) # axis = 1 = row means
# strategy also accepts "median" or "most_frequent"
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data





import pandas as pd
df = pd.DataFrame([
        ["green", "M", 10.1, "class1"],
        ["red", "L", 13.5, "class2"],
        ["blue", "XL", 15.3, "class1"]])

df.columns = ["color", "size", "price", "classlabel"]
df

size_mapping = {"XL" : 3, "L" : 2, "M" : 1}
df["size"] = df["size"].map(size_mapping)
df
# inv_size_mapping = {v : k for k, v in size_mapping.items()}
# inv_size_mapping
# df["size"] = df["size"].map(inv_size_mapping)
# df
