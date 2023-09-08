import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from lazypredict.Supervised import LazyRegressor

data = pd.read_csv("StudentScore.xls")
# print(data["test preparation course"].unique())
target = "math score"
x = data.drop(target, axis=1)
y = data[target]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="median")), #điền vào các khoảng trống
    ('scaler', StandardScaler())
])


education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                    "master's degree"] #sắp xếp thứ tự
gender_values = ["female", "male"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()


ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('scaler', OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('scaler', OneHotEncoder(sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", numerical_transformer, ["reading score", "writing score"]),
    ("ord_features", ordinal_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_features", nominal_transformer, ["race/ethnicity"]),
]) #xử lí dữ liệu theo dạng cột


reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ("model", LinearRegression())
])
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
# for i, j in zip(y_predict, y_test):
#     print("Predict: {}. Actual: {}".format(i, j))
print("MSE {}".format(mean_squared_error(y_test, y_predict)))
print("MAE {}".format(mean_absolute_error(y_test, y_predict)))
print("R2 {}".format(r2_score(y_test, y_predict)))

# Random forest
# MSE 35.23067096333333
# MAE 4.798896666666667
# R2 0.8299907579402186

# reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = reg.fit(x_train, x_test, y_train, y_test)
# print(models)


