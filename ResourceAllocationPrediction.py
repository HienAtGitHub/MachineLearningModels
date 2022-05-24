# Import all needed libraries
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Mute warnings
warnings.filterwarnings("ignore")

# Get the average of WD, GCSVolume, VolumeShare, ValueShare, and Spont Awareness,
# aggregated by Country+ProductCategory+Brand
def group_data(df):
    cols_grouped = ["Country", "ProductCategory", "Brand", "WD", "GCSVolume", "VolumeShare", "ValueShare", "SpontAwareness"]
    df_transformed = df[cols_grouped].groupby(["Country", "ProductCategory", "Brand"]).mean()
    return df_transformed

# Change datatype of categorical cols to "category"
def change_datatype(df):
    cat_cols = ["Country","ProductCategory","Brand"]
    for name in cat_cols:
        df[name] = df[name].astype("category")
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace = True)
    return df

def cleanse_data(df):
    # Rename column headers to the approriate format
    df = df.rename(columns={"Top 40/NTop 40":"Top40?","04 Country": "Country","Product Category": "ProductCategory",
                            "Key_Brand":"KeyBrand","GCS Volume":"GCSVolume","Volume Share":"VolumeShare",
                            "Value Share":"ValueShare","Spont Awareness":"SpontAwareness"})

    # Convert WD value to numeric instead of percentage
    df["WD"] = df["WD"]/100

    # Filter out letters or special characters in numerical-only columns
    df =  df[~df['GCSVolume'].str.contains('[A-Za-z]')]

    # Filter out all records where "VolumeShare" = "ValueShare" = 0 --> cant predict
    #df = df.loc[(df["VolumeShare"] !=0) & (df["ValueShare"] !=0)]

    # Convert GCSVolume to float datatype
    df["GCSVolume"] = df["GCSVolume"].str.replace(",","").astype("float")/1.0

    # Group data
    df_grouped = group_data(df)

    return df_grouped

# Function to set aside all records that dont have SpontAwareness values and save it to .csv
def filter_unseen(df):
    spont_col_null = ["SpontAwareness"]
    df_unseen = df[df[spont_col_null].isna().all(1)]
    df_unseen.to_csv("SpontAwareness_testset.csv")
    df_unseen_imported = pd.read_csv("SpontAwareness_testset.csv")
    return df_unseen_imported

# Function to split the dataset into X and y
def split_data(df):
    X = df.drop(["SpontAwareness"], axis=1)
    y = df.pop("SpontAwareness")
    return X, y

# Function to score the model
def score_model(X,y, model=XGBRegressor()):
    for col in X.select_dtypes(["category"]):
        X[col] = X[col].cat.codes
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    scores_mae = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=1)
    # force scores to be possitive
    scores_mae = np.absolute(scores_mae)
    return scores_mae

# Function to train and evaluate the model performance, then predict data where SpontAwareness value is not available
def predict_data(X_train,y_train, X_test, y_test, X_unseen, model=XGBRegressor()):
    for col in X_train.select_dtypes(["category"]):
        X_train[col] = X_train[col].cat.codes
    for col in X_test.select_dtypes(["category"]):
        X_test[col] = X_test[col].cat.codes
    for col in X_unseen.select_dtypes(["category"]):
        X_unseen[col] = X_unseen[col].cat.codes
    # fit model
    model.fit(X_train,y_train)
    # predict on test data
    preds_test = model.predict(X_test)
    # predict new data
    preds_unseen = model.predict(X_unseen)

    # compute RMSE between predicted and actual y on test data
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))

    return preds_test, preds_unseen, rmse

def SpontAwarePred():
    # Read data from .csv file
    file_path = "SpontAwareness_raw.csv"
    df = pd.read_csv(file_path)

    # Cleanse data
    df_cleansed = cleanse_data(df)

    # Set aside records where SpontAwareness value is null and change datatype of categorical cols from "object" to "category"
    df_unseen = filter_unseen(df_cleansed)
    df_unseen = change_datatype(df_unseen)

    # Create a train and validation dataset by dropping the above unseen date, then save to .csv
    df_trainval = df_cleansed.dropna(axis=0)
    df_trainval.to_csv("SpontAwareness_TrainValSet.csv")

    #Read data from clean dataset
    transformed_data = pd.read_csv("SpontAwareness_TrainValSet.csv")

    # Change datatype of categorical cols from "object" to "category"
    transformed_data = change_datatype(transformed_data)

    # Split the above data into X for predicting variables and y for target variable before training
    X, y = split_data(transformed_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state = 1)
    df_train_final = X_train.copy()
    y_train_raw = y_train.copy()
    X_test_raw = X_test.copy()
    y_test_raw = y_test.copy()

    # Split unseen data to X and y for prediction
    X_unseen, y_unseen = split_data(df_unseen)
    X_unseen_raw = X_unseen.copy()

    # Fit the model with train data, validate test data, predict unseen data
    preds_test, preds_unseen, rmse = predict_data(X_train,y_train, X_test, y_test, X_unseen, model=XGBRegressor())

    # export train data to .csv file before appending to final deliverable
    df_train_final["SpontAwarenessActual"] = y_train_raw
    df_train_final["SpontAwarenessPredicted"] = np.nan
    df_train_final.to_csv("Result_TrainDataset.csv")

    # export test data to .csv file before appending to final deliverable
    df_test = X_test_raw.copy()
    df_test["SpontAwarenessActual"] = y_test.values
    df_test["SpontAwarenessPredicted"] = preds_test
    df_test.to_csv("Result_testdataset.csv")

    # export unseen data to .csv file before appending to final deliverable
    df_unseen_final = X_unseen_raw.copy()
    df_unseen_final["SpontAwarenessActual"] = np.nan
    df_unseen_final["SpontAwarenessPredicted"] = preds_unseen
    df_unseen_final.to_csv("Result_UnseenDataset.csv")

    # Append all above .csv files to one final .csv file for deliverable
    train_final = pd.read_csv("Result_TrainDataset.csv")
    test_final = pd.read_csv("Result_testdataset.csv")
    unseen_final = pd.read_csv("Result_UnseenDataset.csv")
    frames = [train_final, test_final, unseen_final]
    deliverable = pd.concat(frames)

    return deliverable.to_csv("Result_SpontAwarenessPrediction.csv")


if __name__ == "__main__":
    SpontAwarePred()
