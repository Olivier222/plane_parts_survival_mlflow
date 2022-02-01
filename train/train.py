import os
import json
import argparse
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest

from urllib.parse import urlparse


import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def to_categorical(df, to_cat_cols):  
  for column in to_cat_cols:
    all_vals = list(df[column])
    unique = set(all_vals)
    conversion_map = {value:i for i, value in enumerate(unique)}
    # TODO: save conversion maps to config files (JSON)
    def _to_number(x):
      return conversion_map[x]

    df[column] = df[column].apply(_to_number)

  return df

def to_datetime(df, to_datetime_cols):
  for column in to_datetime_cols:
    df[column] = pd.to_datetime(df[column])
  return df

def add_datetime_features(df, dt_columns):
    '''
    This function extracts datetime features from datetime pandas variables. 
    '''
    for col in dt_columns:
        df[col+"_year"] = df[col].dt.year
        df[col+"_month"] = df[col].dt.month
        df[col+"_week"] = df[col].dt.week
        df[col+"_day"] = df[col].dt.day
        df[col+"_hour"] = df[col].dt.hour
        df[col+"_minute"] = df[col].dt.minute
        df[col+"_dayofweek"] = df[col].dt.dayofweek
    return df

def get_survival_xy(df, event_name, event_time_name):
    # x,y definition/assignment
    data_y = df.loc[:, [event_name, event_time_name]]
    data_y[event_name] = data_y.loc[:, event_name].astype(bool)
    x = df.drop(columns=[event_name, event_time_name])

    return x, data_y

if __name__ == "__main__":
    
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', nargs="?", const=50, type=int)
    parser.add_argument('--min_samples_split', nargs="?", const=10, type=int)
    parser.add_argument('--min_samples_leaf', nargs="?", const=15, type=int)
    args = parser.parse_args()
    n_estimators = int(args.n_estimators)
    min_samples_split = int(args.min_samples_split)
    min_samples_leaf = int(args.min_samples_leaf)
    
    random_state = 369
    np.random.seed(random_state)


    # define column data types
    # TODO: Setup as API parameters
    EXCLUDE = ["SERIAL_NUMBER", "INSTALL_DT", "REMOVAL_DT", 'REMOVAL_DT_minute', 'REMOVAL_DT_hour']
    CATEGORICAL = ["OPERATOR_CD", "AIRCRAFT_CD", "FLEET_CD", "CHAPTER_CD", "PART_GROUP_CD", "PART_NUMBER", "SERIAL_NUMBER"]
    DATETIME =  ['INSTALL_DT', 'REMOVAL_DT']
    EVENT = "REMOVED_BOOL"
    EVENT_TIME = "TIME_SINCE_NEW_CYCLES"


    # load data
    project_root = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(project_root, "removals", "removals.csv")
    df = pd.read_csv(filepath)
    df = to_categorical(df, CATEGORICAL)
    df = to_datetime(df, DATETIME)
    df['installed_days'] = (df['REMOVAL_DT'] - df['INSTALL_DT']).dt.days.astype('int16').abs()
    df = add_datetime_features(df, DATETIME)
    df = df.drop(columns=EXCLUDE)


    # x,y definition/assignment
    x, data_y = get_survival_xy(df, EVENT, EVENT_TIME)


    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, data_y, test_size=0.2, random_state=random_state)


    # impute missing data
    # TODO: get list of impute columns by using isna()
    # TODO: replace with more advanced imputation
    impute_columns = ['MEL_QTY', 'DELAY_QTY']
    imputer = SimpleImputer().fit(x_train.loc[:, impute_columns])
    x_train_imputed = imputer.transform(x_train.loc[:, impute_columns])
    x_train.loc[:, impute_columns] = x_train_imputed
    x_test_imputed = imputer.transform(x_test.loc[:, impute_columns])
    x_test.loc[:, impute_columns] = x_test_imputed

    # Only use subset of variables which are predictive. See notebook experiment
    with open(os.path.join(project_root, "predictive_features.json")) as json_file:
      predictive_features = json.load(json_file)
    x_train = x_train.loc[:, predictive_features]
    x_test = x_test.loc[:, predictive_features]

    # train model using mlflow
    with mlflow.start_run():
        rsf = RandomSurvivalForest(n_estimators=n_estimators,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features="sqrt",
                                n_jobs=-1,
                                random_state=random_state)


        rsf.fit(x_train, y_train.to_records(index=False))
        concord_index = rsf.score(x_test, y_test.to_records(index=False))

        print("Random Survival Forest")
        print("Concordance Index:", concord_index)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_metric("concord_index", concord_index)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(rsf, "model", registered_model_name="RandomSurvivalForestModel")
        else:
            mlflow.sklearn.log_model(rsf, "model")
