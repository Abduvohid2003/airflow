# <YOUR_IMPORTS>
import datetime
import json
import pandas as pd
import os
import glob
import dill
path = os.environ.get('PROJECT_PATH')
# path = 'C:/Users/User/Desktop/airflow_hw'


def load_model():
    with open(f'{path}/data/models/cars_pipe.pkl','rb') as f:
        model = dill.load(f)
    return model


def transform_json():
    json1 = f'{path}/data/test'
    json_pattern = os.path.join(json1, '*.json')
    f_list = glob.glob(json_pattern)

    model = load_model()
    dfs=[]

    for f in f_list:
        with open(f) as f1:
            json_data = pd.json_normalize(json.loads(f1.read()))
            dfs.append(json_data)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['pred'] = model.predict(combined_df)
    combined_df[['id', 'pred']].to_csv(f'{path}/data/predictions/pred.csv',
                                       index=False)


def predict():
    # <YOUR_CODE>
    transform_json()
    print('Finish')


if __name__ == '__main__':
    predict()
