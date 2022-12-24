# <YOUR_IMPORTS>
from os import listdir
import os
import dill
import pandas as pd
import json
from datetime import datetime

path = os.environ.get('$PROJECT_PATH', '.')
# path = os.path.abspath(os.pardir)


def extraxt_file(path):
    with open(path, 'rb') as js:
        dict_js = json.load(js)
    return pd.DataFrame.from_dict([dict_js])


def extract_model(path):
    name_model = sorted(listdir(f'{path}/data/models'))[-1]
    with open(f'{path}/data/models/{name_model}', 'rb') as file:
        model = dill.load(file)
    return model


def predict():
    df_pred = pd.DataFrame(columns=['car_id','pred'])
    model = extract_model(path)
    for f in listdir(f'{path}/data/test'):
        df = extraxt_file(f'{path}/data/test/{f}')
        y = model.predict(df)
        df_pred = df_pred.append({'car_id': df['id'][0], 'pred': y[0]}, ignore_index=True)
    df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
