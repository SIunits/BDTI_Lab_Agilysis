from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape, mean_squared_error as mse, r2_score as r2, explained_variance_score as ev_score, max_error, confusion_matrix
from joblib import dump
from matplotlib import pyplot as plt
from tqdm import tqdm

zid = [181, 184, 199, 225, 244, 245, 285, 338, 339, 341, 342, 346, 392, 409]
zid_train = []
zid_test = [392, 409]
data = ['NO', 'NO2', 'O3', 'PM1', 'PM10', 'PM25']
df_performance_log = pd.DataFrame(columns=['Algorithm', 'Pollutant', 'MAE', 'MAPE', 'RMSE', 'r2', 'EV_score'])
for d in tqdm(data):
    df = pd.read_csv('data/{}.csv'.format(d))
    df = df.fillna(df.mean())
    df = df.drop(columns=['Timestamp'])
    df_test = df[df['zid'] == zid_test[0]]
    df_test = df_test.append(df[(df['zid'] == zid_test[1])])
    df_train = df[~df.isin(df_test)].dropna()
     
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform([df_train[d]])
    X_train = sc_X.fit_transform(df_train.drop(columns=[d]))
    y_test = sc_y.fit_transform([df_test[d]])
    X_test = sc_X.fit_transform(df_test.drop(columns=[d]))
    reg = svm.SVR(kernel='rbf')
    reg.fit(X=X_train, y=y_train[0])

    '''
    param = {'kernel' : ['rbf'],'C' : [1,2,5,10],'epsilon': [2,5,10,20]}
    reg = GridSearchCV( estimator = svm.SVR(),param_grid = param, cv = 3, n_jobs = -1, verbose = 2)
    reg.fit(X_train,y_train[0])
    '''
    
    y_pred = reg.predict(X_test)
    y_pred = sc_y.inverse_transform(y_pred)
    df_pred =  pd.DataFrame(data=y_pred, columns=['Prediction'])
    
    # plt.figure(figsize=(5, 5))
    plt.scatter(x=y_test[0], y=df_pred['Prediction'], c='crimson', s=2, marker='.')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.savefig('models/{}_performance.png'.format(d))
    
    df_test = df_test.reset_index(drop=True)
    df_test_report = pd.concat([df_test, df_pred], axis=1, join='inner').drop(['Unnamed: 0'], axis=1)
    df_test_report.to_csv('models/{}_test.csv'.format(d), index=None)

    score_mae = mae(y_test[0], y_pred)
    score_mape = mape(y_test[0], y_pred)
    score_rmse = mse(y_test[0], y_pred, squared=True)
    score_r2 = r2(y_test[0], y_pred)
    score_ev = ev_score(y_test[0], y_pred)
    print('MAE for {}: {}'.format(d, score_mae))
    print('MAPE for {}: {}'.format(d, score_mape))
    print('RMSE for {}: {}'.format(d, score_rmse))
    print('r2 for {}: {}'.format(d, score_r2))
    print('Explained Variance Score for {}: {}'.format(d, score_ev))

    mod = dump(reg, 'models/{}.h5'.format(d))


