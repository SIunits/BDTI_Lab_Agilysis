# import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # RandomForestClassfier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import mlflow

mlflow.autolog()
mlflow.set_experiment(experiment_name="agilysis")

def scale_data(X_train, y_train, X_val, y_val, X_test, y_test):
    sc_X_train = StandardScaler()
    sc_y_train = StandardScaler()
    sc_X_val = StandardScaler()
    sc_y_val = StandardScaler()
    sc_X_test = StandardScaler()
    sc_y_test = StandardScaler()
    
    X_train = sc_X_train.fit_transform(X_train)
    y_train = sc_y_train.fit_transform(y_train)
    X_val = sc_X_val.fit_transform(X_val)
    y_val = sc_y_val.fit_transform(y_val)
    X_test = sc_X_test.fit_transform(X_test)
    y_test = sc_y_test.fit_transform(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

    status = 'experiment' # control
def experiment(status='control'):
    # read data
    # ,Timestamp,hour,weekday,Holiday,zid,Humidity,Ambient pressure,Temp,speed,green_area,road_area,buildings,NO
    pollutants = ['NO', 'NO2', 'O3', 'PM1', 'PM10', 'PM25']
    for pol in pollutants:
        features = ['weekday','Holiday','zid','Humidity','Ambient pressure','Temp','speed','green_area','road_area','buildings', pol]
        data = pd.read_csv(f'data/{pol}.csv', usecols=features)
        data = data.fillna(0)
        # print(data.head())

        # data preparation/splitting
        # [181 184 199 225 244 245 285 338 339 341 342 346 392 409]
        train_zid = [199, 225, 244, 245, 285, 338, 339, 341, 342, 346, 392, 409]
        val_zid = [409]
        test_zid = [181, 184]

        # X_train,y_train,X_test,y_test = train_test_split(data,test_size=0.3)
        train = data[data['zid'].isin(train_zid)]
        val = data[data['zid'].isin(val_zid)]
        test = data[data['zid'].isin(test_zid)]
        y_train = train[pol]
        X_train = train.drop(columns=[pol])
        y_val = val[pol]
        X_val = val.drop(columns=[pol])
        y_test = test[pol]
        X_test = test.drop(columns=[pol])

        if status == 'experiment':
            # feature selection (RF)
            # NOTE: examine only the training data to avoid overfitting
            '''
            sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
            sel.fit(X_train, y_train)

            # select features
            selected_feat= X_train.columns[(sel.get_support())]
            len(selected_feat)

            print(selected_feat)

            pd.Series(sel.estimator_,feature_importances_,.ravel()).hist()
            '''


            rf = RandomForestRegressor(random_state=0)
            rf.fit(X_train,y_train)

            f_i = list(zip(features,rf.feature_importances_))
            f_i.sort(key = lambda x : x[1])
            plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
            # plt.show()
            plt.savefig(f'result/{pol}_F_rank.jpg')

            # NOTE: select the best features using feature importance /// Recursive Feature Elimination with Cross-Validation
            rfe = RFECV(rf,cv=5,scoring="neg_mean_squared_error")
            selector = rfe.fit(X_train,y_train)
            print(selector.support_)
            print(selector.ranking_)
            selected_features = np.array(features)[rfe.get_support()]
            
            X_train = train[selected_features]
            X_val = val[selected_features]
            X_test = test[selected_features]
            
        # NOTE: feature scaling
        sc_X_train = StandardScaler()
        sc_y_train = StandardScaler()
        sc_X_val = StandardScaler()
        sc_y_val = StandardScaler()
        sc_X_test = StandardScaler()
        sc_y_test = StandardScaler()
        
        X_train_sc = sc_X_train.fit_transform(X_train)
        y_train_sc = sc_y_train.fit_transform(y_train)
        X_val_sc = sc_X_val.fit_transform(X_val)
        y_val_sc = sc_y_val.fit_transform(y_val)
        X_test_sc = sc_X_test.fit_transform(X_test)
        y_test_sc = sc_y_test.fit_transform(y_test)

        # model training/validation
        with mlflow.start_run():
            # model training
            regressor = SVR(kernel='rbf')
            model = regressor.fit(X_train_sc, y_train_sc)
            
            # model validation
            
        # model evaluation
        y_pred = regressor.predict(X_test_sc)
        y_pred = sc_y_train.inverse_transform(y_pred) 
        
        X_grid = np.arange(min(X_train_sc), max(X_train_sc), 0.01) #this step required because data is feature scaled.
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X_test_sc, y_test_sc, color = 'red')
        plt.plot(X_test_sc, regressor.predict(X_test_sc), color = 'blue')
        plt.title('Truth or Bluff (SVR)')
        plt.xlabel('Independent')
        plt.ylabel('Pollution')
        # plt.show()
        plt.savefig(f'result/{pol}_{status}.jpg')

        # model evaluation

if __name__ == '__main__':
    experiment(status='experiment')
    experiment(status='control')