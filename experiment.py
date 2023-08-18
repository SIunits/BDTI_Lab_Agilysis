# import libraries
import os, random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor # RandomForestClassfier
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import mlflow


 
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

def create_exp_folder(directory_name):
    path = 'result'  # Replace with the desired parent directory path

    # Combine the parent directory path with the directory name
    full_path = os.path.join(path, directory_name)

    # Create the directory
    print(full_path)
    os.makedirs(full_path, exist_ok=True)
    print('[INFO]: created directory for the experiment.')


def experiment(name, status='control'):
    create_exp_folder(name)

    # read data
    # ,Timestamp,hour,weekday,Holiday,zid,Humidity,Ambient pressure,Temp,speed,green_area,road_area,buildings,NO
    pollutants = ['PM1'] # ['NO', 'NO2', 'O3', 'PM1', 'PM10', 'PM25']
    for pol in pollutants:
        features = ['weekday','Holiday','zid','Humidity','Ambient pressure','Temp','speed','green_area','road_area','buildings', pol]
        data = pd.read_csv(f'data/{pol}.csv', usecols=features)
        data = data.dropna() # .fillna(0)
        data = data.drop(data[data[pol] == 0].index)
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
            plt.title(f'Feature ranking for {pol} ({status})')
            # plt.show()
            plt.savefig(f'result/{name}/{name}_{pol}_F_rank.jpg')
            plt.close()

            # NOTE: select the best features using feature importance /// Recursive Feature Elimination with Cross-Validation
            rfe = RFECV(rf,cv=5,scoring="neg_mean_squared_error")
            selector = rfe.fit(X_train,y_train)

            # Get the boolean mask of selected features
            selected_features_mask = list(selector.support_)
            if '' in selected_features_mask:
                selected_features_mask.remove('')
            if ' ' in selected_features_mask:
                selected_features_mask.remove(' ')

            features.remove(pol)
            selected_features = np.array(features)[selected_features_mask]
            
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
        
        X_train_sc = sc_X_train.fit_transform(X_train.values)
        y_train_sc = y_train  # (sc_y_train.fit_transform([y_train.values]))[0]
        X_val_sc = sc_X_val.fit_transform(X_val.values)
        y_val_sc = y_val  # (sc_y_val.fit_transform([y_val.values]))[0]
        X_test_sc = sc_X_test.fit_transform(X_test.values)
        y_test_sc = y_test  # (sc_y_test.fit_transform([y_test.values]))[0]
        
        # model training/validation
        mlflow.autolog(log_models=False)
        mlflow.set_experiment(experiment_name=name)
        
        model = ''
        with mlflow.start_run(tags={'pollutant':pol, 'exp_status':status}):
            # model training
            regressor = SVR(kernel='rbf')
            model = regressor.fit(X_train_sc, y_train_sc)
            
            # model validation
            
        # model evaluation
        # y_pred = model.predict(X_test_sc)
        # y_pred = sc_y_train.inverse_transform([y_pred]) 
        
        # X_grid = np.arange(min(X_train_sc), max(X_train_sc), 0.01) #this step required because data is feature scaled.
        # X_grid = X_grid.reshape((len(X_grid), 1))
        # print(y_test_sc)
        # print(model.predict(X_test_sc))
        plt.scatter(y_test_sc, model.predict(X_test_sc), color = 'red')
        plt.plot(y_test_sc, y_test_sc, color = 'blue')
        plt.title(f'{pol} ({status}) - SVR')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        # plt.show()
        plt.savefig(f'result/{name}/{name}_{pol}_{status}.jpg')
        plt.close()
        

        # model evaluation

if __name__ == '__main__':
    name = 'agilysis_3'
    experiment(name = name, status='control')
    experiment(name = name, status='experiment')
