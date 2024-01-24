import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

if __name__ == '__main__':
    df = pd.read_csv("house_class.csv")
    rows, columns = df.shape
    any_na = np.any(df.isna())
    max_room = df["Room"].max()
    mean_area = round(df["Area"].mean(), 1)
    unique_ziploc = df["Zip_loc"].nunique()

    # print(rows)
    # print(columns)
    # print(any_na)
    # print(max_room)
    # print(mean_area)
    # print(unique_ziploc)

    X = df.iloc[:, 1:]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,
                                                        random_state=1, stratify=X['Zip_loc'].values)

    # print(X_train['Zip_loc'].value_counts().to_dict())

    enc = OneHotEncoder(drop="first")
    enc.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])
    X_train_transformed = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                       index=X_train.index).add_prefix('enc')
    X_test_transformed = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                      index=X_test.index).add_prefix('enc')

    X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
    X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

    model = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best',
                                   max_depth=6, min_samples_split=4, random_state=3)
    model.fit(X_train_final, y_train)

    predictions = model.predict(X_test_final)
    # print(accuracy_score(y_test, predictions))

    o_encoder = OrdinalEncoder()
    o_encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])
    X_train_o_transformed = pd.DataFrame(o_encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]),
                                         index=X_train.index).add_prefix('enc')
    X_test_o_transformed = pd.DataFrame(o_encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                                        index=X_test.index).add_prefix('enc')
    X_train_o_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_o_transformed)
    X_test_o_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_o_transformed)
    o_model = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best',
                                     max_depth=6, min_samples_split=4, random_state=3)
    o_model.fit(X_train_o_final, y_train)
    o_predictions = o_model.predict(X_test_o_final)
    # print(accuracy_score(y_test, o_predictions))

    t_enc = TargetEncoder()
    t_enc.fit(X_train[['Room', 'Zip_area', 'Zip_loc']], y_train)
    X_train_t_transformed = pd.DataFrame(t_enc.transform(X_train[['Room', 'Zip_area', 'Zip_loc']]),
                                         index=X_train.index).add_prefix('enc')
    X_test_t_transformed = pd.DataFrame(t_enc.transform(X_test[['Room', 'Zip_area', 'Zip_loc']]),
                                        index=X_test.index).add_prefix('enc')
    X_train_t_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_t_transformed)
    X_test_t_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_t_transformed)
    t_model = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best',
                                     max_depth=6, min_samples_split=4, random_state=3)
    t_model.fit(X_train_t_final, y_train)
    t_predictions = t_model.predict(X_test_t_final)
    # print(accuracy_score(y_test, t_predictions))

    # print(classification_report(y_test, predictions))
    # print(classification_report(y_test, o_predictions))
    # print(classification_report(y_test, t_predictions))

    f1_score_oh = f1_score(y_test, predictions, average='macro')
    f1_score_o = f1_score(y_test, o_predictions, average='macro')
    f1_score_t = f1_score(y_test, t_predictions, average='macro')

    print(f"OneHotEncoder:{round(f1_score_oh, 2)}")
    print(f"OrdinalEncoder:{round(f1_score_o, 2)}")
    print(f"TargetEncoder:{round(f1_score_t, 2)}")
