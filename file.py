import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

class General:
    def __init__(self):
        self.data = pd.read_csv("data/gnarly_csv.csv")

    def prepare_data(self):
        df = self.data
        df.dropna(inplace = True)
        filtered_garbage = df[~(df["species"] == "Species McSpeciesFace") & \
            ~(df.sex =="Sexy McSexyFace")].copy()
        filtered_garbage.drop(columns = \
            ["avg_wave_ht_day_of_measurement_cm", "researcher_name"], inplace = True)
        reduced_cols = filtered_garbage.iloc[:,:-2]
        self.X = reduced_cols.iloc[:,2:]
        self.y = reduced_cols.iloc[:,1]

        # Making (slightly unnecessary) Pipeline
        scaler_pipe = make_pipeline(make_column_transformer(
            (MinMaxScaler(),
                make_column_selector(dtype_include=np.number)),  # rating
            (OneHotEncoder(),
        make_column_selector(dtype_include=object))))
        self.scaled_data = scaler_pipe.fit_transform(self.X)
        self.scaler = scaler_pipe

    def test_models(self):
        score_dict = {}
        for model in [DecisionTreeClassifier, SVC, KNeighborsClassifier]:
            score = cross_val_score(model(), self.scaled_data, self.y, cv = 5)
            score_dict[score.mean()] = model
        best_score = max(score_dict.keys())
        best_model = score_dict[best_score]
        self.best_model = best_model

class Production(General):

    def set_up_best_model(self):

        self.prepare_data()
        self.test_models()

    def make_pred(self):

        best_model_trained = self.best_model().fit(self.scaled_data, self.y)
        rows_in_self_data = self.scaled_data.shape[0]

        random_row_index = np.random.randint(0,rows_in_self_data)
        random_row = self.scaled_data[random_row_index, :]

        prediction = best_model_trained.predict([random_row])
        print(prediction)
