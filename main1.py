import pickle
import numpy as np
import pandas as pd
from app import top_collections
from xgboost import XGBClassifier

pickled_model = pickle.load(open('Bored Ape Yacht Club.pkl', 'rb'))
# print(xgboost.__version)
inter = np.full(10, 0)
df = pd.DataFrame(inter).T
df.columns = top_collections.keys()

y_scores = pickled_model.predict_proba(df.drop("Bored Ape Yacht Club", axis=1))[0][1]
print(y_scores)
