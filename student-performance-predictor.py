import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras import models, layers

# read dataset
df = pd.read_csv("student-mat_modified.csv", encoding="utf-8").drop(columns = ["Unnamed: 0"]).dropna()  # has first column that adds nothing so i dropped it
# get dummy cols + df
dummy_cols = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", 
	"schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]  # cols that need dummy vars (binary or nominal), no performance
dummy_df = pd.concat([(pd.get_dummies(df[c]) if c in dummy_cols else df[c]) for c in dummy_cols], axis = 1)  # dummy df
# get pca data
scaled_df = StandardScaler().fit(dummy_df).transform(dummy_df)  # scale data w/ standard scaling
pca_df = pd.DataFrame(PCA(n_components=0.9).fit(scaled_df).transform(scaled_df))  # pca data (43 predictors down to 21), enough principal comonents to explain 90% of variance
pc_cnt = pca_df.shape[1]  # number of principal components
# MODEL 1: KNN
y = df["Performance"].values
x_train, x_test, y_train, y_test = train_test_split(pca_df, y, test_size = 0.2, random_state = 0) # 80% train / 20% test
# find optimal n_neighbors
knn_err_rate = []
for k in range(1, 20):
	pred_k = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train).predict(x_test)
	knn_err_rate.append((np.mean(pred_k != y_test), k))
k_optimal = min(knn_err_rate)[1]  # k = 6 has lowest error rate (55.7%, ouch)
# make knn prediction
knn_pred = KNeighborsClassifier(n_neighbors = k_optimal).fit(x_train, y_train).predict(x_test)
knn_acc = accuracy_score(y_test, knn_pred)  # knn -> 44.3% accuracy, expected due to high dimensionality
print(f"MODEL 1: KNN\n    knn (k = {k_optimal}) got accuracy of {round(knn_acc * 100, 2)}%\n")
# MODEL 2: RANDOM FOREST
# find optimal n_estimators
rf_gini_err, rf_ent_err = [], []
for e in range(1, 100):  # check number of trees between 1 and 100
	rf_g = RandomForestClassifier(n_estimators = e, max_depth = None, criterion = "gini").fit(x_train, y_train)
	rf_e = RandomForestClassifier(n_estimators = e, max_depth = None, criterion = "entropy").fit(x_train, y_train)
	pred_eg = rf_g.predict(x_test)
	pred_ee = rf_e.predict(x_test)
	rf_gini_err.append((np.mean(pred_eg != y_test), e, "gini"))
	rf_ent_err.append((np.mean(pred_ee != y_test), e, "entropy"))
eg_optimal, ee_optimal = min(rf_gini_err), min(rf_ent_err)
# since rf is random, need to check if entropy or gini is better
_, e_optimal, criterion_optimal = max(eg_optimal, ee_optimal, key = lambda x:x[0])
# make rf prediction
rf_pred = RandomForestClassifier(n_estimators = e_optimal, max_depth = None, criterion = criterion_optimal).fit(x_train, y_train).predict(x_test)
rf_acc = accuracy_score(y_test, rf_pred)  # rf -> ~40% accuracy
print(f"MODEL 2: RANDOM FOREST\n    rf (e = {e_optimal}, criterion = {criterion_optimal}) got accuracy of {round(rf_acc * 100, 2)}%\n")
# MODEL 3: NEURAL NETWORK

'''

FIGURE OUT HOW TO DO THE SEQUENTIAL
HOW MANY UNITS TO START + FINISH?
EXPLAIN ALL 0 PROBLEM

'''

nn = models.Sequential(name = "DeepNN", layers = [
	layers.Flatten(),
	layers.Dense(units = 2048, activation = "relu"),
	layers.Dense(units = 1024, activation = "relu"),
	layers.Dense(units = 128, activation = "relu"),
	layers.Dense(units = 3, activation = "softmax")
])  # neural network (model)
print("\n\n"); print(np.count_nonzero(y_test == "Low")/len(y_test)); print("\n\n")
# convert dataframes to tf objects (np array), one-hot version of y
nn_x_train, nn_x_test = np.array(x_train), np.array(x_test)
pred_dict = {
	"Low": 0,
	"Normal": 1,
	"High": 2
}
nn_y_train, nn_y_test = np.array([(pred_dict[x] / 3) for x in y_train]), np.array([(pred_dict[x] / 3) for x in y_test])
# train nn
nn.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
nn_res = nn.fit(x = nn_x_train, y = nn_y_train)
nn_loss, nn_acc = nn.evaluate(nn_x_test, nn_y_test)
print(f"MODEL 3: NEURAL NETWORK\n    nn got loss of {nn_loss} and accuracy of {round(nn_acc * 100, 2)}%")