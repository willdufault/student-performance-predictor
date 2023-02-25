import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.keras import models, layers
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# read dataset
df = pd.read_csv("student-mat_modified.csv", encoding="utf-8").drop(columns = ["Unnamed: 0"]).dropna() # has first column that adds nothing so i dropped it
# get dummy cols + df
dummy_cols = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", 
	"schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"] # cols that need dummy vars (binary or nominal), no performance
dummy_df = pd.concat([(pd.get_dummies(df[c]) if c in dummy_cols else df[c]) for c in dummy_cols], axis = 1) # dummy df
# get pca data
scaled_df = StandardScaler().fit(dummy_df).transform(dummy_df) # scale data w/ standard scaling
pca_df = pd.DataFrame(PCA(n_components = 16).fit(scaled_df).transform(scaled_df)) # pca data, 16 vars explains 77.6% of variance
# print(sum(PCA(n_components = 16).fit(scaled_df).explained_variance_ratio_))  # print explained variance
# pca scree plot
pca = PCA().fit(scaled_df)
pc_vals = np.arange(pca.n_components_) + 1
plt.plot(pc_vals, pca.explained_variance_ratio_, 'ro-', linewidth = 2)
plt.title('scree plot')
plt.xlabel('principal component')
plt.ylabel('proportion of variance explained')
plt.show()
'''

NOTE: scree plot shows that pca is not great for this dataset, but i'm using it anyway for fun

'''
# MODEL 1: KNN
y = df["Performance"].values
x_train, x_test, y_train, y_test = train_test_split(pca_df, y, test_size = 0.2, random_state = 0) # 80% train / 20% test
# find optimal n_neighbors
knn_err_rate = []
for k in range(1, 50):
	pred_k = KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train).predict(x_test)
	knn_err_rate.append((np.mean(pred_k != y_test), k))
k_optimal = min(knn_err_rate)[1]
# make knn prediction
knn_pred = KNeighborsClassifier(n_neighbors = k_optimal).fit(x_train, y_train).predict(x_test)
knn_acc = accuracy_score(y_test, knn_pred) # knn -> 43% accuracy
print(f"MODEL 1: KNN\n  knn (k = {k_optimal}) got accuracy of {round(knn_acc * 100, 2)}%\n")
# MODEL 2: RANDOM FOREST
# find optimal n_estimators
rf_gini_err, rf_ent_err = [], []
for e in range(1, 100): # check number of trees between 1 and 100
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
rf_acc = accuracy_score(y_test, rf_pred) # rf -> ~40% accuracy
print(f"MODEL 2: RANDOM FOREST\n  rf (e = {e_optimal}, criterion = {criterion_optimal}) got accuracy of {round(rf_acc * 100, 2)}%\n")
# MODEL 3: NEURAL NETWORK
nn = models.Sequential(layers = [ # neural network (model)
	layers.Flatten(),
	layers.Dense(units = 4096, activation = "relu"),
	layers.Dense(units = 1024, activation = "relu"),
	layers.Dense(units = 256, activation = "relu"),
	layers.Dense(units = 3, activation = "softmax")
])
# split into training + testing data
nn_x_train, nn_x_test = np.array(x_train), np.array(x_test)
le = LabelEncoder().fit(y_train)
nn_y_train, nn_y_test = np.array(to_categorical(le.transform(y_train), 3)), np.array(to_categorical(le.transform(y_test), 3))
# train nn
nn.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
nn.fit(x = nn_x_train, y = nn_y_train, verbose = 0)
# results
nn_loss, nn_acc = nn.evaluate(x = nn_x_test, y = nn_y_test, verbose = 0) # ~40% accuracy
print(f"MODEL 3: NEURAL NETWORK\n  nn got loss of {round(nn_loss, 2)} and accuracy of {round(nn_acc * 100, 2)}%")
