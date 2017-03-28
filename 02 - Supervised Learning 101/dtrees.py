# Data visualization
import pandas as pd

iris_df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/iris.csv")
print("Iris dataframe)",iris_df.head(),sep="\n")
print("\n....\n")
print(iris_df.tail(),sep="\n")
print("\n....\n")
print(iris_df.info())
print("\n....\n")
print(iris_df.describe())

# Encoding the labels to be usabel in our training model
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(iris_df["Name"])
y = le.transform(iris_df["Name"])

 # Split the data into a training and test set
from sklearn.model_selection import train_test_split

X = iris_df.drop("Name",1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Training the model with an decision tree classifier
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=6, min_samples_split=2, 
							min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
							random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, 
							presort=False)
dt.fit(X_train,y_train)

# Predicted labels for our test features
Y_pred = dt.predict(X_test)

print("The predicted values are: {}".format(Y_pred))
print("The values that should be obtained are: {}".format(y_test))

score_train = dt.score(X_train,y_train)
score_test = dt.score(X_test, y_test)

print("Our model has a score of {} for the train set and a score of {} for the test set".format(score_train,score_test))

# Visualize the decision tree

import pydotplus 
from sklearn import tree

features = X.columns[:4]

dot_data = tree.export_graphviz(dt, out_file=None,
								feature_names = features, class_names = ['setosa','versicolor','virginica'],
								filled = True, rounded = True) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf") 