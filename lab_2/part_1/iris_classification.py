import pandas
import matplotlib.pyplot as plt
import seaborn
from sklearn.datasets import load_iris, make_classification
from sklearn.datasets import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
dataframe = pandas.DataFrame(iris.data, columns = iris.feature_names)
dataframe['target'] = iris.target

# Task 1
plt.figure(figsize=(12,6))
colors = {0: 'purple', 1: 'blue', 2: 'cyan'}

def plotting_irises(name) -> None:
    for i, sort in enumerate(iris.target_names):
        plt.scatter(
            dataframe[dataframe['target'] == i][f'{name} length (cm)'],
            dataframe[dataframe['target'] == i][f'{name} width (cm)'],
            c = colors[i],
            label = sort)
    plt.xlabel(f'{name} length in cm')
    plt.ylabel(f'{name} width in cm')
    plt.legend()

# sepal
plt.subplot(1,2,1)
plotting_irises('sepal')
# petal
plt.subplot(1,2,2)
plotting_irises('petal')
plt.show()

# Task 2
seaborn.pairplot(dataframe, hue = 'target', palette = 'viridis')
plt.show()

# Task 3
df_first = dataframe[dataframe['target'].isin([0, 1])] # setosa & versicolor
df_second = dataframe[dataframe['target'].isin([1, 2])] # versicolor & virgincia

# Tasks 4-8 (Machine learning)
def split_train_predict(dataframe) -> None:
    # 4: splitting data
    X = dataframe.drop('target', axis = 1)
    Y = dataframe['target']
    train_predict(X,Y)

def train_predict(X, Y) -> None:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

    # 5-6: training the model
    clf = LogisticRegression(random_state = 0)
    clf.fit(X_train, Y_train)

    # 7: making predictions
    Y_predict = clf.predict(X_test)

    # 8: printing the score
    accuracy = accuracy_score(Y_test, Y_predict)
    print('Accuracy score: ', accuracy)

split_train_predict(df_first) # setosa & versicolor
split_train_predict(df_second) # versicolor & virgincia

# Task 9 (dataset generation & classification)
X, Y = make_classification(
    n_samples = 1000, n_features = 2, 
    n_redundant = 0, n_informative = 2, 
    random_state = 1, n_clusters_per_class = 1)
plt.scatter(
    X[:, 0], X[:, 1], 
    c = 'purple', cmap = 'viridis',
    label = 'New dataset')
plt.xlabel('First feature')
plt.ylabel('Secong feature')
plt.show()

train_predict(X, Y)
