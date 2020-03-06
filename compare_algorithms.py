# compare different machine learning algorithms

from pandas import read_csv
from matplotlib import pyplot
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# load dataset
print("Loading dataset...")
names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = datasets.load_iris()

# split dataset into training and validation
print("Splitting dataset into training and validation...")
x = dataset.data
y = dataset.target
x_train, x_validation, y_train, y_validation = train_test_split(
    x, y, test_size=0.20, random_state=1, shuffle=True)

# define which algorithms/models to use
models = []
models.append(("LR", LogisticRegression(
    solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))

# evaluate each model/algorithm
print("Evaluating models...")
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, x_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("\t%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

# compare algorithms
print("Generating comparison plot...")
pyplot.boxplot(results, labels=names)
pyplot.title("Algorithm Comparison")
pyplot.show()
