# make predictions

from pandas import read_csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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

# make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

# evaluate predictions
print("Evaluating predictions...\n")
print("Accuracy score:\n{0}\n".format(
    accuracy_score(y_validation, predictions)))
print("Confusion matrix:\n{0}\n".format(
    confusion_matrix(y_validation, predictions)))
print("Classification report:\n{0}\n".format(
    classification_report(y_validation, predictions)))
