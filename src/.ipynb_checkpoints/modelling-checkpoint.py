import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
import joblib
import picks as utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_dataset(config_data: dict):
    X_train = utils.pickle_load(config_data["train_set_path"][0])
    y_train = utils.pickle_load(config_data["train_set_path"][1])
    
    X_train_vect = utils.pickle_load(config["vect_set_path"][0])
    X_test_vect = utils.pickle_load(config["vect_set_path"][1])

    X_test = utils.pickle_load(config_data["test_set_path"][0])
    y_test = utils.pickle_load(config_data["test_set_path"][1])
    
    kamusalay_dict = utils.pickle_load(config["kamusalay_dict"])
    
    model = utils.pickle_load(config["production_model_path"])
    
    return X_train, y_train, X_train_vect, X_test_vect, X_test, kamusalay_dict, model