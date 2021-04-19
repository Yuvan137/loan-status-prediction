# loan-status-prediction
Loan status prediction
# scientific computing libraries
import pandas as pd
import numpy as np
from scipy import optimize, stats  
from scipy.stats import norm, skew
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 8)
plt.style.use('ggplot')

# algorithmic library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

# scaling libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PowerTransformer

# evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# warnings
import warnings
warnings.filterwarnings('ignore')
