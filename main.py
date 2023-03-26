import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define the column names for the dataset
column_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab',
    'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415',
    'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm',
    'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;',
    'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total', 'is_spam'
]

# Load the dataset into a pandas DataFrame
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data", header=None, names=column_names)

# Display the first few rows of the DataFrame to verify that the data was loaded correctly
print(data.head())


# Modify the dataset according to the given constraints
data.loc[data['word_freq_george'] > 0, 'is_spam'] = 0
data.loc[data['word_freq_650'] > 0, 'is_spam'] = 0

# Check for duplicate data
print("Number of duplicate rows before removing duplicates:", len(data[data.duplicated()]))

# Remove duplicate data
data.drop_duplicates(inplace=True)

# Check for duplicate data after removing duplicates
print("Number of duplicate rows after removing duplicates:", len(data[data.duplicated()]))

# Split the dataset into input features and target variable
X = data.drop(['is_spam'], axis=1)
y = data['is_spam']

# Split the dataset into training and testing sets
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the dataset using Z-score normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA to reduce the dimensionality of the dataset
pca = PCA(n_components=20)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Convert the labels for spam and non-spam to binary values, such as 0 and 1
y_train = y_train.astype('category').cat.codes
y_test = y_test.astype('category').cat.codes