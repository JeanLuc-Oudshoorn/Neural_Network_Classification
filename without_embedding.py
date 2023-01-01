# Imports
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
import string
# import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

# Silence future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Read in data
train = pd.read_csv('bookings_train.csv')
test = pd.read_csv('bookings_test_solutions.csv')

train['set'] = 'train'
test['set'] = 'test'

data = pd.concat([train, test])

###############################################################################
#                  1. Feature Engineering & Preprocessing                     #
###############################################################################

print("\n Start Preprocessing")

# Feature Engineering
data.rename(columns={'arrival_date_day_of_month': 'day',
                     'arrival_date_year': 'year',
                     'arrival_date_month': 'month',
                     'arrival_date_week_number': 'weeknum'},
            inplace=True)

# Convert month to number
data['month'] = data['month'].apply(lambda x: datetime.strptime(x, "%B").month)

# Create date
data['date'] = pd.to_datetime(data[['year', 'month', 'day']],
                              format="%Y%B%d")

# Extract day of week
data['weekday'] = data['date'].dt.dayofweek

# Binary: Customer got reserved room
data['got_reserved_room'] = np.multiply((data['reserved_room_type'] == data['assigned_room_type']), 1)

# Total visitors
data['total_visitors'] = data['adults'] + data['children'] + data['babies']

# Check for missing values
np.sum(data.isna())
data['country'] = data['country'].fillna(value='Other')

# Convert types to category
for col in ['meal', 'country', 'market_segment', 'reserved_room_type', 'assigned_room_type',
            'deposit_type', 'customer_type', 'year']:
    data[col] = data[col].astype('category')

# Drop unnecessary columns
data.drop(columns=['date', 'babies', 'day', 'days_in_waiting_list'], inplace=True)

# Create train/test split from data files
train = data.loc[data['set'] == 'train']
test = data.loc[data['set'] == 'test']

# Remove seemingly wrong observations from train set
train = train.loc[~((train['stays_in_weekend_nights'] == 0) & (train['stays_in_week_nights'] == 0))]
train = train.loc[train['adults'] != 0]
train = train.loc[train['adr'] >= 0]

# Split predictors and outcome
y_train = np.multiply(train['is_cancelled'] == 'yes', 1)
X_train = train.drop('is_cancelled', axis=1)
y_test = np.multiply(test['is_cancelled'] == 'yes', 1)
X_test = test.drop('is_cancelled', axis=1)

# Drop the set column
X_train.drop(columns=['set'], inplace=True)
X_test.drop(columns=['set'], inplace=True)

# Define Variable types
non_cat = ['country']
cat_feats = list(X_train.columns[np.where(X_train.dtypes == 'category')])
cat_feats = list(set(cat_feats).difference(non_cat))
num_feats = list(np.setdiff1d(X_train.columns, (cat_feats + non_cat)))

# Remap values of reserved room type and assigned room type
letters = list(string.ascii_uppercase)
numbers = np.arange(0, 26)
room_mapping = dict(zip(letters, numbers))


# Preprocess data
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

country_transformer = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=200)

numeric_transformer = SimpleImputer(strategy='most_frequent')

preproccessor = ColumnTransformer(
    transformers=[
        ("numerical", numeric_transformer, num_feats),
        ("categorical", categorical_transformer, cat_feats),
        ("country", country_transformer, ["country"]),
    ],
)

# Fit the preprocessor to the training data
X_train_prepped = preproccessor.fit_transform(X_train)
X_test_prepped = preproccessor.transform(X_test)

print("\n Preprocessing Done")

###############################################################################
#                     2. Create Neural Network in Keras                       #
###############################################################################

# Define the input layers for the keras model
input_ = keras.layers.Input(shape=(X_train_prepped.shape[1],), name='input')

# Normalize input
normalized = keras.layers.Normalization()(input_)

# Five hidden layers with Batch Normalization
# batch_norm1 = keras.layers.BatchNormalization()(concat)
hidden1 = keras.layers.Dense(20, activation='leaky_relu', kernel_initializer='he_normal')(normalized)

# Insert dropout layer
dropout1 = keras.layers.Dropout(0.25)(hidden1)

batch_norm2 = keras.layers.BatchNormalization()(dropout1)
hidden2 = keras.layers.Dense(20, activation='leaky_relu', kernel_initializer='he_normal')(batch_norm2)

batch_norm3 = keras.layers.BatchNormalization()(hidden2)
hidden3 = keras.layers.Dense(20, activation='leaky_relu', kernel_initializer='he_normal')(batch_norm3)

# Insert dropout layer
dropout2 = keras.layers.Dropout(0.25)(hidden3)

batch_norm4 = keras.layers.BatchNormalization()(dropout2)
hidden4 = keras.layers.Dense(20, activation='leaky_relu', kernel_initializer='he_normal')(batch_norm4)

batch_norm5 = keras.layers.BatchNormalization()(hidden4)
hidden5 = keras.layers.Dense(20, activation='leaky_relu', kernel_initializer='he_normal')(batch_norm5)

# Skip connection -- wide and deep network
concat = keras.layers.Concatenate()([input_, hidden5])

# Output layer for binary classification
output = keras.layers.Dense(1, activation='sigmoid')(concat)

# Create the model
model = keras.Model(inputs=[input_], outputs=output)

# Compile the model
model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# Print model summary
print(model.summary())

# Set training seed
tf.random.set_seed(7)

# Train the model on GPU
with tf.device('/gpu:0'):

    # Create Callback to save model
    checkpoint_cb = keras.callbacks.ModelCheckpoint('my_checkpoints',
                                                    save_weights_only=True)

    # Create early stopping callback
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=15,
                                                      restore_best_weights=True)

    # Create function to detect overfitting based on loss
    class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            ratio = logs["val_loss"] / logs["loss"]
            print(f" - epoch:{epoch} - val/train:{ratio:.2f}")

    val_ratio_cb = PrintValTrainRatioCallback()

    # Create performance scheduler for learning rate
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=6)

    # Train the model
    history = model.fit(X_train_prepped, y_train, epochs=150, batch_size=32, validation_split=0.2,
                        callbacks=[checkpoint_cb, early_stopping_cb, val_ratio_cb, lr_scheduler])

###############################################################################
#                           3. Evaluate the Model                             #
###############################################################################

# Standard Evaluation Function
loss, accuracy, auc = model.evaluate(X_test_prepped, y_test)
print("\n Model evaluation statistics\n")
print("Model accuracy: {acc} - auc: {auc}".format(acc=accuracy, auc=auc))

# Save the model
model.save("neural_net_classifier_vanilla", save_format="tf")
