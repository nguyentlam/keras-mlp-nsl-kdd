import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

def parseNumber(s):
    try:
        return float(s)
    except ValueError:
        return s

data_train = np.loadtxt('./KDDTrain+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
data_test = np.loadtxt('./KDDTest+.txt', dtype =object, delimiter=',', encoding='latin1', converters=parseNumber)
print('len(data_train)', len(data_train))
print('len(data_test)', len(data_test))

X_train_raw = data_train[:, 0:41]
y_train_raw = data_train[:, [41]]
print('X_train_raw[0:3]===========', X_train_raw[0:3])
print('y_train_raw[0:3]===========', y_train_raw[0:3])
print('=================')

X_test_raw = data_test[:, 0:41]
y_test_raw = data_test[:, [41]]
print('X_test_raw[0:3]===========', X_test_raw[0:3])
print('y_test_raw[0:3]===========', y_test_raw[0:3])
print('=================')

x_columns = np.array(list(range(41)))
print('x_columns', x_columns)
x_categorical_columns = np.array([1, 2, 3])
x_numberic_columns = np.delete(x_columns, x_categorical_columns)
print('x_numberic_columns', x_numberic_columns)

inputs = {}

for column in x_columns:
    dtype = tf.float32
    if column in x_categorical_columns:
        dtype = tf.string
    inputs[column] = keras.Input(shape=(1,), name=str(column), dtype=dtype)

numeric_inputs = {}
for name, input in inputs.items():
    if input.dtype == tf.float32:
        numeric_inputs[name] = input

#print(numeric_inputs)

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(X_train_raw[:, x_numberic_columns].astype(np.float32)))
all_numeric_inputs = norm(x)

#print(all_numeric_inputs)

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue
    categories = np.unique(X_train_raw[:, [name]])
    lookup = layers.StringLookup(vocabulary=categories)
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

kdd_preprocessing = keras.Model(inputs, preprocessed_inputs_cat)

kdd_train_dict = {}
kdd_test_dict = {}
for column in x_columns:
    if column in x_numberic_columns:
        kdd_train_dict[column] = X_train_raw[:, [column]].astype(np.float32)
        kdd_test_dict[column] = X_test_raw[:, [column]].astype(np.float32)
    else:
        kdd_train_dict[column] = X_train_raw[:, [column]]
        kdd_test_dict[column] = X_test_raw[:, [column]]


label_categories = np.unique(y_train_raw).tolist()

def find_index(categories, data):
    index = -1
    try:
        index = categories.index(data)
    except ValueError:
        index = len(categories)
    return index

def preprocess_label(y_data):
    print(y_data[0])
    return np.array(list(map(lambda x : 0 if x[0] == 'normal' else 1, y_data))).astype(np.float32)
    # y_transformed = []
    # for data in y_data:
    #     index = find_index(label_categories, data)
    #     y_transformed.append(index)
    # return np.array(y_transformed).astype(np.float32)

y_train_transformed = preprocess_label(y_train_raw)
y_test_transformed = preprocess_label(y_test_raw)

print('y_train_transformed[0:3]===========', y_train_transformed[0:3])
print('y_test_transformed[0:3]===========', y_test_transformed[0:3])

def kdd_model(preprocessing_head, inputs):
    # body = keras.Sequential([
    #     layers.Dense(40, activation='relu'),
    #     layers.Dense(20, activation='relu'),
    #     layers.Dense(40, activation='relu'),
    #     layers.Dense(1, activation='sigmoid')
    # ])
    body = keras.Sequential([
        layers.Dense(64, activation='tanh'),
        layers.Dense(32, activation='tanh'),
        layers.Dense(16, activation='tanh'),
        layers.Dense(8, activation='tanh'),
        layers.Dense(16, activation='tanh'),
        layers.Dense(32, activation='tanh'),
        layers.Dense(64, activation='tanh'),
        layers.Dense(1, activation='sigmoid')
    ])
    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = keras.Model(inputs, result)

    model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
    return model

kdd_model = kdd_model(kdd_preprocessing, inputs)

kdd_model.fit(x=kdd_train_dict, y=y_train_transformed, batch_size=64, epochs=10)

predictions_train = kdd_model.predict(kdd_train_dict)
train_re = mean_squared_error(y_train_transformed, predictions_train)
threshold = np.mean(train_re) * 0.5
print(threshold)

kdd_model.evaluate(kdd_test_dict,  y_test_transformed, verbose=1) 



predictions_test = kdd_model.predict(kdd_test_dict)
test_re = mean_squared_error(y_test_transformed, predictions_test)
test_label_predict = np.array(list(map(lambda x : 0 if x < threshold else 1, test_re)))
test_acc = accuracy_score(y_test_transformed, test_label_predict)

print(predictions_test[0])
# print(predictions[0].argmax())
# print(label_categories[predictions[0].argmax()])



