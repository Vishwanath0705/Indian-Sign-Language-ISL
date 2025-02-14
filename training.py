import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('/home/skullboxml/ISL/data.pickle', 'rb'))

raw_data = data_dict['data']  
labels = np.asarray(data_dict['labels'])

max_features = max(len(sample) for sample in raw_data)

data = np.array([np.pad(sample, (0, max_features - len(sample)), mode="constant") for sample in raw_data])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

correct_samples = (y_predict == y_test).sum()
total_samples = len(y_test)
score = accuracy_score(y_test, y_predict)

print(f'{correct_samples}/{total_samples} samples were classified correctly ({score * 100:.2f}%)')

print("\nSample Predictions:")
for i in range(min(5, len(y_test))):
    print(f"Actual: {y_test[i]} → Predicted: {y_predict[i]}")

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
