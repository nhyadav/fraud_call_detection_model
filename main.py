
import re
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

data_fraud = pd.read_csv("E:\machine_learning_project\\fraud_detection_model\\fraud_call.file", sep='\t',
                   names=['label', 'content'])
ps = PorterStemmer()
cv = CountVectorizer(max_features=2000)
# copy the cv
copy_cv = cv

"""remove unnecessary digit and marks"""


def remove_digit(data) :
    corpos = []
    for i in range(0, len(data)) :
        review = re.sub('[^a-zA-Z]', ' ', data['content'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
        review = ' '.join(review)
        corpos.append(review)
    return corpos


"""load the naive byce model and train it."""

def detect_model(corpos, data):
    x = cv.fit_transform(corpos).toarray()
    y = pd.get_dummies(data['label'])
    y = y.iloc[:, 1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    fraud_detect = MultinomialNB().fit(x_train, y_train)
    print("model has trained.")
    y_ped = fraud_detect.predict(x_test)
    cong_m = confusion_matrix(y_test, y_ped)
    acc = accuracy_score(y_test, y_ped)
    print("Confusion matrix:", cong_m)
    print("Accuracy_score:", acc)
    return fraud_detect


"""save model."""
def save_model(model):
    joblib.dump(model, "fraud_detection_model.sav")
    print("model has saved.....")


def validate_model(text,cv):
    word = [text]
    vector_word = cv.transform(word)
    get_model = joblib.load("fraud_detection_model.sav")
    predicted_data = get_model.predict(vector_word)
    print("this is normal call.") if(predicted_data[0] == 1) else print("this is fraud call.")



"""implement all function."""
# proper_list = remove_digit(data_fraud)
#
# get_model = detect_model(proper_list,data_fraud)
# save_model(get_model)

# fraud_text = input("Don't worry, we are listing......")
# validate_model(fraud_text,cv)

