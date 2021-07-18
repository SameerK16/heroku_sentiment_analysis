import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from  joblib import load
from nltk.stem import PorterStemmer
import nltk
import re
from nltk.corpus import stopwords


class MulticlassSVM:

    def __init__(self, n_iter = 1000):

        self.labels = None
        self.binary_svm = None
        self.W = None
        self.n_iter = n_iter
        
    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        
        learning_rate = 1e-8
        for i in range(self.n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W
        
    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return self.labels[np.argmax(self.W.dot(X_intercept.T), axis=0)]

    def loss_student(self, W, X, y, C=1.0):
        """
        Compute loss function given W, X, y.
        For exact definitions, please check the MP document.
        Arguments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.
        Returns:
            The value of loss function given W, X and y.
        """
        if self.labels is None:
            self.labels = np.unique(y)

        # loss of regularization term
        l2_loss = 0.5 * np.sum(W**2)

        # gradient of the other term
        # get the matrix of term 1 - delta(j, y_i) + w_j^T * x_i
        loss_aug_inf = 1 - (self.labels[:, None] == y[None, :]) + np.matmul(W, np.transpose(X))  # (K, N)
        # sum over N of max value in loss_aug_inf
        loss_aug_inf_max_sum = np.sum(np.max(loss_aug_inf, axis=0))
        # sum over N of w_{y_i}^T * x_i
        wx_sum = np.sum(W[y] * X)
        multiclass_loss = C * (loss_aug_inf_max_sum - wx_sum)

        total_loss = l2_loss + multiclass_loss
        return total_loss
    def grad_student(self, W, X, y, C=1.0):
        """
        Compute gradient function w.r.t. W given W, X, y.
        For exact definitions, please check the MP document.
        Arguments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.
        Returns:
            The gradient of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        """
        if self.labels is None:
            self.labels = np.unique(y)

        # gradient of regularization term
        l2_grad = W

        # gradient of the other term
        # get the matrix of term 1 - delta(j, y_i) + w_j^T * x_i
        loss_aug_inf = 1 - (self.labels[:, None] == y[None, :]) + np.matmul(W, np.transpose(X))  # (K, N)
        # get the j_max that maximizes the above matrix for every sample
        j_max = np.argmax(loss_aug_inf, axis=0)  # (N,)
        # gradient of sum(...) is:   x_i, if k == j_max_i and k != y_i  (pos_case)
        #                           -x_i, if k != j_max_i and k == y_i  (neg_case)
        #                              0, otherwise
        pos_case = np.logical_and((self.labels[:, None] == j_max[None, :]), (self.labels[:, None] != y[None, :]))
        neg_case = np.logical_and((self.labels[:, None] != j_max[None, :]), (self.labels[:, None] == y[None, :]))
        multiclass_grad = C * np.matmul(pos_case.astype(int) - neg_case.astype(int) , X)

        total_grad = l2_grad + multiclass_grad
        return total_grad


app = Flask(__name__)
# model = pickle.load(open('randomForestRegressor.pkl','rb'))
model = load(open("./models/model_Scratch_2gram.pkl", "rb"))
cv2 = pickle.load(open("./models/tf-idf_vectorizer_2gm.pk", "rb"))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    
    Test_review = request.form['review']
    sentiment = ["Negative", "Positive"]
    
    ps = PorterStemmer()
    sent = [Test_review]
    review = []
    
    stopwords_eng = stopwords.words('english')
    stopwords_eng.append('br')
    
    for i in sent:
        review.extend(nltk.word_tokenize(re.sub('[^a-zA-Z]', ' ', i).lower()))
        #ps.stem(word)
        review = [ps.stem(word) for word in review if word not in stopwords_eng]

    review = ' '.join(review)    
    # print("Review: ", review)
    test_vect = cv2.transform([review]).toarray()
    pred = model.predict_cs(test_vect)
    
    return render_template('home.html', prediction_text="Review has a {} sentiment".format(sentiment[pred[0]]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)