import streamlit as st
import pickle


def main():
    # Loading the vectorizer
    with open(r'./models/vectorizer.pkl', 'rb') as f:
        loaded_vectorizer = pickle.load(f)

    # Loading the Models
    with open(r'./models/classifier_svm.pkl', 'rb') as a:
        loaded_clf_svm = pickle.load(a)
    with open(r'./models/classifier_dec.pkl', 'rb') as b:
        loaded_clf_dec = pickle.load(b)
    with open(r'./models/classifier_log.pkl', 'rb') as d:
        loaded_clf_log = pickle.load(d)

    st.title('Positive and Negative Sentiment Analysis')
    classifiers = ['SVM Classifier', 'Decision Tree Classifier', 'Logistic Regression']
    choice = st.sidebar.selectbox('Select Classifier', classifiers)
    form_text = st.text_input('Enter a Sentence...')
    form_text = [form_text]
    if st.button('Get Sentiment'):
        test_1 = loaded_vectorizer.transform(form_text)

        if choice == choice[0]:
            svm = loaded_clf_svm.predict(test_1)
            st.success(svm[0])

        elif choice == choice[1]:
            dec = loaded_clf_dec.predict(test_1)
            st.success(dec[0])
        else:
            log = loaded_clf_log.predict(test_1)
            st.success(log[0])


if __name__ == "__main__":
    main()