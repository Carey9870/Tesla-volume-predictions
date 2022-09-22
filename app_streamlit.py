# to run type - streamlit run app-streamlit.py

import streamlit as st
import pickle

pickle_in = open('regmodel.pkl', 'rb')
classifier = pickle.load(pickle_in)

def Tesla_Volume_Prediction(Open,	High, Low):
    prediction = classifier.predict([[Open,	High,	Low]])
    print(prediction)
    return prediction

def main():
    st.title('Tesla Volume Prediction')
    html_temp = """
    <div style="background-color:teal;padding:10px">
    <h2 style="color:white;text-align:center;"> Streamlit Tesla Volume Prediction ML App </h2>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html=True)
    Open = st.text_input("Open", "")
    High = st.text_input("High", "")
    Low = st.text_input("Low", "")
    
    result = ""
    
    if st.button("Predict"):
        result = Tesla_Volume_Prediction(Open, High, Low)
    st.success('The Volume Output is {}'.format(result))

if __name__ == "__main__":
    main()