import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd


model_names =['Basic CNN','ResNet101V2','InceptionV3','InceptionResNetV2']
precision =[0.495,0.48,0.49,0.47]
recall=[0.583,0.82,0.58,0.99]
accuracy=[0.463,0.5,0.52,0.41]
f1_score=[0.54,0.61,0.64,0.53]

data = {
    'Model': model_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1_score,
    'Accuracy': accuracy
}

# Define the Streamlit app
def app():
    # Set the title of the app
    st.title("Cardiomegaly vs Others Classification 🫀")
   
    st.subheader('Performance of various models')
    df= pd.DataFrame(data)
    st.dataframe(df)
    st.markdown('---')

    st.subheader('Classification')
    # Load the trained model based on user preference
    model_preference=st.selectbox('Select preferred model: ',['Basic CNN','ResNet101V2','InceptionV3','InceptionResNetV2'])
    if model_preference == 'Basic CNN':
        model_path = r'D:\5C_Networks\models\cardiomaagly_1.h5'        
    elif model_preference == 'InceptionV3':
        model_path=r'D:\5C_Networks\models\cardiomagly_InceptionV3.h5'
    elif model_preference == 'InceptionResNetV2':
        model_path=r'models/cardiomagly_InceptionResNetV2.h5'
    else:
        model_path = r'D:\5C_Networks\models\cardiomagly_ResNet101V2.h5'
    model = tf.keras.models.load_model(model_path)

    # Define the classes that the model can predict
    if model_preference == 'Basic CNN':
        classes = ['Not Cardiomegaly','Cardiomegaly']
    else:
        classes = ['Cardiomegaly','Not Cardiomegaly']


    
    # Ask the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=['png',"jpg"])

    # If the user has uploaded an image
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file).convert('RGB')

        # Display the image to the user
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        image = image.resize((256, 256))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)

        # Make a prediction using the model
        with st.spinner('Predicting...'):
            prediction = model.predict(image)

        # Interpret the predictions

        class_label = 1 if prediction[0][0] > 0.5 else 0  # Using 0.5 as the threshold for binary classification
        confidence = prediction[0][0] if class_label == 1 else 1 - prediction[0][0]

        # Display the results
        if class_label == 1:
            st.success(f"Prediction: {classes[class_label]} with confidence {confidence:.2f}")
        else:
            st.success(f"Prediction: {classes[class_label]} with confidence {1-confidence:.2f}")

# Run the Streamlit app
if __name__ == '__main__':
    app()
