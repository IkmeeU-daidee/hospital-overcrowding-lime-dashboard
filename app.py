import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
import requests
import os

# Streamlit UI
st.title('Hospital Overcrowding Prediction')

# Load the CSV file directly
file_path = 'https://raw.githubusercontent.com/IkmeeU-daidee/hospital-overcrowding-lime-dashboard/main/Diagnosis_data.csv'

try:
    # Load the data
    df = pd.read_csv(file_path)

    # Encode categorical columns to numeric values
    diagnosis_categories_encoder = LabelEncoder()
    df['Diagnosis_categories'] = diagnosis_categories_encoder.fit_transform(df['Diagnosis_categories'])

    diagnosis_encoder = LabelEncoder()
    df['Diagnosis'] = diagnosis_encoder.fit_transform(df['Diagnosis'])

    available_diagnosis_categories = diagnosis_categories_encoder.classes_

    # Splitting Features and Target
    X = df[['Diagnosis_categories', 'Diagnosis', 'Finished Admission Episodes',
            'Emergency', 'Waiting list', 'Planned', 'Other', 'Mean_time_waited',
            'Mean_length_of_stay', 'Young Children',
            'Older Children and Adolescents', 'Young Adults', 'Middle-Aged Adults',
            'Older Adults', 'Elderly 90+']]
    y = df['Overcrowding_Status']
    feature_names = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Download and load model
    model_url = 'https://github.com/IkmeeU-daidee/hospital-overcrowding-lime-dashboard/blob/main/gbm_model.pkl'
    model_local_path = 'gbm_model.pkl'
    try:
        if not os.path.exists(model_local_path):
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_local_path, 'wb') as f:
                f.write(response.content)

        # Use joblib to load the model
        gbm_model = joblib.load(model_local_path)

    except Exception as e:
        st.error(f"Error loading the model: {e}")
        gbm_model = None

    if gbm_model:
        st.subheader("What-If Analysis")

        # Diagnosis category selection
        diagnosis_category_selected = st.selectbox(
            "Select a Diagnosis Category",
            options=available_diagnosis_categories,
            index=0,
        )

        # Filter diagnoses based on the selected category
        filtered_diagnoses = df[df['Diagnosis_categories'] == diagnosis_categories_encoder.transform([diagnosis_category_selected])[0]]['Diagnosis'].unique()
        diagnosis_selected = st.multiselect(
            "Select Diagnoses",
            options=diagnosis_encoder.inverse_transform(filtered_diagnoses),
            default=[]
        )

        # User inputs in a grid with 3 columns, displaying features sequentially
        cols = st.columns(3)  # Create 3 columns
        user_inputs = {}  # Dictionary to store user inputs

        # Explicitly define the order of features
        ordered_features = [
            'Finished Admission Episodes', 'Emergency', 'Waiting list', 'Planned', 'Other', 
            'Mean_time_waited', 'Mean_length_of_stay', 'Young Children', 
            'Older Children and Adolescents', 'Young Adults', 
            'Middle-Aged Adults', 'Older Adults', 'Elderly 90+'
        ]

        # Divide features evenly across 3 columns
        num_columns = 3
        features_per_column = len(ordered_features) // num_columns + (len(ordered_features) % num_columns > 0)
        feature_groups = [ordered_features[i * features_per_column:(i + 1) * features_per_column] for i in range(num_columns)]

        # Display inputs in columns
        for col_idx, column_features in enumerate(feature_groups):
            with cols[col_idx]:
                for feature in column_features:
                    user_inputs[feature] = st.number_input(
                        f"Enter {feature}",
                        value=5,  # Default value
                        step=1,   # Step value
                        format="%d"
                    )

        # Process each selected diagnosis
        if st.button("Predict"):
            if not diagnosis_selected:  # Check if any diagnosis is selected
                st.warning("Please select at least one Diagnosis before making a prediction!")
                st.stop()
            else:
                st.subheader("Prediction and LIME Explanation")
                for diagnosis in diagnosis_selected:
                    st.markdown(f"<h4 style='color:#0F2C67;'>Diagnosis Category: {diagnosis_category_selected}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<h5 style='color:#E48257;'>Diagnosis: {diagnosis}</h5>", unsafe_allow_html=True)

                    user_inputs['Diagnosis_categories'] = diagnosis_categories_encoder.transform([diagnosis_category_selected])[0]
                    user_inputs['Diagnosis'] = diagnosis_encoder.transform([diagnosis])[0]
                    user_input_data = np.array([list(user_inputs.values())])
                    input_data = pd.DataFrame(user_input_data, columns=feature_names)

                    prediction = gbm_model.predict(input_data)
                    prediction_proba = gbm_model.predict_proba(input_data)

                    st.write(f"Predicted Class: **{'Overcrowding' if prediction[0] == 1 else 'Normal'}**")
                    st.write(f"Prediction Confidence: Normal = {prediction_proba[0][0]*100:.2f}%, Overcrowding = {prediction_proba[0][1]*100:.2f}%")

                    explainer = LimeTabularExplainer(
                        X_train.values,
                        feature_names=feature_names,
                        class_names=['Normal', 'Overcrowding'],
                        discretize_continuous=True
                    )

                    explanation = explainer.explain_instance(
                        input_data.iloc[0],
                        gbm_model.predict_proba,
                        num_features=8
                    )
                    
                    st.components.v1.html(explanation.as_html(), height=800)

except Exception as e:
    st.error(f"Error loading or processing the file: {e}")
