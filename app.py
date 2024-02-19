import joblib
import streamlit as st
import pandas as pd

# Load the label encoders and pipeline from joblib files
label_encoders = joblib.load('label_encoder_mappings_check.joblib')
rf_model = joblib.load('multi_output_rf_model_pipe_check.h5')
import streamlit as st

# Define the columns to be label encoded
cols_to_encode = ['ground_name', 'floodlit_name', 'team1_abbreviation', 'team2_abbreviation',
                  'town_name', 'short_description', 'name', 'condition', 'position']

def apply_label_encoding(df, cols_to_encode, label_encoders):
    for col in cols_to_encode:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df

def main():
    st.title('Model Prediction App')

    # Sidebar with user inputs
    st.sidebar.header('Enter Input Data')
# Load the encodings from the joblib file
    

all_encodings = joblib.load('label_encoder_mappings_check.joblib')

# Get all keys from the dictionary
all_keys = {key: all_encodings[key].classes_.tolist() for key in all_encodings}

# ...

# Create input boxes for each feature in new_data
new_data = {
    'away_team_id': st.sidebar.number_input('Away Team ID', min_value=0),
    'batting_first_team_id': st.sidebar.number_input('Batting First Team ID', min_value=0),
    'ground_name': st.sidebar.selectbox('Ground Name', all_keys['ground_name']),
    'home_team_id': st.sidebar.number_input('Home Team ID', min_value=0),
    'floodlit_name': st.sidebar.selectbox('Floodlit Name', all_keys['floodlit_name']),
    'team1_abbreviation': st.sidebar.selectbox('Team 1 Abbreviation', all_keys['team1_abbreviation']),
    'team2_abbreviation': st.sidebar.selectbox('Team 2 Abbreviation', all_keys['team2_abbreviation']),
    'balls_faced': st.sidebar.number_input('Balls Faced', min_value=0),
    'batting_position': st.sidebar.number_input('Batting Position', min_value=0),
    'innings_number': st.sidebar.number_input('Innings Number', min_value=0),
    'minutes': st.sidebar.number_input('Minutes', min_value=0),
    'strike_rate': st.sidebar.number_input('Strike Rate', min_value=0),
    'short_description': st.sidebar.selectbox('Short Description', all_keys['short_description']),
    'name': st.sidebar.selectbox('Name', all_keys['name']),
    'avgtemp_c': st.sidebar.number_input('Average Temperature (C)', min_value=0.0),
    'maxwind_kph': st.sidebar.number_input('Max Wind Speed (kph)', min_value=0.0),
    'condition': st.sidebar.selectbox('Condition', all_keys['condition']),
    'age': st.sidebar.number_input('Age', min_value=0),
    'position': st.sidebar.selectbox('Position', all_keys['position'])
}

#    # Convert user input dictionary to a DataFrame
user_input_df = pd.DataFrame([new_data])

    # Apply label encodings to selected columns
user_input_df_encoded = apply_label_encoding(user_input_df.copy(), cols_to_encode, label_encoders)

    # Use the pipeline for scaling and prediction
prediction = rf_model.predict(user_input_df_encoded)

    # Display the prediction
st.write(f'Model Prediction: {prediction[0]}')  # Assuming the prediction is a single value

if __name__ == '__main__':
    main()