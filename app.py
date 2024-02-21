import joblib
import streamlit as st
import pandas as pd
import numpy as np
df= pd.read_csv('notencoded.csv', index_col=0)
# Load the label encoders and pipeline from joblib files
label_encoders = joblib.load('label_encoder_mappings__check1.joblib')
rf_model = joblib.load('multi_output_rf_model_check1.joblib')
import streamlit as st
# Create a unique set of player names


 # Create mapping dictionaries for dropdowns
team_abbreviation_to_id = { 'KKR': 4341, 'RCB': 4340, 'RR': 4345,
                            'SRH': 5143, 'MI': 4346, 'CSK': 4343, 'DC': 4344,
                            'GT': 6904, 'LSG': 6903, 'PBKS': 4342}

# Define the columns to be label encoded
columns_to_encode = ['floodlit_name', 'team1_abbreviation', 'team2_abbreviation',
       'weather_location_code', 'description', 'name',
       'position', 'ground_name', 'condition']

def apply_label_encoding(df, cols_to_encode, label_encoders):
    for col in cols_to_encode:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df


def main():
  #  st.title('Model Prediction App')
     print("Initialized main")
    # Sidebar with user inputs
  #  st.sidebar.header('Enter Input Data')
# Load the encodings from the joblib file
    

all_encodings = joblib.load('label_encoder_mappings__check1.joblib')

# Get all keys from the dictionary
all_keys = {key: all_encodings[key].classes_.tolist() for key in all_encodings}

# ...
# Sidebar with user inputs
selected_name = st.sidebar.selectbox('Name', df['name'].unique())

# Fetch player information based on the selected name
selected_player_info = df[df['name'] == selected_name].iloc[0]

# Create input boxes for each feature in new_data
new_data = {
    'away_team_id': list(team_abbreviation_to_id.keys()),
    'floodlit_name': st.sidebar.selectbox('Floodlit Name', all_keys['floodlit_name']),
    'home_team_id': list(team_abbreviation_to_id.keys()),
    'team1_abbreviation': st.sidebar.selectbox('Team 1 Abbreviation', all_keys['team1_abbreviation']),
    'team2_abbreviation': st.sidebar.selectbox('Team 2 Abbreviation', all_keys['team2_abbreviation']),
    'weather_location_code': st.sidebar.selectbox('weather code', all_keys['weather_location_code']),
    'daily_will_it_rain': st.sidebar.number_input('will it rain', min_value=0),
    'batting_position': st.sidebar.number_input('Batting Position', min_value=0),
    'innings_number': st.sidebar.number_input('Innings Number', min_value=0),
    'description': st.sidebar.selectbox('Description', all_keys['description']),
    'age': st.sidebar.number_input('age',value=int(selected_player_info['age']), min_value=0),
    'Avg_Balls_Faced':int(selected_player_info['Avg_Balls_Faced']),
    'Avg_Strike_Rate':int(selected_player_info['Avg_Strike_Rate']), 
    'name': st.sidebar.selectbox('Name', all_keys['name']),
    'position': st.sidebar.selectbox('Position', all_keys['position']),
    'ground_name': st.sidebar.selectbox('Town Name', all_keys['ground_name']),
    'avgtemp_c': st.sidebar.number_input('Average Temperature (C)', min_value=0.0),
    'maxwind_kph': st.sidebar.number_input('Max Wind Speed (kph)', min_value=0.0),
    'condition': st.sidebar.selectbox('Condition', all_keys['condition']),

}

#    # Convert user input dictionary to a DataFrame
user_input_df = pd.DataFrame([new_data])

    # Apply label encodings to selected columns
user_input_df_encoded = apply_label_encoding(user_input_df.copy(), columns_to_encode, label_encoders)

    # Use the pipeline for scaling and prediction
prediction = rf_model.predict(user_input_df_encoded)

rounded_prediction = np.round(prediction[0]).astype(int)

# Display the rounded prediction
# st.write(f'Model Prediction: {rounded_prediction}')
runs, fours, sixes = prediction[0]  # Unpack the values

# Round the values to the nearest whole number
runs_rounded = round(runs)
fours_rounded = round(fours)
sixes_rounded = round(sixes)
st.title("Cricket Runs Prediction")
# Print the output
st.write(f'Number of runs: {runs_rounded}')
st.write(f'Number of fours: {fours_rounded}')
st.write(f'Number of sixes: {sixes_rounded}')

if __name__ == '__main__':
    main()
