import joblib
import streamlit as st
import pandas as pd
import numpy as np
df= pd.read_csv('mainalldatafinalll.csv', index_col=0,low_memory=False)
# Load the label encoders and pipeline from joblib files
label_encoders = joblib.load('label_encoders_main123_check.joblib')
rf_model = joblib.load('multi_output_rf_model__check123__check.joblib')
import streamlit as st
# Create a unique set of player names

 # Create mapping dictionaries for dropdowns
team_abbreviation_to_id = { 'KKR': 4341, 'RCB': 4340, 'RR': 4345,
                            'SRH': 5143, 'MI': 4346, 'CSK': 4343, 'DC': 4344,
                            'GT': 6904, 'LSG': 6903, 'PBKS': 4342}

# Define the columns to be label encoded
columns_to_encode = ['floodlit_name', 'team1_abbreviation', 'team2_abbreviation',
                    'description', 'name',
                    'position', 'ground_name', 'condition']

def apply_label_encoding(df, cols_to_encode, label_encoders):
    for col in cols_to_encode:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df


def main():
    # st.title('Model Prediction App')
    print("Compiling model")
    # # Sidebar with user inputs
    # st.sidebar.header('Enter Input Data')
# Load the encodings from the joblib file
    

all_encodings = joblib.load('label_encoders_main123_check.joblib')

# Get all keys from the dictionary
all_keys = {key: all_encodings[key].classes_.tolist() for key in all_encodings}

# ...
# Sidebar with user inputs
selected_name = st.sidebar.selectbox('Name', df['name'].unique())

# Fetch player information based on the selected name
selected_player_info = df[df['name'] == selected_name].iloc[0]
selected_player = selected_name if selected_name in all_keys['name'] else all_keys['name'][0]
# Create input boxes for each feature in new_data
new_data = {
    'away_team_id': st.sidebar.selectbox('Away Team ID',list(team_abbreviation_to_id.keys())),
    'floodlit_name': st.sidebar.selectbox('Floodlit Name', all_keys['floodlit_name']),
    'home_team_id': st.sidebar.selectbox('Home Team ID', list(team_abbreviation_to_id.keys())),
    'team1_abbreviation': st.sidebar.selectbox('Team 1 Abbreviation', all_keys['team1_abbreviation']),
    'team2_abbreviation': st.sidebar.selectbox('Team 2 Abbreviation', all_keys['team2_abbreviation']),
    'batting_position': st.sidebar.number_input('Batting Position', min_value=0),
    'innings_number': st.sidebar.number_input('Innings Number', min_value=0),
    'age': int(selected_player_info['age']),
    'name': st.sidebar.selectbox('Name', all_keys['name'],index=all_keys['name'].index(selected_name)),
    'position': st.sidebar.selectbox('Position', all_keys['position']),
    'ground_name': st.sidebar.selectbox('Town Name', all_keys['ground_name']),
    'avgtemp_c': st.sidebar.number_input('Average Temperature (C)', min_value=0.0),
    'maxwind_kph': st.sidebar.number_input('Max Wind Speed (kph)', min_value=0.0),
    'condition': st.sidebar.selectbox('Condition', all_keys['condition']),
    'Avg_Balls_Faced':selected_player_info['Avg_Balls_Faced'],
    'Avg_Strike_Rate':selected_player_info['Avg_Strike_Rate'],
    'Total_Fifties_Mean':selected_player_info['Total_Fifties_Mean'],
    'Total_Hundreds_Mean':selected_player_info['Total_Hundreds_Mean'],
    'Total_Fifties':selected_player_info['Total_Fifties'],
    'Total_Hundreds':selected_player_info['Total_Hundreds'],
    'Total_Fours_sum':selected_player_info['Total_Fours_sum'],
    'Total_Sixes_sum':selected_player_info['Total_Sixes_sum'],    
    'Total_Fours_mean':selected_player_info['Total_Fours_mean'],
    'Total_Sixes_mean':selected_player_info['Total_Sixes_mean'],
    'Total_runs':selected_player_info['Total_runs'],
    'Total_runs_mean':selected_player_info['Total_runs_mean'],
}

#    # Convert user input dictionary to a DataFrame
user_input_df = pd.DataFrame([new_data])
user_input_df['away_team_id'] = user_input_df['away_team_id'].apply(lambda x: team_abbreviation_to_id.get(x, x))
user_input_df['home_team_id'] = user_input_df['home_team_id'].apply(lambda x: team_abbreviation_to_id.get(x, x))

    # Apply label encodings to selected columns
user_input_df_encoded = apply_label_encoding(user_input_df.copy(), columns_to_encode, label_encoders)

    # Use the pipeline for scaling and prediction
prediction = rf_model.predict(user_input_df_encoded)

rounded_prediction = np.round(prediction[0]).astype(int)
    # Display the rounded prediction
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

# Display the rounded prediction
# st.write(f'Model Prediction: {rounded_prediction}')


if __name__ == '__main__':
    main()