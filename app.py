import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os
df= pd.read_csv('mainalldatafinalll.csv', index_col=0,low_memory=False)
# Load the label encoders and pipeline from joblib files
label_encoders = joblib.load('vscode_label_encoder.joblib')
rf_model = joblib.load('vscode_prediction_model.joblib')
import streamlit as st
# Create a unique set of player names
import striputils
 # Create mapping dictionaries for dropdowns
team_abbreviation_to_id = { 'KKR': 4341, 'RCB': 4340, 'RR': 4345,
                            'SRH': 5143, 'MI': 4346, 'CSK': 4343, 'DC': 4344,
                            'GT': 6904, 'LSG': 6903, 'PBKS': 4342}

# Define the columns to be label encoded
columns_to_encode = ['floodlit_name', 'team1_abbreviation', 'team2_abbreviation',
                    'description', 'name',
                    'position', 'ground_name', 'condition']
player_names = ['AB de Villiers', 'Aaron Finch', 'Abhijeet Tomar', 'Abhinav Manohar', 'Abhishek Sharma', 'Adam Milne', 'Aiden Markram', 'Ajinkya Rahane', 'Akshdeep Nath', 'Alex Hales', 'Alzarri Joseph', 'Aman Hakim Khan', 'Ambati Rayudu', 'Amit Mishra', 'Andre Russell', 'Andrew Tye', 'Ankit Rajpoot', 'Anmolpreet Singh', 'Anrich Nortje', 'Anuj Rawat', 'Anukul Roy', 'Anureet Singh', 'Arshdeep Singh', 'Ashton Turner', 'Avesh Khan', 'Axar Patel', 'Ayush Badoni', 'Baba Indrajith', 'Basil Thampi', 'Ben Cutting', 'Ben Laughlin', 'Ben Stokes', 'Bhanuka Rajapaksa', 'Bhuvneshwar Kumar', 'Billy Stanlake', 'Carlos Brathwaite', 'Chetan Sakariya', 'Chris Gayle', 'Chris Jordan', 'Chris Lynn', 'Chris Morris', 'Chris Woakes', 'Colin Ingram', 'Colin Munro', 'Colin de Grandhomme', 'Corey Anderson', "D'Arcy Short", 'Dale Steyn', 'Dan Christian', 'Daniel Sams', 'Darshan Nalkande', 'Daryl Mitchell', 'David Miller', 'David Warner', 'David Willey', 'Dawid Malan', 'Deepak Chahar', 'Deepak Hooda', 'Devdutt Padikkal', 'Devon Conway', 'Dewald Brevis', 'Dhawal Kulkarni', 'Dinesh Karthik', 'Dushmantha Chameera', 'Dwaine Pretorius', 'Dwayne Bravo', 'Eoin Morgan', 'Evin Lewis', 'Fabian Allen', 'Faf du Plessis', 'Fazalhaq Farooqi', 'Gautam Gambhir', 'Glenn Maxwell', 'Gurkeerat Singh Mann', 'Hanuma Vihari', 'Harbhajan Singh', 'Hardik Pandya', 'Hardus Viljoen', 'Harry Gurney', 'Harshal Patel', 'Heinrich Klaasen', 'Hrithik Shokeen', 'Imran Tahir', 'Ish Sodhi', 'Ishan Kishan', 'Ishant Sharma', 'Jagadeesha Suchith', 'James Neesham', 'Jason Behrendorff', 'Jason Holder', 'Jason Roy', 'Jasprit Bumrah', 'Jayant Yadav', 'Jaydev Unadkat', 'Jean-Paul Duminy', 'Jhye Richardson', 'Jitesh Sharma', 'Joe Denly', 'Jofra Archer', 'Jonny Bairstow', 'Jos Buttler', 'Josh Hazlewood', 'Junior Dala', 'KC Cariappa', 'KL Rahul', 'Kagiso Rabada', 'Kamlesh Nagarkoti', 'Kane Richardson', 'Kane Williamson', 'Karn Sharma', 'Kartik Tyagi', 'Karun Nair', 'Kedar Jadhav', 'Keemo Paul', 'Khaleel Ahmed', 'Kieron Pollard', 'Krishnappa Gowtham', 'Krunal Pandya', 'Kuldeep Sen', 'Kuldeep Yadav', 'Kyle Jamieson', 'Lasith Malinga', 'Liam Plunkett', 'Lockie Ferguson', 'Lungi Ngidi', 'M Shahrukh Khan', 'MS Dhoni', 'Maheesh Theekshana', 'Mahipal Lomror', 'Manan Vohra', 'Mandeep Singh', 'Manish Pandey', 'Manoj Tiwary', 'Marco Jansen', 'Marcus Stoinis', 'Mark Wood', 'Martin Guptill', 'Matheesha Pathirana', 'Matthew Wade', 'Mayank Agarwal', 'Mayank Markande', 'Mitchell Johnson', 'Mitchell Marsh', 'Mitchell McClenaghan', 'Mitchell Santner', 'Moeen Ali', 'Mohammad Nabi', 'Mohammed Shami', 'Mohammed Siraj', 'Mohit Sharma', 'Moises Henriques', 'Mujeeb Ur Rahman', 'Murali Vijay', 'Murugan Ashwin', 'Mustafizur Rahman', 'Naman Ojha', 'Narayan Jagadeesan', 'Nathan Coulter-Nile', 'Nathan Ellis', 'Navdeep Saini', 'Nicholas Pooran', 'Obed McCoy', 'Odean Smith', 'Oshane Thomas', 'Parthiv Patel', 'Pat Cummins', 'Piyush Chawla', 'Pradeep Sangwan', 'Prashant Chopra', 'Prasidh Krishna', 'Prayas Ray Barman', 'Prerak Mankad', 'Prithvi Raj', 'Prithvi Shaw', 'Priyam Garg', 'Quinton de Kock', 'Rahul Chahar', 'Rahul Tewatia', 'Rahul Tripathi', 'Rajat Patidar', 'Raj\xa0Bawa', 'Rashid Khan', 'Rassie van der Dussen', 'Ravi Bishnoi', 'Ravichandran Ashwin', 'Ravindra Jadeja', 'Ricky Bhui', 'Riley Meredith', 'Rinku Singh', 'Rishabh Pant', 'Rishi Dhawan', 'Riyan Parag', 'Robin Uthappa', 'Rohit Sharma', 'Romario Shepherd', 'Rovman Powell', 'Ruturaj Gaikwad', 'Sai Kishore', 'Sai Sudharsan', 'Sam Billings', 'Sam Curran', 'Sandeep Lamichhane', 'Sandeep Sharma', 'Sandeep Warrier', 'Sanjay Yadav', 'Sanju Samson', 'Sarfaraz Khan', 'Scott Kuggeleijn', 'Sean Abbott', 'Shahbaz Ahmed', 'Shahbaz Nadeem', 'Shakib Al Hasan', 'Shane Watson', 'Shardul Thakur', 'Sheldon Jackson', 'Sherfane Rutherford', 'Shikhar Dhawan', 'Shimron Hetmyer', 'Shivam Dube', 'Shivam Mavi', 'Shreevats Goswami', 'Shreyas Gopal', 'Shubman Gill', 'Siddarth Kaul', 'Srikar Bharat', 'Steven Smith', 'Stuart Binny', 'Sunil Narine', 'Suresh Raina', 'T Natarajan', 'Tilak Varma', 'Tim David', 'Tim Seifert', 'Tim Southee', 'Tom Curran', 'Trent Boult', 'Tristan Stubbs', 'Tushar Deshpande', 'Tymal Mills', 'Umesh Yadav', 'Umran Malik', 'Varun Aaron', 'Varun Chakravarthy', 'Venkatesh Iyer', 'Vijay Shankar', 'Vinay Kumar', 'Virat Kohli', 'Virat Singh', 'Wanindu Hasaranga', 'Washington Sundar', 'Wriddhiman Saha', 'Yash Dayal', 'Yashasvi Jaiswal', 'Yusuf Pathan', 'Yuvraj Singh', 'Yuzvendra Chahal']
ground_name = ['Andhra Cricket Association-Visakhapatnam District Cricket Association Stadium, Visakhapatnam', 'Arun Jaitley Stadium, Delhi', 'Brabourne Stadium, Mumbai', 'Dr DY Patil Sports Academy, Mumbai', 'Eden Gardens, Kolkata', 'Feroz Shah Kotla, Delhi', 'Holkar Cricket Stadium, Indore', 'M Chinnaswamy Stadium, Bangalore', 'M Chinnaswamy Stadium, Bengaluru', 'MA Chidambaram Stadium, Chepauk, Chennai', 'Maharashtra Cricket Association Stadium, Pune', 'Narendra Modi Stadium, Ahmedabad', 'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh', 'Rajiv Gandhi International Stadium, Uppal, Hyderabad', 'Sawai Mansingh Stadium, Jaipur', 'Wankhede Stadium, Mumbai']
def apply_label_encoding(df, cols_to_encode, label_encoders):
    for col in cols_to_encode:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df
def calculate_and_display_mae(display_df):
    # Calculate MAE for Runs, Fours, and Sixes
    mae_runs = mean_absolute_error(display_df['Actual Runs'], display_df['Predicted Runs'])
    mae_fours = mean_absolute_error(display_df['Actual Fours'], display_df['Predicted Fours'])
    mae_sixes = mean_absolute_error(display_df['Actual Sixes'], display_df['Predicted Sixes'])
    
    # Display the MAE values under the table
    st.write("### Mean Absolute Error (MAE)")
    st.write(f"MAE for Runs: {mae_runs:.2f}")
    st.write(f"MAE for Fours: {mae_fours:.2f}")
    st.write(f"MAE for Sixes: {mae_sixes:.2f}")

    # Alternatively, you can use st.metric for a more stylized display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="MAE for Runs", value=f"{mae_runs:.2f}")
    with col2:
        st.metric(label="MAE for Fours", value=f"{mae_fours:.2f}")
    with col3:
        st.metric(label="MAE for Sixes", value=f"{mae_sixes:.2f}")

def filter_player_names(input_csv_path, player_names, output_csv_path="name_filtered.csv",colname="name",nrows=10):
    """
    Filters a CSV file to include only rows with player names from the provided list.
    
    Args:
    - input_csv_path (str): Path to the input CSV file.
    - player_names (list of str): List of player names to filter by.
    - output_csv_path (str): Path where the filtered CSV file will be saved.
    """
    # Read the input CSV file
    print(f"Method Called {colname} amd Input: {input_csv_path} For {output_csv_path}")
    default_df = pd.read_csv(input_csv_path)
    if colname != "name":
        first_row = pd.read_csv(input_csv_path, nrows=1)
        print(f"Row: {first_row}")
        column_names = default_df.columns.tolist()
        print(f"Inputs: input csv {input_csv_path}, cols: {colname} output_csv:{output_csv_path} Column Names: {column_names}")
        exclude_indices = [0,1,9,10,11,12]  # Indices of columns to exclude (8th, 9th, and 10th columns)
        total_columns = len(first_row.columns)
        use_columns = [i for i in range(total_columns) if i not in exclude_indices]
        filtered_default = default_df[default_df[colname].isin(player_names)]
        df = pd.read_csv(input_csv_path,usecols=use_columns) 
        df = df[df[colname].isin(player_names)]

        # df.to_csv(output_csv_path, index=False)
    else:
        df = pd.read_csv(input_csv_path)
        df = df[df[colname].isin(player_names)]

    # Filter the DataFrame based on the 'name' column
    
    filtered_default = default_df[default_df[colname].isin(player_names)]
    
    df.to_csv(output_csv_path, index=False)
    filtered_default.to_csv("testing_input.csv", index=False)

    # Save the filtered DataFrame to a new CSV file
    print(f"Filtered CSV saved to {output_csv_path}")


def main():
    # st.title('Model Prediction App')
    print("Compiling model")
    files_path = ['ground_filtered.csv','Input_data.csv','name_filtered.csv','testing_input.csv']
    # Create each file in the list
    for file_path in files_path:
        with open(file_path, 'w') as file:
            file.write('Hello, world!')  # Example content
        print(f"File created at {file_path}")
    # Now, delete each file in the list
    for file_path in files_path:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"The file {file_path} has been deleted.")
        else:
            print(f"The file {file_path} does not exist.")

        # # Sidebar with user inputs
        # st.sidebar.header('Enter Input Data')
    # Load the encodings from the joblib file
    


all_encodings = joblib.load('label_encoders_main123_check.joblib')

# Get all keys from the dictionary
all_keys = {key: all_encodings[key].classes_.tolist() for key in all_encodings}
print(all_keys['ground_name'])

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

st.title('CSV File Uploader and Prediction Display')
def hide_zero_values(val):
    # If the value is an empty string or None, return a CSS attribute to hide the text
    return 'display: none' if val == 0 or val is None else ''

def colorize(row):
    colors = ['','','','','']  # Start with an empty string for the 'Row Index' column if it's part of the row
    for actual, predicted in zip(row[5::2], row[6::2]):  # Start from the 5th element to skip 'Row Index', 'Name', and two other columns
        if actual == 0 and predicted == 0:
            color = 'background-color: #ccc'  # Set color to lime green if both actual and predicted are zero
        else:
            difference = abs(actual - predicted)
            if difference <= 5:
                color = 'background-color: green'
            elif 5 < difference <= 15:
                color = 'background-color: orange'
            else:
                color = 'background-color: red'
        colors.extend([color, color])  # Apply the same color to both actual and predicted cells
    return colors

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
if uploaded_file is not None:
    # first_row = pd.read_csv(uploaded_file, nrows=1)
    # print(f"First Row: {first_row}")
    # exclude_indices = [8, 9,10]  # Indices of columns to exclude (8th, 9th, and 10th columns)
    # total_columns = len(first_row.columns)
    # use_columns = [i for i in range(total_columns) if i not in exclude_indices]
    # print(f"Length of total_columns {total_columns}")
    # # print(f"Length of input_df {use_columns}")
    st.title('Enter a Year')
# Create a number input widget for year input
    year = st.number_input('Enter a year', min_value=1, step=1, format='%d')
    player_names = ['AB de Villiers', 'Aaron Finch', 'Abhijeet Tomar', 'Abhinav Manohar', 'Abhishek Sharma', 'Adam Milne', 'Aiden Markram', 'Ajinkya Rahane', 'Akshdeep Nath', 'Alex Hales', 'Alzarri Joseph', 'Aman Hakim Khan', 'Ambati Rayudu', 'Amit Mishra', 'Andre Russell', 'Andrew Tye', 'Ankit Rajpoot', 'Anmolpreet Singh', 'Anrich Nortje', 'Anuj Rawat', 'Anukul Roy', 'Anureet Singh', 'Arshdeep Singh', 'Ashton Turner', 'Avesh Khan', 'Axar Patel', 'Ayush Badoni', 'Baba Indrajith', 'Basil Thampi', 'Ben Cutting', 'Ben Laughlin', 'Ben Stokes', 'Bhanuka Rajapaksa', 'Bhuvneshwar Kumar', 'Billy Stanlake', 'Carlos Brathwaite', 'Chetan Sakariya', 'Chris Gayle', 'Chris Jordan', 'Chris Lynn', 'Chris Morris', 'Chris Woakes', 'Colin Ingram', 'Colin Munro', 'Colin de Grandhomme', 'Corey Anderson', "D'Arcy Short", 'Dale Steyn', 'Dan Christian', 'Daniel Sams', 'Darshan Nalkande', 'Daryl Mitchell', 'David Miller', 'David Warner', 'David Willey', 'Dawid Malan', 'Deepak Chahar', 'Deepak Hooda', 'Devdutt Padikkal', 'Devon Conway', 'Dewald Brevis', 'Dhawal Kulkarni', 'Dinesh Karthik', 'Dushmantha Chameera', 'Dwaine Pretorius', 'Dwayne Bravo', 'Eoin Morgan', 'Evin Lewis', 'Fabian Allen', 'Faf du Plessis', 'Fazalhaq Farooqi', 'Gautam Gambhir', 'Glenn Maxwell', 'Gurkeerat Singh Mann', 'Hanuma Vihari', 'Harbhajan Singh', 'Hardik Pandya', 'Hardus Viljoen', 'Harry Gurney', 'Harshal Patel', 'Heinrich Klaasen', 'Hrithik Shokeen', 'Imran Tahir', 'Ish Sodhi', 'Ishan Kishan', 'Ishant Sharma', 'Jagadeesha Suchith', 'James Neesham', 'Jason Behrendorff', 'Jason Holder', 'Jason Roy', 'Jasprit Bumrah', 'Jayant Yadav', 'Jaydev Unadkat', 'Jean-Paul Duminy', 'Jhye Richardson', 'Jitesh Sharma', 'Joe Denly', 'Jofra Archer', 'Jonny Bairstow', 'Jos Buttler', 'Josh Hazlewood', 'Junior Dala', 'KC Cariappa', 'KL Rahul', 'Kagiso Rabada', 'Kamlesh Nagarkoti', 'Kane Richardson', 'Kane Williamson', 'Karn Sharma', 'Kartik Tyagi', 'Karun Nair', 'Kedar Jadhav', 'Keemo Paul', 'Khaleel Ahmed', 'Kieron Pollard', 'Krishnappa Gowtham', 'Krunal Pandya', 'Kuldeep Sen', 'Kuldeep Yadav', 'Kyle Jamieson', 'Lasith Malinga', 'Liam Plunkett', 'Lockie Ferguson', 'Lungi Ngidi', 'M Shahrukh Khan', 'MS Dhoni', 'Maheesh Theekshana', 'Mahipal Lomror', 'Manan Vohra', 'Mandeep Singh', 'Manish Pandey', 'Manoj Tiwary', 'Marco Jansen', 'Marcus Stoinis', 'Mark Wood', 'Martin Guptill', 'Matheesha Pathirana', 'Matthew Wade', 'Mayank Agarwal', 'Mayank Markande', 'Mitchell Johnson', 'Mitchell Marsh', 'Mitchell McClenaghan', 'Mitchell Santner', 'Moeen Ali', 'Mohammad Nabi', 'Mohammed Shami', 'Mohammed Siraj', 'Mohit Sharma', 'Moises Henriques', 'Mujeeb Ur Rahman', 'Murali Vijay', 'Murugan Ashwin', 'Mustafizur Rahman', 'Naman Ojha', 'Narayan Jagadeesan', 'Nathan Coulter-Nile', 'Nathan Ellis', 'Navdeep Saini', 'Nicholas Pooran', 'Obed McCoy', 'Odean Smith', 'Oshane Thomas', 'Parthiv Patel', 'Pat Cummins', 'Piyush Chawla', 'Pradeep Sangwan', 'Prashant Chopra', 'Prasidh Krishna', 'Prayas Ray Barman', 'Prerak Mankad', 'Prithvi Raj', 'Prithvi Shaw', 'Priyam Garg', 'Quinton de Kock', 'Rahul Chahar', 'Rahul Tewatia', 'Rahul Tripathi', 'Rajat Patidar', 'Raj\xa0Bawa', 'Rashid Khan', 'Rassie van der Dussen', 'Ravi Bishnoi', 'Ravichandran Ashwin', 'Ravindra Jadeja', 'Ricky Bhui', 'Riley Meredith', 'Rinku Singh', 'Rishabh Pant', 'Rishi Dhawan', 'Riyan Parag', 'Robin Uthappa', 'Rohit Sharma', 'Romario Shepherd', 'Rovman Powell', 'Ruturaj Gaikwad', 'Sai Kishore', 'Sai Sudharsan', 'Sam Billings', 'Sam Curran', 'Sandeep Lamichhane', 'Sandeep Sharma', 'Sandeep Warrier', 'Sanjay Yadav', 'Sanju Samson', 'Sarfaraz Khan', 'Scott Kuggeleijn', 'Sean Abbott', 'Shahbaz Ahmed', 'Shahbaz Nadeem', 'Shakib Al Hasan', 'Shane Watson', 'Shardul Thakur', 'Sheldon Jackson', 'Sherfane Rutherford', 'Shikhar Dhawan', 'Shimron Hetmyer', 'Shivam Dube', 'Shivam Mavi', 'Shreevats Goswami', 'Shreyas Gopal', 'Shubman Gill', 'Siddarth Kaul', 'Srikar Bharat', 'Steven Smith', 'Stuart Binny', 'Sunil Narine', 'Suresh Raina', 'T Natarajan', 'Tilak Varma', 'Tim David', 'Tim Seifert', 'Tim Southee', 'Tom Curran', 'Trent Boult', 'Tristan Stubbs', 'Tushar Deshpande', 'Tymal Mills', 'Umesh Yadav', 'Umran Malik', 'Varun Aaron', 'Varun Chakravarthy', 'Venkatesh Iyer', 'Vijay Shankar', 'Vinay Kumar', 'Virat Kohli', 'Virat Singh', 'Wanindu Hasaranga', 'Washington Sundar', 'Wriddhiman Saha', 'Yash Dayal', 'Yashasvi Jaiswal', 'Yusuf Pathan', 'Yuvraj Singh', 'Yuzvendra Chahal']
    ground_name = ['Andhra Cricket Association-Visakhapatnam District Cricket Association Stadium, Visakhapatnam', 'Arun Jaitley Stadium, Delhi', 'Brabourne Stadium, Mumbai', 'Dr DY Patil Sports Academy, Mumbai', 'Eden Gardens, Kolkata', 'Feroz Shah Kotla, Delhi', 'Holkar Cricket Stadium, Indore', 'M Chinnaswamy Stadium, Bangalore', 'M Chinnaswamy Stadium, Bengaluru', 'MA Chidambaram Stadium, Chepauk, Chennai', 'Maharashtra Cricket Association Stadium, Pune', 'Narendra Modi Stadium, Ahmedabad', 'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh', 'Rajiv Gandhi International Stadium, Uppal, Hyderabad', 'Sawai Mansingh Stadium, Jaipur', 'Wankhede Stadium, Mumbai']

    input_df = pd.read_csv(uploaded_file,index_col=0)
    input_df = striputils.stripFile(input_df,year)
    input_df.to_csv("Input_data.csv",index=False)
    filter_player_names("Input_data.csv",player_names,"name_filtered.csv","name")
    filter_player_names("name_filtered.csv",ground_name,"ground_filtered.csv","ground_name",10)
    created_csv = pd.read_csv("ground_filtered.csv")
    not_excluded = pd.read_csv("testing_input.csv")
    print(f"Shape of Excluded: {not_excluded.columns}")
    if st.button('Predict'):
        # Apply label encoding to the input DataFrame
        print(f"Inputs: {columns_to_encode} label_encoders: {label_encoders}")
        input_df_encoded = apply_label_encoding(created_csv.copy(), columns_to_encode, label_encoders)

        # Make predictions
        try:
            predictions = rf_model.predict(input_df_encoded)
            
            # Create a DataFrame for displaying predictions and actual values
            predictions_df = pd.DataFrame(predictions, columns=['Predicted Runs', 'Predicted Fours', 'Predicted Sixes'])
            display_df = pd.DataFrame({
                'Row Index': created_csv.index,
                'CricInfo ID': not_excluded['cricinfo_id'],
                'Start Time': not_excluded['start_time'],
                'Name' : not_excluded['name'],
                'Balls Faced':not_excluded['balls_faced'],
                'Actual Runs': not_excluded['runs'],
                'Predicted Runs': predictions_df['Predicted Runs'],
                'Actual Fours': not_excluded['fours'],
                'Predicted Fours': predictions_df['Predicted Fours'],
                'Actual Sixes': not_excluded['sixes'],
                'Predicted Sixes': predictions_df['Predicted Sixes'],


            })

            st.write('## Predictions')
            display_df = display_df.sort_values(by='CricInfo ID')
            styled_df = display_df.style.applymap(hide_zero_values)
            styled_df = display_df.style.apply(colorize, axis=1)
            st.write('## Predictions with Conditional Formatting')
            st.dataframe(styled_df)
            calculate_and_display_mae(display_df)
        except ValueError as e:
            st.title(f"File Upload Error {ValueError}")
    else:
        st.write("Upload a CSV file to get started.")


def prepare_data_for_prediction(input_csv_path, trained_feature_names):
    # Load the data
    df = pd.read_csv(input_csv_path)
    
    # Remove unseen features by keeping only the columns that were present during training
    df_for_prediction = df[trained_feature_names]
    
    return df_for_prediction

def generate_comparison_csv(input_csv_path, output_csv_path, model, trained_feature_names):
    # Prepare the data for prediction
    df_for_prediction = prepare_data_for_prediction(input_csv_path, trained_feature_names)
    
    # Generate predictions
    predictions = model.predict(df_for_prediction)
    
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Actual Runs': df['runs'],  # Assuming 'runs' is a column in your input CSV
        'Predicted Runs': predictions[:, 0],  # Assuming the first prediction corresponds to 'runs'
        # Repeat for 'fours' and 'sixes' if needed
    })
    
    # Save the comparison DataFrame to a CSV
    comparison_df.to_csv(output_csv_path, index=False)
    print(f"Comparison CSV saved to {output_csv_path}")

def read_csv_exclude_columns_by_index(input_csv_path, exclude_indices):
    # Read only the first row to get column names
    df_temp = pd.read_csv(input_csv_path, nrows=0)  # nrows=0 loads no data, only headers
    
    # Get all column names
    all_columns = df_temp.columns.tolist()
    
    # Determine columns to exclude by their indices
    exclude_columns = [all_columns[i] for i in exclude_indices if i < len(all_columns)]
    
    # Read the CSV excluding specified columns by name
    df = pd.read_csv(input_csv_path, usecols=lambda column: column not in exclude_columns)
    
    return df

if __name__ == '__main__':
    main()