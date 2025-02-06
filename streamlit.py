import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder



# Sidebar for project selection
st.sidebar.title('Project Selection')
project_1 = st.sidebar.checkbox('Flight Price Prediction')  
project_2 = st.sidebar.checkbox('Customer Satisfaction Prediction ')
# Apply custom styles
st.markdown(
    """
    <style>
    /* Change background color of the sidebar */
    [data-testid="stSidebar"] {
        background-color:#8c6908;
        color:#fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if project_1:
        # Load the trained machine learning model
        model = joblib.load('modelpricev1.joblib')  # Replace with your model file path

        # Title of the app
        st.title('Flight Price Prediction')
        st.write('Enter the details below to predict the flight price.')

        # User inputs
        st.header('Flight Details')
        airline = st.selectbox('Airline', ['Indigo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia','Multiple carriers Premium economy','Jet Airways Business', 'Vistara Premium economy', 'Trujet'])
        source = st.selectbox('Source', ['Delhi', 'Kolkata', 'Mumbai', 'Chennai', 'Bangalore'])
        destination = st.selectbox('Destination', ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'Bangalore'])
        total_stops = st.selectbox('Total Stops', ['0', '1', '2', '3', '4'])
        additional_info = st.selectbox('Additional Info', ['No info', 'In-flight meal not included', 'No check-in baggage included', '1 Long layover', 'Change airports', 'Business class', 'Red-eye flight', '2 Long layover'])

        # Input for date
        date_input = st.date_input("Select a date")

        # Extract day, month, and year
        date = date_input.day
        month = date_input.month
        year = date_input.year

        # Time input for hours and minutes (three different time inputs with unique keys)
        time_input1 = st.time_input("Duration_time", value=datetime(2025, 1, 1, 12, 0).time(), key="time_input_1")
        time_input2 = st.time_input("Deparature_time", value=datetime(2025, 1, 1, 13, 0).time(), key="time_input_2")
        time_input3 = st.time_input("Arrivial_time", value=datetime(2025, 1, 1, 14, 0).time(), key="time_input_3")
        #hour, and minute

        duration_hours= time_input1.hour
        duration_minutes= time_input1.minute

        #hour, and minute
        departure_hour= time_input2.hour
        departure_minute= time_input2.minute



        #hour, and minute
        arrival_hour= time_input3.hour
        arrival_minute= time_input3.minute

        # Prepare input data
        if st.button('Predict'):
            input_data = {
                'Airline': [airline],
                'Source': [source],
                'Destination': [destination],
                'Total_Stops': [total_stops],
                'Additional_Info': [additional_info],
                'Date': [date],
                'Month': [month],
                'Year': [year],
                'Duration_hours': [duration_hours],
                'Duration_minutes': [duration_minutes],
                'Departure_hour': [departure_hour],
                'Departure_minute': [departure_minute],
                'Arrival_hour': [arrival_hour],
                'Arrival_minute': [arrival_minute]
            }

            # Convert input into DataFrame
            df = pd.DataFrame(input_data)


            # Preprocess the input data (if required)
            # Example: Encode categorical variables
            from sklearn.preprocessing import LabelEncoder

            categorical_columns = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info']
            for col in categorical_columns:
                label_encoder = LabelEncoder()
                df[col] = label_encoder.fit_transform(df[col])

            # Make prediction
            prediction = model.predict(df)

            # Display the prediction result
            st.write(f'Predicted Flight Price: **{prediction[0]:.2f}**')

            



elif project_2:
    # Load the trained model
    model = joblib.load('modelv2.joblib')

    # Title and description of the app
    st.title('Customer Satisfaction Prediction')
    st.write('Provide the details to predict the flight satisfaction level.')

    # User inputs
    gender = st.selectbox('Gender', ['Male', 'Female'])
    customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'Disloyal Customer'])
    age = st.number_input('Age', min_value=1, max_value=100, step=5)
    type_of_travel = st.selectbox('Type of Travel', ['Business', 'Personal'])
    flight_class = st.selectbox('Class', ['Eco', 'Eco Plus', 'Business'])
    flight_distance = st.number_input("Enter a value", min_value=300, max_value=2000,step=50)
    # Survey-related inputs (assuming a scale of 1-5 for rating)
    inflight_wifi_service = st.slider('Inflight Wifi Service', 1, 5)
    departure_arrival_time = st.number_input('Departure/Arrival time convenient', min_value=0, max_value=5)
    online_booking = st.number_input('Ease of Online booking', min_value=0, max_value=5)
    gate_location = st.slider('Gate location', 1, 5)
    food_and_drink = st.number_input('Food and drink', min_value=0, max_value=5)
    online_boarding = st.slider('Online boarding', 1, 5)
    seat_comfort = st.slider('Seat comfort', 1, 5)
    inflight_entertainment = st.number_input('Inflight entertainment', min_value=0, max_value=5)
    onboard_service = st.slider('On-board service', 1, 5)
    leg_room_service = st.number_input('Leg room service', min_value=0, max_value=5)
    baggage_handling = st.slider('Baggage handling', 1, 5)
    checkin_service = st.number_input('Checkin service', min_value=0, max_value=5)
    inflight_service = st.slider('Inflight service', 1, 5)
    cleanliness = st.slider('Cleanliness', 1, 5)

    # Add departure and arrival delay inputs
    time_input4 = st.time_input("Duration_time", value=datetime(2025, 1, 1, 12, 0).time(), key="time_input_1")
    time_input5 = st.time_input("Deparature_time", value=datetime(2025, 1, 1, 13, 0).time(), key="time_input_2")
    departure_delay = time_input4.minute
    arrival_delay =time_input5.minute


    # Prepare input data, add one-hot encoding for categorical variables
    if st.button('Predict'):
            
        input_dict = {
            'Gender': [gender],
            'Customer Type': [customer_type],
            'Age': [age],
            'Type of Travel': [type_of_travel],
            'Class': [flight_class],
            'Flight Distance': [flight_distance],
            'Inflight wifi service': [inflight_wifi_service],
            'Departure/Arrival time convenient': [departure_arrival_time],
            'Ease of Online booking': [online_booking],
            'Gate location': [gate_location],
            'Food and drink': [food_and_drink],
            'Online boarding': [online_boarding],
            'Seat comfort': [seat_comfort],
            'Inflight entertainment': [inflight_entertainment],
            'On-board service': [onboard_service],
            'Leg room service': [leg_room_service],
            'Baggage handling': [baggage_handling],
            'Checkin service': [checkin_service],
            'Inflight service': [inflight_service],
            'Cleanliness': [cleanliness],
            'Departure Delay in Minutes': [departure_delay],
            'Arrival Delay in Minutes': [arrival_delay]
        }

        # Convert input into DataFrame (required for one-hot encoding)
        df = pd.DataFrame(input_dict)
        #Encodeing data for Gender
        category={'Male':1,'Female':0}
        df['Gender']=df['Gender'].map(category)

        #Encoding data for customer type
        from sklearn.preprocessing import LabelEncoder
        model1=LabelEncoder()
        df['Customer Type']=model1.fit_transform(df['Customer Type'])

        #Encoding data for class column
        df['Class']=model1.fit_transform(df['Class'])

        #Encoder data for Type of Travel column
        df['Type of Travel']=model1.fit_transform(df['Type of Travel'])

        # Make prediction
        prediction = model.predict(df)
        satisfaction = 'Satisfied' if prediction == 1 else 'Not Satisfied'
        st.write(f'The predicted customer satisfaction is: **{satisfaction}**')
        
else:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Flight Price Prediction Description", "Customer Satisfaction Prediction Description"])

    with tab1:
        # Title and description of the app
        st.markdown(
            """
            <style>
            .header {
                text-align: center;
                font-size: 40px;
                font-weight: bold;
                font-family: 'caveat';
                color: #454033;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.markdown('<p class="header">Machine Learning Projects</p>', unsafe_allow_html=True)
        # Inject custom CSS to change the full page background color
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #c4c4c2; 
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    with tab2:

        st.markdown(
            """
            <style>
            .header1 {
                text-align: center;
                font-size: 30px;
                font-weight: bold;
                font-family: 'caveat';
                color: #000;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.markdown('<h1 class="header1">Problem Statement</h1>', unsafe_allow_html=True)


        st.write("Build an end-to-end project to predict flight ticket prices based on multiple factors such as departure time, source, destination, and airline type. Use the provided dataset to process, clean, and perform feature engineering. Train a regression model to predict flight prices and deploy the model in a Streamlit application. The app should allow users to input filters (route, time, and date) and get a predicted price for their flight.")

        st.markdown('<h1 class="header1">Business Use Cases</h1>', unsafe_allow_html=True)

        # Bulleted list using markdown
        st.markdown("""
        - Helping travelers plan trips by predicting flight prices based on their preferences.
        - Assisting travel agencies in price optimization and marketing strategies.
        - Enabling businesses to budget for employee travel by forecasting ticket prices.
        - Supporting airline companies in identifying trends and optimizing pricing strategies.
        """)
        
        st.markdown('<h1 class="header1">Technical Tags</h1>', unsafe_allow_html=True)
        # Bulleted list using markdown
        st.markdown("""
        - Python
        - Data Cleaning
        - Feature Engineering
        - Machine Learning
        - Regression
        - Streamlit
        - MLflow
        """)

        st.markdown('<h1 class="header1">Dataset</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <style>
            .header2 {
                font-size: 24px;
                font-weight: bold;
                font-family: 'caveat';
                color: #000;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.markdown('<p class="header2">Dataset includes:</>', unsafe_allow_html=True)
        # Bulleted list using markdown
        st.markdown("""
        - **Airline**: Name of the airline.
        - **Date_of_Journey**: Date of takeoff.
        - **Source**: Starting airport location.
        - **Destination**: Final landing airport location.
        - **Route**: The route from where the plane will go and stops.
        - **Dep_Time**: Departure time.
        - **Arrival_Time**: Arrival time of the plane landing.
        - **Duration**: How long the flight lasted.
        - **Total_Stops**: Number of stops between flights for fuel, etc.
        - **Additional_Info**: Additional notes from the airline (e.g., meal not included).
        """)

        st.markdown('<h1 class="header1">Result</h1>', unsafe_allow_html=True)
        # Bulleted list using markdown
        st.markdown("""
        - Cleaned and processed dataset for analysis.
        - Built regression models with predictions achieving high accuracy.
        - Developed a user-friendly Streamlit app to analyze and predict flight prices.
        """)

    with tab3:
        st.markdown(
            """
            <style>
            .header1 {
                text-align: center;
                font-size: 30px;
                font-weight: bold;
                font-family: 'caveat';
                color: #000;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.markdown('<h1 class="header1">Problem Statement</h1>', unsafe_allow_html=True)


        st.write("Build a classification model to predict customer satisfaction levels based on features such as customer feedback, demographics, and service ratings. Use the provided dataset to process, clean, and perform feature engineering. Deploy the model in a Streamlit application, allowing users to input customer data and receive a predicted satisfaction level.")

        st.markdown('<h1 class="header1">Business Use Cases</h1>', unsafe_allow_html=True)

        # Bulleted list using markdown
        st.markdown("""
        - Enhancing customer experience by predicting and addressing dissatisfaction.
        - Providing actionable insights for businesses to improve services.
        - Supporting marketing teams in identifying target customer groups.
        - Assisting management in decision-making for customer retention strategies.
        """)
        
        st.markdown('<h1 class="header1">Technical Tags</h1>', unsafe_allow_html=True)
        # Bulleted list using markdown
        st.markdown("""
        - Python
        - Data Cleaning
        - Feature Engineering
        - Machine Learning
        - Regression
        - Streamlit
        - MLflow
        """)

        st.markdown('<h1 class="header1">Dataset</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <style>
            .header2 {
                font-size: 24px;
                font-weight: bold;
                font-family: 'caveat';
                color: #000;
            }
            </style>
            """, unsafe_allow_html=True
        )
        st.markdown('<p class="header2">Dataset includes:</>', unsafe_allow_html=True)
        # Bulleted list using markdown
        # Bulleted list using markdown
        st.markdown("""
        - **Gender**: Gender of the passengers (Female, Male)
        - **Customer Type**: The customer type (Loyal customer, disloyal customer)
        - **Age**: The actual age of the passengers
        - **Type of Travel**: Purpose of the flight of the passengers (Personal Travel, Business Travel)
        - **Class**: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
        - **Flight distance**: The flight distance of this journey
        - **Inflight wifi service**: Satisfaction level of the inflight wifi service (0:Not Applicable; 1-5)
        - **Departure/Arrival time convenient**: Satisfaction level of Departure/Arrival time convenient
        - **Ease of Online booking**: Satisfaction level of online booking
        - **Gate location**: Satisfaction level of Gate location
        - **Food and drink**: Satisfaction level of Food and drink
        - **Online boarding**: Satisfaction level of online boarding
        - **Seat comfort**: Satisfaction level of Seat comfort
        - **Inflight entertainment**: Satisfaction level of inflight entertainment
        - **On-board service**: Satisfaction level of On-board service
        - **Leg room service**: Satisfaction level of Leg room service
        - **Baggage handling**: Satisfaction level of baggage handling
        - **Check-in service**: Satisfaction level of Check-in service
        - **Inflight service**: Satisfaction level of inflight service
        - **Cleanliness**: Satisfaction level of Cleanliness
        """)


        st.markdown('<h1 class="header1">Result</h1>', unsafe_allow_html=True)
        # Bulleted list using markdown
        st.markdown("""
        - Cleaned and processed dataset for analysis.
        - Built classification models with predictions achieving high accuracy, tracked using MLflow.
        - Developed a user-friendly Streamlit app to analyze and predict customer satisfaction levels.
        """)





