import os
import openai
import pickle
import pandas as pd

from flask import Flask, render_template, request, session
from flask_session import Session
from dotenv import load_dotenv

app = Flask(__name__)

# Configure session
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')  
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data in the filesystem
Session(app)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the trained model pipeline
with open('budget_prediction_model.pkl', 'rb') as file:
    budget_model = pickle.load(file)


@app.route("/")
def home():
    """Renders the home page.

    Returns:
        The rendered home page template.
    """
    return render_template("home.html")




@app.route("/submit", methods=["POST"])
def submit():
    """Handles the submission of the form on the home page.

    Retrieves the location, activities and length of trip from the form data.
    Sends an API request to OpenAI to generate a travel itinerary based on the form data.
    Processes the response from OpenAI and renders the response page with the generated itinerary.

    Returns:
        The rendered response page template with the generated itinerary.
    """
    location = request.form.get("location")
    activities = request.form.get("activities")
    length = request.form.get("length")

    # Send the API request to OpenAI
    prompt = f"Generate a {length}-day travel for {location} with {activities}. For each day, try to recommend some locations along with the activities for that location. Make sure to include a short 2 - 3 sentence description for the locations!" \
             f"Each day MUST look exactly like this: " \
             f"Day 4: Roatán Island. Take a ferry or a short flight to Roatán Island, one of Honduras' most popular tourist destinations. Roatán Island is a Caribbean paradise located off the northern coast of Honduras. Known for its stunning beaches, crystal-clear waters, and vibrant coral reefs, it is a popular destination for snorkeling, scuba diving, and other water activities. The island also offers a range of restaurants, bars, and accommodations to suit any budget. Spend the day exploring the island, snorkeling, or scuba diving in the coral reefs."
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # or "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful travel itinerary planner."},
        {"role": "user", "content": prompt},
    ],
    max_tokens=1024,
    temperature=0.7
).choices[0].message["content"]

    response = process_response(response)

    return render_template("response.html",
                           response=response,
                           location=location,
                           activities=activities,
                           length=length,
                           title=f'Itinerary for {location}',
                           )




@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Handle chat interactions."""
    # Initialize chat history in session if it doesn't exist
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_message = request.form['message']

        # Generate response from OpenAI
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Use GPT-3.5-turbo or GPT-4
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_message}
                ]
            )
            bot_response = response['choices'][0]['message']['content']
        except Exception as e:
            bot_response = f"Error: {str(e)}"

        # Save the message and response in session chat history
        session['chat_history'].append({'user': user_message, 'bot': bot_response})
        session.modified = True

    # Render the chat page with chat history
    return render_template('chat.html', chat_history=session['chat_history'])

@app.route('/clear', methods=['GET'])
def clear_chat():
    """Clear the chat history for the current session."""
    print("Before clearing:", session.get('chat_history'))  # Debugging
    if 'chat_history' in session:
        session.pop('chat_history')
    print("After clearing:", session.get('chat_history'))  # Debugging
    session.modified = True
    return render_template('chat.html', chat_history=[])



@app.route("/predict_budget", methods=["GET", "POST"])
def predict_budget():
    if request.method == "POST":
        try:
            # Collect form inputs
            destination = request.form.get("destination")
            trip_duration = int(request.form.get("trip_duration"))
            accommodation_type = request.form.get("accommodation_type")
            accommodation_cost = float(request.form.get("accommodation_cost"))
            activity_preference = request.form.get("activity_preference")
            activity_cost = float(request.form.get("activity_cost"))
            dining_preference = request.form.get("dining_preference")
            dining_cost = float(request.form.get("dining_cost"))
            transportation_cost = float(request.form.get("transportation_cost"))
            flight_cost = float(request.form.get("flight_cost"))
            seasonality_factor = float(request.form.get("seasonality_factor"))

            # Create input DataFrame
            input_data = pd.DataFrame([{
                "Destination": destination,
                "Trip Duration (Days)": trip_duration,
                "Accommodation Type": accommodation_type,
                "Accommodation Cost (per day)": accommodation_cost,
                "Activity Preference": activity_preference,
                "Activity Cost (per day)": activity_cost,
                "Dining Preference": dining_preference,
                "Dining Cost (per day)": dining_cost,
                "Transportation Cost": transportation_cost,
                "Flight Cost": flight_cost,
                "Seasonality Factor": seasonality_factor
            }])

            # Predict the budget
            predicted_budget = budget_model.predict(input_data)[0]

            return render_template(
                "budget.html",
                budget=predicted_budget,
                destination=destination,
                trip_duration=trip_duration,
                accommodation_type=accommodation_type,
                accommodation_cost=accommodation_cost,
                activity_preference=activity_preference,
                activity_cost=activity_cost,
                dining_preference=dining_preference,
                dining_cost=dining_cost,
                transportation_cost=transportation_cost,
                flight_cost=flight_cost,
                seasonality_factor=seasonality_factor
            )
        except Exception as e:
            return str(e)

    return render_template("budget.html", budget=None)






def process_response(response):
    """Processes the response from OpenAI.

    Splits the response into a list of lists where each inner list contains the day number and the itinerary for that day.

    Args:
        response: The response from OpenAI.

    Returns:
        The processed response as a list of lists.
    """
    response = response.replace('\n', '').split('Day')[1:]
    response = list([[item.split('.')[0], '.'.join(item.split('.')[1:])] for item in response])
    return response


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
