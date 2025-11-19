"""
Event handlers module for the Abético application
"""
import sys
import os
import time
import math
import gi

from gi.repository import GLib
from src.predictor import DiabetesPredictor

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

# Add parent directory to path to import src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class EventHandlers:
    """Class that contains the application's event handlers"""

    def __init__(self, window):
        self.window = window
        self.progress_start_time = 0  # To control the animation time
        self.predictor = DiabetesPredictor()  # Initialize the predictor
        self.prediction_result = None  # Store the prediction result

    # Helper methods
    def validate_questions(self):
        """Validates the questionnaire answers"""
        if (not self.window.age_entry.get_text().isdigit() or
            not self.window.bmi_entry.get_text().replace('.', '', 1).isdigit() or
            not self.window.glucose_entry.get_text().isdigit() or
            not self.window.family_history_entry.get_text().isdigit()):
            return False
        return True

    def convert_family_history_to_score(self, num_people):
        """Converts number of family members with diabetes to a score"""
        # Convert number of people to DiabetesPedigreeFunction score
        # 0 people = 0.2 (low genetic risk)
        # 1-2 people = 0.5 (moderate genetic risk)
        # 3+ people = 0.8 (high genetic risk)
        if num_people == 0:
            return 0.2
        elif num_people <= 2:
            return 0.5
        else:
            return 0.8

    def simulate_progress(self):
        """Simulates calculation progress with smooth curve (ease-in-out)"""

        # Calculate how much time has passed since the start
        elapsed = time.time() - self.progress_start_time
        duration = 0.5  # Total duration in seconds

        # If time has passed, finish
        if elapsed >= duration:
            self.window.progress_bar.set_fraction(1.0)
            self.window.progress_bar.set_visible(False)
            self.window.progress_bar.set_fraction(0.0)

            # Update result labels on the result page
            if self.prediction_result:
                probability = self.prediction_result['probability']
                risk_description = self.prediction_result['risk_description']

                # Update result page widgets
                self.window.result_percentage_label.set_text(f"A probabilidade é de {probability * 100:.1f}% de ter diabetes.")
                self.window.result_description_label.set_text(risk_description)

                # Remove previous classes
                style_context = self.window.result_percentage_label.get_style_context()
                style_context.remove_class("success")
                style_context.remove_class("warning")
                style_context.remove_class("error")

                # Add class based on probability
                percentage = probability * 100
                if percentage < 30:
                    style_context.add_class("success")
                elif percentage < 70:
                    style_context.add_class("warning")
                else:
                    style_context.add_class("error")

            # Switch to result page
            self.window.main_stack.set_visible_child_name("result")

            return False  # Stop the timeout

        # Calculate progress with ease-in-out curve (smooth)
        # t goes from 0 to 1 during animation
        t = elapsed / duration

        # Ease-in-out function: starts slow, accelerates in the middle, decelerates at the end
        # Uses sine function to create smooth curve
        smooth_progress = (1 - math.cos(t * math.pi)) / 2

        self.window.progress_bar.set_fraction(smooth_progress)

        return True  # Continue the timeout


    # Event handler methods
    def on_go_result(self, _button):
        """Validates and processes the questionnaire answers"""
        error_label = self.window.error_label

        if not self.validate_questions():
            error_label.set_visible(True)
            error_label.set_text("Preencha todos os campos corretamente.")

            # Hide the error after 2 seconds (2000ms)
            GLib.timeout_add(2000, lambda: error_label.set_visible(False))
            return
        else:
            error_label.set_visible(False)

            # Collect field values
            glucose = float(self.window.glucose_entry.get_text())
            bmi = float(self.window.bmi_entry.get_text())
            num_family_members = int(self.window.family_history_entry.get_text())
            age = int(self.window.age_entry.get_text())

            # Convert number of family members to DiabetesPedigreeFunction score
            family_history = self.convert_family_history_to_score(num_family_members)

            # Make prediction
            probability = self.predictor.predict(glucose, bmi, family_history, age)
            risk_description = self.predictor.get_risk_description(probability)

            # Store the result
            self.prediction_result = {
                'probability': probability,
                'risk_description': risk_description
            }

            print(f"Prediction: {probability * 100:.1f}% - Description: {risk_description}")

            # Mark animation start time
            self.progress_start_time = time.time()

            # Show progress bar and start from zero
            self.window.progress_bar.set_fraction(0.0)
            self.window.progress_bar.set_visible(True)

            # Update every 16ms (~60fps) for smooth animation
            GLib.timeout_add(16, self.simulate_progress)

            print("Processing answers...")

    def on_go_questions(self, _button):
        """Switch to questions page"""

        self.window.main_stack.set_visible_child_name("questions")
        self.window.back_header_button.set_visible(True)

    def on_back_to_home(self, _button):
        """Go back to home page"""
        self.window.main_stack.set_visible_child_name("home")
        self.window.back_header_button.set_visible(False)

    def on_submit_questions(self, button):
        """Alias for on_go_result (compatibility)"""
        self.on_go_result(button)
