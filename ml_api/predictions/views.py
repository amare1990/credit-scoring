import pickle
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionInputSerializer

# Directory containing the saved models
MODELS_PATH = "../../data/models/saved_models/"

# A dictionary to hold model names and their corresponding file paths
models = {
    "logistic_regression": "Logistic_Regression.pkl",
    "decision_tree": "Decision_Tree.pkl",
    "random_forest": "Random_Forest.pkl",
    "gradient_boosting": "Gradient_Boosting.pkl",
    "tuned_random_forest": "Tuned_Random_Forest.pkl"
}

# Load all the models into a dictionary
loaded_models = {}
for model_name, model_filename in models.items():
    try:
        with open(MODELS_PATH + model_filename, 'rb') as model_file:
            loaded_models[model_name] = pickle.load(model_file)
            print(f"Model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{model_name}': {str(e)}")

class PredictView(APIView):
    """
    View to handle prediction requests.
    """
    def post(self, request):
        serializer = PredictionInputSerializer(data=request.data)

        # Validate input
        if serializer.is_valid():
            try:
                # Get model name from the request
                model_name = request.data.get("model_name", "").lower()

                # Check if the model name is valid
                if model_name not in loaded_models:
                    return Response({"error": "Invalid model name. Please choose a valid model."},
                                    status=status.HTTP_400_BAD_REQUEST)

                # Get the input features
                features = serializer.validated_data["features"]
                input_df = pd.DataFrame([features])

                # Get the selected model
                model = loaded_models[model_name]

                # Make prediction
                prediction = model.predict(input_df)[0]
                probability = None
                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(input_df)[0].max()

                return Response({
                    "prediction": int(prediction),
                    "probability": float(probability) if probability else None
                }, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"error": f"Prediction error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
