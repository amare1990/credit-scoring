import pickle
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionInputSerializer

# Load the trained model using pickle
MODEL_PATH = "../data/saved_models/trained_model.pkl"
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

class PredictView(APIView):
    """
    View to handle prediction requests.
    """
    def post(self, request):
        serializer = PredictionInputSerializer(data=request.data)

        # Validate input
        if serializer.is_valid():
            try:
                # Convert input to DataFrame
                features = serializer.validated_data["features"]
                input_df = pd.DataFrame([features])

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
