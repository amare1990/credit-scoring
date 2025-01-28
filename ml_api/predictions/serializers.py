from rest_framework import serializers

class PredictionInputSerializer(serializers.Serializer):
    features = serializers.ListField(
        child=serializers.FloatField(),
        min_length=1,
        help_text="List of numerical features for prediction."
    )
