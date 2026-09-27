"""
Django API views for the Senti-Pred project.
"""
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
import json

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'sentiment_model.pkl')


class SentimentPredictionView(APIView):
    """
    API for sentiment prediction.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the model if it exists
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = None
    
    def post(self, request):
        """
        Endpoint for sentiment prediction.
        
        Expects a JSON with the 'text' field containing the text to analyze.
        Returns the sentiment prediction and the probabilities.
        """
        if self.model is None:
            return Response(
                {"error": "Model not found. Train the model first."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        try:
            # Get the text from the request
            data = json.loads(request.body)
            text = data.get('text', '')
            
            if not text:
                return Response(
                    {"error": "The 'text' field is required."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Make the prediction
            sentiment = self.model.predict([text])[0]
            
            # Get probabilities if available
            try:
                probabilities = self.model.predict_proba([text])[0].tolist()
                classes = self.model.classes_.tolist()
                probs_dict = {str(cls): prob for cls, prob in zip(classes, probabilities)}
            except:
                probs_dict = {}
            
            # Return the result
            return Response({
                "text": text,
                "sentiment": sentiment,
                "probabilities": probs_dict
            })
            
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ModelInfoView(APIView):
    """
    API for information about the model.
    """
    
    def get(self, request):
        """
        Returns information about the loaded model.
        """
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            
            # Extract model information
            model_type = type(model).__name__
            
            # Check whether it is a pipeline
            if hasattr(model, 'steps'):
                steps = [step[0] for step in model.steps]
                classifier = type(model.steps[-1][1]).__name__
            else:
                steps = []
                classifier = model_type
            
            return Response({
                "model_loaded": True,
                "model_type": model_type,
                "pipeline_steps": steps,
                "classifier": classifier,
                "model_path": MODEL_PATH
            })
        else:
            return Response({
                "model_loaded": False,
                "error": "Model not found"
            })


def health_check(request):
    """
    Simple endpoint to check that the API is working.
    """
    return JsonResponse({"status": "ok"})