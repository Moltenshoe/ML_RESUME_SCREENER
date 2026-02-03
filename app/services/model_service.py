import os
import joblib
import numpy as np

class ModelService:
    def __init__(self):
        self.model, self.vectorizer = self._load_latest_model()

    def _load_latest_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        artifacts_dir = os.path.join(base_dir, "artifacts")

        if not os.path.exists(artifacts_dir):
            raise Exception("Artifacts folder not found.")

        folders = sorted(os.listdir(artifacts_dir), reverse=True)

        if not folders:
            raise Exception("No trained models found.")

        latest_folder = os.path.join(artifacts_dir, folders[0])

        model = joblib.load(os.path.join(latest_folder, "resume_model.pkl"))
        vectorizer = joblib.load(os.path.join(latest_folder, "vectorizer.pkl"))

        return model, vectorizer

    def predict_top3(self, text: str):
        vectorized = self.vectorizer.transform([text])

        scores = self.model.decision_function(vectorized)[0]
        classes = self.model.classes_

        top_indices = np.argsort(scores)[::-1][:3]

        results = []
        for idx in top_indices:
            results.append((classes[idx], round(float(scores[idx]), 3)))

        return results
