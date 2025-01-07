from joblib import dump
import pickle

# Load the original model
try:
    with open('models/fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
        print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: fake_news_model.pkl not found in the models directory.")
    exit()

# Compress and save the model
try:
    dump(model, 'models/fake_news_model_compressed.joblib', compress=3)
    print("Model compressed and saved successfully as fake_news_model_compressed.joblib.")
except Exception as e:
    print(f"Error while compressing the model: {e}")
