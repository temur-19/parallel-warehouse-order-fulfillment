import dotenv
import os

dotenv.load_dotenv()

print({"BOT_TOKEN":os.getenv("TELEGRAM_BOT_TOKEN"),
       "ADMIN_ID":os.getenv("TELEGRAM_ADMIN_ID")})

import joblib

file_path = "app/ml/model.pkl"

try:
    model = joblib.load(file_path)

    print("Model muvaffaqiyatli yuklandi!")
    print("Model turi:", type(model))

    if hasattr(model, "steps"):
        print("\nPipeline bosqichlari:")
        for name, step in model.steps:
            print(f"{name}: {type(step).__name__}")

except Exception as error:
    print("Xatolik:", repr(error))

import joblib

model = joblib.load("app/ml/model.pkl")

print("Model:", type(model).__name__)

print("\nPipeline bosqichlari:")
for name, step in model.steps:
    print(f"{name}: {type(step).__name__}")

print("\nModel parametrlari:")
print(model.named_steps["classifier"].get_params())

print("\nKutilayotgan xususiyatlar:")
print(model.feature_names_in_
      if hasattr(model, "feature_names_in_")
      else "Pipeline darajasida mavjud emas")