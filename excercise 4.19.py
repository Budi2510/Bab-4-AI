# Pastikan dijalankan di lingkungan virtual/Python yang memiliki transformers dan torch
from transformers import pipeline

# Inisialisasi pipeline sentiment-analysis
classifier = pipeline("sentiment-analysis")

sentences = [
    "I really like this application.",
    "This movie is amazing.",
    "I hate this product.",
    "The service is very disappointing."
]

for sentence in sentences:
    result = classifier(sentence)

    print("Sentence:", sentence)
    print("Result:", result)
    print()