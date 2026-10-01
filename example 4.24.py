# Example 4.24 - Question Answering (Text-based)
# Pastikan dijalankan di dalam venv Python 3.11 yang sudah menginstal:
# pip install transformers torch

from transformers import pipeline

# Inisialisasi pipeline question-answering
question_answerer = pipeline('question-answering')

# Format pemanggilan dengan argumen standar yang bersih dari karakter tersembunyi
result = question_answerer(
    question='What is the name of the company?',
    context='We created Biox Systems Ltd company back in the year of 2000.'
)

print(result)