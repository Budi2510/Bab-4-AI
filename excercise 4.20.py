from transformers import pipeline

question_answerer = pipeline("question-answering")

context = '''
Politeknik Elektronika Negeri Surabaya is a university
in Indonesia. PENS focuses on electronics,
engineering, information technology, and multimedia.
The campus is located in Surabaya.
'''

questions = [
    "Where is PENS located?",
    "What does PENS focus on?",
    "What is the name of the university?"
]

for question in questions:
    result = question_answerer(
        question=question,
        context=context
    )

    print("Question:", question)
    print("Answer:", result["answer"])
    print("Score:", result["score"])
    print()
