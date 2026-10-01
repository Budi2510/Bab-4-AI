import nltk

text = '''
Artificial intelligence is a technology that allows
computers to learn from data and solve problems.
Deep learning is one important part of artificial intelligence.
'''

# Jalankan sekali jika resource NLTK belum tersedia:
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("averaged_perceptron_tagger_eng")

tokens = nltk.word_tokenize(text)
print("Tokens:")
print(tokens)

tagged = nltk.pos_tag(tokens)

print("\nPOS Tag:")
print(tagged)
