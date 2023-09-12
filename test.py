from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Пример текстовых данных (корпус)
corpus = [
    "яблоко красное",
    "груша зеленая",
    "яблоко вкусное",
    "груша сочная",
    "банан желтый"
]

# Разбиваем текст на слова (токенизация)
tokenized_corpus = [sentence.split() for sentence in corpus]

# Создаем модель Word2Vec
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=0)

# Сохраняем модель (опционально)
model.save("word2vec.model")

# Загружаем сохраненную модель (при необходимости)
# model = Word2Vec.load("word2vec.model")

# Извлекаем вектор для слова "яблоко"
vector_apple = model.wv["яблоко"]
print("Вектор слова 'яблоко':", vector_apple)

# Находим наиболее близкие слова к "яблоко"
similar_words = model.wv.most_similar("яблоко", topn=3)
print("Наиболее близкие слова к 'яблоко':", similar_words)