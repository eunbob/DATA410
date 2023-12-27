import os
import pandas as pd
import matplotlib.pyplot as plt

from konlpy.tag import Mecab
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE


dir_path = os.getcwd()
os.chdir(dir_path)
train_data = pd.read_csv('ratings_train.txt', sep="\t", encoding='utf-8-sig', header=0)

#데이터 전처리
print(train_data.isnull().values.any())
train_data = train_data.dropna(how = 'any')

train_data['document'] = train_data['document'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣]', " ")

stop_words = []
with open('stopwords.txt', 'r', encoding='utf-8') as file:
    for line in file:
        stop_words.append(line.strip())

tokenized_data = []
for sentence in train_data['document']:
    temp = Mecab().morphs(sentence)
    temp = [word for word in temp if word not in stop_words and len(word) > 1] 
    tokenized_data.append(temp)
    
model = Word2Vec(sentences = tokenized_data, vector_size = 128, window = 5, min_count = 5, workers = 4, sg = 0)
word_vectors = model.wv.vectors

#차원 축소
tsne = TSNE(n_components=2)
reduced_vec = tsne.fit_transform(word_vectors)

plt.rc('font', family='AppleGothic')
plt.figure(figsize=(100,100))
plt.scatter(reduced_vec[:,0], reduced_vec[:,1], c = 'r', marker = 'x')

for i, word in enumerate(model.wv.index_to_key):
    plt.annotate(word, xy=(reduced_vec[i, 0], reduced_vec[i, 1]))

plt.savefig("output.png")
