import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from pymongo.mongo_client import MongoClient
import fastapi

app = fastapi.FastAPI()


uri = "mongodb+srv://nandini_a:test123A456@cluster0.xdjalu6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client["music_db"]
songs_collection = db["songs"]

df = pd.DataFrame(list(songs_collection.find({}, {"_id": 0})))
df = df.dropna(subset=["Song_Name"])
df["Song_Name"] = df["Song_Name"].astype(str)

print("Data loaded. Total songs:", len(df))

text = " ".join(df['Song_Name'].tolist())
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1


input_sequences = []
for line in df['Song_Name']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_sequences.append(n_gram_seq)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:, :-1]
y = input_sequences[:, -1]


model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len - 1),
    LSTM(150),
    Dense(total_words, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X, y, epochs=5, steps_per_epoch=50,verbose=1)





predicted_songs = set()

def predict_next_song(seed_text, df, next_words=5):
   
    global predicted_songs

    song_row = df[df['Song_Name'].str.lower() == seed_text.lower()]
    if song_row.empty:
        print(" Song not found in dataset.")
        return None, None

    emotion = song_row['Sentiment_Label'].iloc[0]
    Genre = song_row['Genre'].iloc[0] if 'Genre' in df.columns else None

    print(f"🎭 Emotion detected for '{seed_text}': {emotion}")
    if Genre:
        print(f" genre detected: {Genre}")

    
    generated_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted = np.argmax(predicted_probs)

        output_word = next((w for w, i in tokenizer.word_index.items() if i == predicted), "")
        if output_word == "" or output_word in generated_text.split():
            break
        generated_text += " " + output_word
    if Genre:
        candidate_songs = df[
            (df['Genre'].str.lower() == Genre.lower()) &
            (df['Sentiment_Label'].str.lower() == emotion.lower())
        ]
    else:
        candidate_songs = df[df['Sentiment_Label'].str.lower() == emotion.lower()]

        candidate_songs = candidate_songs[
        ~candidate_songs['Song_Name'].isin(predicted_songs) &
        (candidate_songs['Song_Name'].str.lower() != seed_text.lower())
    ]

    if candidate_songs.empty:
        candidate_songs = df[
            (df['Sentiment_Label'].str.lower() == emotion.lower()) &
            ~df['Song_Name'].isin(predicted_songs) &
            (df['Song_Name'].str.lower() != seed_text.lower())
        ]
        print(" No more songs found for this genre — using emotion-only prediction.")

    if candidate_songs.empty:
        print(" No unseen songs left for this emotion.")
        return None, emotion

    matches = candidate_songs[
        candidate_songs['Song_Name'].str.contains(output_word, case=False, na=False)
    ]

    if not matches.empty:
        full_name = matches.iloc[0]['Song_Name']
    else:
       
        full_name = candidate_songs.iloc[0]['Song_Name']

    predicted_songs.add(full_name)
    songs_collection.insert_one({
        "Input_Song": seed_text,
        "Predicted_Song": full_name,
        "Emotion": emotion,
        "Genre": Genre,
        "Generated_Text": generated_text
    })

    return {"predicted_song": predicted_songs,"emotion": emotion}

