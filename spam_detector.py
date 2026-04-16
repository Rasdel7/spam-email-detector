import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
import urllib.request
import warnings
warnings.filterwarnings('ignore')

url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print("Dataset loaded! Shape:", df.shape)
print("\nSample data:")
print(df.head())
print("\nSpam vs Ham distribution:")
print(df['label'].value_counts())


df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})


df['message_length'] = df['message'].apply(len)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df[df['label'] == 'ham']['message_length'].hist(
    bins=50, ax=axes[0], color='#2ecc71', edgecolor='black')
axes[0].set_title('Ham Messages')
axes[0].set_xlabel('Message Length')

df[df['label'] == 'spam']['message_length'].hist(
    bins=50, ax=axes[1], color='#e74c3c', edgecolor='black')
axes[1].set_title('Spam Messages')
axes[1].set_xlabel('Message Length')

plt.suptitle('Message Length: Ham vs Spam', fontsize=14)
plt.tight_layout()
plt.savefig('message_length.png')
print("\nMessage length chart saved!")

X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)


models = {
    'Naive Bayes':        MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc   = accuracy_score(y_test, preds)
    results[name] = round(acc * 100, 2)
    print(f"\n{name} — Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds,
                                target_names=['Ham', 'Spam']))

best_name = max(results, key=results.get)
best_model = list(models.values())[list(models.keys()).index(best_name)]
best_preds = best_model.predict(X_test_tfidf)

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title(f'Confusion Matrix — {best_name}', fontsize=13)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved!")


print("\n--- Testing custom messages ---")
test_messages = [
    "Congratulations! You won a free iPhone. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your bank account has been compromised. Call now!",
    "Can you send me the notes from today's class?"
]

best_vectorized = tfidf.transform(test_messages)
predictions = best_model.predict(best_vectorized)
for msg, pred in zip(test_messages, predictions):
    label = "🚨 SPAM" if pred == 1 else "✅ HAM"
    print(f"{label} → {msg[:60]}...")

print("\nDone! Check your folder for both charts.")