# ─── Instalações ───────────────────────────────────────────────────────────────
# ─── Imports ───────────────────────────────────────────────────────────────────
import kagglehub
from kagglehub import KaggleDatasetAdapter

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Reprodutibilidade
SEED = 42
np.random.seed(SEED)

print('Imports OK ✔')
# ─── Carrega treino ────────────────────────────────────────────────────────────
train_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "jp797498e/twitter-entity-sentiment-analysis",
    "twitter_training.csv",
)

# ─── Carrega validação / teste ─────────────────────────────────────────────────
val_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "jp797498e/twitter-entity-sentiment-analysis",
    "twitter_validation.csv",
)

print(f'Treino : {train_df.shape}')
print(f'Validação: {val_df.shape}')
train_df.head()
# ─── Nomes de colunas padronizados ─────────────────────────────────────────────
# O dataset não possui header; as colunas são: id, entity, sentiment, text
COL_NAMES = ['id', 'entity', 'sentiment', 'text']

if train_df.shape[1] == 4:
    train_df.columns = COL_NAMES
    val_df.columns   = COL_NAMES

print('Sentimentos únicos:', train_df['sentiment'].unique())
print('Distribuição no treino:')
print(train_df['sentiment'].value_counts())
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribuição de sentimentos
sentiment_counts = train_df['sentiment'].value_counts()
colors = ['#4CAF50', '#F44336', '#2196F3', '#FF9800']
sentiment_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='white', linewidth=0.8)
axes[0].set_title('Distribuição de Sentimentos (Treino)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sentimento')
axes[0].set_ylabel('Quantidade')
axes[0].tick_params(axis='x', rotation=30)
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height()):,}',
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha='center', va='bottom', fontsize=10)

# Comprimento dos tweets
train_df['text_len'] = train_df['text'].fillna('').apply(len)
train_df.groupby('sentiment')['text_len'].plot(kind='kde', ax=axes[1], legend=True)
axes[1].set_title('Distribuição do Tamanho dos Tweets por Sentimento', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Número de caracteres')
axes[1].set_ylabel('Densidade')

plt.tight_layout()
plt.show()
def clean_tweet(text: str) -> str:
    """Limpeza básica de tweet para análise de sentimento."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # URLs
    text = re.sub(r'@\w+', '', text)                     # Menções
    text = re.sub(r'#(\w+)', r'\1', text)               # Hashtags → palavra
    text = re.sub(r'[^a-z0-9\s!?.,\'\-]', '', text)    # Caracteres especiais
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Aplica limpeza
for df in (train_df, val_df):
    df['clean_text'] = df['text'].apply(clean_tweet)

# Remove registros sem texto e sem sentimento válido
valid_sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
train_df = train_df[
    train_df['clean_text'].str.len() > 0 &
    train_df['sentiment'].isin(valid_sentiments)
].copy()

val_df = val_df[
    val_df['clean_text'].str.len() > 0 &
    val_df['sentiment'].isin(valid_sentiments)
].copy()

print(f'Treino após limpeza : {train_df.shape}')
print(f'Validação após limpeza: {val_df.shape}')
train_df[['text', 'clean_text', 'sentiment']].head(3)
# ─── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(valid_sentiments)

y_train = le.transform(train_df['sentiment'])
y_val   = le.transform(val_df['sentiment'])

X_train_text = train_df['clean_text'].values
X_val_text   = val_df['clean_text'].values

print('Classes:', dict(zip(le.classes_, le.transform(le.classes_))))
# ─── TF-IDF Vectorizer compartilhado ──────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=70_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    strip_accents='unicode',
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_val_tfidf   = tfidf.transform(X_val_text)

print(f'Shape TF-IDF treino : {X_train_tfidf.shape}')
print(f'Shape TF-IDF validação: {X_val_tfidf.shape}')
# ─── Definição dos modelos ─────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=11.0, solver='lbfgs', multi_class='multinomial', random_state=SEED
    ),
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=100, random_state=SEED, n_jobs=-1
    ),
    'Linear SVC': LinearSVC(
        C=19.0, max_iter=1000, random_state=SEED
    ),
    'Passive Aggressive': PassiveAggressiveClassifier(
        C=1.0, max_iter=1000, random_state=SEED
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=7, metric='cosine', n_jobs=-1
    ),
    'Ridge Classifier': RidgeClassifier(
        alpha=1.0
    ),
    'SGD Classifier': SGDClassifier(
        loss='modified_huber', max_iter=1000, random_state=SEED, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f'Treinando {name}...', end=' ')
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_val_tfidf)
    acc   = accuracy_score(y_val, preds)
    f1    = f1_score(y_val, preds, average='weighted')
    results[name] = {'acc': acc, 'f1': f1, 'preds': preds}
    print(f'Acc={acc:.4f}  F1={f1:.4f} ✔')
print('='*60)
print('Treinando Mamba (state-spaces/mamba-130m-hf)')
print('='*60)

import os
os.environ['USE_TF'] = '0' # Previne conflitos de protobuf

import torch
import torch.nn as nn
from transformers import MambaModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn.metrics import accuracy_score, f1_score

mamba_name = "state-spaces/mamba-130m-hf"
tokenizer_mamba = AutoTokenizer.from_pretrained(mamba_name)
tokenizer_mamba.pad_token = tokenizer_mamba.eos_token

class MambaClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(MambaClassifier, self).__init__()
        self.mamba = MambaModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.mamba.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.mamba(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            pooled_output = sum_hidden / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
            
        return self.classifier(pooled_output)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assumindo que num_classes é len(le.classes_) que é 4
num_classes = len(le.classes_)
mamba_model = MambaClassifier(mamba_name, num_classes=num_classes).to(device)

# Vamos amostrar um pouco dos dados de treino para ser rápido no notebook, se for muito grande
# O dataset tem milhares de linhas. Vamos usar as listas X_train_text e y_train (já formatadas e limitadas se for o caso)
train_encodings = tokenizer_mamba(list(X_train_text), truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer_mamba(list(X_val_text), truncation=True, padding=True, max_length=128, return_tensors='pt')

train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(list(y_train)))
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(list(y_val)))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(mamba_model.parameters(), lr=1e-5)

start_time = time.time()
mamba_model.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        optimizer.zero_grad()
        outputs = mamba_model(b_input_ids, attention_mask=b_attention_mask)
        loss = criterion(outputs, b_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

mamba_time = time.time() - start_time

mamba_model.eval()
mamba_preds = []
with torch.no_grad():
    for batch in test_loader:
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        outputs = mamba_model(b_input_ids, attention_mask=b_attention_mask)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        mamba_preds.extend(preds)

mamba_acc = accuracy_score(y_val, mamba_preds)
mamba_f1 = f1_score(y_val, mamba_preds, average='weighted')
print(f"Mamba Accuracy: {mamba_acc:.4f} | F1: {mamba_f1:.4f} in {mamba_time:.2f}s")

# Adicionar aos results
results['Mamba'] = {'acc': mamba_acc, 'f1': mamba_f1, 'preds': mamba_preds}

# ─── Tabela comparativa ────────────────────────────────────────────────────────
comparison = pd.DataFrame(
    {k: {'Accuracy': v['acc'], 'F1-Weighted': v['f1']} for k, v in results.items()}
).T.sort_values('F1-Weighted', ascending=False)

print('\n=== Comparação de Modelos (Conjunto de Validação) ===')
print(comparison.to_string(float_format='{:.4f}'.format))

best_model_name = comparison.index[0]
print(f'\n🏆 Melhor modelo: {best_model_name}')
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Barras comparativas
x = np.arange(len(comparison))
w = 0.35
axes[0].bar(x - w/2, comparison['Accuracy'], w, label='Accuracy', color='#2196F3', alpha=0.85)
axes[0].bar(x + w/2, comparison['F1-Weighted'], w, label='F1-Weighted', color='#4CAF50', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison.index, rotation=20, ha='right')
axes[0].set_ylim(0, 1)
axes[0].set_title('Métricas por Modelo', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# Confusion matrix do melhor modelo
best_preds = results[best_model_name]['preds']
cm = confusion_matrix(y_val, best_preds)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
    xticklabels=le.classes_, yticklabels=le.classes_
)
axes[1].set_title(f'Matriz de Confusão — {best_model_name}', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Real')
axes[1].set_xlabel('Predito')

plt.tight_layout()
plt.show()
# ─── Classification report do melhor modelo ────────────────────────────────────
print(f'=== Classification Report — {best_model_name} ===')
print(classification_report(y_val, best_preds, target_names=le.classes_))
# ─── Amostras de erros de classificação ───────────────────────────────────────
val_df = val_df.copy()
val_df['pred_label'] = le.inverse_transform(best_preds)
errors = val_df[val_df['sentiment'] != val_df['pred_label']][['text', 'sentiment', 'pred_label']]

print(f'Total de erros: {len(errors)} / {len(val_df)} ({len(errors)/len(val_df)*100:.1f}%)')
print('\nAmostra de erros:')
errors.sample(min(10, len(errors)), random_state=SEED).reset_index(drop=True)
best_clf = models[best_model_name]

def predict_sentiment(texts):
    """Prediz sentimento para uma lista de textos."""
    cleaned = [clean_tweet(t) for t in texts]
    feats   = tfidf.transform(cleaned)
    preds   = best_clf.predict(feats)
    return le.inverse_transform(preds)


exemplos = [
    "I absolutely love this product, it changed my life!",
    "Terrible experience, would not recommend to anyone.",
    "The weather today is cloudy.",
    "Just saw the new update, not sure how I feel about it.",
    "This company is amazing and their support team rocks!",
]

preds_exemplo = predict_sentiment(exemplos)

resultado = pd.DataFrame({'Texto': exemplos, 'Sentimento Predito': preds_exemplo})
print('=== Predições em novos tweets ===')
resultado
submission = val_df[['id', 'entity', 'text', 'sentiment', 'pred_label']].copy()
submission.columns = ['id', 'entity', 'text', 'true_sentiment', 'predicted_sentiment']
submission.to_csv('predictions_validation.csv', index=False)

print('Arquivo salvo: predictions_validation.csv')
print(f'Shape: {submission.shape}')
submission.head()