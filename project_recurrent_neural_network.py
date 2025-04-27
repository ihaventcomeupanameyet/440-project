import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch


import spacy

from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pad_sequence



clickbaitDF = pd.read_csv('data/clickbait-train1.csv')
isClickbait = clickbaitDF['clickbait']

headlines = clickbaitDF['headline']

isClickbait_train, isClickbait_temp, headlines_train, headlines_temp = train_test_split(
    isClickbait, headlines, test_size=0.3, random_state=440)
isClickbait_validate, isClickbait_test, headlines_validate, headlines_test = train_test_split(
    isClickbait_temp, headlines_temp, test_size=0.5, random_state=440)

# Tokenizer (using spaCy)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


isClickbait_train = isClickbait_train.tolist()
isClickbait_validate = isClickbait_validate.tolist()
isClickbait_test = isClickbait_test.tolist()


headlines_train = headlines_train.tolist()
headlines_validate = headlines_validate.tolist()
headlines_test = headlines_test.tolist()




class ClickbaitNN(torch.nn.Module):
    def __init__(self,train_texts,embed_dim,hidden_dim):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        tokenized_train = [self._tokenize_single(t) for t in train_texts]
        self.vocab = build_vocab_from_iterator(tokenized_train, specials=["<UNK>"])
        self.vocab.set_default_index(self.vocab["<UNK>"])
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)


    def _tokenize_single(self, text):
        return [tok.text.lower() for tok in self.nlp(text)]

    def tokenize(self, texts):
        return [self._tokenize_single(t) for t in texts]

    def encode_tokens(self, token_lists):
        return [[self.vocab[t] for t in tokens] for tokens in token_lists]


    def forward(self, x_padded, lengths):
        emb = self.embedding(x_padded)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out_packed, (h_n, c_n) = self.lstm(packed)
        logits = self.fc(h_n[-1])
        return logits




model = ClickbaitNN(headlines_train, embed_dim=128, hidden_dim=64)


class TextDataset(Dataset):
    def __init__(self, texts, labels, model):
        self.texts = texts
        self.labels = labels
        self.model = model

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)

    tokenized = model.tokenize(texts)
    encoded = model.encode_tokens(tokenized)

    tensors = [torch.LongTensor(ids) for ids in encoded]
    lengths = torch.tensor([t.size(0) for t in tensors], dtype=torch.long)

    x_padded = pad_sequence(tensors, batch_first=True, padding_value=0)
    y = torch.tensor(labels, dtype=torch.float32)
    return x_padded, lengths, y



def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for x_padded, lengths, y in loader:
            x_padded, y = x_padded.to(device), y.to(device)
            logits = model(x_padded, lengths).squeeze(1)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            total_correct += (preds == y.long()).sum().item()
            total_count += y.size(0)
    return total_loss / total_count, total_correct / total_count



def train_model(model, train_texts, train_labels,
                val_texts=None, val_labels=None,
                epochs=10, batch_size=32, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    train_ds = TextDataset(train_texts, train_labels, model)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)

    if val_texts is not None:
        val_ds = TextDataset(val_texts, val_labels, model)
        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)
    else:
        val_loader = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x_padded, lengths, y in train_loader:
            x_padded, y = x_padded.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x_padded, lengths).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

        train_loss = running_loss / len(train_ds)

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch:2d} — train_loss: {train_loss:.4f}  "
                  f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch:2d} — train_loss: {train_loss:.4f}")

    return model

    
train_model(model,train_texts=headlines_train,
            train_labels=isClickbait_train,val_texts=headlines_validate,
            val_labels=isClickbait_validate)

test_ds = TextDataset(headlines_test, isClickbait_test, model)
test_loader = DataLoader(test_ds, batch_size=len(test_ds),
                            shuffle=False, collate_fn=collate_fn)

criterion = nn.BCEWithLogitsLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"test1_acc: {test_acc:.4f}")


clickbaitDF = pd.read_csv('data/clickbait-train2.csv')
isClickbait = clickbaitDF['label']

isClickbait[isClickbait=="news"] = 0
isClickbait[isClickbait=="clickbait"] = 1
headline = clickbaitDF['title']

test_ds = TextDataset(headline, isClickbait, model)
test_loader = DataLoader(test_ds, batch_size=int(0.1*len(test_ds)),
                            shuffle=False, collate_fn=collate_fn)

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"test2_acc: {test_acc:.4f}")


clickbaitDF = pd.read_csv('data/clickbait-train1.csv')
clickbaitDF = clickbaitDF[clickbaitDF['clickbait'] == 1]
isClickbait = clickbaitDF['clickbait']
headlines = clickbaitDF['headline']

test_ds = TextDataset(headline, isClickbait, model)
test_loader = DataLoader(test_ds, batch_size=int(0.1*len(test_ds)),
                            shuffle=False, collate_fn=collate_fn)

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"test3_acc: {test_acc:.4f}")


