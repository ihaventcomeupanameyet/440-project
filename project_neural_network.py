import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

# Define classes
class ClickbaitDataset(Dataset):
    def __init__(self, isClickbait, headlines):
        super().__init__()
        self.isClickbait = isClickbait
        self.headlines = headlines

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, index):
        return {"headlines": self.headlines[index], "isClickbait": self.isClickbait[index]}
    
class ClickbaitNN(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.modelSequence = torch.nn.Sequential(
            torch.nn.Linear(length, 3 * length),
            torch.nn.Linear(3 * length, 100),
            torch.nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.modelSequence(x)

def train(loader, model, device, loss_func, optimizer):
    model.train()

    for _, batch in enumerate(loader):
        X = batch["headlines"].float().to(device)
        y = batch["isClickbait"].float().to(device)

        prediction = model(X).squeeze()
        loss = loss_func(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def testValidate(parameters, labels, model, device):
    model.eval()

    parameters = parameters.to(device)
    labels = labels.squeeze().to(device)

    prediction = torch.sigmoid(model(parameters).squeeze())
    prediction = (prediction > 0.5).float()
    
    print(f"prediction: {prediction.shape}")
    print(f"labels: {labels.shape}")

    accuracy = (prediction == labels).sum().item() / len(labels)
    print(f"Test/Validation accuracy: {100 * accuracy}%")

def trainWithEpochs(epochs, loader, model, device, loss_func, optimizer, parameters, labels):
    for i in range(epochs):
        print(f"Epoch {i + 1}:")

        train(loader, model, device, loss_func, optimizer)
        testValidate(parameters, labels, model, device)
    print("Done")

# Pre-process data
clickbaitDF = pd.read_csv('data/clickbait-train1.csv')

countVectorizer = CountVectorizer()
bagOfWords = countVectorizer.fit_transform(clickbaitDF.loc[:, 'headline'])
bagOfWordsDF = pd.DataFrame(bagOfWords.toarray(), columns=countVectorizer.get_feature_names_out())

isClickbait = clickbaitDF['clickbait']
headlines = bagOfWordsDF

isClickbait_train, isClickbait_temp, headlines_train, headlines_temp = train_test_split(isClickbait, headlines, test_size=0.3, random_state=440)
isClickbait_validate, isClickbait_test, headlines_validate, headlines_test = train_test_split(isClickbait_temp, headlines_temp, test_size=0.5, random_state=440)

batch_size = 224
trainSet = ClickbaitDataset(torch.from_numpy(isClickbait_train.values).float(), torch.from_numpy(headlines_train.values).float())
trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)

isClickbaitTensor_validate = torch.from_numpy(isClickbait_validate.values).float()
headlinesTensor_validate = torch.from_numpy(headlines_validate.values).float()
isClickbaitTensor_test = torch.from_numpy(isClickbait_test.values).float()
headlinesTensor_test = torch.from_numpy(headlines_test.values).float()

# Train model
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
theNN = ClickbaitNN(headlines.shape[1]).to(device)
loss_func = torch.nn.BCEWithLogitsLoss(reduction="sum")
optimizer = torch.optim.SGD(theNN.parameters())

epochs = 7
trainWithEpochs(epochs, trainLoader, theNN, device, loss_func, optimizer, headlinesTensor_validate, isClickbaitTensor_validate)

# Test model
print("Test model with new dataset")
testValidate(headlinesTensor_test, isClickbaitTensor_test, theNN, device)

# Test model with new dataset
clickbaitDFTest = pd.read_csv('data/clickbait-train2.csv')

bagOfWordsTest = countVectorizer.transform(clickbaitDFTest.loc[:, 'title'])
bagOfWordsDFTest = pd.DataFrame(bagOfWordsTest.toarray(), columns=countVectorizer.get_feature_names_out())

isClickbait_test_2 = clickbaitDFTest['label']
isClickbait_test_2_temp = isClickbait_test_2.to_numpy()
isClickbait_test_2_temp = np.where(isClickbait_test_2 == 'clickbait', 1, 0)
isClickbait_test_2 = pd.DataFrame(isClickbait_test_2_temp, columns=['label'])
headlines_test_2 = bagOfWordsDFTest

isClickbaitTensor_test_2 = torch.from_numpy(isClickbait_test_2.values).float()
headlinesTensor_test_2 = torch.from_numpy(headlines_test_2.values).float()

print("Testing model with new dataset")
testValidate(headlinesTensor_test_2, isClickbaitTensor_test_2, theNN, device)