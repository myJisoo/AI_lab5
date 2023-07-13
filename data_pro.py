import json
import chardet
import pandas as pd

def add2dataset(data, dataset):
    x = 'neutral'
    for i in range(data.shape[0]):
        guid = str(int(data[i, 0]))
        with open("dataset/data/" + guid + ".txt", "rb") as f:
            text_byte = f.read()
            text = text_byte.decode('gb18030')
            encoding = chardet.detect(f.read())["encoding"]
        text = text.strip('\n').strip('\r').strip(' ').strip()
        if encoding == "GB2312":
            encoding = "GBK"
        # print(text)
        with open("dataset/data/" + guid + ".txt", encoding = 'gb18030') as f:
            # try:
            #     text = f.read().rstrip("\n")
            # except UnicodeDecodeError:
            #     print(f"Error decoding file: {guid}.txt")
            #     continue
            # print(encoding)
            # print((x if data[i, 1] == "NaN" else data[i, 1]))
            dataset.append({
                "text": f.read().rstrip("\n"),
                "label": (x if data[i, 1] == "NaN" else data[i, 1]),
                "img": "dataset/data/" + guid + ".jpg"
            })
    # print(encoding)
    return dataset
train_dataset = []
dev_dataset = []
test_dataset = []

train_index_label = pd.read_csv("dataset/train.txt", keep_default_na=False)
test_index_label = pd.read_csv("dataset/test_without_label.txt", keep_default_na=False).values
# print(test_index_label)
train_count = dict(train_index_label["tag"].value_counts() * 0.8 // 1)

train_neg = train_index_label.loc[train_index_label["tag"] == "negative"].sample(n=int(train_count["negative"])).values
train_pos = train_index_label.loc[train_index_label["tag"] == "positive"].sample(n=int(train_count["positive"])).values
train_neu = train_index_label.loc[train_index_label["tag"] == "neutral"].sample(n=int(train_count["neutral"])).values
dev_neg = train_index_label[~train_index_label["guid"].isin(train_neg[:, 0])].loc[
    train_index_label["tag"] == "negative"].values
dev_pos = train_index_label[~train_index_label["guid"].isin(train_pos[:, 0])].loc[
    train_index_label["tag"] == "positive"].values
dev_neu = train_index_label[~train_index_label["guid"].isin(train_neu[:, 0])].loc[
    train_index_label["tag"] == "neutral"].values

train_dataset = add2dataset(train_neg, train_dataset)
train_dataset = add2dataset(train_pos, train_dataset)
train_dataset = add2dataset(train_neu, train_dataset)

dev_dataset = add2dataset(dev_neg, dev_dataset)
dev_dataset = add2dataset(dev_pos, dev_dataset)
dev_dataset = add2dataset(dev_neu, dev_dataset)

test_dataset = add2dataset(test_index_label, test_dataset)

with open("dataset/train.json", "w", encoding="utf-8") as f:
    json.dump(train_dataset, f)

with open("dataset/dev.json", "w", encoding="utf-8") as f:
    json.dump(dev_dataset, f)

with open("dataset/test.json", "w", encoding="utf-8") as f:
    json.dump(test_dataset, f)