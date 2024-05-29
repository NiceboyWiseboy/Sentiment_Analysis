from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax


def preprocess(input_text):
    new_text = []

    for t in input_text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def process_text(input_text: str):
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    labels = ['negative', 'neutral', 'positive']
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # text = "You are a great person."
    text = preprocess(input_text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    print(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    sent_list = []
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        sent_list.append({l: f'{s}'})

    return sent_list[0]
