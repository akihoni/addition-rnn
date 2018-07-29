# coding:utf-8

import keras
import numpy as np
import add_encode

DIGIT_LEN = 3
REVERSE = True
MAX_LEN = DIGIT_LEN + 1 + DIGIT_LEN
MODEL_SAVE_PATH = 'model'


def predict():
    chars = '0123456789+ '
    ctable = add_encode.Table(chars)

    questions = []

    query = input('please input a question:')

    # reform data
    query = query + ' ' * (MAX_LEN - len(query))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    x = np.zeros((len(questions), MAX_LEN, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAX_LEN)

    # prediction
    model = keras.models.load_model("model/model.h5")
    prediction = model.predict_classes(x, verbose=0)
    q = ctable.decode(x[0])
    pred = ctable.decode(prediction[0], calc_argmax=False)
    print((q[::-1] if REVERSE else q) + '= ' + pred)


def main():
    predict()


if __name__ == '__main__':
    main()
