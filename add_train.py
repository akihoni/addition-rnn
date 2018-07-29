# coding:utf-8

from keras import Sequential
from keras import layers
import numpy as np
import add_encode

TRAINING_SIZE = 50000
DIGIT_LEN = 3
REVERSE = True
MAX_LEN = DIGIT_LEN + 1 + DIGIT_LEN

HIDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1


def train():
    chars = '0123456789+ '
    ctable = add_encode.Table(chars)

    questions = []
    answers = []
    seen = set()

    # generate data
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789'))
                                for i in range(np.random.randint(1, DIGIT_LEN + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        q = '{}+{}'.format(a, b)
        query = q + ' ' * (MAX_LEN - len(q))
        ans = str(a + b)
        ans += ' ' * (DIGIT_LEN + 1 - len(ans))
        if REVERSE:
            query = query[::-1]
        questions.append(query)
        answers.append(ans)

    # vectorization
    x = np.zeros((len(questions), MAX_LEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(questions), DIGIT_LEN + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAX_LEN)
    for i, sentence in enumerate(answers):
        y[i] = ctable.encode(sentence, DIGIT_LEN + 1)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    split = len(x) - len(x) // 10
    (x_train, x_val) = x[:split], x[split:]
    (y_train, y_val) = y[:split], y[split:]

    model = Sequential()
    model.add(layers.LSTM(HIDEN_SIZE, input_shape=(MAX_LEN, len(chars))))
    model.add(layers.RepeatVector(DIGIT_LEN + 1))
    for _ in range(LAYERS):
        model.add(layers.LSTM(HIDEN_SIZE, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    for i in range(1, 200):
        print()
        print('-' * 50)
        print('iteration ', i)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  validation_data=(x_val, y_val))
    model.save("model/model.h5")


def main():
    train()


if __name__ == '__main__':
    main()
