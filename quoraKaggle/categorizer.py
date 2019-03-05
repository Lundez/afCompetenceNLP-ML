#!/usr/bin/env python
# coding: utf8
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import csv
import spacy
from spacy.util import minibatch, compounding


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None,
         output_dir=None,
         n_iter=20,
         n_texts=2000):

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    textcat.add_label('INSINCERE')

    print("Loading Quora data...")
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data()
    print("Using {} examples ({} training, {} evaluation)"
          .format(n_texts, len(train_texts), len(dev_texts)))
    train_data = list(zip(train_texts,
                          [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,
                           annotations,
                           sgd=optimizer,
                           drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))

    ## test the trained model
    test_text = "Kill all people?"
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)


def load_data():
    """Load data from the Quora dataset."""
    # Partition off part of the train data for evaluation
    with open('quora/train.csv', 'r') as train_file:
        train_data = [row for row in csv.reader(train_file, delimiter=',', quotechar='"')][1:]
        random.shuffle(train_data)
        train_data = train_data[:50000]
        train_texts = [line[1] for line in train_data]
        train_cats = [{'INSINCERE': line[2] == '1'} for line in train_data]

    split = int(0.8*len(train_data))

    return (train_texts[:split], train_cats[:split]), (train_texts[split:], train_cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0   # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0   # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + 1e-8 + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


if __name__ == '__main__':
    plac.call(main)

# Using 2000 examples (20000 training, 5000 evaluation)
# Training the model...
# LOSS      P       R       F
# 96.458  0.673   0.121   0.205
# 73.351  0.593   0.252   0.354
# 58.390  0.599   0.324   0.421
# 47.129  0.564   0.334   0.420
# 38.019  0.556   0.324   0.410
# 33.645  0.528   0.324   0.402
# 28.565  0.537   0.328   0.407
# 26.210  0.514   0.328   0.400
# 25.106  0.505   0.331   0.400
# 22.143  0.482   0.331   0.393
# 20.824  0.480   0.324   0.387
# 21.733  0.487   0.324   0.389
# 20.255  0.487   0.324   0.389
# 19.674  0.487   0.334   0.397
# 20.321  0.472   0.324   0.384
# 17.538  0.490   0.345   0.405
# 17.666  0.495   0.328   0.394
# 18.839  0.500   0.324   0.393
# 17.263  0.489   0.321   0.387
# 16.867  0.497   0.338   0.402

# TODO
# 50k --> 0.48F1
# 0.563, 568 & raising!!
#1181.828        0.713   0.473   0.568
