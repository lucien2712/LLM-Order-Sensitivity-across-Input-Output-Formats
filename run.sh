#!/bin/bash

echo "Downloading data..."
python download.py

echo "Shuffling data..."
python shuffle.py

echo "Running experiment with original data..."
python experiment.py -- input mmlu_17subjects_2langs_100samples.json

echo "Running experiment with shuffled data..."
python experiment.py -- input shuffle_mmlu_17subjects_2langs_100samples.json

echo "Evaluating model..."
python eval.py

echo "Generating visualizations..."
python visualize.py

echo "Done!"

