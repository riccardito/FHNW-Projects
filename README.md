# FHNW-Projects

My coursework from the BSc Data Science at FHNW (Fachhochschule Nordwestschweiz, Windisch), 2019–2022 — Switzerland's first Data Science bachelor. One folder per course, each with its own README.

This is an archive, not a showcase. The notebooks are the ones I handed in — outputs stripped, credentials removed, data left where it came from. Some of it is finished work with a number at the end. Some of it is a Minichallenge I ran out of time on. The folder READMEs say which is which.

## Course map

| Folder | Course | What's in it | Status |
|---|---|---|---|
| [`DataWrangling/`](DataWrangling/) | Data Wrangling | An FTPS → pandas → Postgres pipeline for MeteoSwiss CH2018 climate scenarios | exploratory |
| [`DeepLearning/`](DeepLearning/) | Deep Learning · bachelor thesis | CIFAR-10 from an MLP to 80 %, image captioning on Flickr8k, a chest X-ray training pipeline and the XAI heatmap evaluation from my thesis | finished, plus thesis code |
| [`MachineLearning/`](MachineLearning/) | Supervised + Unsupervised Learning | PCA, K-Means and ridge regression written from the maths; `regX`, my first packaged module | two finished, one draft |
| [`Probabilities/`](Probabilities/) | Probability & Statistics Challenge | A Monte-Carlo rockfall risk report for a road in Graubünden (R, PDF) | finished report |
| [`Recommendersystems/`](Recommendersystems/) | Recommender Systems | A genre-based content recommender on MovieLens in R Markdown | finished |
| [`internetOfThings/`](internetOfThings/) | IoT Data Analysis | A CircuitPython humidifier controller with a Flask API, MariaDB and a Dash board | finished prototype |
| [`nlp/`](nlp/) | Natural Language Processing | Amazon-review sentiment (SVM vs CNN: macro-F1 0.85, 95 % accuracy) and four German Rasa chatbots for Davos-Klosters | finished, team work |

## Conventions

- **No outputs in notebooks.** `nbstripout` runs as a git filter (`.gitattributes`). After cloning, run `nbstripout --install` once and it stays that way.
- **No data in the repo.** Every dataset is named in its folder README with its source. Nothing is committed except two small hand-made files (`events.csv`, `data.txt`).
- **No credentials.** Database, FTPS and API access read from environment variables — `PG_*`, `DB_*`, `FTP_*`, `RAPIDAPI_KEY`. Each README lists the ones its code needs.
- **Not runnable as a whole.** Each folder says what it needs; most need a dataset download and a Python 3.8–3.9 environment of its time.
- **German inside.** The course language was German, so several notebooks and the PDF report are written in German. The READMEs are English.

## Credits

The work is mine unless a folder README says otherwise. Two Minichallenges and one report were team work; the folder READMEs say so.

## Licence

To be decided before this repository goes public. Until then: all rights reserved for the code; every dataset keeps its own terms.
