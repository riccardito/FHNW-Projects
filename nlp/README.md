# nlp

**Course:** Natural Language Processing (FHNW BSc Data Science) · **When:** 2022 · **Authors:** Riccard Nef; Mini-Challenge 1 with a classmate; Mini-Challenge 2 with four classmates

## What this is

Two Minichallenges and a folder of exploration. Mini-Challenge 1 classifies Amazon reviews by sentiment, a classic model against a fine-tuned deep one. Mini-Challenge 2 is a team of five building German chatbots for Davos-Klosters ski information — one bot per person, mine among them. `Exploratory/` is where I learned the tooling.

## Contents

| Path | What it does | Result (as written) |
|---|---|---|
| `Notebooks/Minichallenge1/mc1_Notebook.ipynb` | Binary sentiment on Amazon Toys & Games reviews (1–2 negative, 4–5 positive, 3 dropped; about 92 % positive, so class weights and a balanced test subset). NLTK preprocessing, TF-IDF 1–3-grams with LinearSVC against a Keras Conv1D CNN of about 2.1 M parameters, a one-at-a-time grid search, a parallel CNN tower, error analysis | SVM macro-F1 0.78 → 0.85 (best 0.847, trigrams). CNN test accuracy 95.27 % after tuning, train 99.4 % — overfit, and the notebook says so. Misclassified reviews are longer than average |
| `Notebooks/Minichallenge1/archive/`, `test1.ipynb` | The March draft and a partner's exploration | — |
| `Notebooks/Minichallenge2/bot-Gian/` | A Rasa bot: ticket types and prices scraped live from davosklostersmountains.ch, weather via RapidAPI, piste status, events from `events.csv`; spaCy `de_core_news_sm`, DIET, TED, RulePolicy | — |
| `Notebooks/Minichallenge2/bot-Jakob/` | A Rasa bot for tickets only, with fuzzy matching (difflib) and a Duckling number extractor | — |
| `Notebooks/Minichallenge2/bot-Nicka/` | A Rasa bot: weather forecast, piste-open percentage, events; a regex entity extractor; a commented-out GPT fallback | — |
| `Notebooks/Minichallenge2/bot-Manuel/` | No Rasa: a spaCy intent pipeline and NER written from scratch, plus two notebooks trying BERT and a DistilBERT fine-tune on GLUE CoLA | unfinished — the intent model is a stub |
| `Notebooks/Exploratory/Riccard/` | Skip-gram and CBOW in PyTorch on a toy sentence, the NLTK chapter-3 exercises, re-run Kaggle tutorials (text cleaning, sklearn preprocessing, a Jigsaw LSTM pipeline), a RapidAPI weather call | — |

## How to run

Not runnable as is. MC1 downloads the Amazon reviews at run time; its `Pipfile` pins Python 3.9 and little else. The Rasa bots carry no version pin (Rasa 3.x, domain schema 3.1) and need `rasa`, `rasa_sdk` and `spacy` with `de_core_news_sm`; Jakob's also needs a Duckling server. The weather actions read `RAPIDAPI_KEY` from the environment, and Gian's bot reads a `keys.json` you create. The `credentials.yml` files are Rasa's untouched templates.

## Data

- **Amazon product reviews**, Toys & Games 5-core (McAuley, UCSD) — public, downloaded at run time.
- **davosklostersmountains.ch** — scraped live for ticket prices; the site's terms apply.
- **events.csv** — seven Davos events of 2022, typed by the team.
- **Yelp reviews, GLUE** — via Hugging Face `datasets` in Manuel's notebooks.
- The Kaggle tutorial inputs are not committed; those notebooks point at `../input/`.

## What it shows, and what it doesn't

Classic and deep text classification side by side, with an honest error analysis. Rasa dialogue design, scraping and team delivery, in German. The exploration folder shows learning rather than finished work, and says so.

## Status

MC1 finished, a team of two. MC2 finished as a team of five; each bot is its author's own.
