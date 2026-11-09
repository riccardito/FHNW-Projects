# Recommendersystems

**Course:** Recommender Systems (`rsy`, FHNW BSc Data Science) · **When:** 2022 · **Authors:** Riccard Nef; Mini-Challenge 2 with a classmate

## What this is

Two R Markdown challenges on the MovieLense data. Challenge 1 is a recommenderlab warm-up. Mini-Challenge 2 (April 2022) is a content-based recommender built from genre profiles, with its rendered HTML report kept beside the source.

## Contents

| File | What it does | Result (as written) |
|---|---|---|
| `Challenge1.rmd` | User-based collaborative filtering (`UBCF`) with recommenderlab on users with more than 100 ratings, trained on 50 users; top-10 lists for two held-out users | none — five chunks, no evaluation, a scaffold |
| `challenge2_rsy_riccard_nef.Rmd` | A binary liked-items matrix (rating > 3), movie-genre and user-genre profiles, cosine similarity users × films with an own `calc_cos_similarity_twomtrx` tested on a 2 × 2 case, masking of already-rated films, top-N lists. Min-similarity histograms for N = 10 / 20 / 50 / 100 and radar plots of seen against recommended genres for six users | written findings: the median user has seen 64 or fewer films (from 19 up to all of them); user 133 gets only Drama and Comedy; user 555 looks like a shared account |
| `challenge2_rsy_riccard_nef.html` | The rendered report of the Rmd above — 8 MB of plots and embedded JS libraries | — |

## How to run

R with `recommenderlab`, `tidyverse`, `plotly` and `fmsb`. The data ships with the package (`data(MovieLense)`); there is no renv lockfile.

## Data

**MovieLense** — the 943 users × 1 664 films set bundled with recommenderlab (GroupLens MovieLens 100k). Nothing is committed; the MovieLens terms apply (research use, no redistribution).

## What it shows, and what it doesn't

Matrix reshaping in the tidyverse, similarity by hand, visual evaluation and interpretive commentary — in German. No train/test split, no RMSE or precision: the evaluation is visual, by design of the assignment. Challenge 1 is a scaffold and reads like one.

## Status

Mini-Challenge 2 finished, a team of two. Challenge 1 a warm-up.
