# Probabilities

**Course:** Probability & Statistics Challenge (HS19 C4, "Steinschlag") · **When:** autumn 2019 · **Authors:** Riccard Nef, ⟨name, consent pending⟩, ⟨name, consent pending⟩

## What this is

One report, `Steinschlag_new.pdf` — 100 pages rendered from an R notebook, English text with German code comments. The question: what is the probability that a rockfall kills someone on the road below a slope near Schiers, Graubünden, and should the road be closed until the protection nets are replaced?

## Contents

| File | What it does | Result (as written in the report) |
|---|---|---|
| `Steinschlag_new.pdf` | Fits distributions to two zones of rockfall records — mass, speed, energy, time between events; lognormal, exponential, normal, checked with QQ plots. An analytic estimate of P(car hit \| net breaks) from hourly rockfall counts × hourly traffic × a 9.8 m danger zone. Then a Monte-Carlo simulation of 100 × 100 000 years with daily net clearing and road closure after a breach | P(rock through the net in a year) ≈ 0.016. P(car hit per year) 1.2–2.9 × 10⁻⁴ depending on the traffic assumption, analytic 1.5 × 10⁻⁴ — all above the 10⁻⁴ threshold, so the report recommends closing the road until the nets are replaced |

## How to run

There is nothing to run. The source `.Rmd` and the two CSVs are not in this folder; the report was built with R — tidyverse, fitdistrplus, MASS, psych.

## Data

Two rockfall observation sets (zone 1 with 68 events, zone 2), January–March 2019, provided by the course, and hourly traffic density from the Swiss Federal Statistical Office (2015). The report reproduces the course tables; their licence is not stated.

## What it shows

Distribution fitting with honest diagnostics. A simulation designed around the actual failure mechanics — energy over 1 000 kJ breaks the net, 500–1 000 kJ breaks it only when the net is already full. Three traffic scenarios, a convergence plot, and a dead end admitted in the text: the net-mass distribution could not be fitted, so the approach changed.

## Status

Finished report, as submitted. Team work by three authors.
