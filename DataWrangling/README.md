# DataWrangling

**Course:** Data Wrangling (FHNW BSc Data Science) · **When:** 2020 · **Author:** Riccard Nef

## What this is

One notebook, `pipeline.ipynb`: a data-injection pipeline that pulls MeteoSwiss CH2018 climate-scenario files from an FTPS server, turns them into one pandas DataFrame and writes it to a Postgres database. The assignment, in the course's words: "Datainjectionpipeline for meteo schweiz data."

## Contents

| File | What it does | Result |
|---|---|---|
| `pipeline.ipynb` | Downloads two CH2018 zips (`tas` temperature, `pr` precipitation; DMI-HIRHAM / EC-EARTH, RCP2.6, QM stations, 1981–2099) over explicit FTPS with a session-reusing `FTP_TLS` subclass. Unzips, skips each file's header to the first blank line, reads `;`-separated values, indexes by date, tags STATION and ATTRIBUTE from the file name, concatenates, and loads with `DataFrame.to_sql("ClimaValues")` | none stated — the notebook ends at the load |

## How to run

Not runnable as is. It needs the course's FTPS account and a Postgres server, both read from the environment:

```
FTP_HOST  FTP_USER  FTP_PASS              # the CH2018 FTPS server
PG_USER   PG_PASSWORD  PG_HOST  PG_DB     # the target database
```

plus `pandas`, `sqlalchemy` and `psycopg2`. There is no requirements file.

## Data

CH2018 Swiss climate scenarios (MeteoSwiss / NCCS), station data, served over a course-provisioned FTPS account. Nothing is committed — the zips are deleted after extraction. The CH2018 terms of use apply to anything you download.

## What it shows, and what it doesn't

FTPS access, file-name-driven batch parsing, DataFrame assembly and a relational load. Course-grade code: only two of the available files are used, and an unused SQLAlchemy `stations` class, a dead `CREATE TABLE` and a few scratch cells are still in there. The comments mix German and English.

## Status

Exploratory. Kept as handed in, minus the credentials.
