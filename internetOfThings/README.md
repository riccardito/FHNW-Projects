# internetOfThings

**Course:** Internet of Things Data Analysis (`idb`) · **When:** 2021 · **Author:** Riccard Nef

## What this is

One project, `idb-main/`. An Adafruit microcontroller reads a humidity sensor and switches a humidifier, with a manual button to override it. The readings go to a Flask REST API on a Raspberry Pi, into MariaDB and onto a Dash board — or to ThingSpeak. The 2021 README and two hand-drawn sketches are in `idb-main/`.

## Contents

| Path | What it does |
|---|---|
| `idb-main/microcontroller/code.py` | The final CircuitPython firmware (Adafruit nRF52840 + ESP32 AirLift): a DHT11 read every 10 s, a four-state relay/LED/button machine — relay on below 35 % humidity, the button forces it on for an hour — and a send to both the Flask API and ThingSpeak |
| `idb-main/microcontroller/code_flaskapi.py`, `code_thinSpeak.py` | The two earlier single-target variants |
| `idb-main/flaskWebService/api.py` | A Flask-RESTful endpoint that stores temperature, humidity and timestamp in MariaDB and appends to `data.txt` |
| `idb-main/flaskWebService/dash_app.py` | A Dash app with two Plotly line charts read from `data.txt` |
| `idb-main/flaskWebService/data.txt` | 852 readings from my own room — the one data file in this folder |
| `idb-main/flaskWebService/mysqlconnection.ipynb` | Scratch: a connection test, a test insert, a test GET |
| `idb-main/raspberry/mariadbsetup` | SQL notes for the database and table on the Pi |
| `idb-main/sketch1-1.png`, `sketch2-1.png` | Sketch 1 is the first design and still says InfluxDB. Sketch 2 is what was built: CircuitPython → Flask on the Pi (port 5000) → MariaDB, plus a POST to ThingSpeak |

## How to run

Hardware first: the board, a DHT11, a relay. The firmware reads a `secrets.py` you create yourself (Wi-Fi SSID and password, the ThingSpeak write key) — it was never committed. The API reads `DB_HOST`, `DB_USER`, `DB_PASSWORD` and `DB_NAME` from the environment. Libraries: `flask`, `flask_restful`, `mysql-connector`, `dash`, `plotly`, `pandas`; the Dash imports are the pre-2.0 style.

## Data

My own sensor readings. No third-party data.

## What it shows, and what it doesn't

An edge-to-dashboard pipeline end to end: sensor and actuator control, a REST API, a schema, time-series logging, a live chart. Against that: the table-creation script and the insert disagree on column names, `code_flaskapi.py` posts to `localhost` and could never have reached the Pi, and the data collected was never analysed.

## Status

Finished prototype, as built in 2021.
