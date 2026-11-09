# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import requests
from bs4 import BeautifulSoup
import pandas as pd
import copy
import re
import difflib


def create_dataframe(table):
    df = pd.DataFrame()
    string_tabelle = ""
    for i, tr in enumerate(table.findAll('tr')):
        string_tabelle += "\n"
        for j, td in enumerate(tr.findAll('td')):
            string_tabelle += td.getText().replace("\n", " ").replace("\t", "").replace("\tab", "").ljust(30)
            string_tabelle += "\t"
            df.loc[i, j] = td.getText().replace("\n", " ").replace("\t", "").replace("\tab", "")
    df.index = df.iloc[:, 0]
    df = df.iloc[: , 1:]
    df.columns = df.iloc[0]
    df = df.iloc[1: ,:]
    df.index.name = ""
    return df, string_tabelle

def read_prices(season="winter"):
    if season == "winter":
        URL = "https://www.davosklostersmountains.ch/de/mountains/winter/tarife-tickets/ski-regionalpass"

    if season == "summer":
        URL = "https://www.davosklostersmountains.ch/de/mountains/sommer/tarife-tickets"

    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    body = soup.find('body')
    # create ticket dict
    tickets = dict()
    ticket_area = body.find_all("div",  attrs={'class':"pimcore_area_content-accordion-area"})
    for area in ticket_area:
        # get title of area
        title = area.find("h2").getText().replace("\\x", " ").lower()
        tickets[title] = dict()
        # get info of area
        info_text = area.find("div", attrs={'class':"wysiwyg intro text-center"}).getText().replace("\n", "")
        tickets[title]["info"] = info_text
        # read cards in area
        cards = area.find_all("div", attrs={'class':"card"})
        for card in cards:
            header = card.find("div", attrs={'class':"card-header"})
            body = card.find("div", attrs={'class':"card-body"})
            card_title = re.search(r"(\w.+\w.+)+", header.getText()).group(1).lower()
            table = body.find("tbody")
            if table:
                df, str_tbl = create_dataframe(table)
                tickets[title][card_title] = str_tbl
            else:
                tickets[title][card_title] = "Keine Informationen gefunden."
    return tickets

tickets = read_prices(season="winter")


class GetTicketTypes(Action):

    def name(self) -> Text:
        return "action_get_ticket_types"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            ticket_types = ""
            for key in tickets.keys():
                ticket_types += "\n- " + key 
            dispatcher.utter_message(text="Ich habe die folgenden Ticketarten gefunden: {}".format(ticket_types))
        except:
            dispatcher.utter_message(text="Ich kann deine Frage leider aktuell nicht beantorten.")
        return []


class GetInfoOnTicketType(Action):

    def name(self) -> Text:
        return "action_get_information_on_ticket_type"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            ticket_type = list(tracker.get_latest_entity_values("ticket_type_request"))[-1]
            ticket_types = [key for key in tickets.keys()]
            ticket_type_key = difflib.get_close_matches(ticket_type, ticket_types, n=1, cutoff=0.0)[0]
            # dispatcher.utter_message(text="Original:{} Guess:{}".format(ticket_type, ticket_type_key))
            dispatcher.utter_message(text="Ich habe die folgenden Infos gefunden: {}".format(tickets[ticket_type_key]["info"]))
        except:
            dispatcher.utter_message(text="Ich kann deine Frage leider aktuell nicht beantorten (action_get_information_on_ticket_type)\nEventuell muss der Ticket-Typ genauer angegeben werden.")
        return []


class GetInfoOnTicketPrice(Action):

    def name(self) -> Text:
        return "action_get_information_on_ticket_price"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            ticket_type = tracker.get_slot('ticket_type')
            ticket_types = [key for key in tickets.keys()]
            ticket_type_key = difflib.get_close_matches(ticket_type, ticket_types, n=1, cutoff=0.0)[0]

            ticket_place = tracker.get_slot('ticket_place')
            ticket_places = [key for key in tickets[ticket_type_key].keys()]
            ticket_place_key = difflib.get_close_matches(ticket_place, ticket_places, n=1, cutoff=0.0)[0]
            dispatcher.utter_message(text="Ich habe die folgenden Infos für {} bei {} gefunden: {}".format(ticket_type_key, ticket_place_key, tickets[ticket_type_key][ticket_place_key]))
        except:
            dispatcher.utter_message(text="Ich kann deine Frage leider aktuell nicht beantorten (action_get_information_on_ticket_price)")
        return []