# import sys
# adding Folder_2/subfolder to the system path
# sys.path.insert(0, '/home/daliu/Documents/master-thesis/code/dliu-ds-ma')

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, BertTokenizer

from data.data_processing import store_jointslu_labels
from pretrain.model_lightning import LitBertTokenClassification
from pretrain.jointslu_data_module import JointsluDataModule


# Update labels again if training dataset is generated
def update_label():
    store_jointslu_labels()


def define_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True,
        sep_token='EOS',
        cls_token='BOS')
    # read all tokens and add them into tokenizer
    special_tokens = [
        "I-depart_date.today_relative",
        "B-arrive_time.start_time",
        "atis_distance",
        "O",
        "B-stoploc.airport_name",
        "I-restriction_code",
        "B-depart_date.today_relative",
        "atis_ground_fare",
        "B-arrive_date.day_number",
        "I-arrive_time.end_time",
        "B-depart_time.period_mod",
        "B-connect",
        "atis_ground_service",
        "atis_flight_time",
        "B-days_code",
        "I-fromloc.city_name",
        "B-arrive_time.end_time",
        "B-arrive_date.day_name",
        "atis_airfare",
        "B-meal_description",
        "I-arrive_date.day_number",
        "B-or",
        "I-arrive_time.time_relative",
        "B-airline_code",
        "B-fromloc.state_name",
        "B-depart_time.time_relative",
        "atis_airport",
        "I-depart_time.time",
        "atis_city",
        "B-depart_date.year",
        "B-time",
        "B-month_name",
        "I-arrive_time.start_time",
        "I-flight_stop",
        "B-arrive_time.period_of_day",
        "I-toloc.state_name",
        "B-toloc.country_name",
        "atis_flight_no",
        "B-flight_days",
        "I-return_date.day_number",
        "atis_quantity",
        "B-day_name",
        "B-depart_date.day_number",
        "B-meal",
        "B-airport_code",
        "B-toloc.state_code",
        "I-airport_name",
        "atis_cheapest",
        "I-airline_name",
        "B-period_of_day",
        "B-today_relative",
        "I-round_trip",
        "B-fromloc.airport_code",
        "B-toloc.airport_code",
        "B-toloc.city_name",
        "B-arrive_date.today_relative",
        "I-city_name",
        "B-arrive_time.time_relative",
        "B-economy",
        "atis_ground_service#atis_ground_fare",
        "B-return_time.period_of_day",
        "B-fare_amount",
        "B-state_name",
        "B-meal_code",
        "B-flight_mod",
        "B-class_type",
        "I-class_type",
        "I-meal_code",
        "atis_airline#atis_flight_no",
        "B-flight_time",
        "B-depart_date.month_name",
        "B-depart_time.end_time",
        "atis_restriction",
        "B-toloc.airport_name",
        "atis_aircraft",
        "B-airport_name",
        "B-toloc.state_name",
        "B-fromloc.airport_name",
        "B-mod",
        "I-flight_mod",
        "B-fromloc.state_code",
        "atis_aircraft#atis_flight#atis_flight_no",
        "I-economy",
        "B-state_code",
        "atis_meal",
        "B-fromloc.city_name",
        "B-flight_stop",
        "I-transport_type",
        "I-depart_time.period_of_day",
        "I-depart_date.day_number",
        "atis_airfare#atis_flight_time",
        "B-stoploc.city_name",
        "I-toloc.airport_name",
        "atis_flight",
        "B-depart_date.day_name",
        "B-flight_number",
        "B-depart_time.start_time",
        "B-arrive_time.time",
        "I-fromloc.airport_name",
        "I-fare_basis_code",
        "I-flight_time",
        "B-day_number",
        "B-return_date.date_relative",
        "I-fare_amount",
        "B-stoploc.state_code",
        "B-depart_time.time",
        "B-fare_basis_code",
        "B-arrive_date.date_relative",
        "I-arrive_time.period_of_day",
        "B-airline_name",
        "B-arrive_time.period_mod",
        "atis_flight#atis_airfare",
        "I-cost_relative",
        "atis_capacity",
        "B-depart_time.period_of_day",
        "B-cost_relative",
        "I-toloc.city_name",
        "B-return_date.month_name",
        "B-arrive_date.month_name",
        "I-stoploc.city_name",
        "B-return_date.day_number",
        "I-depart_time.time_relative",
        "B-aircraft_code",
        "I-time",
        "B-restriction_code",
        "B-city_name",
        "I-depart_time.start_time",
        "I-depart_time.end_time",
        "I-fromloc.state_name",
        "B-time_relative",
        "B-depart_date.date_relative",
        "B-round_trip",
        "atis_abbreviation",
        "I-arrive_time.time",
        "I-today_relative",
        "B-transport_type",
        "B-return_time.period_mod",
        "atis_airline"
    ]
    labels_dict = {'I-depart_date.today_relative': 0,
                   'B-arrive_time.start_time': 1,
                   'atis_distance': 2,
                   'O': 3,
                   'B-stoploc.airport_name': 4,
                   'I-restriction_code': 5,
                   'B-depart_date.today_relative': 6,
                   'atis_ground_fare': 7,
                   'B-arrive_date.day_number': 8,
                   'I-arrive_time.end_time': 9,
                   'B-depart_time.period_mod': 10,
                   'B-connect': 11,
                   'atis_ground_service': 12,
                   'atis_flight_time': 13,
                   'B-days_code': 14,
                   'I-fromloc.city_name': 15,
                   'B-arrive_time.end_time': 16,
                   'B-arrive_date.day_name': 17,
                   'atis_airfare': 18,
                   'B-meal_description': 19,
                   'I-arrive_date.day_number': 20, 'B-or': 21, 'I-arrive_time.time_relative': 22, 'B-airline_code': 23, 'B-fromloc.state_name': 24, 'B-depart_time.time_relative': 25, 'atis_airport': 26, 'I-depart_time.time': 27, 'atis_city': 28, 'B-depart_date.year': 29, 'B-time': 30, 'B-month_name': 31, 'I-arrive_time.start_time': 32, 'I-flight_stop': 33, 'B-arrive_time.period_of_day': 34, 'I-toloc.state_name': 35, 'B-toloc.country_name': 36, 'atis_flight_no': 37, 'B-flight_days': 38, 'I-return_date.day_number': 39, 'atis_quantity': 40, 'B-day_name': 41, 'B-depart_date.day_number': 42, 'B-meal': 43, 'B-airport_code': 44, 'B-toloc.state_code': 45, 'I-airport_name': 46, 'atis_cheapest': 47, 'I-airline_name': 48, 'B-period_of_day': 49, 'B-today_relative': 50, 'I-round_trip': 51, 'B-fromloc.airport_code': 52, 'B-toloc.airport_code': 53, 'B-toloc.city_name': 54, 'B-arrive_date.today_relative': 55, 'I-city_name': 56, 'B-arrive_time.time_relative': 57, 'B-economy': 58, 'atis_ground_service#atis_ground_fare': 59, 'B-return_time.period_of_day': 60, 'B-fare_amount': 61, 'B-state_name': 62, 'B-meal_code': 63, 'B-flight_mod': 64, 'B-class_type': 65, 'I-class_type': 66, 'I-meal_code': 67, 'atis_airline#atis_flight_no': 68, 'B-flight_time': 69, 'B-depart_date.month_name': 70, 'B-depart_time.end_time': 71, 'atis_restriction': 72, 'B-toloc.airport_name': 73, 'atis_aircraft': 74, 'B-airport_name': 75, 'B-toloc.state_name': 76, 'B-fromloc.airport_name': 77, 'B-mod': 78, 'I-flight_mod': 79, 'B-fromloc.state_code': 80, 'atis_aircraft#atis_flight#atis_flight_no': 81, 'I-economy': 82, 'B-state_code': 83, 'atis_meal': 84, 'B-fromloc.city_name': 85, 'B-flight_stop': 86, 'I-transport_type': 87, 'I-depart_time.period_of_day': 88, 'I-depart_date.day_number': 89, 'atis_airfare#atis_flight_time': 90, 'B-stoploc.city_name': 91, 'I-toloc.airport_name': 92, 'atis_flight': 93, 'B-depart_date.day_name': 94, 'B-flight_number': 95, 'B-depart_time.start_time': 96, 'B-arrive_time.time': 97, 'I-fromloc.airport_name': 98, 'I-fare_basis_code': 99, 'I-flight_time': 100, 'B-day_number': 101, 'B-return_date.date_relative': 102, 'I-fare_amount': 103, 'B-stoploc.state_code': 104, 'B-depart_time.time': 105, 'B-fare_basis_code': 106, 'B-arrive_date.date_relative': 107, 'I-arrive_time.period_of_day': 108, 'B-airline_name': 109, 'B-arrive_time.period_mod': 110, 'atis_flight#atis_airfare': 111, 'I-cost_relative': 112, 'atis_capacity': 113, 'B-depart_time.period_of_day': 114, 'B-cost_relative': 115, 'I-toloc.city_name': 116, 'B-return_date.month_name': 117, 'B-arrive_date.month_name': 118, 'I-stoploc.city_name': 119, 'B-return_date.day_number': 120, 'I-depart_time.time_relative': 121, 'B-aircraft_code': 122, 'I-time': 123, 'B-restriction_code': 124, 'B-city_name': 125, 'I-depart_time.start_time': 126, 'I-depart_time.end_time': 127, 'I-fromloc.state_name': 128, 'B-time_relative': 129, 'B-depart_date.date_relative': 130, 'B-round_trip': 131, 'atis_abbreviation': 132, 'I-arrive_time.time': 133, 'I-today_relative': 134, 'B-transport_type': 135, 'B-return_time.period_mod': 136, 'atis_airline': 137}

    special_tokens_dict = {
        'additional_special_tokens':
            special_tokens
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    # Setup tokenizer
    return tokenizer


if __name__ == '__main__':
    tokenizer = define_tokenizer()
    data_module = JointsluDataModule(tokenizer=tokenizer)
    model = LitBertTokenClassification(tokenizer=tokenizer)

    logger = TensorBoardLogger("pretrain/tb_logger", name="bert_jointslu")
    trainer = Trainer(max_epochs=3, logger=logger)
    trainer.fit(model, datamodule=data_module)
