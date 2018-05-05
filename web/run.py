from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import glob
import pickle
import requests
import json

WEATHER_API_KEY = ''
FL_INFO_FOLDER = 'flight_info.data'
DELAYS_INFO_FOLDER = 'delays.data'
CLFS_INFO_FOLDER = 'clfs'
COLS_FOLDER = 'values_dicts'
INFO_DATA = {}

app = Flask(__name__)


@app.before_request
def before_request():
    global INFO_DATA

    if len(INFO_DATA) == 0:
        INFO_DATA = load_initial_data()

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/predict_flight", methods=['GET'])
def predict_flight():
    origin = request.args.get('origin', default='', type=str)
    dest = request.args.get('dest', default='', type=str)
    carrier = request.args.get('carrier', default='', type=str)
    fl_num = request.args.get('fl_num', default='', type=str)
    flight_date = datetime.strptime(request.args.get('flight_date'), "%Y-%m-%d").date()
    features = get_features(INFO_DATA, WEATHER_API_KEY, origin, dest, carrier, fl_num, flight_date)

    return jsonify(INFO_DATA['clfs'][origin]['clf'].predict_proba(features).tolist())


def load_initial_data():
    flight_info = load_file(FL_INFO_FOLDER)
    delays = load_file(DELAYS_INFO_FOLDER)
    files_list = get_file_list(CLFS_INFO_FOLDER)
    clfs = {}
    for file in files_list:
        airport = file.replace(CLFS_INFO_FOLDER + '/', '').replace('.data', '')
        clfs[airport] = {'clf': load_file(file), 'cols': load_file(COLS_FOLDER + '/' + airport + '.data')}
    return {'clfs': clfs, 'fl_info': flight_info, 'delays': delays}


def get_features(data, weather_api_key, origin, dest, carrier, flight, date):
    fl_info = data['fl_info'][(data['fl_info']['fl_num'] == int(flight)) & (data['fl_info']['carrier'] == carrier) & (data['fl_info']['origin'] == origin)]
    df = pd.DataFrame(columns=data['clfs'][origin]['cols'])
    df = df.append({'average_wind_speed': get_wind_speed_for_city(get_series_value(fl_info['origin_city_name']), weather_api_key),
                    'crs_dep_time': get_series_value(fl_info['crs_dep_time']),
                    'crs_elapsed_time': get_series_value(fl_info['crs_elapsed_time']),'dest_' + dest: 1,
                    'carrier_' + carrier: 1,
                    'month': date.month,
                    'quarter': date.month // 4,
                    'day_of_month': date.day,
                    'day_of_year': date.timetuple().tm_yday,
                    'airline_delay_index': get_series_value(data['delays'][data['delays']['Carrier'] == carrier]['Delay index'])}
                   , ignore_index=True)
    df = df.drop('status', axis=1)
    df = df.fillna(0)
    return df


def get_wind_speed_for_city(city_name, api_key):
    try:
        api_url = 'http://api.openweathermap.org/data/2.5/weather'
        r = requests.get(url=api_url, params=dict(q=city_name, APPID=api_key))
        result_json = json.loads(r.text)
        return result_json['wind']['speed'] * 10
    except Exception as e:
        print(e)
        return 0


def get_series_value(series):
    return series.values[0]


def get_file_list(folder_name, prefix=''):
    return glob.glob(folder_name + '/*' + prefix + '.data')


def load_file(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    app.run(port=5001)
