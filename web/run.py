from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import glob
import pickle
import requests
import json
from werkzeug.contrib.cache import SimpleCache

WEATHER_API_KEY = ''
FL_INFO_FOLDER = '../data/flight_info.data'
DELAYS_INFO_FOLDER = '../data/delays.data'
AVG_DELAYS_INFO_FOLDER = '../data/avg_delays.data'
CLFS_INFO_FOLDER = '../data/clfs'
COLS_FOLDER = '../data/values_dicts'
INFO_DATA = {}
CACHE_TIMEOUT = 100
cache = SimpleCache()

app = Flask(__name__)


@app.before_request
def before_request():
    global INFO_DATA

    if len(INFO_DATA) == 0:
        INFO_DATA = load_initial_data()

    response = cache.get(request.path)
    if response:
        return response


@app.after_request
def cache_response(response):
    cache.set(request.path, response, CACHE_TIMEOUT)
    return response


@app.route('/')
def main_route():
    return 'Web service is up and running'


@app.route('/api/last_update')
def last_update():
    data = {'last_update': '08-05-2018'}
    return jsonify(get_formatted_json(data))


@app.route("/api/flights/predict", methods=['GET'])
def predict_flight():
    try:
        origin = request.args.get('origin', default='', type=str)
        dest = request.args.get('dest', default='', type=str)
        carrier = request.args.get('carrier', default='', type=str)
        fl_num = request.args.get('fl_num', default='', type=str)
        flight_date = datetime.strptime(request.args.get('flight_date'), "%Y-%m-%d").date()
        features = get_features(INFO_DATA, WEATHER_API_KEY, origin, dest, carrier, int(fl_num), flight_date)

        result = INFO_DATA['clfs'][origin]['clf'].predict_proba(features).tolist()[0]
        result = [{'cancelled_flight': result[0], 'delay': result[1], 'no_delay': result[2]}]
        return jsonify(get_formatted_json(result))
    except:
        return jsonify(get_formatted_json([], 'Could not check flight status. Please, check input params.'))


@app.route("/api/airlines", methods=['GET'])
def get_airlines_list():
    data = {'airlines': INFO_DATA['delays']['Carrier'].to_dict()}
    return jsonify(get_formatted_json(data))


@app.route("/api/airlines/flights", methods=['GET'])
def get_airlines_flights():
    try:
        carrier = request.args.get('carrier', default='', type=str)
        data = {'flights': INFO_DATA['fl_info'][INFO_DATA['fl_info']['carrier'] == carrier][['origin', 'dest', 'fl_num']].to_dict('index')}
        if len(data) == 0:
            raise ValueError('Incorrect carrier code')
        return jsonify(get_formatted_json(data))
    except:
        return jsonify(get_formatted_json([], 'Could not find selected airline. Please, check your request.'))


@app.route("/api/airlines/delay_rating", methods=['GET'])
def get_airlines_delay_ratings():
    data = {'airlines': INFO_DATA['delays'][['Carrier', 'Delay index']].to_dict('index')}
    return jsonify(get_formatted_json(data))


def load_initial_data():
    flight_info = load_file(FL_INFO_FOLDER)
    delays = load_file(DELAYS_INFO_FOLDER)
    avg_delays = load_file(AVG_DELAYS_INFO_FOLDER)
    files_list = get_file_list(CLFS_INFO_FOLDER)
    clfs = {}
    for file in files_list:
        airport = file.replace(CLFS_INFO_FOLDER + '/', '').replace('.data', '')
        clfs[airport] = {'clf': load_file(file), 'cols': load_file(COLS_FOLDER + '/' + airport + '.data')}
    return {'clfs': clfs, 'fl_info': flight_info, 'delays': delays, 'avg_delays': avg_delays}


def get_features(data, weather_api_key, origin, dest, carrier, flight, date):
    fl_info = data['fl_info'][(data['fl_info']['fl_num'] == flight) & (data['fl_info']['carrier'] == carrier)
                              & (data['fl_info']['origin'] == origin)]
    df = pd.DataFrame(columns=data['clfs'][origin]['cols'])
    df = df.append({'average_wind_speed': get_wind_speed_for_city(get_series_value(fl_info['origin_city_name']),
                                                                  weather_api_key),
                    'crs_dep_time': get_series_value(fl_info['crs_dep_time']),
                    'crs_elapsed_time': get_series_value(fl_info['crs_elapsed_time']),
                    'day_of_month': date.day,
                    'day_of_week': date.weekday(),
                    'month': date.month,
                    'quarter': date.month // 4,
                    'previous_flight_delay': 0,
                    'airline_delay_index': get_series_value(data['delays']
                                                            [data['delays']['Carrier'] == carrier]['Delay index']),
                    'airline_avg_delay': get_series_value(data['avg_delays']
                                                          [data['avg_delays']['carrier'] == carrier]['carrier_delay']),
                    'dest_' + dest: 1,
                    'day_of_year': int(date.strftime("%j"))}, ignore_index=True)
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


def get_formatted_json(data, message='ok'):
    formatted_data = {'message': message, 'date': datetime.now(), 'result': data}
    return formatted_data


if __name__ == '__main__':
    app.run()
