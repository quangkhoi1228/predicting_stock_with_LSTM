import pandas as pd
import os
import numpy as np
import model_handle_util
import json
import webbrowser
import threading
import random


from model_handle_util import main


def view_all_result(options):
    json_path = options['json_path'] if 'json_path' in options else f'{os.getcwd()}/web/json/VN30/'
    all_file = os.listdir(json_path)
    save_file_name = 'vn30_result'

    list_statistic = {}
    for file_name in all_file:
        symbol = file_name.replace('.json', '')
        json_data_file_path = f'{json_path}{file_name}'
        json_data_file = open(json_data_file_path, "r")
        symbol_statistic = json.loads(json_data_file.read())
        json_data_file.close()
        list_statistic[symbol] = symbol_statistic

    json.dumps(list_statistic, sort_keys=True)
    template_file = open(f'{os.getcwd()}/web/template/template.html', "r")
    content = template_file.read()
    template_file.close()

    js_file = open(f'{os.getcwd()}/web/static/js/index.js', "r")
    js_content = js_file.read()
    js_file.close()

    css_file = open(f'{os.getcwd()}/web/static/css/bulma.min.css', "r")
    css_content = css_file.read()
    css_file.close()

    result_file_name = f'{os.getcwd()}/web/html/0_all_result_{save_file_name}.html'
    result_file = open(result_file_name, "w")
    result_file.write(
        f'{content}<style>{css_content}</style><script>{js_content}</script><script>var type="all";  var data = {list_statistic}; index.build(data);</script>')
    result_file.close()

    url = "file://"+result_file_name
    webbrowser.open(url, new=1)


def view_result(options):

    list_statistic = {}
    symbol = options['symbol']
    json_data_file_path = options['json_data_file_path']
    save_file_name = options['save_file_name'] if 'save_file_name' in options else symbol
    json_data_file = open(json_data_file_path, "r")
    symbol_statistic = json.loads(json_data_file.read())
    json_data_file.close()
    list_statistic[symbol] = symbol_statistic

    json.dumps(list_statistic, sort_keys=True)
    template_file = open(f'{os.getcwd()}/web/template/template.html', "r")
    content = template_file.read()
    template_file.close()

    js_file = open(f'{os.getcwd()}/web/static/js/index.js', "r")
    js_content = js_file.read()
    js_file.close()

    css_file = open(f'{os.getcwd()}/web/static/css/bulma.min.css', "r")
    css_content = css_file.read()
    css_file.close()

    result_file_name = f'{os.getcwd()}/web/html/{save_file_name}.html'
    result_file = open(result_file_name, "w")
    result_file.write(
        f'{content}<style>{css_content}</style><script>{js_content}</script><script>var type="all";  var data = {list_statistic}; index.build(data);</script>')
    result_file.close()

    url = "file://"+result_file_name
    webbrowser.open(url, new=1)
