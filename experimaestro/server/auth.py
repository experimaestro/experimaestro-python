# Handles user authentification
# Adapted from https://github.com/aaugustin/websockets/blob/main/experiments/authentication/app.py

import urllib.parse
import http
import http.cookies


def get_cookie(raw, key):
    cookie = http.cookies.SimpleCookie(raw)
    morsel = cookie.get(key)
    if morsel is not None:
        return morsel.value


def get_query_param(path, key):
    query = urllib.parse.urlparse(path).query
    params = urllib.parse.parse_qs(query)
    values = params.get(key, [])
    if len(values) == 1:
        return values[0]
