import geoip2.database
import numpy as np
import pandas as pd

reader = geoip2.database.Reader("../data/GeoLite2-City.mmdb")


def extract_ip_features(ip):
    if pd.isna(ip) or not isinstance(ip, str):
        return pd.Series([np.nan, np.nan, np.nan])

    try:
        response = reader.city(ip)

        country = response.country.name
        timezone = response.location.time_zone
        continent = response.continent.name

        return pd.Series([country, timezone, continent])
    except:
        return pd.Series([np.nan, np.nan, np.nan])
