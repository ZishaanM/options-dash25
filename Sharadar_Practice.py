import nasdaqdatalink
import pandas as pd

# Set your API key
nasdaqdatalink.ApiConfig.api_key = 'FhxPgXcvBjc7fxfFAyLa'

data = nasdaqdatalink.get_table('QDL/FON', qopts={'columns': ['type', 'market_participation']})

print(data.head(15))