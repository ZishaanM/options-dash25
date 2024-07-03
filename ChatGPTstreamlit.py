# import os
# import pandas as pd
# import sqlalchemy as sa
# import streamlit as st

# def establish_connection_source():
#     server = os.getenv('sra_azure_mysql_server')
#     username = os.getenv('sra_azure_mysql_username')
#     userpwd = os.getenv('sra_azure_mysql_pwd')
#     con = {}
#     db = 'source'
#     connect_string = ''.join(['mysql+mysqldb://', username, ':', userpwd, '@', server, '/', db])
#     con['connect_string'] = connect_string
#     con['engine'] = sa.create_engine(connect_string, echo=False)
#     con['metadata'] = sa.MetaData()
#     Session = sa.orm.sessionmaker(bind=con['engine'])
#     con['session'] = Session()
#     con['yahoo_options'] = sa.Table('yahoo_options', con['metadata'], autoload=True,
#                                     autoload_with=con['engine'])
#     con['yahoo_intraday'] = sa.Table('yahoo_intraday', con['metadata'], autoload=True,
#                                     autoload_with=con['engine'])
#     con['av_minute'] = sa.Table('av_minute', con['metadata'], autoload=True,
#                                     autoload_with=con['engine'])
#     con['sharadar_marketdata_sfp'] = sa.Table('sharadar_marketdata_sfp', con['metadata'], autoload=True,
#                                               autoload_with=con['engine'])
#     return con

# def fetch_yahoo_intraday(con, ticker, cdate):
#     qry = con['session'].query(con['yahoo_intraday'])
#     qry = qry.filter(con['yahoo_intraday'].c.ticker == ticker)
#     qry = qry.filter(con['yahoo_intraday'].c.asof_date == cdate)
#     qry = qry.order_by(con['yahoo_intraday'].c.asof.desc())
#     res = pd.read_sql(qry.statement, qry.session.bind, parse_dates=['asof','asof_date'])
#     return res

# def main():
#     st.title('Yahoo Finance Data Visualization')
    
#     # User inputs
#     ticker = st.text_input('Enter Ticker Symbol (e.g., SPY)', 'SPY')
#     cdate = st.date_input('Select Date')
    
#     if st.button('Fetch Data'):
#         con = establish_connection_source()
#         data = fetch_yahoo_intraday(con, ticker, cdate)
        
#         if not data.empty:
#             st.write('Data Retrieved:')
#             st.write(data)
            
#             st.line_chart(data.set_index('asof')['close'])
#         else:
#             st.write('No data available for the selected date and ticker.')

# if __name__ == '__main__':
#     main()

import os
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import streamlit as st

def establish_connection_source():
    server = os.getenv('sra_azure_mysql_server')
    username = os.getenv('sra_azure_mysql_username')
    userpwd = os.getenv('sra_azure_mysql_pwd')
    con = {}
    db = 'source'
    connect_string = ''.join(['mysql+mysqldb://', username, ':', userpwd, '@', server, '/', db])
    con['connect_string'] = connect_string
    con['engine'] = sa.create_engine(connect_string, echo=False)
    con['metadata'] = sa.MetaData()
    Session = sessionmaker(bind=con['engine'])
    con['session'] = Session()
    con['yahoo_options'] = sa.Table('yahoo_options', con['metadata'], autoload=True,
                                    autoload_with=con['engine'])
    con['yahoo_intraday'] = sa.Table('yahoo_intraday', con['metadata'], autoload=True,
                                    autoload_with=con['engine'])
    con['av_minute'] = sa.Table('av_minute', con['metadata'], autoload=True,
                                    autoload_with=con['engine'])
    con['sharadar_marketdata_sfp'] = sa.Table('sharadar_marketdata_sfp', con['metadata'], autoload=True,
                                              autoload_with=con['engine'])
    return con

def fetch_yahoo_intraday(con, ticker, cdate):
    qry = con['session'].query(con['yahoo_intraday'])
    qry = qry.filter(con['yahoo_intraday'].c.ticker == ticker)
    qry = qry.filter(con['yahoo_intraday'].c.asof_date == cdate)
    qry = qry.order_by(con['yahoo_intraday'].c.asof.desc())
    res = pd.read_sql(qry.statement, qry.session.bind, parse_dates=['asof','asof_date'])
    return res

def main():
    st.title('Yahoo Finance Data Visualization')
    
    # User inputs
    ticker = st.text_input('Enter Ticker Symbol (e.g., SPY)', 'SPY')
    cdate = st.date_input('Select Date')
    
    if st.button('Fetch Data'):
        con = establish_connection_source()
        data = fetch_yahoo_intraday(con, ticker, cdate)
        
        if not data.empty:
            st.write('Data Retrieved:')
            st.write(data)
            
            st.line_chart(data.set_index('asof')['close'])
        else:
            st.write('No data available for the selected date and ticker.')

if __name__ == '__main__':
    main()
