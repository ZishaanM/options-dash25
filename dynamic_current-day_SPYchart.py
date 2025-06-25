import os, pdb
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import streamlit as st
from datetime import date
from pandas.tseries.offsets import BDay

import make_charts as mc

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
    con['metadata'].reflect(bind=con['engine'])
    
    Session = sessionmaker(bind=con['engine'])
    con['session'] = Session()
    con['yahoo_options'] = con['metadata'].tables['yahoo_options']
    con['yahoo_intraday'] = con['metadata'].tables['yahoo_intraday']
    con['av_minute'] = con['metadata'].tables['av_minute']
    con['sharadar_marketdata_sfp'] = con['metadata'].tables['sharadar_marketdata_sfp']
    return con

def fetch_yahoo_intraday(con, ticker, cdate):
    print(f"Fetching data for ticker: {ticker} on date: {cdate}")
    
    qry = con['session'].query(con['yahoo_intraday'])
    qry = qry.filter(con['yahoo_intraday'].c.ticker == ticker)
    qry = qry.filter(con['yahoo_intraday'].c.asof_date == cdate)
    qry = qry.order_by(con['yahoo_intraday'].c.asof.desc())
    
    print(f"Constructed query: {qry}")
    
    res = pd.read_sql(qry.statement, qry.session.bind, parse_dates=['asof','asof_date'])
    print(f"Query results: {res}")
    
    return res


def main():
    st.title('Yahoo Finance Data Visualization')
    
    # User inputs
    ticker = st.text_input('Enter Ticker Symbol (e.g., SPY)', 'SPY')
    # Still does not print last business day
    cdate = date.today()
    con = establish_connection_source()
    data = fetch_yahoo_intraday(con, ticker, cdate)
    # last_b_day = cdate
    # while data == "Empty DataFrame":
    #     print("no data")
    #     last_b_day = last_b_day - 1
    #     data = fetch_yahoo_intraday(con, ticker, last_b_day)
    # print("----------")
    # print(data)
    # print("----------")
    to_graph = data[['asof','close']].sort_values(by=['asof']).set_index('asof')
    to_graph = to_graph.sort_index()
    intraday_chart = mc.plotly_intraday_chart(to_graph, '{}'.format(ticker.upper()), '', height=300, pct=False)

    if not data.empty:
        st.plotly_chart(intraday_chart, use_container_width=True)
    else:
        st.write('No data available for the selected date and ticker.')

if __name__ == '__main__':
    main()


#Always pull latest data