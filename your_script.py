import streamlit as st
import os
import time
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from google.cloud import bigquery
import pandas as pd
import json
from google.oauth2 import service_account
from google.cloud.bigquery import SchemaField
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib

def main():
    st.title("キーワード分析")
    keyword = st.text_input("キーワードを入力（アルファベットは「小文字」で入力してください）")
    execute_button = st.button("分析を実行") 

    project_id = 'mythical-envoy-386309'

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(credentials=credentials)

    @st.cache_data(ttl=600)
    def run_query(query):
        query_job = client.query(query)
        rows_raw = query_job.result()
        rows = [dict(row) for row in rows_raw]
        return rows

    if execute_button and keyword: 
        
        pytrend = TrendReq(hl='ja', tz=540)
        pytrend.build_payload(kw_list=[keyword])
        time.sleep(5)
        df_trends = pytrend.interest_over_time()
        df_trends_quarterly = df_trends[keyword].resample('Q').sum()
        
        query = f"""
        SELECT *
        FROM `mythical-envoy-386309.majisemi.business_it_article_api`
        WHERE tag = "{keyword}"
           OR CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
           OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
           OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
        """
        
        query2 = f"""
        SELECT *
        FROM `mythical-envoy-386309.majisemi.bussiness_it_seminar_api`
        WHERE CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
           OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
           OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
        """
        
        rows = run_query(query)
        rows2 = run_query(query2)

        if not rows:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(rows)

        if not rows2:
            df2 = pd.DataFrame()
        else:
            df2 = pd.DataFrame(rows2)

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df_quarterly = df.resample('Q').count()['title'].loc['2021':]
        
        if not df2.empty:
            df2['date'] = pd.to_datetime(df2['date'])
            df2 = df2.set_index('date')
            df2_quarterly = df2.resample('Q').count()['title'].loc['2021':]

        df_trends_quarterly = df_trends[keyword].resample('Q').sum().loc['2021':]

        st.subheader('他社メディア記事・セミナー数&Googleトレンド')

        plt.rcParams['font.size'] = 15
        fig, ax1 = plt.subplots(figsize=(14,7))

        ax2 = ax1.twinx()
        ax2.plot(df_trends_quarterly.index, df_trends_quarterly, color='tab:blue', marker='o', label='Google Trends')

        if not df.empty:
            ax1.plot(df_quarterly.index, df_quarterly, color='tab:red', marker='o', label='Number of articles')
        
        if not df2.empty:
            ax1.plot(df2_quarterly.index, df2_quarterly, color='tab:green', marker='o', label='Number of seminars')

        ax1.set_xlabel('Quarter')
        ax1.set_ylabel('Number of articles and seminars', color='tab:red')
        ax2.set_ylabel('Google Trends', color='tab:blue')
        plt.title('Quarterly trends for keyword: {}'.format(keyword))
        ax1.legend(loc="upper left")
        st.pyplot(fig)

        # ... [略] 以下のコードは変更していません。

if __name__ == "__main__":
    main()
