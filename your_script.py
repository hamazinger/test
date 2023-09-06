import streamlit as st
import os
import time
import numpy as np
from google.cloud import bigquery
import pandas as pd
import json
from google.oauth2 import service_account
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import japanize_matplotlib
import unicodedata

@st.cache(ttl=600)
def run_query(query, client):
    query_job = client.query(query)
    rows_raw = query_job.result()
    rows = [dict(row) for row in rows_raw]
    return rows

def fetch_and_plot_data(keyword, client):
    if not keyword:
        st.warning("キーワードが入力されていません。")
        return

    keywords = [unicodedata.normalize('NFKC', k.strip().lower()) for k in keyword.split(",")]

    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=2)
    current_date = pd.Timestamp.now().normalize()

    pytrend = TrendReq(hl='ja', tz=540)
    pytrend.build_payload(kw_list=keywords)
    time.sleep(10)
    df_trends = pytrend.interest_over_time()
    df_trends_combined = df_trends[keywords].sum(axis=1)
    df_trends_quarterly = df_trends_combined.resample('Q').sum().loc[start_date:end_date]
    df_trends_quarterly = df_trends_quarterly.loc[:current_date]

    # For `business_it_article_api`
    queries_article = [f"""
    tag = "{k}"
    OR CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
    OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
    OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
    """ for k in keywords]
    combined_query_article = " AND ".join(queries_article)

    # For `bussiness_it_seminar_api` and `majisemi_seminar_api`
    queries_seminar = [f"""
    CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
    OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
    OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
    """ for k in keywords]
    combined_query_seminar = " AND ".join(queries_seminar)

    query_article = f"""
    SELECT *
    FROM `mythical-envoy-386309.majisemi.business_it_article_api`
    WHERE {combined_query_article}
    """
    query_bussiness_it_seminar = f"""
    SELECT *
    FROM `mythical-envoy-386309.majisemi.bussiness_it_seminar_api`
    WHERE {combined_query_seminar}
    """
    query_majisemi_seminar = f"""
    SELECT *
    FROM `mythical-envoy-386309.majisemi.majisemi_seminar_api`
    WHERE {combined_query_seminar}
    """

    rows_article = run_query(query_article, client)
    rows_bussiness_it_seminar = run_query(query_bussiness_it_seminar, client)
    rows_majisemi_seminar = run_query(query_majisemi_seminar, client)

    df_article = pd.DataFrame(rows_article)
    df_bussiness_it_seminar = pd.DataFrame(rows_bussiness_it_seminar)
    df_majisemi_seminar = pd.DataFrame(rows_majisemi_seminar)

    # Google Trends data plot
    st.subheader('Googleトレンド')
    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(df_trends_quarterly.index, df_trends_quarterly, marker='o')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Google Trends')
    ax.set_title(f'Googleトレンド: {keyword}')
    st.pyplot(fig)

    # Plotting data from `business_it_article_api`
    if not df_article.empty:
        st.subheader('business_it_article_api データ')
        st.write(df_article)

    # Plotting data from `bussiness_it_seminar_api`
    if not df_bussiness_it_seminar.empty:
        st.subheader('bussiness_it_seminar_api データ')
        st.write(df_bussiness_it_seminar)

    # Plotting data from `majisemi_seminar_api`
    if not df_majisemi_seminar.empty:
        st.subheader('majisemi_seminar_api データ')
        st.write(df_majisemi_seminar)

def main():
    st.title("キーワード分析")

    keyword1 = st.text_input("キーワード1をカンマで区切って複数入力可能（例：python,java）")
    keyword2 = st.text_input("キーワード2をカンマで区切って複数入力可能（比較するキーワードを入力）")
    execute_button = st.button("分析を実行")

    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
    client = bigquery.Client(credentials=credentials)

    if execute_button:
        col1, col2 = st.columns(2)
        with col1:
            fetch_and_plot_data(keyword1, client)
        with col2:
            fetch_and_plot_data(keyword2, client)

if __name__ == "__main__":
    main()
