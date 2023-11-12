import streamlit as st
import os
import time
import numpy as np
from google.cloud import bigquery
import pandas as pd
import json

from google.oauth2 import service_account
from google.cloud.bigquery import SchemaField

from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import japanize_matplotlib
import unicodedata

# 認証情報の設定
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

def search_titles_with_keyword(keyword, table_name):
    # 正規表現を使用してタイトルにキーワードが含まれるレコードを検索するクエリを構築
    query = f"""
    SELECT *
    FROM `{table_name}`
    WHERE REGEXP_CONTAINS(title, r'\\b{keyword}\\b')
    """
    results = run_query(query)
    return results

@st.cache(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    rows = [dict(row) for row in rows_raw]
    return rows

def main():
    st.title("キーワード分析")

    # 2つのキーワード入力ボックスを配置
    keyword_input1 = st.text_input("キーワード1を入力")
    keyword_input2 = st.text_input("キーワード2を入力")
    execute_button = st.button("分析を実行")

    if execute_button:
        # それぞれのキーワードの取得と正規化
        keyword1 = unicodedata.normalize('NFKC', keyword_input1.strip().lower())
        keyword2 = unicodedata.normalize('NFKC', keyword_input2.strip().lower())

        # キーワード1に基づくデータ取得および分析
        st.write("## キーワード1の結果")
        result1 = analyze_keyword(keyword1)

        # キーワード1に基づくタイトルの検索結果
        st.write("### キーワード1に基づく記事の検索結果")
        results_articles_kw1 = search_titles_with_keyword(keyword1, 'mythical-envoy-386309.ex_media.article')
        if results_articles_kw1:
            st.write(pd.DataFrame(results_articles_kw1))
        else:
            st.write("キーワード1にマッチする記事はありませんでした。")

        # キーワード1に基づくセミナーの検索結果
        st.write("### キーワード1に基づくセミナーの検索結果")
        results_seminars_kw1 = search_titles_with_keyword(keyword1, 'mythical-envoy-386309.ex_media.seminar')
        if results_seminars_kw1:
            st.write(pd.DataFrame(results_seminars_kw1))
        else:
            st.write("キーワード1にマッチするセミナーはありませんでした。")

        # キーワード2に基づくデータ取得および分析
        st.write("## キーワード2の結果")
        result2 = analyze_keyword(keyword2)

        # キーワード2に基づくタイトルの検索結果
        st.write("### キーワード2に基づく記事の検索結果")
        results_articles_kw2 = search_titles_with_keyword(keyword2, 'mythical-envoy-386309.ex_media.article')
        if results_articles_kw2:
            st.write(pd.DataFrame(results_articles_kw2))
        else:
            st.write("キーワード2にマッチする記事はありませんでした。")

        # キーワード2に基づくセミナーの検索結果
        st.write("### キーワード2に基づくセミナーの検索結果")
        results_seminars_kw2 = search_titles_with_keyword(keyword2, 'mythical-envoy-386309.ex_media.seminar')
        if results_seminars_kw2:
            st.write(pd.DataFrame(results_seminars_kw2))
        else:
            st.write("キーワード2にマッチするセミナーはありませんでした。")

def analyze_keyword(keyword):
    keywords = [keyword]  # 単一のキーワードをリストに変換

    # Google Trendsのデータ取得
    pytrend = TrendReq(hl='ja', tz=540)
    pytrend.build_payload(kw_list=keywords)
    time.sleep(10)  # APIのレートリミットに注意
    df_trends = pytrend.interest_over_time()
    df_trends_combined = df_trends[keywords].sum(axis=1)
    df_trends_quarterly = df_trends_combined.resample('Q').sum()

    # タイトル検索
    articles_query = f"""
    SELECT *
    FROM `mythical-envoy-386309.ex_media.article`
    WHERE REGEXP_CONTAINS(title, r'\\b{keyword}\\b')
    """
    articles = run_query(articles_query)

    seminar_query = f"""
    SELECT *
    FROM `mythical-envoy-386309.ex_media.seminar`
    WHERE REGEXP_CONTAINS(title, r'\\b{keyword}\\b')
    """
    seminars = run_query(seminar_query)

    # 結果をデータフレームに変換
    df_articles = pd.DataFrame(articles)
    df_seminars = pd.DataFrame(seminars)

    # Google Trendsグラフの描画
    st.subheader('Google Trends')
    plt.figure(figsize=(10, 4))
    plt.plot(df_trends_quarterly.index, df_trends_quarterly, color='blue', marker='o')
    plt.title(f'Google Trends for "{keyword}"')
    plt.xlabel('Date')
    plt.ylabel('Trends Score')
    st.pyplot(plt)

    # 記事とセミナーの結果の表示
    if not df_articles.empty:
        st.subheader('関連記事')
        st.dataframe(df_articles)

    if not df_seminars.empty:
        st.subheader('関連セミナー')
        st.dataframe(df_seminars)

    if df_articles.empty and df_seminars.empty:
        st.write(f'"{keyword}"に関連する記事やセミナーは見つかりませんでした。')

    return f"{keyword}の分析結果"


if __name__ == "__main__":
    main()
