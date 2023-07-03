import streamlit as st
import os

import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from google.cloud import bigquery
import pandas as pd
import json

from google.oauth2 import service_account
from google.cloud.bigquery import SchemaField

from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import japanize_matplotlib

# ここからコードを追加します。
def main():
    st.title("Google Trends and Article Analysis")
    keyword = st.text_input("Enter a keyword")
    execute_button = st.button("Execute Query") 

    # 変数の設定
    project_id = 'mythical-envoy-386309'
    destination_table = 'mythical-envoy-386309.majisemi.bussiness_it_article'

    # 認証情報の設定
    # Create API client.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(credentials=credentials)

    # 残りのコードをここに追加します。
    import openai

    openai.api_key = st.secrets["openai"]["api_key"]
    
    def get_related_terms(token, topn=20):
        """
        Look up the topn most similar terms to token and print them as a formatted list.
        """
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"'{token}' と同じ意味で使われる単語（日本語）を {topn} 個リストアップして、改行せずにカンマ区切りで出力してください。",
            temperature=0.3,
            max_tokens=60
        )
        
        related_terms = response.choices[0].text.strip().split(', ')
    
        return related_terms
    
    # Use st.cache_data to only rerun when the query changes or after 10 min.
    @st.cache_data(ttl=600)
    def run_query(query):
        query_job = client.query(query)
        rows_raw = query_job.result()
        # Convert to list of dicts. Required for st.cache_data to hash the return value.
        rows = [dict(row) for row in rows_raw]
        return rows

    # if execute_button:  
    if execute_button and keyword: 
        
        # OpenAIより関連キーワードを取得
        related_terms = [keyword] + get_related_terms(keyword, topn=5)

        # Googleトレンドのデータ取得（キーワードのトレンド）
        pytrend = TrendReq(hl='ja', tz=540)
        pytrend.build_payload(kw_list=[keyword])
        df_trends = pytrend.interest_over_time()
        
        # データフレームからキーワードの列のみを取り出し、3ヶ月ごとにリサンプリング
        df_trends_quarterly = df_trends[keyword].resample('3M').sum()
        
        # Google Trendsのデータから最初と最後の日付を取得
        start_date = df_trends_quarterly.index.min().strftime("%Y-%m-%d")
        end_date = df_trends_quarterly.index.max().strftime("%Y-%m-%d")
        
        # 関連キーワードについての記事数をBigQueryから取得
        rows = []
        for term in related_terms:
            query = f"""
            SELECT date, COUNT(title) as count 
            FROM `{destination_table}` 
            WHERE title LIKE '%{term}%' AND date BETWEEN '{start_date}' AND '{end_date}' 
            GROUP BY date
            ORDER BY date
            """
            rows.extend(run_query(query))
        
        # 取得したデータをデータフレームに変換
        df_articles = pd.DataFrame(rows)
        df_articles['date'] = pd.to_datetime(df_articles['date'])
        df_articles.set_index('date', inplace=True)
        df_articles_quarterly = df_articles.resample('3M').sum()

        # プロット
        plt.plot(df_trends_quarterly.index, df_trends_quarterly, label='Google Trends')
        plt.plot(df_articles_quarterly.index, df_articles_quarterly, label='Article Counts')
        plt.legend(loc='best')
        st.pyplot(plt)

if __name__ == "__main__":
    main()
