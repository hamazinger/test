import streamlit as st
import os

import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from google.cloud import bigquery
import pandas as pd
import json

from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.bigquery import SchemaField

from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import japanize_matplotlib
# %matplotlib inline  # Streamlitではこの行は不要です。

# ここからコードを追加します。
def main():
    st.title("Google Trends and Article Analysis")
    #keyword = st.text_input("Enter a keyword", value='マネーフォワード')  # ユーザがキーワードを入力できるテキストボックスを作成します。
    keyword = st.text_input("Enter a keyword")

    # 変数の設定
    project_id = 'mythical-envoy-386309'
    destination_table = 'mythical-envoy-386309.majisemi.bussiness_it_article'

    # 認証情報の設定
    # Create API client.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(credentials=credentials)


    # クライアントの作成
    # client = bigquery.Client(credentials=credentials, project=project_id)

    # 残りのコードをここに追加します。
    import openai

    openai.api_key = 'OPENAI_API_KEY'
    
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
    
    
    related_terms = [keyword] + get_related_terms(keyword, topn=5)
    
    
    pytrend = TrendReq(hl='ja', tz=540)
    # Googleトレンドのデータ取得（キーワードのトレンド）
    pytrend.build_payload(kw_list=[keyword])
    df_trends = pytrend.interest_over_time()
    
    # データフレームからキーワードの列のみを取り出し、3ヶ月ごとにリサンプリング
    df_trends_quarterly = df_trends[keyword].resample('3M').sum()
    
    # Google Trendsのデータから最初と最後の日付を取得
    start_date = df_trends_quarterly.index.min().strftime("%Y-%m-%d")
    end_date = df_trends_quarterly.index.max().strftime("%Y-%m-%d")
    
    # WHERE句を作成
    # where_clause = " OR ".join([f"title LIKE '%{term}%' OR tag LIKE '%{term}%'" for term in related_terms])
    
    # query = f"""
    # SELECT date, title, tag
    # FROM `mythical-envoy-386309.majisemi.bussiness_it_article`
    # WHERE (date BETWEEN '{start_date}' AND '{end_date}') AND ({where_clause})
    # """
    
    #キーワードを部分文字列として含む単語は抽出しないよう改善
    where_clause = " OR ".join([f"REGEXP_CONTAINS(title, r'\\b{term}\\b') OR REGEXP_CONTAINS(tag, r'\\b{term}\\b')" for term in related_terms])
    
    query = f"""
    SELECT date, title, tag
    FROM `mythical-envoy-386309.majisemi.bussiness_it_article`
    WHERE (date BETWEEN '{start_date}' AND '{end_date}') AND ({where_clause})
    """
    
    
    # クエリの実行と結果の取得
    df = client.query(query).to_dataframe()
    
    # 日付列をDatetime型に変換
    # df['date'] = pd.to_datetime(df['date'])
    
    # # 3ヶ月単位での集計
    # df_quarterly = df.resample('3M', on='date').count()['title']
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # 3ヶ月単位での集計
    df_quarterly = df.resample('3M').count()['title']
    
    # 近似曲線の描画
    fig, ax1 = plt.subplots(figsize=(14,7))
    
    # プロットのデータポイント数
    n_plot_points = 10000
    
    # x軸の値を等間隔に補間（日時をエポック秒に変換）
    xnew = np.linspace(df_trends_quarterly.index.astype(int).min(), df_trends_quarterly.index.astype(int).max(), n_plot_points)
    
    # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
    spl_trends = make_interp_spline(df_trends_quarterly.index.astype(int), df_trends_quarterly)
    ynew_trends = spl_trends(xnew)
    
    # Googleトレンドの近似曲線の描画（エポック秒を日時に戻す）
    ax2 = ax1.twinx()
    ax2.plot(pd.to_datetime(xnew), ynew_trends, color='tab:blue')
    
    if not df_quarterly.empty:
        xnew = np.linspace(df_quarterly.index.astype(int).min(), df_quarterly.index.astype(int).max(), n_plot_points)
        spl_quarterly = make_interp_spline(df_quarterly.index.astype(int), df_quarterly)
        ynew_quarterly = spl_quarterly(xnew)
        ax1.plot(pd.to_datetime(xnew), ynew_quarterly, color='tab:red')
    
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of articles', color='tab:red')
    ax2.set_ylabel('Google Trends', color='tab:blue')
    plt.title('Quarterly trends for keyword: {}'.format(keyword))
    #plt.show()
    # ただし、最後の plt.show() は以下のように書き換える必要があります。
    st.pyplot(fig)

if __name__ == "__main__":
    main()
