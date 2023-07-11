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
        time.sleep(5) # 5秒待つ
        df_trends = pytrend.interest_over_time()
        
        # データフレームからキーワードの列のみを取り出し、3ヶ月ごとにリサンプリング
        df_trends_quarterly = df_trends[keyword].resample('3M').sum()
        
        
        
        # Google Trendsのデータから最初と最後の日付を取得
        start_date = df_trends_quarterly.index.min().strftime("%Y-%m-%d")
        end_date = df_trends_quarterly.index.max().strftime("%Y-%m-%d")

        where_clause = " OR ".join([f"title LIKE '%{term}%'" for term in related_terms])

        query = f"""
        SELECT date, title
        FROM `{destination_table}` 
        WHERE ({where_clause}) AND date BETWEEN '{start_date}' AND '{end_date}' 
        ORDER BY date
        """
        
        rows = run_query(query)
        
        df_articles = pd.DataFrame(rows)

        # プロット
        fig, ax1 = plt.subplots()
        
        color = 'tab:red'
        ax1.set_xlabel('Time (Quarterly)')
        ax1.set_ylabel('Article Counts', color=color)
        
        # プロットのデータポイント数
        n_plot_points = 10000
        xnew = np.linspace(df_trends_quarterly.index.astype(int).min(), df_trends_quarterly.index.astype(int).max(), n_plot_points)
        
        if not df_articles.empty:
            df_articles['date'] = pd.to_datetime(df_articles['date'])
            df_articles.set_index('date', inplace=True)
            df_articles_quarterly = df_articles.resample('3M').count()
        
            # 近似曲線の描画
            # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
            # spl_articles = make_interp_spline(df_articles_quarterly.index.astype(int), df_articles_quarterly)
            # ynew_articles = spl_articles(xnew)
            # 記事数の近似曲線の描画（エポック秒を日時に戻す）
            # ax1.plot(pd.to_datetime(xnew), ynew_articles, color=color)
            ax1.plot(df_articles_quarterly.index, df_articles_quarterly, color=color)
            # 目盛り幅を設定します。
            # ax1.yaxis.set_ticks(np.arange(0, df_articles_quarterly.max() + 1, 1))
            ax1.yaxis.set_ticks(np.arange(0, df_articles_quarterly.iloc[:, 0].max() + 1, 1))
        
        ax1.tick_params(axis='y', labelcolor=color)
        
        # x軸の日付のフォーマットをyyyyに変更
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Google Trends', color=color)
        
        # Google Trendsの近似曲線の描画
        # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
        # spl_trends = make_interp_spline(df_trends_quarterly.index.astype(int), df_trends_quarterly)
        # ynew_trends = spl_trends(xnew)
        # Googleトレンドの近似曲線の描画（エポック秒を日時に戻す）
        # ax2.plot(pd.to_datetime(xnew), ynew_trends, color=color)
        
        ax2.plot(df_trends_quarterly.index, df_trends_quarterly, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        st.pyplot(fig)
        if not df_articles.empty:
            st.dataframe(df_articles.reset_index())
        st.write(f"Related terms: {', '.join(related_terms)}")

        # # プロット
        # fig, ax1 = plt.subplots()
        
        # color = 'tab:red'
        # ax1.set_xlabel('Time (Quarterly)')
        # ax1.set_ylabel('Article Counts', color=color)
        
        # if not df_articles.empty:
        #     df_articles['date'] = pd.to_datetime(df_articles['date'])
        #     df_articles.set_index('date', inplace=True)
        #     df_articles_quarterly = df_articles.resample('3M').count()
        
        #     # 近似曲線の描画
        #     # プロットのデータポイント数
        #     n_plot_points = 10000
        #     # x軸の値を等間隔に補間（日時をエポック秒に変換）
        #     xnew = np.linspace(df_articles_quarterly.index.astype(int).min(), df_articles_quarterly.index.astype(int).max(), n_plot_points)
        #     # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
        #     spl_articles = make_interp_spline(df_articles_quarterly.index.astype(int), df_articles_quarterly)
        #     ynew_articles = spl_articles(xnew)
        #     # 記事数の近似曲線の描画（エポック秒を日時に戻す）
        #     ax1.plot(pd.to_datetime(xnew), ynew_articles, color=color)
        
        # ax1.tick_params(axis='y', labelcolor=color)
        
        # # x軸の日付のフォーマットをyyyyに変更
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # ax2 = ax1.twinx()
        # color = 'tab:blue'
        # ax2.set_ylabel('Google Trends', color=color)
        
        # # Google Trendsの近似曲線の描画
        # # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
        # spl_trends = make_interp_spline(df_trends_quarterly.index.astype(int), df_trends_quarterly)
        # ynew_trends = spl_trends(xnew)
        # # Googleトレンドの近似曲線の描画（エポック秒を日時に戻す）
        # ax2.plot(pd.to_datetime(xnew), ynew_trends, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        # fig.tight_layout()
        # st.pyplot(fig)
        # if not df_articles.empty:
        #     st.dataframe(df_articles.reset_index())
        # st.write(f"Related terms: {', '.join(related_terms)}")

        
        # df_articles['date'] = pd.to_datetime(df_articles['date'])
        # df_articles.set_index('date', inplace=True)
        # df_articles_quarterly = df_articles.resample('3M').count()
        
        # # プロット
        # fig, ax1 = plt.subplots()
        
        # color = 'tab:red'
        # ax1.set_xlabel('Time (Quarterly)')
        # ax1.set_ylabel('Article Counts', color=color)
        
        # if not df_articles_quarterly.empty:
        #     # 近似曲線の描画
        #     # プロットのデータポイント数
        #     n_plot_points = 10000
        #     # x軸の値を等間隔に補間（日時をエポック秒に変換）
        #     xnew = np.linspace(df_articles_quarterly.index.astype(int).min(), df_articles_quarterly.index.astype(int).max(), n_plot_points)
        #     # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
        #     spl_articles = make_interp_spline(df_articles_quarterly.index.astype(int), df_articles_quarterly)
        #     ynew_articles = spl_articles(xnew)
        #     # 記事数の近似曲線の描画（エポック秒を日時に戻す）
        #     ax1.plot(pd.to_datetime(xnew), ynew_articles, color=color)
        
        # ax1.tick_params(axis='y', labelcolor=color)
        
        # # x軸の日付のフォーマットをyyyyに変更
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # ax2 = ax1.twinx()
        # color = 'tab:blue'
        # ax2.set_ylabel('Google Trends', color=color)
        
        # # Google Trendsの近似曲線の描画
        # # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
        # spl_trends = make_interp_spline(df_trends_quarterly.index.astype(int), df_trends_quarterly)
        # ynew_trends = spl_trends(xnew)
        # # Googleトレンドの近似曲線の描画（エポック秒を日時に戻す）
        # ax2.plot(pd.to_datetime(xnew), ynew_trends, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        # fig.tight_layout()
        # st.pyplot(fig)
        # st.dataframe(df_articles.reset_index())
        # st.write(f"Related terms: {', '.join(related_terms)}")


        # if not df_articles.empty:
        #     df_articles['date'] = pd.to_datetime(df_articles['date'])
        #     df_articles.set_index('date', inplace=True)
        #     df_articles_quarterly = df_articles.resample('3M').count()
    
        #     # プロット
        #     fig, ax1 = plt.subplots()
    
        #     color = 'tab:red'
        #     ax1.set_xlabel('Time (Quarterly)')
        #     ax1.set_ylabel('Article Counts', color=color)
            
        #     # 近似曲線の描画
        #     # プロットのデータポイント数
        #     n_plot_points = 10000
        #     # x軸の値を等間隔に補間（日時をエポック秒に変換）
        #     xnew = np.linspace(df_articles_quarterly.index.astype(int).min(), df_articles_quarterly.index.astype(int).max(), n_plot_points)
        #     # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
        #     spl_articles = make_interp_spline(df_articles_quarterly.index.astype(int), df_articles_quarterly)
        #     ynew_articles = spl_articles(xnew)
        #     # 記事数の近似曲線の描画（エポック秒を日時に戻す）
        #     ax1.plot(pd.to_datetime(xnew), ynew_articles, color=color)
            
        #     ax1.tick_params(axis='y', labelcolor=color)

        #     # x軸の日付のフォーマットをyyyyに変更
        #     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
        #     ax2 = ax1.twinx()
        #     color = 'tab:blue'
        #     ax2.set_ylabel('Google Trends', color=color)
    
        #     # Google Trendsの近似曲線の描画
        #     # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
        #     spl_trends = make_interp_spline(df_trends_quarterly.index.astype(int), df_trends_quarterly)
        #     ynew_trends = spl_trends(xnew)
        #     # Googleトレンドの近似曲線の描画（エポック秒を日時に戻す）
        #     ax2.plot(pd.to_datetime(xnew), ynew_trends, color=color)
        #     ax2.tick_params(axis='y', labelcolor=color)
        #     fig.tight_layout()
        #     st.pyplot(fig)
        #     st.dataframe(df_articles.reset_index())
        #     st.write(f"Related terms: {', '.join(related_terms)}")

        # else:
        #     st.write("No articles found for the related terms.")



if __name__ == "__main__":
    main()
