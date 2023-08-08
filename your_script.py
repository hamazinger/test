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
    st.title("キーワード分析")
    keyword = st.text_input("キーワードを入力（アルファベットは「小文字」で入力してください）")
    execute_button = st.button("分析を実行") 

    # 変数の設定
    project_id = 'mythical-envoy-386309'
    #destination_table = 'mythical-envoy-386309.majisemi.bussiness_it_article'

    # 認証情報の設定
    # Create API client.
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(credentials=credentials)

    # 残りのコードをここに追加します。
    # import openai

    # openai.api_key = st.secrets["openai"]["api_key"]
    
    # def get_related_terms(token, topn=20):
    #     """
    #     Look up the topn most similar terms to token and print them as a formatted list.
    #     """
    #     response = openai.Completion.create(
    #         engine="text-davinci-002",
    #         prompt=f"'{token}' と同じ意味で使われる単語（日本語）を {topn} 個リストアップして、改行せずにカンマ区切りで出力してください。",
    #         temperature=0.3,
    #         max_tokens=60
    #     )
        
    #     related_terms = response.choices[0].text.strip().split(', ')
    
    #     return related_terms
    
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
        
        # # OpenAIより関連キーワードを取得
        # related_terms = [keyword] + get_related_terms(keyword, topn=5)

        # Googleトレンドのデータ取得（キーワードのトレンド）
        # pytrend = TrendReq(hl='ja', tz=540)
        # pytrend.build_payload(kw_list=[keyword])
        # time.sleep(5) # 5秒待つ
        # df_trends = pytrend.interest_over_time()
        # # データフレームからキーワードの列のみを取り出し、3ヶ月ごとにリサンプリング
        # df_trends_quarterly = df_trends[keyword].resample('3M').sum()

        
        #------------Googleトレンドのデータ取得---------------------
        pytrend = TrendReq(hl='ja', tz=540)
        pytrend.build_payload(kw_list=[keyword])
        time.sleep(5) # 5秒待つ
        df_trends = pytrend.interest_over_time()
        df_trends_quarterly = df_trends[keyword].resample('Q').sum()
                
        # # Google Trendsのデータから最初と最後の日付を取得
        # start_date = df_trends_quarterly.index.min().strftime("%Y-%m-%d")
        # end_date = df_trends_quarterly.index.max().strftime("%Y-%m-%d")

        # where_clause = " OR ".join([f"title LIKE '%{term}%'" for term in related_terms])

        # query = f"""
        # SELECT date, title
        # FROM `{destination_table}` 
        # WHERE ({where_clause}) AND date BETWEEN '{start_date}' AND '{end_date}' 
        # ORDER BY date
        # """

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
        
        df = pd.DataFrame(rows)
        df2 = pd.DataFrame(rows2)

        # 日付列をDatetime型に変換
        df['date'] = pd.to_datetime(df['date'])
        df2['date'] = pd.to_datetime(df2['date'])
        
        #
        # 3ヶ月単位での集計
        df = df.set_index('date')
        df_quarterly = df.resample('Q').count()['title'].loc['2021':]
        
        df2 = df2.set_index('date')
        df2_quarterly = df2.resample('Q').count()['title'].loc['2021':]
        
        df_trends_quarterly = df_trends[keyword].resample('Q').sum().loc['2021':]
        #

        st.subheader('他社メディア記事・セミナー数&Googleトレンド')
        
        # 折れ線グラフの描画
        plt.rcParams['font.size'] = 15 # 文字サイズを14に設定
        fig, ax1 = plt.subplots(figsize=(14,7))
        
        # Googleトレンドのデータを描画
        ax2 = ax1.twinx()
        ax2.plot(df_trends_quarterly.index, df_trends_quarterly, color='tab:blue', marker='o', label='Google Trends')
        
        # 集計した記事数を描画
        ax1.plot(df_quarterly.index, df_quarterly, color='tab:red', marker='o', label='Number of articles')
        ax1.plot(df2_quarterly.index, df2_quarterly, color='tab:green', marker='o', label='Number of seminars')
        
        ax1.set_xlabel('Quarter')
        ax1.set_ylabel('Number of articles and seminars', color='tab:red')
        ax2.set_ylabel('Google Trends', color='tab:blue')
        plt.title('Quarterly trends for keyword: {}'.format(keyword))
        ax1.legend(loc="upper left") # 凡例の追加
        st.pyplot(fig)


        

        #----------以下不要--------------

        # # プロット
        # fig, ax1 = plt.subplots()
        
        # color = 'tab:red'
        # ax1.set_xlabel('Time (Quarterly)')
        # ax1.set_ylabel('Article Counts', color=color)
        
        # # プロットのデータポイント数
        # n_plot_points = 10000
        # xnew = np.linspace(df_trends_quarterly.index.astype(int).min(), df_trends_quarterly.index.astype(int).max(), n_plot_points)
        
        # if not df_articles.empty:
        #     df_articles['date'] = pd.to_datetime(df_articles['date'])
        #     df_articles.set_index('date', inplace=True)
        #     df_articles_quarterly = df_articles.resample('3M').count()
        
        #     # 近似曲線の描画
        #     # スプライン補間関数の生成（日時をエポック秒に変換）と近似曲線の値を生成
        #     # spl_articles = make_interp_spline(df_articles_quarterly.index.astype(int), df_articles_quarterly)
        #     # ynew_articles = spl_articles(xnew)
        #     # 記事数の近似曲線の描画（エポック秒を日時に戻す）
        #     # ax1.plot(pd.to_datetime(xnew), ynew_articles, color=color)
        #     ax1.plot(df_articles_quarterly.index, df_articles_quarterly, color=color)
        #     # 目盛り幅を設定します。
        #     # ax1.yaxis.set_ticks(np.arange(0, df_articles_quarterly.max() + 1, 1))
        #     ax1.yaxis.set_ticks(np.arange(0, df_articles_quarterly.iloc[:, 0].max() + 1, 1))
        
        # ax1.tick_params(axis='y', labelcolor=color)
        
        # # x軸の日付のフォーマットをyyyyに変更
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # ax2 = ax1.twinx()
        # color = 'tab:blue'
        # ax2.set_ylabel('Google Trends', color=color)
    
        
        # ax2.plot(df_trends_quarterly.index, df_trends_quarterly, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        # fig.tight_layout()
        # st.pyplot(fig)
        # if not df_articles.empty:
        #     st.dataframe(df_articles.reset_index())
        # st.write(f"Related terms: {', '.join(related_terms)}")


        #-----------セミナー結果出力----------------------
        query = f"""
        SELECT *
        FROM `mythical-envoy-386309.majisemi.majisemi_seminar_api`
        WHERE Major_Category = "{keyword}"
           OR Category = "{keyword}"
           OR CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
           OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
           OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
        """
        
        
        # クエリの実行と結果の取得
        rows = run_query(query)
        df = pd.DataFrame(rows)
        
        
        # 四半期ごとにデータを集計
        df['Quarter'] = df['Seminar_Date'].dt.to_period('Q')
        df_grouped = df.groupby('Quarter').agg({
            'Acquisition_Speed': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
            'Seminar_Title': 'count'
        }).rename(columns={'<lambda_0>': '1Q', '<lambda_1>': '3Q', 'Seminar_Title': 'セミナー開催数'})

        st.subheader('マジセミ開催実績')
        # プロット作成
        plt.rcParams['font.size'] = 15 # 文字サイズを設定
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        color = 'tab:blue'
        ax1.set_xlabel('Quarter')
        ax1.set_ylabel('集客速度', color=color)
        ax1.plot(df_grouped.index.to_timestamp(), df_grouped[('Acquisition_Speed', 'median')], marker='o',color=color, label='Median Acquisition Speed')
        ax1.fill_between(df_grouped.index.to_timestamp(), df_grouped[('Acquisition_Speed', '1Q')], df_grouped[('Acquisition_Speed', '3Q')], color=color, alpha=0.1, label='Interquartile Range')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('マジセミ開催数', color=color)
        ax2.plot(df_grouped.index.to_timestamp(), df_grouped[('セミナー開催数', 'count')], marker='o',color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title(f'Average Acquisition Speed and Number of Seminars Containing "{keyword}"')
        plt.rcParams['font.size'] = 14 # 文字サイズを設定
        st.pyplot(fig)


        # 'parse_api_result', 'keyword_extraction_api_result', 'entity_extraction_api_result'を除く
        columns_to_exclude = ['parse_api_result', 'keyword_extraction_api_result', 'entity_extraction_api_result']
        df_filtered = df.drop(columns=columns_to_exclude)
        
        # カラム名を英語から日本語に変更
        df_filtered.rename(columns={
            'Seminar_Date': 'セミナー開催日',
            'Seminar_Title': 'セミナータイトル',
            'Organizer_Name': '主催企業名',
            'Major_Category': '大分類',
            'Category': 'カテゴリ',
            'Total_Participants': '合計集客人数',
            'Acquisition_Speed': '集客速度',
            'Action_Response_Count': 'アクション回答数',
            'Action_Response_Rate': 'アクション回答率（%）'
        }, inplace=True)

        # 'Seminar_Date'カラムの日付を "yyyy-mm-dd" 形式に変換
        df_filtered['セミナー開催日'] = pd.to_datetime(df_filtered['セミナー開催日']).dt.strftime('%Y-%m-%d')
        df_filtered['集客速度'] = df_filtered['集客速度'].round(2)
        df_filtered['アクション回答率（%）'] = df_filtered['アクション回答率（%）'].round(2)
        # 'Quarter' カラムを先頭に持ってくる
        cols = ['Quarter'] + [col for col in df_filtered.columns if col != 'Quarter']
        df_filtered = df_filtered[cols]
        
        # 表形式でStreamlitに出力
        st.dataframe(df_filtered)


        st.write("""
        ※集客速度は、1日あたりの平均申し込み数を表しています。
        """)




       



if __name__ == "__main__":
    main()
