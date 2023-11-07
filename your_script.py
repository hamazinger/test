import streamlit as st
import os
import time
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestRegressor

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
from matplotlib.ticker import MaxNLocator
import japanize_matplotlib
import unicodedata


# 認証情報の設定
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

@st.cache(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    rows = [dict(row) for row in rows_raw]
    return rows

# def main():
#     st.title("キーワード分析")

#     # 2つのキーワード入力ボックスを配置
#     keyword_input1 = st.text_input("キーワード1を入力")
#     keyword_input2 = st.text_input("キーワード2を入力")
#     execute_button = st.button("分析を実行")

#     if execute_button:
#         # それぞれのキーワードの取得と正規化
#         keyword1 = unicodedata.normalize('NFKC', keyword_input1.strip().lower())
#         keyword2 = unicodedata.normalize('NFKC', keyword_input2.strip().lower())

#         # キーワード1に基づくデータ取得および分析
#         st.write("## キーワード1の結果")
#         result1 = analyze_keyword(keyword1)
#         # st.write(result1)
#         # st.write("### キーワード1に関連するキーワードの集客速度への影響")
#         # input_keyword_importance1, related_keywords_result1 = get_related_keywords_for_input(keyword1)
#         # st.write(related_keywords_result1)

#         # キーワード2に基づくデータ取得および分析
#         st.write("## キーワード2の結果")
#         result2 = analyze_keyword(keyword2)
#         # st.write("### キーワード2に関連するキーワードの集客速度への影響")
#         # input_keyword_importance2, related_keywords_result2 = get_related_keywords_for_input(keyword2)
#         # st.write(related_keywords_result2)
#         # # st.write(result2)


#---------------------------------

def main():
    st.title("キーワード分析")

    # キーワード入力ボックスの配置と注意書きの追加
    st.write("※複数キーワードを検索する場合はカンマ区切りで入力してください。")
    keyword_input1 = st.text_input("キーワード1を入力（例: AI,機械学習）")
    keyword_input2 = st.text_input("キーワード2を入力（例: データ分析,ビッグデータ）")
    execute_button = st.button("分析を実行")

    if execute_button:
        # キーワードの取得と正規化、さらにカンマで分割してリスト化
        keywords1 = [unicodedata.normalize('NFKC', k.strip().lower()) for k in keyword_input1.split(',')]
        keywords2 = [unicodedata.normalize('NFKC', k.strip().lower()) for k in keyword_input2.split(',')]
        
        # キーワード1とキーワード2のリストを結合
        combined_keywords = list(set(keywords1 + keywords2))  # 重複を避けるためsetに変換してからリストに戻す

        # キーワードに基づくデータ取得および分析
        st.write("## キーワードの結果")
        result = analyze_keyword(combined_keywords)
        st.write(result)



def analyze_keyword(keyword):
    keywords = [keyword]  # 単一のキーワードをリストに変換

    # 現在の日付を取得し、過去2年分の範囲を計算します。
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=2)
    current_date = pd.Timestamp.now().normalize()

    #------------Googleトレンドのデータ取得---------------------
    pytrend = TrendReq(hl='ja', tz=540)
    pytrend.build_payload(kw_list=keywords)
    time.sleep(10) 
    df_trends = pytrend.interest_over_time()
    df_trends_combined = df_trends[keywords].sum(axis=1)
    df_trends_quarterly = df_trends_combined.resample('Q').sum().loc[start_date:end_date]
    df_trends_quarterly = df_trends_quarterly.loc[:current_date]

    # 以下はキーワードに基づいたデータ取得および分析のロジック
    # ここで各種データベースへのクエリを実行し、結果をグラフや表としてStreamlitに表示します。
    # ...
    queries_for_article = []
    for k in keywords:
        condition_for_k = f"""
        (
            tag = "{k}"
            OR CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
            OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
            OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
        )
        """
        queries_for_article.append(condition_for_k)
    
    combined_query_for_article = " AND ".join(queries_for_article)


    # query = f"""
    # SELECT *
    # FROM `mythical-envoy-386309.majisemi.business_it_article_api`
    # WHERE tag = "{keyword}"
    #    OR CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
    #    OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
    #    OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
    # """

    query = f"""
    SELECT *
    FROM `mythical-envoy-386309.majisemi.business_it_article_api`
    WHERE {combined_query_for_article}
    """
    
    # query2 = f"""
    # SELECT *
    # FROM `mythical-envoy-386309.majisemi.bussiness_it_seminar_api`
    # WHERE CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
    #    OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
    #    OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
    # """

    # queries_for_seminar = []
    # for k in keywords:
    #     queries_for_seminar.append(f"""
    #     CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
    #     OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
    #     OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
    #     """)

    # combined_query = " AND ".join(queries_for_seminar)

    queries_for_seminar = []
    for k in keywords:
        condition_for_k = f"""
        (
            CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
            OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
            OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
        )
        """
        queries_for_seminar.append(condition_for_k)
    
    combined_query_for_seminar = " AND ".join(queries_for_seminar)

    query2 = f"""
    SELECT *
    FROM `mythical-envoy-386309.majisemi.bussiness_it_seminar_api`
    WHERE {combined_query_for_seminar}
    """
    
    rows = run_query(query)
    rows2 = run_query(query2)

    # この部分を追加
    if not rows:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(rows)
    
    if not rows2:
        df2 = pd.DataFrame()
    else:
        df2 = pd.DataFrame(rows2)
    
    # df = pd.DataFrame(rows)
    # df2 = pd.DataFrame(rows2)

    if not df.empty:
        # 日付列をDatetime型に変換
        df['date'] = pd.to_datetime(df['date'])
        # 3ヶ月単位での集計
        df = df.set_index('date')
        # df_quarterly = df.resample('Q').count()['title'].loc['2021':]
        df_quarterly = df.resample('Q').count()['title'].loc[start_date:end_date]
        df_quarterly = df_quarterly.loc[:current_date]
    

    if not df2.empty:
        df2['date'] = pd.to_datetime(df2['date'])
        df2 = df2.set_index('date')
        # df2_quarterly = df2.resample('Q').count()['title'].loc['2021':]
        df2_quarterly = df2.resample('Q').count()['title'].loc[start_date:end_date]
        df2_quarterly = df2_quarterly.loc[:current_date]
    
    # df_trends_quarterly = df_trends[keyword].resample('Q').sum().loc['2021':]
    #

    st.subheader('他社メディア記事・セミナー数&Googleトレンド')
    
    # 折れ線グラフの描画
    plt.rcParams['font.size'] = 15 # 文字サイズを14に設定
    fig, ax1 = plt.subplots(figsize=(14,7))
    
    # Googleトレンドのデータを描画
    ax2 = ax1.twinx()
    ax2.plot(df_trends_quarterly.index, df_trends_quarterly, color='tab:blue', marker='o', label='Google Trends')
    
    # 集計した記事数を描画
    if not df.empty:
        ax1.plot(df_quarterly.index, df_quarterly, color='tab:red', marker='o', label='Number of articles')
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    if not df2.empty:
        ax1.plot(df2_quarterly.index, df2_quarterly, color='tab:green', marker='o', label='Number of seminars')
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Number of articles and seminars', color='tab:red')
    ax2.set_ylabel('Google Trends', color='tab:blue')
    #plt.title('Quarterly trends for keyword: {}'.format(keyword))
    plt.title('Quarterly trends for keyword: {}'.format(", ".join(keywords)))
    ax1.legend(loc="upper left") # 凡例の追加
    st.pyplot(fig)


    


    #-----------セミナー結果出力----------------------
    # query = f"""
    # SELECT *
    # FROM `mythical-envoy-386309.majisemi.majisemi_seminar_api`
    # WHERE Major_Category = "{keyword}"
    #    OR Category = "{keyword}"
    #    OR CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
    #    OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
    #    OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{keyword}", ',%')
    #    AND Seminar_Date BETWEEN '{start_date}' AND '{end_date}'
    # """

    queries_for_seminar = []
    for k in keywords:
        condition_for_k = f"""
        (
            Major_Category = "{k}"
            OR Category = "{k}"
            OR CONCAT(',', parse_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
            OR CONCAT(',', keyword_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
            OR CONCAT(',', entity_extraction_api_result, ',') LIKE CONCAT('%,', "{k}", ',%')
        )
        """
        queries_for_seminar.append(condition_for_k)
    
    combined_query_for_seminar = " AND ".join(queries_for_seminar)

    
    query = f"""
    SELECT *
    FROM `mythical-envoy-386309.majisemi.majisemi_seminar_api`
    WHERE {combined_query_for_seminar}
       AND Seminar_Date BETWEEN '{start_date}' AND '{end_date}'
    """
    
    
    # クエリの実行と結果の取得
    rows = run_query(query)
    if not rows:
        df = pd.DataFrame()
        st.warning("セミナーのデータが見つかりませんでした。")
    else:
        df = pd.DataFrame(rows)
    
    
        # 四半期ごとにデータを集計
        # df['Quarter'] = df['Seminar_Date'].dt.to_period('Q')
        # df_grouped = df.groupby('Quarter').agg({
        #     'Acquisition_Speed': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        #     'Seminar_Title': 'count'
        # }).rename(columns={'<lambda_0>': '1Q', '<lambda_1>': '3Q', 'Seminar_Title': 'セミナー開催数'})

        # 四半期ごとにデータを集計
        df['Quarter'] = df['Seminar_Date'].dt.to_period('Q')
        df_grouped = df.groupby('Quarter').agg({
            'Acquisition_Speed': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
            'Seminar_Title': 'count'
        }).rename(columns={'<lambda_0>': '1Q', '<lambda_1>': '3Q', 'Seminar_Title': 'セミナー開催数'}).loc[start_date.to_period('Q'):end_date.to_period('Q')]

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
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        # plt.title(f'Average Acquisition Speed and Number of Seminars Containing "{keyword}"')
        plt.title(f'Average Acquisition Speed and Number of Seminars Containing: {", ".join(keywords)}')
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

        # セミナー開催日列の降順でソート
        df_filtered = df_filtered.sort_values(by='セミナー開催日', ascending=False)

        
        # 表形式でStreamlitに出力
        st.dataframe(df_filtered)


        st.write("""
        ※集客速度は、1日あたりの平均申し込み数を表しています。
        """)


    return f"{keyword}の分析結果"



# def get_related_keywords_for_input(input_keyword, num_related_keywords=5):
#     data_cleaned = data.dropna(subset=['Acquisition_Speed'])
#     subset_data = data_cleaned[data_cleaned['keyword_extraction_api_result'].str.contains(input_keyword, case=False, na=False)]
#     custom_stopwords = ["セミナー", "web", "開催", "日時", "参加", "申し込み", "無料", "オンライン"]
#     tfidf_vectorizer_subset = TfidfVectorizer(max_df=1.0, min_df=0.005, ngram_range=(1, 2), stop_words=custom_stopwords)
#     tfidf_matrix_subset = tfidf_vectorizer_subset.fit_transform(subset_data['keyword_extraction_api_result'].fillna(""))
#     X_subset = pd.DataFrame(tfidf_matrix_subset.toarray(), columns=tfidf_vectorizer_subset.get_feature_names_out())
#     y_speed_subset = subset_data['Acquisition_Speed'].values
#     rf_speed_subset = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_speed_subset.fit(X_subset, y_speed_subset)
#     feature_importances_speed_subset = rf_speed_subset.feature_importances_
#     keywords_importance_speed_subset_df = pd.DataFrame({
#         'Keyword': X_subset.columns,
#         'Importance': feature_importances_speed_subset
#     }).sort_values(by='Importance', ascending=False)
#     input_keyword_importance = keywords_importance_speed_subset_df[keywords_importance_speed_subset_df['Keyword'] == input_keyword]
#     related_keywords_subset = keywords_importance_speed_subset_df[keywords_importance_speed_subset_df['Keyword'] != input_keyword].head(num_related_keywords)
#     return input_keyword_importance, related_keywords_subset

    
if __name__ == "__main__":
    main()
