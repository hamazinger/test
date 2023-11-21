import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
# from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import japanize_matplotlib
import unicodedata
from matplotlib.ticker import MaxNLocator

# 認証情報の設定
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

def search_titles_with_keyword(keyword, table_name):
    query = f"""
    SELECT *
    FROM `{table_name}`
    WHERE REGEXP_CONTAINS(title, r'(?i)(^|\\W){keyword}(\\W|$)')
    """
    results = run_query(query)
    return results

@st.cache(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    rows = [dict(row) for row in rows_raw]
    return rows

def analyze_keyword(keywords):
    keywords = [unicodedata.normalize('NFKC', k.strip().lower()) for k in keywords.split(',')]
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=2)

    conditions = [f"REGEXP_CONTAINS(title, r'(?i)(^|\\W){k}(\\W|$)')" for k in keywords]
    combined_condition = ' AND '.join(conditions)

    articles_query = f"""
    SELECT TIMESTAMP_TRUNC(date, QUARTER) as quarter, COUNT(*) as count
    FROM `mythical-envoy-386309.ex_media.article`
    WHERE {combined_condition}
    GROUP BY quarter
    ORDER BY quarter
    """
    df_articles = pd.DataFrame(run_query(articles_query))
    # df_articles_quarterly = df_articles.set_index('quarter').resample('Q').sum().loc[start_date:end_date]

    # エラーハンドリングを追加
    if not df_articles.empty and 'quarter' in df_articles.columns:
        df_articles_quarterly = df_articles.set_index('quarter').resample('Q').sum().loc[start_date:end_date]
    else:
        df_articles_quarterly = pd.DataFrame()  # 空のデータフレームを作成

    seminars_query = f"""
    SELECT TIMESTAMP_TRUNC(date, QUARTER) as quarter, COUNT(*) as count
    FROM `mythical-envoy-386309.ex_media.seminar`
    WHERE {combined_condition}
    GROUP BY quarter
    ORDER BY quarter
    """
    df_seminars = pd.DataFrame(run_query(seminars_query))
    # df_seminars_quarterly = df_seminars.set_index('quarter').resample('Q').sum().loc[start_date:end_date]

    # エラーハンドリングを追加
    if not df_seminars.empty and 'quarter' in df_seminars.columns:
        df_seminars_quarterly = df_seminars.set_index('quarter').resample('Q').sum().loc[start_date:end_date]
    else:
        df_seminars_quarterly = pd.DataFrame()  # 空のデータフレームを作成



    # プロットの描画
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    if not df_articles.empty:
        ax1.plot(df_articles_quarterly.index, df_articles_quarterly['count'], color='green', marker='x', label='Articles')
    else:
        st.write("No article count data found.")
    
    if not df_seminars.empty:
        ax2.plot(df_seminars_quarterly.index, df_seminars_quarterly['count'], color='red', marker='^', label='Seminars')
    else:
        st.write("No seminar count data found.")
    
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Counts of Articles', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.legend(loc='upper left')
    
    ax2.set_ylabel('Counts of Seminars', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(f'Number of Articles and Seminars for "{", ".join(keywords)}"')
    st.pyplot(plt)

    # 検索条件に一致した記事・セミナーの一覧
    articles_full_query = f"""
    SELECT *
    FROM `mythical-envoy-386309.ex_media.article`
    WHERE {combined_condition}
    """
    df_articles_full = pd.DataFrame(run_query(articles_full_query))

    seminars_full_query = f"""
    SELECT *
    FROM `mythical-envoy-386309.ex_media.seminar`
    WHERE {combined_condition}
    """
    df_seminars_full = pd.DataFrame(run_query(seminars_full_query))

    if not df_articles_full.empty:
        st.subheader('検索条件に一致した記事')
        st.dataframe(df_articles_full.sort_values(by='date', ascending=False))
    else:
        st.write("No matched articles found.")
    
    if not df_seminars_full.empty:
        st.subheader('検索条件に一致したセミナー')
        st.dataframe(df_seminars_full.sort_values(by='date', ascending=False))
    else:
        st.write("No matched seminars found.")

    # マジセミセミナーの検索条件
    conditions_majisemi = [f"REGEXP_CONTAINS(Seminar_Title, r'(?i)(^|\\W){k}(\\W|$)')" for k in keywords]
    combined_condition_majisemi = ' AND '.join(conditions_majisemi)

    seminar_query = f"""
    SELECT *
    FROM `mythical-envoy-386309.majisemi.majisemi_seminar`
    WHERE {combined_condition_majisemi}
       AND Seminar_Date BETWEEN '{start_date}' AND '{end_date}'
    """
    df_seminar = pd.DataFrame(run_query(seminar_query))

    st.subheader('マジセミ開催実績')

    if not df_seminar.empty:
        df_seminar['Quarter'] = pd.to_datetime(df_seminar['Seminar_Date']).dt.to_period('Q')
        df_grouped = df_seminar.groupby('Quarter').agg({
            'Acquisition_Speed': ['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
            'Seminar_Title': 'count'
        })

        df_grouped.columns = ['Median_Speed', 'Q1_Speed', 'Q3_Speed', 'Num_Seminars']

        plt.figure(figsize=(14, 7))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.plot(df_grouped.index.to_timestamp(), df_grouped['Median_Speed'], marker='o', color='blue', label='Median Acquisition Speed')
        ax1.fill_between(df_grouped.index.to_timestamp(), df_grouped['Q1_Speed'], df_grouped['Q3_Speed'], color='blue', alpha=0.1, label='Interquartile Range')
        ax1.set_xlabel('Quarter')
        ax1.set_ylabel('Acquisition Speed', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc='upper left')

        ax2.plot(df_grouped.index.to_timestamp(), df_grouped['Num_Seminars'], marker='o', color='red', label='Number of Seminars')
        ax2.set_ylabel('Number of Seminars', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')

        st.pyplot(plt)

        st.subheader('検索条件に一致したマジセミのセミナー一覧')
        df_seminar.rename(columns={
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
        # 'セミナー開催日'列を日付型に変換
        df_seminar['セミナー開催日'] = pd.to_datetime(df_seminar['セミナー開催日'])
        df_seminar['セミナー開催日'] = df_seminar['セミナー開催日'].dt.strftime('%Y-%m-%d')
        #st.dataframe(df_seminar)
        st.dataframe(df_seminar.sort_values(by='セミナー開催日', ascending=False))
        st.write("""
        ※集客速度は、1日あたりの平均申し込み数を表しています。
        """)
    else:
        st.write("No matched seminars found.")

    return f"Analysis results for {', '.join(keywords)}"



def main():
    st.title("キーワード分析")

    keyword_input1 = st.text_input("キーワード1を入力【カンマ区切りでand検索可能（例：AI, ChatGPT）】")
    keyword_input2 = st.text_input("キーワード2を入力【カンマ区切りでand検索可能（例：AI, ChatGPT）】")
    execute_button = st.button("分析を実行")

    # if execute_button:
    #     if keyword_input1:
    #         keyword1 = unicodedata.normalize('NFKC', keyword_input1.strip().lower())
    #         st.write("## キーワード1の結果")
    #         result1 = analyze_keyword(keyword1)
    #         st.write(result1)
        
    #     if keyword_input2:
    #         keyword2 = unicodedata.normalize('NFKC', keyword_input2.strip().lower())
    #         st.write("## キーワード2の結果")
    #         result2 = analyze_keyword(keyword2)
    #         st.write(result2)

    # Streamlitのページ設定
    st.set_page_config(layout="wide")

    if execute_button:
        # 画面を2つの列に分ける
        col1, col2 = st.columns(2)

        # キーワード1の分析結果を左の列に表示
        if keyword_input1:
            keyword1 = unicodedata.normalize('NFKC', keyword_input1.strip().lower())
            with col1:
                st.write(f"## キーワード1: {keyword1} の結果")
                result1 = analyze_keyword(keyword1)
                st.write(result1)

        # キーワード2の分析結果を右の列に表示
        if keyword_input2:
            keyword2 = unicodedata.normalize('NFKC', keyword_input2.strip().lower())
            with col2:
                st.write(f"## キーワード2: {keyword2} の結果")
                result2 = analyze_keyword(keyword2)
                st.write(result2)

if __name__ == "__main__":
    main()
