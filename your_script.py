import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from pytrends.request import TrendReq
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

    pytrend = TrendReq(hl='ja', tz=540)
    pytrend.build_payload(kw_list=keywords, timeframe=f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}')
    df_trends = pytrend.interest_over_time()
    df_trends_combined = df_trends[keywords].sum(axis=1)
    df_trends_quarterly = df_trends_combined.resample('Q').sum().loc[start_date:end_date]

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
    df_articles_quarterly = df_articles.set_index('quarter').resample('Q').sum().loc[start_date:end_date]

    seminars_query = f"""
    SELECT TIMESTAMP_TRUNC(date, QUARTER) as quarter, COUNT(*) as count
    FROM `mythical-envoy-386309.ex_media.seminar`
    WHERE {combined_condition}
    GROUP BY quarter
    ORDER BY quarter
    """
    df_seminars = pd.DataFrame(run_query(seminars_query))
    df_seminars_quarterly = df_seminars.set_index('quarter').resample('Q').sum().loc[start_date:end_date]

    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(df_articles_quarterly.index, df_articles_quarterly['count'], color='green', marker='x', label='Articles')
    ax1.plot(df_seminars_quarterly.index, df_seminars_quarterly['count'], color='red', marker='^', label='Seminars')
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('Counts of Articles and Seminars', color='black')
    ax1.legend(loc='upper left')

    ax2.plot(df_trends_quarterly.index, df_trends_quarterly, color='blue', marker='o', label='Google Trends')
    ax2.set_ylabel('Google Trends Score', color='blue')
    ax2.legend(loc='upper right')

    plt.title(f'Google Trends and Number of Articles/Seminars for "{", ".join(keywords)}"')
    st.pyplot(plt)

    # マジセミセミナー情報の取得
    conditions_majisemi = [f"REGEXP_CONTAINS(Seminar_Title, r'(?i)(^|\\W){k}(\\W|$)')" for k in keywords]
    combined_condition_majisemi = ' AND '.join(conditions_majisemi)
    seminar_query = f"""
    SELECT *
    FROM `mythical-envoy-386309.majisemi.majisemi_seminar`
    WHERE {combined_condition_majisemi}
       AND Seminar_Date BETWEEN '{start_date}' AND '{end_date}'
    """
    df_seminar = pd.DataFrame(run_query(seminar_query))

    # セミナー情報の表示
    if not df_seminar.empty:
        st.subheader('Matched Majisemi Seminars')
        st.dataframe(df_seminar)
    else:
        st.write("No matched seminars found.")

    return f"Analysis results for {', '.join(keywords)}"

def main():
    st.title("キーワード分析")

    keyword_input1 = st.text_input("キーワード1を入力")
    keyword_input2 = st.text_input("キーワード2を入力")
    execute_button = st.button("分析を実行")

    if execute_button:
        keyword1 = unicodedata.normalize('NFKC', keyword_input1.strip().lower())
        keyword2 = unicodedata.normalize('NFKC', keyword_input2.strip().lower())

        st.write("## キーワード1の結果")
        result1 = analyze_keyword(keyword1)
        st.write(result1)

        st.write("## キーワード2の結果")
        result2 = analyze_keyword(keyword2)
        st.write(result2)

if __name__ == "__main__":
    main()
