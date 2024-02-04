import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
import matplotlib.pyplot as plt
import japanize_matplotlib
import unicodedata
import re
import requests 
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
from PIL import Image
from io import BytesIO
from collections import defaultdict
# from datetime import datetime, timedelta

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

# Streamlitのページ設定をワイドモードに設定
st.set_page_config(page_title="Keyword Analytics", layout="wide")

# 認証関数（APIを使用）
def authenticate(username, password):
    url = 'https://stg1.majisemi.com/e/api/check_user'
    data = {'name': username, 'pass': password}
    response = requests.post(url, data=data)
    response_json = response.json()
    # if response.status_code == 200:
    if response_json.get('status') == 'ok':
        return True
    else:
        return False

        
# メインページの関数
def main_page():
    
    # 直近3ヶ月のワードクラウドを生成する関数
    def generate_three_month_wordcloud():
        if 'wordcloud_image' in st.session_state:
            # 保存されたワードクラウド画像を表示
            st.image(st.session_state['wordcloud_image'], use_column_width=True)
        else:
            # 記事のクエリ
            articles_3m_query = f"""
            SELECT *
            FROM `mythical-envoy-386309.ex_media.article`
            WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH) AND CURRENT_DATE()
            """
            df_articles_3m = pd.DataFrame(run_query(articles_3m_query))
            # セミナーのクエリ
            seminars_3m_query = f"""
            SELECT *
            FROM `mythical-envoy-386309.ex_media.seminar`
            WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH) AND CURRENT_DATE()
            """
            df_seminars_3m = pd.DataFrame(run_query(seminars_3m_query))
            combined_titles = ' '.join(df_articles_3m['title']) + ' ' + ' '.join(df_seminars_3m['title'])
            # 形態素解析の実行
            t = Tokenizer()
            tokens = t.tokenize(combined_titles)
            words = [token.surface for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞']]  # 名詞と動詞のみを抽出
            
            # フィルタリング条件
            # 1文字の単語を除外
            words = [word for word in words if len(word) > 1]
            # ひらがな2文字の単語を除外
            words = [word for word in words if not re.match('^[ぁ-ん]{2}$', word)]
            words = [word for word in words if not re.match('^[一-龠々]{1}[ぁ-ん]{1}$', word)]
            # キーワードの除外
            exclude_words = {'する'}
            words = [word for word in words if word not in exclude_words]
            # フォントファイルのパス指定
            font_path = 'NotoSansJP-Regular.ttf'
            
            # ワードクラウドの生成
            wordcloud = WordCloud(
                font_path=font_path,
                background_color='white',
                width=1600,  # 幅を増やす
                height=800   # 高さを増やす
            ).generate(' '.join(words))
    
            # Pillow画像に変換
            image = wordcloud.to_image()
    
            # セッション状態に画像を保存
            with BytesIO() as output:
                image.save(output, format="PNG")
                data = output.getvalue()
            st.session_state['wordcloud_image'] = data
    
            # 画像を表示
            st.image(image, use_column_width=True)
        
            
    # 年別のワードクラウドを生成する関数
    def generate_yearly_wordcloud(year):
        # セッション状態のキーを年別にする
        session_key = f'wordcloud_image_{year}'
        
        if session_key in st.session_state:
            # 保存されたワードクラウド画像を表示
            st.image(st.session_state[session_key], use_column_width=True)
        else:
            articles_query = f"""
            SELECT title
            FROM `mythical-envoy-386309.ex_media.article`
            WHERE EXTRACT(YEAR FROM date) = {year}
            AND EXTRACT(MONTH FROM date) BETWEEN 1 AND 12
            """
            df_articles = pd.DataFrame(run_query(articles_query))
        
            seminars_query = f"""
            SELECT title
            FROM `mythical-envoy-386309.ex_media.seminar`
            WHERE EXTRACT(YEAR FROM date) = {year}
            AND EXTRACT(MONTH FROM date) BETWEEN 1 AND 12
            """
            df_seminars = pd.DataFrame(run_query(seminars_query))
            combined_titles = ' '.join(df_articles['title']) + ' ' + ' '.join(df_seminars['title'])
        
            t = Tokenizer()
            tokens = t.tokenize(combined_titles)
            words = [token.surface for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞']]
            words = [word for word in words if len(word) > 1]
            words = [word for word in words if not re.match('^[ぁ-ん]{2}$', word)]
            words = [word for word in words if not re.match('^[一-龠々]{1}[ぁ-ん]{1}$', word)]
            exclude_words = {'する'}
            words = [word for word in words if word not in exclude_words]
        
            wordcloud = WordCloud(
                font_path = 'NotoSansJP-Regular.ttf',
                background_color='white',
                width=1600,
                height=800
            ).generate(' '.join(words))
        
            image = wordcloud.to_image()

            # Pillow画像に変換し、セッション状態に画像を保存
            with BytesIO() as output:
                image.save(output, format="PNG")
                data = output.getvalue()
            st.session_state[session_key] = data
    
            # 画像を表示
            st.image(image, use_column_width=True)
        
            # with BytesIO() as output:
            #     image.save(output, format="PNG")
            #     data = output.getvalue()
        
            # st.image(image, use_column_width=True)
    
    def get_max_count(keywords, data_type):
        # keywords_conditions = [f"REGEXP_CONTAINS(title, r'(?i)(^|\\W){k}(\\W|$)')" for k in keywords.split(',')]
        # combined_keywords_condition = ' AND '.join(keywords_conditions)
        keywords = [unicodedata.normalize('NFKC', k.strip().lower()) for k in keywords.split(',')]
        conditions = [f"REGEXP_CONTAINS(title, r'(?i)(^|\\W){k}(\\W|$)')" for k in keywords]
        combined_condition = ' AND '.join(conditions)
    
        # マジセミセミナーの検索条件
        conditions_majisemi = [f"REGEXP_CONTAINS(Seminar_Title, r'(?i)(^|\\W){k}(\\W|$)')" for k in keywords]
        combined_condition_majisemi = ' AND '.join(conditions_majisemi)
    
        query = ""
        if data_type == "articles":
            query = f"""
            SELECT MAX(count) as max_count
            FROM (
                SELECT TIMESTAMP_TRUNC(date, QUARTER) as quarter, COUNT(*) as count
                FROM `mythical-envoy-386309.ex_media.article`
                WHERE {combined_condition}
                GROUP BY quarter
            )
            """
        elif data_type == "seminars":
            query = f"""
            SELECT MAX(count) as max_count
            FROM (
                SELECT TIMESTAMP_TRUNC(date, QUARTER) as quarter, COUNT(*) as count
                FROM `mythical-envoy-386309.ex_media.seminar`
                WHERE {combined_condition}
                GROUP BY quarter
            )
            """
        # 他のデータタイプについても同様にクエリを設定
        elif data_type == "majisemi_seminars":
            query = f"""
            SELECT MAX(count) as max_count
            FROM (
                SELECT TIMESTAMP_TRUNC(Seminar_Date, QUARTER) as quarter, COUNT(*) as count
                FROM `mythical-envoy-386309.majisemi.majisemi_seminar`
                WHERE {combined_condition_majisemi}
                GROUP BY quarter
            )
            """
        elif data_type == "acquisition_speed":
            query = f"""
            SELECT MAX(Acquisition_Speed) as max_count
            FROM `mythical-envoy-386309.majisemi.majisemi_seminar`
            WHERE {combined_condition_majisemi}
            """
        df = pd.DataFrame(run_query(query))
        if df.empty or 'max_count' not in df.columns:
            return 0
        return df['max_count'].max()*1.1
    
    
    def analyze_keyword(keywords,max_counts):
        keywords = [unicodedata.normalize('NFKC', k.strip().lower()) for k in keywords.split(',')]
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=3)
    
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
    
        # st.write(articles_query)
        # st.write(seminars_query)
    
        # プロットの描画
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()
    
        ax1.set_ylim(0, max_counts["articles"])
        ax2.set_ylim(0, max_counts["seminars"])
        
        if not df_articles.empty:
            # ax1.plot(df_articles_quarterly.index, df_articles_quarterly['count'], color='green', marker='x', label='Articles')
            x_articles = np.arange(len(df_articles_quarterly))
            y_articles = df_articles_quarterly['count'].values
    
            try:
                # 多項式回帰（例：2次）
                z_articles = np.polyfit(x_articles, y_articles, 2)
                p_articles = np.poly1d(z_articles)
    
                # 近似曲線のプロット
                ax1.plot(df_articles_quarterly.index, p_articles(x_articles), color='green', linestyle='--', label='Articles Approximation')
            except np.linalg.LinAlgError:
                st.write("※多項式回帰が収束しませんでした。Articlesの近似曲線は表示されません。")
    
            # 元のデータのプロット
            ax1.plot(df_articles_quarterly.index, y_articles, color='green', marker='x', label='Articles')       
            
        else:
            st.write("No article count data found.")
        
        if not df_seminars.empty:
            # ax2.plot(df_seminars_quarterly.index, df_seminars_quarterly['count'], color='red', marker='^', label='Seminars')
            x_seminars = np.arange(len(df_seminars_quarterly))
            y_seminars = df_seminars_quarterly['count'].values
    
            try:
                # 多項式回帰（例：2次）
                z_seminars = np.polyfit(x_seminars, y_seminars, 2)
                p_seminars = np.poly1d(z_seminars)
                
                # 近似曲線のプロット
                ax2.plot(df_seminars_quarterly.index, p_seminars(x_seminars), color='red', linestyle='--', label='Seminars Approximation')
            except np.linalg.LinAlgError:
                st.write("※多項式回帰が収束しませんでした。Seminarsの近似曲線は表示されません。")
    
            # 元のデータのプロット
            ax2.plot(df_seminars_quarterly.index, y_seminars, color='red', marker='^', label='Seminars')
            
        else:
            st.write("No seminar count data found.")
        
        # ax1.set_xlabel('Quarter')
        # ax1.set_ylabel('Counts of Articles', color='green')
        # ax1.tick_params(axis='y', labelcolor='green')
        # ax1.legend(loc='upper left')
        
        # ax2.set_ylabel('Counts of Seminars', color='red')
        # ax2.tick_params(axis='y', labelcolor='red')
        # ax2.legend(loc='upper right')
    
        # # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        # ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
        # plt.title(f'Number of Articles and Seminars for "{", ".join(keywords)}"')
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
        
        # ワードクラウド作成
        if not df_articles_full.empty or not df_seminars_full.empty:
    
            if df_seminars_full.empty:
                combined_titles = ' '.join(df_articles_full['title']) + ' '
    
            elif df_articles_full.empty:
                combined_titles = ' '.join(df_seminars_full['title']) + ' '
    
            elif not df_articles_full.empty and not df_seminars_full.empty:
                # 記事とセミナーのタイトルを結合
                combined_titles = ' '.join(df_articles_full['title']) + ' ' + ' '.join(df_seminars_full['title'])
        
            # 形態素解析の実行
            t = Tokenizer()
            tokens = t.tokenize(combined_titles)
            words = [token.surface for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞']]  # 名詞と動詞のみを抽出
    
            # フィルタリング条件
            # 1文字の単語を除外
            words = [word for word in words if len(word) > 1]
            # ひらがな2文字の単語を除外
            words = [word for word in words if not re.match('^[ぁ-ん]{2}$', word)]
            words = [word for word in words if not re.match('^[一-龠々]{1}[ぁ-ん]{1}$', word)]
        
            # キーワードの除外
            exclude_words = set(keywords)
            words = [word for word in words if word not in exclude_words]
        
            # フォントファイルのパス指定
            font_path = 'NotoSansJP-Regular.ttf'
        
            # ワードクラウドの生成（記事とセミナーのタイトル用）
            # wordcloud_combined = WordCloud(font_path=font_path,width=800, height=800, background_color='white', min_font_size=10).generate(combined_titles)
        
            # ワードクラウドの生成
            wordcloud = WordCloud(
                font_path=font_path,
                background_color='white',
                width=1600,  # 幅を増やす
                height=800   # 高さを増やす
            ).generate(' '.join(words))
            
            st.subheader('ワードクラウド：外部メディア')
            # ワードクラウドの表示
            plt.figure(figsize=(10, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.show()
            st.pyplot(plt)
    
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
    
            ax1.set_ylim(0, max_counts["acquisition_speed"])
            ax2.set_ylim(0, max_counts["majisemi_seminars"])
    
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
    
            # マジセミのセミナータイトルに対するワードクラウド生成
            seminar_titles = ' '.join(df_seminar['セミナータイトル'])  # セミナータイトルの結合
            
            # 形態素解析の実行
            t = Tokenizer()
            tokens = t.tokenize(seminar_titles)
            words_seminar = [token.surface for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞']]  # 名詞と動詞のみを抽出
    
            # フィルタリング条件
            # 1文字の単語を除外
            words_seminar = [word for word in words_seminar if len(word) > 1]
            # ひらがな2文字の単語を除外
            words_seminar = [word for word in words_seminar if not re.match('^[ぁ-ん]{2}$', word)]
            # 漢字1文字とひらがな1文字から成る2文字の単語を除外
            words_seminar = [word for word in words_seminar if not re.match('^[一-龠々]{1}[ぁ-ん]{1}$', word)]
            
            # キーワードの除外
            words_seminar = [word for word in words_seminar if word not in exclude_words]
            
            # ワードクラウドの生成（セミナータイトル用）
            wordcloud_seminar = WordCloud(
                font_path=font_path,
                background_color='white',
                width=1600,  # 幅
                height=800   # 高さ
            ).generate(' '.join(words_seminar))
            
            # ワードクラウドの表示（セミナータイトル用）
            st.subheader('ワードクラウド：マジセミ')
            plt.figure(figsize=(10, 10))
            plt.imshow(wordcloud_seminar, interpolation='bilinear')
            plt.axis('off')
            plt.show()
            st.pyplot(plt)
    
            # ワードクラウド：マジセミ（集客重み付け版）
            st.subheader('ワードクラウド：マジセミ（集客重み付け版）')
            # 単語の重み付け頻度を格納する辞書
            weighted_word_freqs = defaultdict(int)
            
            # 各セミナーのタイトルと集客速度をループ処理
            for title, speed in zip(df_seminar['セミナータイトル'], df_seminar['集客速度']):
                # 形態素解析
                tokens = t.tokenize(title)
                words = [token.surface for token in tokens if token.part_of_speech.split(',')[0] in ['名詞', '動詞']]
                
                # フィルタリング条件
                words = [word for word in words if len(word) > 1 and not re.match('^[ぁ-ん]{2}$', word) and not re.match('^[一-龠々]{1}[ぁ-ん]{1}$', word)]
                words = [word for word in words if word not in exclude_words]
            
                # 単語に対して重み（集客速度）を掛け、辞書に加算
                for word in words:
                    weighted_word_freqs[word] += speed
            
            # ワードクラウドの生成
            wordcloud_seminar = WordCloud(
                font_path=font_path,
                background_color='white',
                width=1600,
                height=800
            ).generate_from_frequencies(weighted_word_freqs)
    
            
            plt.figure(figsize=(10, 10))
            plt.imshow(wordcloud_seminar, interpolation='bilinear')
            plt.axis('off')
            plt.show()
            st.pyplot(plt)
    
        else:
            st.write("No matched seminars found.")
    
        return f"Analysis results for {', '.join(keywords)}"
    
    
    def show_analytics():
        st.title("Keyword Analytics")
    
        # キーワード入力ボックスを配置
        col_input1, col_input2 = st.columns([2, 2])
        with col_input1:
            st.subheader("＜Latest Updates＞")
            st.markdown("""
            - 2024/01/16(火) 認証機能を実装
            - 2024/01/09(火) 2023年12月分の外部メディアのデータを追加
            - 2023/12/17(日) 集客重み付け版ワードクラウドを実装
            """)
            st.markdown("---")
            keyword_input1 = st.text_input("キーワード1を入力【カンマ区切りでand検索可能（例：AI, ChatGPT）】")
        # with col_input2:
            keyword_input2 = st.text_input("キーワード2を入力【カンマ区切りでand検索可能（例：AI, ChatGPT）】")
        with col_input2:
            st.subheader('ワードクラウドによるトレンド分析')
            # ボタンが押された場合、セッション状態を更新
            if st.button('直近3ヶ月のワードクラウドを表示'):
                # st.session_state['show_three_month_wordcloud'] = True
                generate_three_month_wordcloud()
            # 年別ワードクラウドの表示
            if st.button('年別ワードクラウドを表示'):
                for year in [2023, 2022, 2021]:  # 対象年を変更する場合は、このリストを更新
                    st.subheader(f'{year}年のワードクラウド')
                    generate_yearly_wordcloud(year)
        
            # # セッション状態に基づいてワードクラウドを表示
            # if st.session_state.get('show_three_month_wordcloud'):
            #     generate_three_month_wordcloud()
        
        execute_button = st.button("キーワード分析を実行")
    
        if execute_button:
            # 各指標の最大値を格納する辞書
            max_counts = {"articles": 0, "seminars": 0, "majisemi_seminars": 0, "acquisition_speed": 0}
    
            if keyword_input1:
                keyword1 = unicodedata.normalize('NFKC', keyword_input1.strip().lower())
                for data_type in max_counts.keys():
                    max_count = get_max_count(keyword1, data_type)
                    max_counts[data_type] = max(max_counts[data_type], max_count)
    
            if keyword_input2:
                keyword2 = unicodedata.normalize('NFKC', keyword_input2.strip().lower())
                for data_type in max_counts.keys():
                    max_count = get_max_count(keyword2, data_type)
                    max_counts[data_type] = max(max_counts[data_type], max_count)
    
            # 画面を2つの列に分ける
            col1, col2 = st.columns([1,1])
    
            # キーワード1の結果を左の列に表示
            if keyword_input1:
                with col1:
                    st.write(f"## キーワード1: {keyword1} の結果")
                    analyze_keyword(keyword1, max_counts)
    
            # キーワード2の結果を右の列に表示
            if keyword_input2:
                with col2:
                    st.write(f"## キーワード2: {keyword2} の結果")
                    analyze_keyword(keyword2, max_counts)
    
    show_analytics()
    



# ログインページの関数
def login_page():
    
    
    # ログイン状態を保持するための非表示のチェックボックス
    if "login_checked" not in st.session_state:
        st.session_state.login_checked = False

    # 認証が成功していない場合のみ、ユーザー名とパスワードの入力欄を表示
    if not st.session_state.login_checked:
        # 空のカラム（左）、ログインフォームのカラム（中央）、空のカラム（右）を作成
        col1, col2, col3 = st.columns([1,2,1])

        with col2:
            title_placeholder = st.empty()
            title_placeholder.title("Keyword Analytics")
            username_placeholder = st.empty()
            password_placeholder = st.empty()
            username = username_placeholder.text_input("ユーザー名")
            password = password_placeholder.text_input("パスワード", type="password")
            login_message_placeholder = st.empty()
            login_message_placeholder.write("※マジカンのアカウントでログインできます")

            login_button_placeholder = st.empty()
            if login_button_placeholder.button("ログイン"):
                if authenticate(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state.login_checked = True  # 認証成功時にチェック
                    title_placeholder.empty()  # タイトルをクリア
                    username_placeholder.empty()  # ユーザー名入力欄をクリア
                    password_placeholder.empty()  # パスワード入力欄をクリア
                    login_button_placeholder.empty()  # ログインボタンをクリア
                    login_message_placeholder.empty()  # ログインメッセージをクリア
                else:
                    st.error("認証に失敗しました。")

    # 認証後にセッション状態が更新されたことを確認し、メインページに遷移する
    if st.session_state.login_checked:
        main_page()

def main():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if st.session_state['authenticated']:
        main_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
