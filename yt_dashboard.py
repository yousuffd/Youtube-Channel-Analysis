# import required libraries
from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import isodate
import warnings
import re
import streamlit as st

warnings.filterwarnings('ignore')

# ----------------------------------------------------
# 1. CONNECT TO YOUTUBE API
# ----------------------------------------------------

api_key = st.secrets["API_KEY"]

channel_ids=['UCmTM_hPCeckqN3cPWtYZZcg',
             'UCzI8K9xO_5E-4iCP7Km6cRQ',
             'UC5fcjujOsqD-126Chn_BAuA',
             'UC-CSyyi47VX1lD9zyeABW3w',
             'UC0yXUUIaPVAqZLgRjvtMftw',
             'UCWtlPzcP989da26sVyHPzqQ']

youtube = build('youtube', 'v3', developerKey=api_key)

# ----------------------------------------------------
# 2. FUNCTIONS TO FETCH DATA
# ----------------------------------------------------
@st.cache_data(show_spinner="Fetching channel stats...")
def get_channel_stats_cached(channel_ids, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    all_channels = []
    response = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=",".join(channel_ids)
    ).execute()
    for item in response['items']:
        all_channels.append({
            "channel_name": item["snippet"]["title"],
            "channel_id": item["id"],
            "ch_subscribers": item["statistics"]["subscriberCount"],
            "views": item["statistics"]["viewCount"],
            "total_videos": item["statistics"]["videoCount"],
            "playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
        })
    return pd.DataFrame(all_channels)


@st.cache_data(show_spinner="Fetching video IDs from playlists...")
def get_videos_from_playlists_cached(playlist_ids, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_ids = []
    for playlist_id in playlist_ids:
        next_page_token = None
        while True:
            response = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            for item in response.get("items", []):
                video_ids.append(item["contentDetails"]["videoId"])
            next_page_token = response.get("nextPageToken")
            if next_page_token is None:
                break
    return video_ids


@st.cache_data(show_spinner="Fetching video details...")
def get_videos_data_cached(video_ids, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_details = []
    for i in range(0, len(video_ids), 50):
        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i + 50])
        ).execute()
        for video in response['items']:
            stats = video.get("statistics", {})
            video_details.append({
                "video_id": video["id"],
                "title": video["snippet"]["title"],
                "channel_id": video["snippet"]["channelId"],
                "publish_date": video["snippet"]["publishedAt"],
                "duration": video["contentDetails"]["duration"],
                "views": stats.get("viewCount"),
                "likes": stats.get("likeCount")
            })
    return pd.DataFrame(video_details)


# Helper functions
def iso_duration_to_minutes(duration):
    if not isinstance(duration, str):
        return np.nan
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return np.nan
    h = int(match.group(1) or 0)
    m = int(match.group(2) or 0)
    s = int(match.group(3) or 0)
    return h * 60 + m + s / 60


def minutes_to_mmss(value):
    if pd.isna(value): return None
    minutes = int(value)
    seconds = round((value - minutes) * 60)
    return f"{minutes:02d}:{seconds:02d}"


def posting_period(date):
    d = date.day
    return "First" if d <= 10 else "Middle" if d <= 20 else "Last"

def format_millions(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:,}"
    
# ----------------------------------------------------
# 3. FETCH AND PREPARE DATA
# ----------------------------------------------------
st.set_page_config(page_title="YouTube Channel Analytics", layout="wide")
st.title("📊 YouTube Channel Analytics Dashboard")

yt_channels = get_channel_stats_cached(channel_ids, api_key)
video_ids = get_videos_from_playlists_cached(yt_channels['playlist_id'].tolist(), api_key)
yt_video_data = get_videos_data_cached(video_ids, api_key)

yt_data = pd.merge(
    yt_video_data,
    yt_channels[['channel_id', 'channel_name', 'ch_subscribers']],
    on='channel_id',
    how='left'
)

yt_data['views'] = pd.to_numeric(yt_data['views'], errors='coerce')
yt_data['likes'] = pd.to_numeric(yt_data['likes'], errors='coerce')
yt_data['ch_subscribers'] = pd.to_numeric(yt_data['ch_subscribers'], errors='coerce')
yt_data['publish_date'] = pd.to_datetime(yt_data['publish_date'])
yt_data['duration2'] = yt_data['duration'].astype(str).apply(iso_duration_to_minutes).round(2)
yt_data['posting_period'] = yt_data['publish_date'].apply(posting_period)
yt_data['publish_month'] = yt_data['publish_date'].dt.to_period('M').astype(str)

df = (
    yt_data.groupby(['channel_id', 'channel_name'])
    .agg(
        videos=('video_id', 'count'),
        total_views=('views', 'sum'),
        total_likes=('likes', 'sum'),
        avg_likes=('likes', 'mean'),
        avg_duration=('duration2', 'mean'),
        subscribers=('ch_subscribers', 'first'),
    )
    .reset_index()
)

# Engagement metrics
df['avg_views_per_video'] = df['total_views'] / df['videos']
df['likes_per_1000_views'] = (df['total_likes'] / df['total_views']) * 1000
df['views_per_subscriber'] = df['total_views'] / df['subscribers']
df['engagement_score'] = (df['total_likes'] / df['total_views']) + df['views_per_subscriber']


# ----------------------------------------------------
# 4. SIDEBAR FILTER
# ----------------------------------------------------
channel_options = ["All Channels"] + sorted(df['channel_name'].unique().tolist())
selected_channel = st.sidebar.selectbox("Select Channel", channel_options)

if st.sidebar.button("🔄 Refresh YouTube Data"):
    st.cache_data.clear()
    st.experimental_rerun()


# ----------------------------------------------------
# 5. DASHBOARD TABS
# ----------------------------------------------------
tabs = st.tabs(["📌 Channel Metrics", "🔥 Engagement Metrics"])


# -------------------------------------------------------------------------
# TAB 1 — CHANNEL METRICS
# -------------------------------------------------------------------------
with tabs[0]:
    if selected_channel == "All Channels":
        st.subheader("📌 Overall Channel Summary")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Videos", f"{df['videos'].sum():,}")
        col2.metric("Total Subscribers", format_millions(df['subscribers'].sum()))
        col3.metric("Total Views", format_millions(df['total_views'].sum()))
        col4.metric("Total Likes", format_millions(df['total_likes'].sum()))
        col5.metric("Avg Video Duration", f"{df['avg_duration'].mean():.1f} mins")

        fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig1.subplots_adjust(wspace=0.25, hspace=0.40)
        sns.barplot(df, x='channel_name', y='videos', ax=axes[0, 0])
        axes[0, 0].set_title("Total Videos")

        sns.barplot(df, x='channel_name', y='total_views', ax=axes[0, 1])
        axes[0, 1].set_title("Total Views")

        sns.barplot(df, x='channel_name', y='total_likes', ax=axes[1, 0])
        axes[1, 0].set_title("Total Likes")

        sns.barplot(df, x='channel_name', y='avg_duration', ax=axes[1, 1])
        axes[1, 1].set_title("Average Video Duration (mins)")
        for ax in axes.flat: ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

        st.markdown("#### 🎬 Top Videos (All Channels)")
        top_videos = yt_data.sort_values('views', ascending=False).head(10)
        top_videos['duration_mmss'] = top_videos['duration2'].apply(minutes_to_mmss)
        st.dataframe(top_videos[['title', 'channel_name', 'views', 'likes', 'publish_date', 'duration_mmss']])

    else:
        st.subheader(f"📌 Channel Summary – {selected_channel}")

        channel_data = yt_data[yt_data['channel_name'] == selected_channel].copy()
        df_month = (
            channel_data.groupby('publish_month')
            .agg(videos=('video_id', 'count'),
                 total_views=('views', 'sum'),
                 total_likes=('likes', 'sum'),
                 avg_likes=('likes', 'mean'),
                 avg_duration=('duration2', 'mean'),
                 subscribers=('ch_subscribers', 'max'))
        ).reset_index().sort_values('publish_month').tail(8)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Videos", f"{df[df['channel_name']==selected_channel]['videos'].sum():,}")
        col2.metric("Total Subscribers", format_millions(df[df['channel_name']==selected_channel]['subscribers'].sum()))
        col3.metric("Total Views", format_millions(df[df['channel_name']==selected_channel]['total_views'].sum()))
        col4.metric("Total Likes", format_millions(df[df['channel_name']==selected_channel]['total_likes'].sum()))

        col5.metric("Avg Video Duration", f"{df[df['channel_name']==selected_channel]['avg_duration'].mean():.1f} mins")

        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
        fig2.subplots_adjust(wspace=0.25, hspace=0.40)
        sns.barplot(df_month, x='publish_month', y='videos', ax=axes2[0, 0])
        axes2[0, 0].set_title("Monthly Videos Count")

        sns.barplot(df_month, x='publish_month', y='total_views', ax=axes2[0, 1])
        axes2[0, 1].set_title("Monthly Total Views")

        sns.barplot(df_month, x='publish_month', y='avg_likes', ax=axes2[1, 0])
        axes2[1, 0].set_title("Average Likes per Month")

        sns.barplot(df_month, x='publish_month', y='avg_duration', ax=axes2[1, 1])
        axes2[1, 1].set_title("Average Video Duration per Month")
        for ax in axes2.flat: ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

        st.markdown("#### 🎬 Top Videos")
        top_videos_ch = channel_data.sort_values('views', ascending=False).head(10)
        st.dataframe(top_videos_ch[['title', 'publish_date', 'views', 'likes']])


# -------------------------------------------------------------------------
# TAB 2 — ENGAGEMENT METRICS (MONTH-WISE FOR SINGLE CHANNEL)
# -------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🔥 Engagement Metrics")

    # ALL CHANNELS — CHANNEL-LEVEL
    if selected_channel == "All Channels":
        df_eng = df.copy()
        corr_data = yt_data.copy()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Views per Subscriber", f"{df_eng['views_per_subscriber'].mean():.2f}")
        col2.metric("Likes per 1000 Views", f"{df_eng['likes_per_1000_views'].mean():.2f}")
        col3.metric("Avg Views per Video", f"{df_eng['avg_views_per_video'].mean():.2f}")
        col4.metric("Engagement Score", f"{df_eng['engagement_score'].mean():.2f}")

        fig_eng, axes_eng = plt.subplots(1, 2, figsize=(18, 7))
        fig_eng.subplots_adjust(wspace=0.25, hspace=0.40)
        sns.barplot(df_eng, x='channel_name', y='views_per_subscriber', ax=axes_eng[0])
        axes_eng[0].set_title("Views per Subscriber")

        sns.barplot(df_eng, x='channel_name', y='likes_per_1000_views', ax=axes_eng[1])
        axes_eng[1].set_title("Likes per 1000 Views")
        for ax in axes_eng.flat: ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig_eng)

        fig_corr, axes_corr = plt.subplots(1, 2, figsize=(18, 6))
        sns.regplot(corr_data, x='views', y='likes', ax=axes_corr[0])
        sns.regplot(df, x='total_views', y='subscribers', ax=axes_corr[1])
        st.pyplot(fig_corr)

        corr1 = corr_data[['views', 'likes']].corr().iloc[0, 1]
        corr2 = df[['total_views', 'subscribers']].corr().iloc[0, 1]
        st.info(f"📌 Correlation between **Views & Likes:** `{corr1:.2f}`\n📌 Correlation between **Views & Subscribers:** `{corr2:.2f}`")

    # SINGLE CHANNEL — MONTH-WISE LAST 8 MONTHS
    else:
        channel_data = yt_data[yt_data['channel_name'] == selected_channel].copy()

        df_eng_month = (
            channel_data.groupby('publish_month')
            .agg(
                videos=('video_id', 'count'),
                month_views=('views', 'sum'),
                month_likes=('likes', 'sum'),
                subscribers=('ch_subscribers', 'max')
            )
        ).reset_index().sort_values('publish_month').tail(12)

        df_eng_month['avg_views_per_video'] = df_eng_month['month_views'] / df_eng_month['videos']
        df_eng_month['likes_per_1000_views'] = (df_eng_month['month_likes'] / df_eng_month['month_views']) * 1000
        df_eng_month['views_per_subscriber'] = df_eng_month['month_views'] / df_eng_month['subscribers']
        df_eng_month['engagement_score'] = \
            (df_eng_month['month_likes'] / df_eng_month['month_views']) + df_eng_month['views_per_subscriber']
        
    
        channel_full = yt_data[yt_data['channel_name'] == selected_channel]

        total_views = channel_full['views'].sum()
        total_likes = channel_full['likes'].sum()
        subscribers = channel_full['ch_subscribers'].max()
        videos = channel_full['video_id'].count()

        kpi_views_per_sub = total_views / subscribers
        kpi_likes_per_1000 = (total_likes / total_views) * 1000
        kpi_avg_views = total_views / videos
        kpi_eng_score = (total_likes / total_views) + kpi_views_per_sub

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Views per Subscriber", f"{kpi_views_per_sub:.2f}")
        col2.metric("Likes per 1000 Views", f"{kpi_likes_per_1000:.2f}")
        col3.metric("Avg Views per Video", f"{kpi_avg_views:.2f}")
        col4.metric("Engagement Score", f"{kpi_eng_score:.2f}")


        fig_m, axes_m = plt.subplots(2, 2, figsize=(18, 8))
        fig_m.subplots_adjust(wspace=0.25, hspace=0.40)
        sns.lineplot(df_eng_month, x='publish_month', y='views_per_subscriber', marker='o', ax=axes_m[0, 0])
        axes_m[0, 0].set_title("Views per Subscriber")

        sns.lineplot(df_eng_month, x='publish_month', y='likes_per_1000_views', marker='o', ax=axes_m[0, 1])
        axes_m[0, 1].set_title("Likes per 1000 Views")

        sns.lineplot(df_eng_month, x='publish_month', y='avg_views_per_video', marker='o', ax=axes_m[1, 0])
        axes_m[1, 0].set_title("Average Views per Video")

        sns.lineplot(df_eng_month, x='publish_month', y='engagement_score', marker='o', ax=axes_m[1, 1])
        axes_m[1, 1].set_title("Engagement Score")
        for ax in axes_m.flat: ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig_m)

        fig_corr, axes_corr = plt.subplots(1, 2, figsize=(16, 6))
        sns.regplot(channel_data, x='views', y='likes', ax=axes_corr[0])
        sns.regplot(channel_data, x='views', y='ch_subscribers', ax=axes_corr[1])
        st.pyplot(fig_corr)

        corr1 = channel_data[['views', 'likes']].corr().iloc[0, 1]
        corr2 = channel_data[['views', 'ch_subscribers']].corr().iloc[0, 1]
        st.info(f"📌 Correlation between **Views & Likes:** `{corr1:.2f}`\n📌 Correlation between **Views & Subscribers:** `{corr2:.2f}`")
