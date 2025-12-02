# yt_dashboard.py
# Full YouTube Analytics Dashboard using Streamlit

import os
import re
from googleapiclient.discovery import build
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# 1. STREAMLIT PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="YouTube Channel Analysis Dashboard",
    layout="wide"
)

st.title("📊 YouTube Channel Analysis Dashboard")

st.markdown(
    """
This dashboard pulls data from multiple YouTube channels using the YouTube Data API,
aggregates it, and visualizes channel performance, engagement, and posting patterns.
"""
)

# -----------------------------
# 2. YOUTUBE API CONFIG
# -----------------------------

# 👉 IMPORTANT: Put your own API key here or set it as env variable YOUTUBE_API_KEY
API_KEY = 'AIzaSyCyLuZ0crF5wlJNA_A-NUdj8fgjR7gCHpY'

# List of channel IDs you want to analyze
CHANNEL_IDS = [
    "UCmTM_hPCeckqN3cPWtYZZcg",
    "UCzI8K9xO_5E-4iCP7Km6cRQ",
    "UC5fcjujOsqD-126Chn_BAuA",
    "UC-CSyyi47VX1lD9zyeABW3w",
    "UC0yXUUIaPVAqZLgRjvtMftw",
    "UCWtlPzcP989da26sVyHPzqQ"
]

# Build the YouTube API client
youtube = build('youtube','v3',developerKey=API_KEY)


# -----------------------------
# 3. HELPER FUNCTIONS
# -----------------------------

def get_channel_stats(youtube_client, channel_ids):
    """Fetch channel-level statistics."""
    all_channels = []

    request = youtube_client.channels().list(
        part="snippet,contentDetails,statistics",
        id=",".join(channel_ids)
    )
    response = request.execute()

    for item in response["items"]:
        data = dict(
            channel_name=item["snippet"]["title"],
            channel_id=item["id"],
            ch_subscribers=item["statistics"].get("subscriberCount", 0),
            views=item["statistics"].get("viewCount", 0),
            total_videos=item["statistics"].get("videoCount", 0),
            playlist_id=item["contentDetails"]["relatedPlaylists"]["uploads"]
        )
        all_channels.append(data)

    return all_channels


def get_videos_from_playlists(youtube_client, playlist_ids):
    """Fetch all video IDs from a list of playlist IDs."""
    all_video_ids = []

    for playlist_id in playlist_ids:
        video_ids = []
        next_page_token = None
        more_pages = True

        while more_pages:
            request = youtube_client.playlistItems().list(
                part="contentDetails",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get("items", []):
                video_ids.append(item["contentDetails"]["videoId"])

            next_page_token = response.get("nextPageToken")
            if next_page_token is None:
                more_pages = False

        all_video_ids.extend(video_ids)

    return all_video_ids


def get_videos_data(youtube_client, video_ids):
    """Fetch details for a list of video IDs."""
    video_details = []

    # YouTube API accepts max 50 IDs at a time
    for i in range(0, len(video_ids), 50):
        request = youtube_client.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i + 50])
        )
        response = request.execute()

        for video in response["items"]:
            stats = video.get("statistics", {})
            video_data = dict(
                video_id=video["id"],
                title=video["snippet"]["title"],
                channel_id=video["snippet"]["channelId"],
                publish_date=video["snippet"]["publishedAt"],
                duration=video["contentDetails"]["duration"],
                views=stats.get("viewCount"),
                likes=stats.get("likeCount"),
                fav_count=stats.get("favoriteCount")
            )
            video_details.append(video_data)

    return video_details


def iso_duration_to_minutes(duration):
    """Convert ISO 8601 duration (e.g., 'PT15M33S') to minutes (float)."""
    hours = minutes = seconds = 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
    return hours * 60 + minutes + seconds / 60


def posting_period(date):
    """Classify a date as First/Middle/Last 10 days of the month."""
    day = date.day
    if day <= 10:
        return "First"
    elif day <= 20:
        return "Middle"
    else:
        return "Last"


# -----------------------------
# 4. LOAD & PREPARE DATA (CACHED)
# -----------------------------

@st.cache_data(show_spinner=True)
def load_data():
    """Fetch data from YouTube API and prepare yt_data and summary df."""

    # --- Channel-level data ---
    channels = get_channel_stats(youtube, CHANNEL_IDS)
    yt_channels = pd.DataFrame(channels)

    # --- Video-level data ---
    playlist_ids = yt_channels["playlist_id"]
    video_ids = get_videos_from_playlists(youtube, playlist_ids)
    video_details = get_videos_data(youtube, video_ids)
    yt_video_data = pd.DataFrame(video_details)

    # --- Merge video data with channel info ---
    channel_sub = yt_channels[["channel_id", "channel_name", "ch_subscribers"]]
    yt_data = pd.merge(yt_video_data, channel_sub, on="channel_id", how="left")

    # --- Cleaning & type conversions ---
    yt_data["views"] = pd.to_numeric(yt_data["views"], errors="coerce")
    yt_data["likes"] = pd.to_numeric(yt_data["likes"], errors="coerce")
    yt_data["fav_count"] = pd.to_numeric(yt_data["fav_count"], errors="coerce")
    yt_data["ch_subscribers"] = pd.to_numeric(yt_data["ch_subscribers"], errors="coerce")

    # Clean publish_date to date only
    yt_data["publish_date"] = yt_data["publish_date"].astype(str).str.split().str[0]
    yt_data["publish_date"] = pd.to_datetime(yt_data["publish_date"], errors="coerce").dt.date

    # Duration in minutes
    yt_data["duration2"] = yt_data["duration"].astype(str).apply(iso_duration_to_minutes)
    yt_data["duration2"] = yt_data["duration2"].round(2)

    # Posting period (First/Middle/Last)
    yt_data["posting_period"] = yt_data["publish_date"].apply(posting_period)

    # --- Aggregate at channel level ---
    df = (
        yt_data.groupby(["channel_id", "channel_name"])
        .agg(
            videos=("video_id", "count"),
            total_views=("views", "sum"),
            total_likes=("likes", "sum"),
            avg_views=("views", "mean"),
            avg_likes=("likes", "mean"),
            subscribers=("ch_subscribers", "first")
        )
        .reset_index()
    )

    # Additional metrics
    df["views_per_subscriber"] = df["total_views"] / df["subscribers"]
    df["likes_per_1000_views"] = (df["total_likes"] / df["total_views"]) * 1000
    df["avg_views_per_video"] = df["total_views"] / df["videos"]
    df["engagement_score"] = (df["total_likes"] / df["total_views"]) + df["views_per_subscriber"]

    return yt_data, df


# Load data
with st.spinner("Loading data from YouTube API..."):
    yt_data, df = load_data()

if yt_data.empty or df.empty:
    st.error("No data could be loaded. Please check your API key and channel IDs.")
    st.stop()

sns.set_theme(style="whitegrid")

# -----------------------------
# 5. SIDEBAR FILTERS
# -----------------------------

st.sidebar.header("Filters")

channels_available = sorted(df["channel_name"].unique())
selected_channel = st.sidebar.selectbox("Select a channel", channels_available)

st.sidebar.markdown("---")
st.sidebar.write("You can pick a channel above to see its detailed top videos.")


# -----------------------------
# 6. TOP-LEVEL KPIs
# -----------------------------

st.subheader("Overall Channel Summary")

col1, col2, col3, col4 = st.columns(4)
total_videos = int(df["videos"].sum())
total_views = int(df["total_views"].sum())
total_likes = int(df["total_likes"].sum())
total_subs = int(df["subscribers"].sum())

col1.metric("Total Videos", f"{total_videos:,}")
col2.metric("Total Views", f"{total_views:,}")
col3.metric("Total Likes", f"{total_likes:,}")
col4.metric("Total Subscribers (sum)", f"{total_subs:,}")

# -----------------------------
# 7. LAYOUT SECTIONS (TABS)
# -----------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📦 Channel Basics", "📅 Posting Behavior",
     "📣 Engagement", "📈 Correlations", "🎥 Top Videos"]
)

# -----------------------------
# TAB 1: CHANNEL BASICS
# -----------------------------
with tab1:
    st.subheader("Videos per Channel")

    sorted_df_videos = df.sort_values("videos", ascending=True)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=sorted_df_videos,
        x="channel_name",
        y="videos",
        ax=ax1
    )
    ax1.set_title("YouTube Channel Videos")
    ax1.set_xlabel("Channel Name")
    ax1.set_ylabel("Number of Videos")
    ax1.tick_params(axis="x", rotation=30)
    st.pyplot(fig1)

    st.subheader("Average Views per Video by Channel")

    sorted_df_avg_views = df.sort_values("avg_views_per_video", ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=sorted_df_avg_views,
        x="channel_name",
        y="avg_views_per_video",
        ax=ax2
    )
    ax2.set_title("Average Views per Video")
    ax2.set_xlabel("Channel Name")
    ax2.set_ylabel("Avg Views per Video")
    ax2.tick_params(axis="x", rotation=30)
    st.pyplot(fig2)

    st.markdown("Raw summary data:")
    st.dataframe(df[["channel_name", "videos", "total_views", "total_likes",
                     "avg_views_per_video", "subscribers"]])


# -----------------------------
# TAB 2: POSTING BEHAVIOR
# -----------------------------
with tab2:
    st.subheader("Videos by Period of the Month")

    video_period_share = yt_data["posting_period"].value_counts()

    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.pie(
        video_period_share,
        labels=video_period_share.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax3.set_title("Videos Posted by Period of Month")
    st.pyplot(fig3)

    st.subheader("Posting Pattern per Channel (First, Middle, Last)")

    video_counts = yt_data.groupby(["channel_name", "posting_period"]).size().unstack(fill_value=0)
    # Ensure logical order of columns
    video_counts = video_counts[["First", "Middle", "Last"]]
    video_perc = video_counts.div(video_counts.sum(axis=1), axis=0) * 100

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    video_perc.plot(
        kind="barh",
        stacked=True,
        ax=ax4
    )
    ax4.set_title("Share of Videos by Posting Period and Channel")
    ax4.set_xlabel("% of Videos")
    ax4.set_ylabel("Channel")
    ax4.legend(title="Posting Period", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig4)

    st.markdown("Underlying data (percentage of videos):")
    st.dataframe(video_perc.round(2))


# -----------------------------
# TAB 3: ENGAGEMENT
# -----------------------------
with tab3:
    st.subheader("Views per Subscriber by Channel")

    sorted_views_sub = df.sort_values("views_per_subscriber", ascending=False)
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=sorted_views_sub,
        x="channel_name",
        y="views_per_subscriber",
        ax=ax5
    )
    ax5.set_title("Views per Subscriber")
    ax5.set_xlabel("Channel Name")
    ax5.set_ylabel("Views per Subscriber")
    ax5.tick_params(axis="x", rotation=30)
    st.pyplot(fig5)

    st.subheader("Likes per 1000 Views by Channel")

    fig6, ax6 = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=df,
        x="channel_name",
        y="likes_per_1000_views",
        ax=ax6
    )
    ax6.set_title("Likes per 1000 Views")
    ax6.set_xlabel("Channel Name")
    ax6.set_ylabel("Likes per 1000 Views")
    ax6.tick_params(axis="x", rotation=30)
    st.pyplot(fig6)

    st.subheader("Overall Engagement Score by Channel")

    sorted_engagement = df.sort_values("engagement_score", ascending=False)
    fig7, ax7 = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=sorted_engagement,
        x="channel_name",
        y="engagement_score",
        ax=ax7
    )
    ax7.set_title("Overall Engagement Score")
    ax7.set_xlabel("Channel Name")
    ax7.set_ylabel("Engagement Score")
    ax7.tick_params(axis="x", rotation=30)
    st.pyplot(fig7)

    st.markdown("Engagement-related metrics:")
    st.dataframe(df[["channel_name", "views_per_subscriber",
                     "likes_per_1000_views", "engagement_score"]])


# -----------------------------
# TAB 4: CORRELATIONS
# -----------------------------
with tab4:
    st.subheader("Correlation: Average Views vs Average Likes per Video")

    fig8, ax8 = plt.subplots(figsize=(6, 5))
    sns.regplot(
        data=df,
        x="avg_views",
        y="avg_likes",
        ax=ax8,
        scatter_kws={"alpha": 0.6}
    )
    ax8.set_title("Avg Views vs Avg Likes (Channel Level)")
    ax8.set_xlabel("Average Views")
    ax8.set_ylabel("Average Likes")
    st.pyplot(fig8)

    st.subheader("Correlation: Subscribers vs Total Views")

    fig9, ax9 = plt.subplots(figsize=(6, 5))
    sns.regplot(
        data=df,
        x="subscribers",
        y="total_views",
        ax=ax9,
        scatter_kws={"alpha": 0.7}
    )
    ax9.set_title("Subscribers vs Total Views (Channel Level)")
    ax9.set_xlabel("Subscribers")
    ax9.set_ylabel("Total Views")
    st.pyplot(fig9)


# -----------------------------
# TAB 5: TOP VIDEOS (BY CHANNEL)
# -----------------------------
with tab5:
    st.subheader(f"Top Videos for: {selected_channel}")

    select_channel_df = yt_data[yt_data["channel_name"] == selected_channel].copy()

    if select_channel_df.empty:
        st.warning("No videos found for the selected channel.")
    else:
        top_tabs = st.tabs(["Top 10 by Views", "Top 10 by Likes"])

        with top_tabs[0]:
            st.markdown("### Top 10 Videos by Views")
            top_by_views = select_channel_df.sort_values("views", ascending=False).head(10)
            st.dataframe(
                top_by_views[["title", "views", "likes", "publish_date", "duration2"]]
                .rename(columns={"duration2": "duration_minutes"})
            )

        with top_tabs[1]:
            st.markdown("### Top 10 Videos by Likes")
            top_by_likes = select_channel_df.sort_values("likes", ascending=False).head(10)
            st.dataframe(
                top_by_likes[["title", "likes", "views", "publish_date", "duration2"]]
                .rename(columns={"duration2": "duration_minutes"})
            )

    st.markdown(
        """
You can change the **selected channel** from the sidebar to see the top-performing videos
for different channels.
"""
    )
