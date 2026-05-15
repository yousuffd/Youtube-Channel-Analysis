# ============================================================================
# ENHANCED YOUTUBE ANALYTICS DASHBOARD WITH PREDICTIONS
# ============================================================================
# New Features Added:
# 1. Growth Metrics (subscriber growth, view velocity)
# 2. Predictive Models (view forecasting, engagement prediction)
# 3. Content Performance Analysis (viral videos, optimal duration)
# 4. Posting Pattern Analysis (best days/times)
# 5. Video Performance Categories
# 6. Advanced Engagement Metrics


# ============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import warnings
import re
from datetime import datetime, timedelta
from googleapiclient.discovery import build

# Machine Learning imports for predictions
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# ============================================================================
# 1. API CONNECTION (Same as original)
# ============================================================================

api_key = st.secrets["API_KEY"]

channel_ids = [
    'UCmTM_hPCeckqN3cPWtYZZcg',
    'UCzI8K9xO_5E-4iCP7Km6cRQ',
    'UC5fcjujOsqD-126Chn_BAuA',
    'UC-CSyyi47VX1lD9zyeABW3w',
    'UC0yXUUIaPVAqZLgRjvtMftw',
    'UCWtlPzcP989da26sVyHPzqQ'
]

youtube = build('youtube', 'v3', developerKey=api_key)

# ============================================================================
# 2. DATA FETCHING FUNCTIONS (Enhanced with comments)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Fetching channel stats...")  # Cache for 1 hour
def get_channel_stats_cached(channel_ids, api_key):
    """Fetch channel-level statistics"""
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
            "channel_created": item["snippet"]["publishedAt"]  # NEW: Channel age
        })
    return pd.DataFrame(all_channels)


@st.cache_data(ttl=3600, show_spinner="Fetching video IDs...")  # Cache for 1 hour
def get_videos_from_playlists_cached(playlist_ids, api_key):
    """Fetch all video IDs from playlists"""
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


@st.cache_data(ttl=3600, show_spinner="Fetching video details...")  # Cache for 1 hour
def get_videos_data_cached(video_ids, api_key):
    """Fetch detailed video statistics"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_details = []
    
    for i in range(0, len(video_ids), 50):
        response = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i + 50])
        ).execute()
        
        for video in response['items']:
            stats = video.get("statistics", {})
            snippet = video["snippet"]
            
            video_details.append({
                "video_id": video["id"],
                "title": snippet["title"],
                "channel_id": snippet["channelId"],
                "publish_date": snippet["publishedAt"],
                "duration": video["contentDetails"]["duration"],
                "views": stats.get("viewCount", 0),
                "likes": stats.get("likeCount", 0),
                "comments": stats.get("commentCount", 0),  # NEW: Comment count
                "tags": snippet.get("tags", []),  # NEW: Tags
                "description": snippet.get("description", "")  # NEW: Description
            })
    
    return pd.DataFrame(video_details)


# ============================================================================
# 3. NEW FUNCTION: Fetch Comments for Sentiment Analysis
# ============================================================================

@st.cache_data(show_spinner="Fetching comments for analysis...")
def get_video_comments(video_id, api_key, max_results=100):
    """Fetch comments for a specific video (for sentiment analysis)"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        ).execute()
        
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment_text": comment["textDisplay"],
                "likes": comment.get("likeCount", 0),
                "published_at": comment["publishedAt"]
            })
    except:
        pass  # Comments might be disabled
    
    return comments


# ============================================================================
# 4. HELPER FUNCTIONS (Enhanced)
# ============================================================================

def iso_duration_to_minutes(duration):
    """Convert ISO 8601 duration to minutes"""
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
    """Convert minutes to MM:SS format"""
    if pd.isna(value):
        return None
    minutes = int(value)
    seconds = round((value - minutes) * 60)
    return f"{minutes:02d}:{seconds:02d}"


def posting_period(date):
    """Categorize posting period in month"""
    d = date.day
    return "First" if d <= 10 else "Middle" if d <= 20 else "Last"


def format_millions(value):
    """Format large numbers with K/M suffix"""
    if pd.isna(value):
        return "0"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:,.0f}"


# NEW: Calculate growth rate
def calculate_growth_rate(df, date_col='publish_date', value_col='views', periods=30):
    """Calculate growth rate over specified periods"""
    df = df.sort_values(date_col)
    df['cumulative'] = df[value_col].cumsum()
    df['days_since_first'] = (df[date_col] - df[date_col].min()).dt.days
    
    # Growth rate = (current - previous) / previous
    if len(df) > periods:
        recent = df.tail(periods)[value_col].sum()
        previous = df.iloc[-periods*2:-periods][value_col].sum()
        if previous > 0:
            return ((recent - previous) / previous) * 100
    return 0


# NEW: Categorize video performance
def categorize_video_performance(views, percentile_75, percentile_25):
    """Categorize videos as viral, average, or underperforming"""
    if views >= percentile_75:
        return "Viral"
    elif views >= percentile_25:
        return "Average"
    else:
        return "Underperforming"


# ============================================================================
# 5. STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="YouTube Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0E1117;
}

/* Section containers */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Metrics styling */
[data-testid="metric-container"] {
    background-color: #1C1F26;
    border: 1px solid #2E3440;
    padding: 15px;
    border-radius: 12px;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1C1F26;
    border-radius: 10px;
    padding: 10px 18px;
    color: white;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #4F8BF9;
}

/* Chart spacing */
.element-container {
    margin-bottom: 1.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161A23;
}

</style>
""", unsafe_allow_html=True)

st.title("📊 YouTube Analytics Dashboard")
st.markdown("*Track growth, engagement, content strategy, and predictive insights across YouTube channels*")

# ============================================================================
# 6. FETCH AND PREPARE DATA
# ============================================================================

try:
    with st.spinner("Loading YouTube data..."):
        # Fetch channel stats
        yt_channels = get_channel_stats_cached(channel_ids, api_key)
        
        # Fetch video IDs
        video_ids = get_videos_from_playlists_cached(
            yt_channels['playlist_id'].tolist(), 
            api_key
        )
        
        # Fetch video details
        yt_video_data = get_videos_data_cached(video_ids, api_key)
        
        # Merge channel and video data
        yt_data = pd.merge(
            yt_video_data,
            yt_channels[['channel_id', 'channel_name', 'ch_subscribers', 'channel_created']],
            on='channel_id',
            how='left'
        )
        
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.info("💡 Please check your API key and internet connection")
    st.stop()  # Stop execution here if data loading fails

# ============================================================================
# 7. DATA PREPROCESSING (Enhanced)
# ============================================================================

# Convert data types
yt_data['views'] = pd.to_numeric(yt_data['views'], errors='coerce').fillna(0)
yt_data['likes'] = pd.to_numeric(yt_data['likes'], errors='coerce').fillna(0)
yt_data['comments'] = pd.to_numeric(yt_data['comments'], errors='coerce').fillna(0)
yt_data['ch_subscribers'] = pd.to_numeric(yt_data['ch_subscribers'], errors='coerce')

# Fix datetime parsing with ISO8601 format
yt_data['publish_date'] = pd.to_datetime(yt_data['publish_date'], format='ISO8601')

# Create channel_created mapping and merge properly
channel_created_map = yt_channels.set_index('channel_id')['channel_created'].to_dict()
yt_data['channel_created'] = yt_data['channel_id'].map(channel_created_map)
yt_data['channel_created'] = pd.to_datetime(yt_data['channel_created'], format='ISO8601')

# Duration processing
yt_data['duration_minutes'] = yt_data['duration'].astype(str).apply(iso_duration_to_minutes).round(2)

# NEW: Time-based features
yt_data['publish_hour'] = yt_data['publish_date'].dt.hour
yt_data['publish_day'] = yt_data['publish_date'].dt.day_name()
yt_data['publish_month'] = yt_data['publish_date'].dt.to_period('M').astype(str)
yt_data['publish_year'] = yt_data['publish_date'].dt.year
yt_data['posting_period'] = yt_data['publish_date'].apply(posting_period)

# NEW: Video age in days
yt_data['video_age_days'] = (pd.Timestamp.now(tz='UTC') - yt_data['publish_date']).dt.days

# NEW: Engagement metrics per video (with safe division)
# Replace 0 views with NaN to avoid division by zero warnings
safe_views = yt_data['views'].replace(0, np.nan)

yt_data['engagement_rate'] = ((yt_data['likes'] + yt_data['comments']) / safe_views * 100).fillna(0)
yt_data['like_rate'] = (yt_data['likes'] / safe_views * 100).fillna(0)
yt_data['comment_rate'] = (yt_data['comments'] / safe_views * 100).fillna(0)
yt_data['likes_per_comment'] = (yt_data['likes'] / yt_data['comments'].replace(0, np.nan)).fillna(0)

# NEW: Views per day (velocity)
yt_data['views_per_day'] = (yt_data['views'] / (yt_data['video_age_days'] + 1)).fillna(0)

# NEW: Title length (for analysis)
yt_data['title_length'] = yt_data['title'].str.len()

# NEW: Performance categories
percentile_75 = yt_data['views'].quantile(0.75)
percentile_25 = yt_data['views'].quantile(0.25)
yt_data['performance_category'] = yt_data['views'].apply(
    lambda x: categorize_video_performance(x, percentile_75, percentile_25)
)

# ============================================================================
# 8. AGGREGATE CHANNEL-LEVEL METRICS
# ============================================================================

df_channels = (
    yt_data.groupby(['channel_id', 'channel_name'])
    .agg(
        videos=('video_id', 'count'),
        total_views=('views', 'sum'),
        total_likes=('likes', 'sum'),
        total_comments=('comments', 'sum'),
        avg_likes=('likes', 'mean'),
        avg_views=('views', 'mean'),
        avg_duration=('duration_minutes', 'mean'),
        avg_engagement_rate=('engagement_rate', 'mean'),
        subscribers=('ch_subscribers', 'first'),
        channel_created=('channel_created', 'first')
    )
    .reset_index()
)

# Calculate channel-level metrics (with safe division)
df_channels['avg_views_per_video'] = df_channels['total_views'] / df_channels['videos']
df_channels['likes_per_1000_views'] = ((df_channels['total_likes'] / df_channels['total_views'].replace(0, np.nan)) * 1000).fillna(0)
df_channels['views_per_subscriber'] = (df_channels['total_views'] / df_channels['subscribers'].replace(0, np.nan)).fillna(0)
df_channels['engagement_score'] = ((df_channels['total_likes'] / df_channels['total_views'].replace(0, np.nan)) + df_channels['views_per_subscriber']).fillna(0)

# NEW: Channel age in days (fixed timezone)
df_channels['channel_age_days'] = (pd.Timestamp.now(tz='UTC') - df_channels['channel_created']).dt.days
df_channels['videos_per_month'] = ((df_channels['videos'] / df_channels['channel_age_days'].replace(0, np.nan)) * 30).fillna(0)

# ============================================================================
# 9. SIDEBAR CONTROLS (Enhanced)
# ============================================================================

st.sidebar.header("🎛️ Dashboard Controls")

# Channel selector
channel_options = ["All Channels"] + sorted(df_channels['channel_name'].unique().tolist())
selected_channel = st.sidebar.selectbox("📺 Select Channel", channel_options)

# NEW: Date range filter
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Date Range Filter")

min_date = yt_data['publish_date'].min().date()
max_date = yt_data['publish_date'].max().date()

# Default = current year
current_year = datetime.now().year
default_start = datetime(current_year, 1, 1).date()
default_end = max_date

# Allow going back 3 years
three_year_limit = datetime(current_year - 3, 1, 1).date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(default_start, default_end),
    min_value=max(min_date, three_year_limit),
    max_value=max_date
)

# Apply date filter
if len(date_range) == 2:
    start_date, end_date = date_range
    yt_data_filtered = yt_data[
        (yt_data['publish_date'].dt.date >= start_date) &
        (yt_data['publish_date'].dt.date <= end_date)
    ]
else:
    yt_data_filtered = yt_data.copy()

# NEW: Performance category filter
st.sidebar.markdown("---")
performance_filter = st.sidebar.multiselect(
    "🎯 Filter by Performance",
    options=["Viral", "Average", "Underperforming"],
    default=["Viral", "Average", "Underperforming"]
)

yt_data_filtered = yt_data_filtered[yt_data_filtered['performance_category'].isin(performance_filter)]

# Refresh button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# ============================================================================
# 10. DASHBOARD TABS (Enhanced)
# ============================================================================

tabs = st.tabs([
    "📊 Overview",
    "📈 Growth & Trends",
    "🎯 Content Performance",
    "🔥 Engagement Analysis",
    "🔮 Channel Predictions",
    "⏰ Posting Patterns",
    "🏆 Top Content"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tabs[0]:
   # if selected_channel == "All Channels":
    #    st.subheader("📊 Overall Performance Summary")

    if selected_channel == "All Channels":

        st.markdown("### Platform Growth Trends")

        # Monthly platform growth
        monthly_growth = yt_data_filtered.groupby('publish_month').agg({
            'views': 'sum',
            'video_id': 'count',
            'likes': 'sum'
        }).reset_index()
    
        monthly_growth['avg_views_per_video'] = (
            monthly_growth['views'] / monthly_growth['video_id']
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                monthly_growth,
                x='publish_month',
                y='views',
                markers=True,
                title="Total Platform Views Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                monthly_growth,
                x='publish_month',
                y='avg_views_per_video',
                markers=True,
                title="Average Views Per Video Trend"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Channel comparison
        growth_compare = yt_data_filtered.groupby('channel_name').agg({
            'views_per_day': 'mean',
            'engagement_rate': 'mean'
        }).reset_index()

        fig = px.scatter(
            growth_compare,
            x='views_per_day',
            y='engagement_rate',
            size='views_per_day',
            color='channel_name',
            title="Channel Growth vs Engagement"
        )

        st.plotly_chart(fig, use_container_width=True)    
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_videos = yt_data_filtered['video_id'].nunique()
        total_views = yt_data_filtered['views'].sum()
        total_likes = yt_data_filtered['likes'].sum()
        total_comments = yt_data_filtered['comments'].sum()
        total_subscribers = df_channels['subscribers'].sum()
        
        col1.metric("📹 Total Videos", f"{total_videos:,}")
        col2.metric("👥 Total Subscribers", format_millions(total_subscribers))
        col3.metric("👁️ Total Views", format_millions(total_views))
        col4.metric("👍 Total Likes", format_millions(total_likes))
        col5.metric("💬 Total Comments", format_millions(total_comments))
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        avg_engagement = yt_data_filtered['engagement_rate'].mean()
        avg_views = yt_data_filtered['views'].mean()
        avg_duration = yt_data_filtered['duration_minutes'].mean()
        viral_videos = len(yt_data_filtered[yt_data_filtered['performance_category'] == 'Viral'])
        
        col1.metric("📊 Avg Engagement Rate", f"{avg_engagement:.2f}%")
        col2.metric("👁️ Avg Views/Video", format_millions(avg_views))
        col3.metric("⏱️ Avg Duration", f"{avg_duration:.1f} min")
        col4.metric("🔥 Viral Videos", f"{viral_videos}")
        
        st.markdown("---")
        
        # Channel comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Channel Performance Comparison")
            fig = px.bar(
                df_channels.sort_values('total_views', ascending=False),
                x='channel_name',
                y='total_views',
                color='total_views',
                title="Total Views by Channel",
                labels={'total_views': 'Total Views', 'channel_name': 'Channel'}
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Engagement Rate Comparison")
            fig = px.bar(
                df_channels.sort_values('avg_engagement_rate', ascending=False),
                x='channel_name',
                y='avg_engagement_rate',
                color='avg_engagement_rate',
                title="Average Engagement Rate by Channel",
                labels={'avg_engagement_rate': 'Engagement Rate (%)', 'channel_name': 'Channel'}
            )
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance distribution
        st.markdown("### Video Performance Distribution")
        perf_dist = yt_data_filtered['performance_category'].value_counts()
        fig = px.pie(
            values=perf_dist.values,
            names=perf_dist.index,
            title="Distribution of Video Performance",
            color=perf_dist.index,
            color_discrete_map={'Viral': '#2ecc71', 'Average': '#f39c12', 'Underperforming': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Single channel overview
        st.subheader(f"📊 {selected_channel} - Performance Overview")
        
        channel_data = yt_data_filtered[yt_data_filtered['channel_name'] == selected_channel]
        channel_info = df_channels[df_channels['channel_name'] == selected_channel].iloc[0]
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("📹 Total Videos", f"{len(channel_data):,}")
        col2.metric("👥 Subscribers", format_millions(channel_info['subscribers']))
        col3.metric("👁️ Total Views", format_millions(channel_data['views'].sum()))
        col4.metric("👍 Total Likes", format_millions(channel_data['likes'].sum()))
        col5.metric("💬 Total Comments", format_millions(channel_data['comments'].sum()))
        
        # Second row
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("📊 Avg Engagement", f"{channel_data['engagement_rate'].mean():.2f}%")
        col2.metric("👁️ Avg Views", format_millions(channel_data['views'].mean()))
        col3.metric("⏱️ Avg Duration", f"{channel_data['duration_minutes'].mean():.1f} min")
        col4.metric("📅 Upload Frequency", f"{channel_info['videos_per_month']:.1f}/month")
        
        st.markdown("---")
        
        # Monthly performance
        df_monthly = channel_data.groupby('publish_month').agg({
            'video_id': 'count',
            'views': 'sum',
            'likes': 'sum',
            'engagement_rate': 'mean'
        }).reset_index()
        df_monthly.columns = ['Month', 'Videos', 'Views', 'Likes', 'Engagement Rate']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Monthly Upload Frequency")
            fig = px.bar(
                df_monthly,
                x='Month',
                y='Videos',
                title="Videos Uploaded Per Month"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Monthly View Performance")
            fig = px.line(
                df_monthly,
                x='Month',
                y='Views',
                markers=True,
                title="Total Views Per Month"
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 2: GROWTH & TRENDS
# ============================================================================

with tabs[1]:
    st.subheader("📈 Growth Analysis & Trends")
    
    if selected_channel == "All Channels":
        st.info("💡 Select a specific channel to view detailed growth trends")
        
        # Show channel growth comparison
        st.markdown("### Channel Growth Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Subscriber efficiency
            df_channels['subs_per_video'] = df_channels['subscribers'] / df_channels['videos']
            fig = px.bar(
                df_channels.sort_values('subs_per_video', ascending=False),
                x='channel_name',
                y='subs_per_video',
                title="Subscriber Efficiency (Subs per Video)",
                labels={'subs_per_video': 'Subscribers/Video'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Upload consistency
            fig = px.bar(
                df_channels.sort_values('videos_per_month', ascending=False),
                x='channel_name',
                y='videos_per_month',
                title="Upload Consistency (Videos/Month)",
                labels={'videos_per_month': 'Videos per Month'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        channel_data = yt_data_filtered[yt_data_filtered['channel_name'] == selected_channel].copy()
        channel_data = channel_data.sort_values('publish_date')
        
        # Calculate cumulative metrics
        channel_data['cumulative_views'] = channel_data['views'].cumsum()
        channel_data['cumulative_videos'] = range(1, len(channel_data) + 1)
        
        # Growth metrics
        col1, col2, col3 = st.columns(3)
        
        # Calculate 30-day growth
        recent_30 = channel_data[channel_data['video_age_days'] <= 30]
        prev_30_60 = channel_data[(channel_data['video_age_days'] > 30) & (channel_data['video_age_days'] <= 60)]
        
        if len(recent_30) > 0 and len(prev_30_60) > 0:
            view_growth = ((recent_30['views'].sum() - prev_30_60['views'].sum()) / prev_30_60['views'].sum()) * 100
        else:
            view_growth = 0
        
        col1.metric("📈 30-Day View Growth", f"{view_growth:+.1f}%")
        col2.metric("🚀 Avg Views/Day", format_millions(channel_data['views_per_day'].mean()))
        col3.metric("📊 Content Consistency", f"{len(channel_data) / (channel_data['video_age_days'].max() / 30):.1f} videos/month")
        
        st.markdown("---")
        
        # Cumulative growth chart
        st.markdown("### Cumulative View Growth")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=channel_data['publish_date'],
            y=channel_data['cumulative_views'],
            mode='lines',
            name='Cumulative Views',
            fill='tozeroy'
        ))
        fig.update_layout(
            title="Cumulative Views Over Time",
            xaxis_title="Date",
            yaxis_title="Total Views",
            hovermode='x'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly growth trends
        col1, col2 = st.columns(2)
        
        with col1:
            df_monthly = channel_data.groupby('publish_month').agg({
                'views': 'sum',
                'video_id': 'count'
            }).reset_index()
            df_monthly['views_per_video'] = df_monthly['views'] / df_monthly['video_id']
            
            fig = px.line(
                df_monthly,
                x='publish_month',
                y='views_per_video',
                markers=True,
                title="Average Views per Video (Monthly Trend)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Engagement trend
            df_monthly_eng = channel_data.groupby('publish_month')['engagement_rate'].mean().reset_index()
            
            fig = px.line(
                df_monthly_eng,
                x='publish_month',
                y='engagement_rate',
                markers=True,
                title="Average Engagement Rate (Monthly Trend)"
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 3: CONTENT PERFORMANCE
# ============================================================================

with tabs[2]:
    st.subheader("🎯 Content Performance Analysis")
    
    if selected_channel != "All Channels":
        channel_data = yt_data_filtered[yt_data_filtered['channel_name'] == selected_channel]
    else:
        channel_data = yt_data_filtered
    
    # Performance breakdown
    col1, col2, col3 = st.columns(3)
    
    viral_count = len(channel_data[channel_data['performance_category'] == 'Viral'])
    avg_count = len(channel_data[channel_data['performance_category'] == 'Average'])
    under_count = len(channel_data[channel_data['performance_category'] == 'Underperforming'])
    
    col1.metric("🔥 Viral Videos", viral_count, f"{(viral_count/len(channel_data)*100):.1f}%")
    col2.metric("📊 Average Videos", avg_count, f"{(avg_count/len(channel_data)*100):.1f}%")
    col3.metric("📉 Underperforming", under_count, f"{(under_count/len(channel_data)*100):.1f}%")
    
    st.markdown("---")
    
    # Duration analysis
    st.markdown("### Video Duration vs Performance")
    
    # Create duration bins
    channel_data['duration_category'] = pd.cut(
        channel_data['duration_minutes'],
        bins=[0, 5, 10, 15, 20, 30, 60, 1000],
        labels=['0-5min', '5-10min', '10-15min', '15-20min', '20-30min', '30-60min', '60+min']
    )
    
    duration_perf = channel_data.groupby('duration_category').agg({
        'views': 'mean',
        'engagement_rate': 'mean',
        'video_id': 'count'
    }).reset_index()
    duration_perf.columns = ['Duration', 'Avg Views', 'Avg Engagement', 'Video Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            duration_perf,
            x='Duration',
            y='Avg Views',
            title="Average Views by Video Duration",
            color='Avg Views'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            duration_perf,
            x='Duration',
            y='Avg Engagement',
            title="Average Engagement by Video Duration",
            color='Avg Engagement',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimal duration insight
    best_duration = duration_perf.loc[duration_perf['Avg Views'].idxmax(), 'Duration']
    st.success(f"💡 **Insight:** Videos in the **{best_duration}** category perform best with an average of **{duration_perf['Avg Views'].max():,.0f}** views")
    
    st.markdown("---")
    
    # Title length analysis
    st.markdown("### Title Length Analysis")
    
    channel_data['title_length_category'] = pd.cut(
        channel_data['title_length'],
        bins=[0, 30, 50, 70, 100, 200],
        labels=['Very Short (<30)', 'Short (30-50)', 'Medium (50-70)', 'Long (70-100)', 'Very Long (100+)']
    )
    
    title_perf = channel_data.groupby('title_length_category').agg({
        'views': 'mean',
        'engagement_rate': 'mean'
    }).reset_index()
    
    fig = px.scatter(
        channel_data,
        x='title_length',
        y='views',
        color='performance_category',
        title="Title Length vs Views",
        labels={'title_length': 'Title Length (characters)', 'views': 'Views'},
        color_discrete_map={'Viral': '#2ecc71', 'Average': '#f39c12', 'Underperforming': '#e74c3c'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Best performing title length
    best_title_length = title_perf.loc[title_perf['views'].idxmax(), 'title_length_category']
    st.success(f"💡 **Insight:** Titles that are **{best_title_length}** characters perform best")


# ============================================================================
# TAB 4: ENGAGEMENT ANALYSIS
# ============================================================================

with tabs[3]:
    st.subheader("🔥 Engagement Analysis")
    
    if selected_channel != "All Channels":
        channel_data = yt_data_filtered[yt_data_filtered['channel_name'] == selected_channel]
    else:
        channel_data = yt_data_filtered
    
    # Engagement metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("📊 Avg Engagement Rate", f"{channel_data['engagement_rate'].mean():.2f}%")
    col2.metric("👍 Avg Like Rate", f"{channel_data['like_rate'].mean():.2f}%")
    col3.metric("💬 Avg Comment Rate", f"{channel_data['comment_rate'].mean():.3f}%")
    col4.metric("🔄 Likes per Comment", f"{channel_data['likes_per_comment'].mean():.1f}")
    
    st.markdown("---")
    
    # Engagement scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Views vs Engagement Rate")
        fig = px.scatter(
            channel_data,
            x='views',
            y='engagement_rate',
            color='performance_category',
            size='likes',
            hover_data=['title'],
            title="Views vs Engagement Rate",
            color_discrete_map={'Viral': '#2ecc71', 'Average': '#f39c12', 'Underperforming': '#e74c3c'}
        )
        fig.update_xaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Likes vs Comments Relationship")
        fig = px.scatter(
            channel_data,
            x='likes',
            y='comments',
            color='performance_category',
            size='views',
            hover_data=['title'],
            title="Likes vs Comments",
            color_discrete_map={'Viral': '#2ecc71', 'Average': '#f39c12', 'Underperforming': '#e74c3c'}
        )
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### Engagement Correlation Matrix")
    corr_cols = ['views', 'likes', 'comments', 'engagement_rate', 'duration_minutes']
    corr_matrix = channel_data[corr_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        title="Correlation Between Engagement Metrics",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top insights
    views_likes_corr = channel_data[['views', 'likes']].corr().iloc[0, 1]
    views_comments_corr = channel_data[['views', 'comments']].corr().iloc[0, 1]
    
    st.info(f"""
    📊 **Key Insights:**
    - Views and Likes correlation: **{views_likes_corr:.2f}** (Strong positive indicates good content quality)
    - Views and Comments correlation: **{views_comments_corr:.2f}** (Higher means more discussion)
    - Average engagement rate: **{channel_data['engagement_rate'].mean():.2f}%**
    """)


# ============================================================================
# TAB 5: PREDICTIONS (NEW!)
# ============================================================================

with tabs[4]:
    st.subheader("🔮 Predictive Analytics")
    
    if selected_channel == "All Channels":
        st.info("💡 Please select a specific channel to view predictions")
    else:
        channel_data = yt_data[yt_data['channel_name'] == selected_channel].copy()
        
        if len(channel_data) < 20:
            st.warning("⚠️ Not enough data for reliable predictions. Need at least 20 videos.")
        else:
            # Prepare data for prediction
            st.markdown("### View Count Prediction Model")
            
            # Feature engineering for ML
            feature_cols = ['duration_minutes', 'title_length', 'publish_hour', 'video_age_days']
            
            # Create day of week numeric
            day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                          'Friday': 4, 'Saturday': 5, 'Sunday': 6}
            channel_data['publish_day_num'] = channel_data['publish_day'].map(day_mapping)
            feature_cols.append('publish_day_num')
            
            # Prepare training data
            X = channel_data[feature_cols].fillna(0)
            y = channel_data['views']
            
            # Train-test split (80-20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            with st.spinner("Training prediction model..."):
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            
            # Display model performance
            col1, col2, col3 = st.columns(3)
            col1.metric("Model Accuracy (R²)", f"{r2:.2%}")
            col2.metric("Mean Abs Error", format_millions(mae))
            col3.metric("Avg Prediction Error", f"{(mae/y_test.mean())*100:.1f}%")
            
            # Prediction vs Actual
            st.markdown("### Model Performance: Predicted vs Actual Views")
            
            pred_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            fig = px.scatter(
                pred_df,
                x='Actual',
                y='Predicted',
                title="Predicted vs Actual Views",
                labels={'Actual': 'Actual Views', 'Predicted': 'Predicted Views'}
            )
            
            # Add perfect prediction line
            max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.markdown("### Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Which Factors Most Influence Views?"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # FUTURE VIEW PREDICTION
            st.markdown("### 📊 Predict Views for Your Next Video")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pred_duration = st.number_input("Video Duration (minutes)", min_value=1, max_value=120, value=10)
                pred_title_length = st.number_input("Title Length", min_value=10, max_value=100, value=50)
            
            with col2:
                pred_hour = st.selectbox("Upload Hour (24h format)", range(24), index=14)
                pred_day = st.selectbox("Upload Day", list(day_mapping.keys()), index=1)
            
            with col3:
                st.markdown("##### Prediction")
                if st.button("🔮 Predict Views", use_container_width=True):
                    # Create feature vector
                    pred_features = pd.DataFrame({
                        'duration_minutes': [pred_duration],
                        'title_length': [pred_title_length],
                        'publish_hour': [pred_hour],
                        'video_age_days': [0],  # New video
                        'publish_day_num': [day_mapping[pred_day]]
                    })
                    
                    # Scale and predict
                    pred_features_scaled = scaler.transform(pred_features)
                    predicted_views = model.predict(pred_features_scaled)[0]
                    
                    st.success(f"### 🎯 Predicted Views")
                    st.markdown(f"## {format_millions(predicted_views)}")
                    
                    # Confidence range (±15%)
                    lower = predicted_views * 0.85
                    upper = predicted_views * 1.15
                    st.caption(f"Confidence Range: {format_millions(lower)} - {format_millions(upper)}")


# ============================================================================
# TAB 6: POSTING PATTERNS
# ============================================================================

with tabs[5]:
    st.subheader("⏰ Posting Pattern Analysis")
    
    if selected_channel != "All Channels":
        channel_data = yt_data_filtered[yt_data_filtered['channel_name'] == selected_channel]
    else:
        channel_data = yt_data_filtered
    
    # Day of week analysis
    st.markdown("### Best Days to Post")
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_perf = channel_data.groupby('publish_day').agg({
        'views': 'mean',
        'engagement_rate': 'mean',
        'video_id': 'count'
    }).reindex(day_order).reset_index()
    day_perf.columns = ['Day', 'Avg Views', 'Avg Engagement', 'Video Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            day_perf,
            x='Day',
            y='Avg Views',
            title="Average Views by Day of Week",
            color='Avg Views',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            day_perf,
            x='Day',
            y='Avg Engagement',
            markers=True,
            title="Average Engagement by Day of Week"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    best_day = day_perf.loc[day_perf['Avg Views'].idxmax(), 'Day']
    st.success(f"💡 **Best day to post:** {best_day} with average {format_millions(day_perf['Avg Views'].max())} views")
    
    st.markdown("---")
    
    # Hour of day analysis
    st.markdown("### Best Time to Post")
    
    hour_perf = channel_data.groupby('publish_hour').agg({
        'views': 'mean',
        'engagement_rate': 'mean',
        'video_id': 'count'
    }).reset_index()
    hour_perf.columns = ['Hour', 'Avg Views', 'Avg Engagement', 'Video Count']
    
    fig = px.line(
        hour_perf,
        x='Hour',
        y='Avg Views',
        markers=True,
        title="Average Views by Upload Hour (24h format)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    best_hour = hour_perf.loc[hour_perf['Avg Views'].idxmax(), 'Hour']
    st.success(f"💡 **Best time to post:** {best_hour:02d}:00 (24h format) with average {format_millions(hour_perf['Avg Views'].max())} views")
    
    # Heatmap of day x hour
    st.markdown("### Posting Heatmap (Day × Hour)")
    
    heatmap_data = channel_data.pivot_table(
        values='views',
        index='publish_day',
        columns='publish_hour',
        aggfunc='mean'
    ).reindex(day_order)
    
    fig = px.imshow(
        heatmap_data,
        title="Average Views by Day and Hour",
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Views"),
        aspect='auto',
        color_continuous_scale='YlOrRd'
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 7: TOP CONTENT
# ============================================================================

with tabs[6]:
    st.subheader("🏆 Top Performing Content")
    
    if selected_channel != "All Channels":
        channel_data = yt_data_filtered[yt_data_filtered['channel_name'] == selected_channel]
    else:
        channel_data = yt_data_filtered
    
    # Top videos by views
    st.markdown("### 🔥 Most Viewed Videos")
    top_views = channel_data.nlargest(10, 'views')[
        ['title', 'views', 'likes', 'comments', 'engagement_rate', 'publish_date', 'duration_minutes']
    ].copy()
    top_views['publish_date'] = top_views['publish_date'].dt.strftime('%Y-%m-%d')
    top_views['views'] = top_views['views'].apply(format_millions)
    top_views['likes'] = top_views['likes'].apply(format_millions)
    top_views['engagement_rate'] = top_views['engagement_rate'].round(2)
    top_views['duration_minutes'] = top_views['duration_minutes'].round(1)
    
    st.dataframe(top_views, use_container_width=True)
    
    st.markdown("---")
    
    # Top videos by engagement
    st.markdown("### 💎 Highest Engagement Videos")
    top_engagement = channel_data.nlargest(10, 'engagement_rate')[
        ['title', 'engagement_rate', 'views', 'likes', 'comments', 'publish_date']
    ].copy()
    top_engagement['publish_date'] = top_engagement['publish_date'].dt.strftime('%Y-%m-%d')
    top_engagement['views'] = top_engagement['views'].apply(format_millions)
    top_engagement['engagement_rate'] = top_engagement['engagement_rate'].round(2)
    
    st.dataframe(top_engagement, use_container_width=True)
    
    st.markdown("---")
    
    # Viral videos
    st.markdown("### 🚀 Viral Content (Top 25% Performers)")
    viral_videos = channel_data[channel_data['performance_category'] == 'Viral'].nlargest(10, 'views')[
        ['title', 'views', 'likes', 'engagement_rate', 'views_per_day', 'publish_date']
    ].copy()
    
    if len(viral_videos) > 0:
        viral_videos['publish_date'] = viral_videos['publish_date'].dt.strftime('%Y-%m-%d')
        viral_videos['views'] = viral_videos['views'].apply(format_millions)
        viral_videos['views_per_day'] = viral_videos['views_per_day'].apply(format_millions)
        viral_videos['engagement_rate'] = viral_videos['engagement_rate'].round(2)
        
        st.dataframe(viral_videos, use_container_width=True)
    else:
        st.info("No viral videos found in the current filter")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📊 YouTube Analytics Dashboard - Enhanced Edition</p>
    <p>Built with Streamlit • Powered by YouTube Data API v3</p>
</div>
""", unsafe_allow_html=True)
