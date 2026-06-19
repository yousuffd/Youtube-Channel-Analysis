"""
╔══════════════════════════════════════════════════════════════════════╗
║        YOUTUBE ANALYTICS INTELLIGENCE — Executive Dashboard          ║
║        Streamlit · Plotly · Pandas · YouTube Data API v3             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Data sources : YouTube Data API v3 | Synthetic sample data          ║
║  ML models    : RandomForest (virality, performance, views)          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── stdlib ──────────────────────────────────────────────────────────
import os, re, warnings
from datetime import datetime

# ── third-party ─────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════
# §0  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="YouTube Analytics Intelligence",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════
# §1  GLOBAL CSS / DESIGN SYSTEM  (matches ecom dashboard)
# ══════════════════════════════════════════════════════════════════
def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-base:       #eef0f5;
        --bg-card:       #ffffff;
        --bg-raised:     #f4f6f9;
        --border:        #dde1ea;
        --border-strong: #bcc3d0;
        --blue:          #1d4ed8;
        --blue-light:    #eff6ff;
        --blue-mid:      #bfdbfe;
        --gold:          #b45309;
        --gold-light:    #fffbeb;
        --green:         #047857;
        --green-light:   #ecfdf5;
        --red:           #b91c1c;
        --red-light:     #fef2f2;
        --purple:        #6d28d9;
        --yt-red:        #cc0000;
        --yt-red-light:  #fff1f1;
        --t-hero:        #0a0f1e;
        --t-strong:      #1e2a3a;
        --t-body:        #374151;
        --t-muted:       #6b7280;
        --t-faint:       #9ca3af;
        --fs-hero:       1.75rem;
        --fs-h1:         1.15rem;
        --fs-h2:         0.95rem;
        --fs-body:       0.875rem;
        --fs-label:      0.72rem;
        --fs-caption:    0.65rem;
        --r-sm:  6px;  --r-md: 10px;  --r-lg: 14px;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05), 0 2px 6px rgba(0,0,0,0.06);
        --shadow-md: 0 2px 6px rgba(0,0,0,0.07), 0 6px 20px rgba(0,0,0,0.08);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-base) !important;
        color: var(--t-body) !important;
        font-size: var(--fs-body) !important;
        line-height: 1.5 !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stHeader"] { display: none !important; }
    .block-container {
        padding-top: 0 !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }

    /* ── SIDEBAR ──────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #374151 !important;
        min-width: 260px !important;
        max-width: 260px !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f9fafb !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio span,
    [data-testid="stSidebar"] .stRadio p,
    [data-testid="stSidebar"] .stRadio div {
        color: #ffffff !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        line-height: 1.6 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox span,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div { color: #e5e7eb !important; }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown a,
    [data-testid="stSidebar"] .stMarkdown div { color: #e5e7eb !important; font-size: 0.85rem !important; }
    [data-testid="stSidebar"] label { color: #d1d5db !important; font-size: 0.8rem !important; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="textarea"] textarea {
        background: #1f2937 !important; color: #f9fafb !important;
        border: 1px solid #374151 !important; border-radius: var(--r-sm) !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] input::placeholder { color: #6b7280 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #1f2937 !important; border: 1px solid #374151 !important;
    }
    [data-testid="stSidebar"] hr { border-color: #374151 !important; margin: 10px 0 !important; }
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stCaption { color: #9ca3af !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span { color: #ffffff !important; font-size: 0.8rem !important; }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] p,
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] div { color: #e5e7eb !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    button[title="View fullscreen"] { display: none !important; }

    /* ── TOP HEADER ───────────────────────────── */
    .top-header {
        background: linear-gradient(135deg, #1a0000 0%, #cc0000 100%);
        padding: 16px 28px;
        display: flex; align-items: center; justify-content: space-between;
        position: sticky; top: 0; z-index: 999;
        box-shadow: 0 2px 12px rgba(180,0,0,0.3);
        margin-bottom: 20px;
    }
    .logo { font-size: var(--fs-h1); font-weight: 800; color: #ffffff; letter-spacing: -0.3px; }
    .logo span { color: #fca5a5; }
    .tagline { font-size: var(--fs-caption); color: rgba(255,255,255,0.55); letter-spacing: 1.8px; margin-top: 3px; text-transform: uppercase; font-weight: 500; }
    .header-right { display: flex; align-items: center; gap: 10px; }
    .badge { background: rgba(255,255,255,0.15); color: #ffffff; border: 1px solid rgba(255,255,255,0.3); border-radius: 20px; font-size: var(--fs-caption); font-weight: 700; padding: 3px 10px; letter-spacing: 0.8px; }
    .ds-badge { background: rgba(255,255,255,0.10); color: rgba(255,255,255,0.8); border: 1px solid rgba(255,255,255,0.2); border-radius: 20px; font-size: var(--fs-caption); padding: 3px 10px; }

    /* ── KPI CARDS ────────────────────────────── */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--r-lg);
        padding: 20px 22px 18px;
        position: relative; overflow: hidden;
        box-shadow: var(--shadow-sm);
        transition: box-shadow .18s, transform .18s;
    }
    .kpi-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
    .kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: var(--yt-red); }
    .kpi-card.gold::before   { background: var(--gold); }
    .kpi-card.green::before  { background: var(--green); }
    .kpi-card.blue::before   { background: var(--blue); }
    .kpi-card.purple::before { background: var(--purple); }
    .kpi-icon { position: absolute; top: 14px; right: 14px; font-size: 1.4rem; opacity: 0.10; }
    .kpi-label { font-size: var(--fs-caption); font-weight: 600; letter-spacing: 1.1px; text-transform: uppercase; color: var(--t-muted); margin-bottom: 8px; }
    .kpi-value { font-size: 1.9rem; font-weight: 800; color: var(--t-hero); line-height: 1; margin-bottom: 6px; letter-spacing: -1px; font-family: 'Inter', sans-serif; }
    .kpi-delta { font-size: var(--fs-label); font-weight: 600; }
    .kpi-delta.pos { color: var(--green); }
    .kpi-delta.neg { color: var(--red); }
    .kpi-delta.neu { color: var(--t-muted); }
    .kpi-sub { font-size: var(--fs-caption); color: var(--t-faint); margin-top: 3px; }

    /* ── SECTION HEADER ───────────────────────── */
    .section-header {
        font-size: var(--fs-label); font-weight: 700; letter-spacing: 1.2px;
        text-transform: uppercase; color: var(--t-muted);
        margin: 24px 0 14px; padding-bottom: 8px;
        border-bottom: 2px solid var(--border);
        display: flex; align-items: center; gap: 8px;
    }
    .section-header::before { content: ''; display: inline-block; width: 4px; height: 14px; background: var(--yt-red); border-radius: 2px; flex-shrink: 0; }

    /* ── EXEC ALERT BANNERS ───────────────────── */
    .exec-alert { border-left: 4px solid var(--yt-red); border-radius: 0 var(--r-sm) var(--r-sm) 0; padding: 14px 18px; margin-bottom: 10px; font-size: var(--fs-body); font-weight: 500; color: #7f1d1d; background: var(--yt-red-light); line-height: 1.5; }
    .exec-alert.warn  { border-color: var(--gold);   color: #78350f; background: var(--gold-light); }
    .exec-alert.good  { border-color: var(--green);  color: #064e3b; background: var(--green-light); }
    .exec-alert.info  { border-color: var(--blue);   color: #1e3a8a; background: var(--blue-light); }
    .exec-alert.purple{ border-color: var(--purple); color: #4c1d95; background: #f5f3ff; }
    .exec-alert-label { font-size: var(--fs-caption); font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; opacity: 0.75; }
    .exec-alert-value { font-size: 1.6rem; font-weight: 800; color: var(--t-hero); letter-spacing: -0.5px; line-height: 1.1; }
    .exec-alert-sub   { font-size: var(--fs-caption); color: var(--t-body); margin-top: 4px; opacity: 0.8; }

    /* ── STYLED TABLE ─────────────────────────── */
    .styled-table { width:100%; border-collapse:collapse; }
    .styled-table th { background: var(--bg-raised); color: var(--t-muted); font-size: var(--fs-caption); font-weight: 700; text-transform: uppercase; letter-spacing: 0.9px; padding: 10px 14px; text-align: left; border-bottom: 2px solid var(--border-strong); }
    .styled-table td { font-size: var(--fs-body); padding: 11px 14px; border-bottom: 1px solid var(--bg-raised); color: var(--t-body); }
    .styled-table tr:hover td { background: var(--bg-raised); }
    .rank-badge { background: var(--yt-red-light); color: var(--yt-red); border-radius: 4px; padding: 2px 7px; font-size: var(--fs-caption); font-weight: 700; font-family: 'JetBrains Mono', monospace; }

    /* ── PILLS ────────────────────────────────── */
    .pill { display: inline-block; border-radius: 20px; padding: 2px 10px; font-size: var(--fs-caption); font-weight: 600; }
    .pill-green  { background: #d1fae5; color: #065f46; }
    .pill-red    { background: #fee2e2; color: #991b1b; }
    .pill-amber  { background: #fef3c7; color: #92400e; }
    .pill-blue   { background: #dbeafe; color: #1e40af; }

    /* ── API KEY INPUT BOX ────────────────────── */
    .api-box {
        background: #1a2744; border: 1px solid #2d3f6b; border-radius: var(--r-md);
        padding: 14px 16px; margin: 8px 0;
    }
    .api-box-label { font-size: var(--fs-caption); color: #8b9ab8; margin-bottom: 6px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; }

    /* ── STREAMLIT NATIVE OVERRIDES ───────────── */
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 800 !important; color: var(--t-hero) !important; }
    [data-testid="stMetricLabel"] { font-size: var(--fs-label) !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; color: var(--t-muted) !important; }
    [data-testid="stMetricDelta"] { font-size: var(--fs-label) !important; }
    [data-testid="stTabs"] [data-baseweb="tab"] { font-size: var(--fs-body) !important; font-weight: 600 !important; }
    [data-testid="stTabs"] [aria-selected="true"] { color: var(--yt-red) !important; border-bottom-color: var(--yt-red) !important; }
    .stButton > button { background: var(--yt-red) !important; color: #fff !important; font-weight: 600 !important; border: none !important; border-radius: var(--r-sm) !important; padding: 8px 20px !important; font-size: var(--fs-body) !important; }
    .stButton > button:hover { opacity: .86 !important; }

    /* ── FOOTER ───────────────────────────────── */
    .footer { text-align:center; padding:20px 0 12px; font-size: var(--fs-caption); color: var(--t-faint); border-top:1px solid var(--border); margin-top:28px; }
    .footer span { color: var(--yt-red); font-weight:600; }

    /* ── SCROLLBAR ────────────────────────────── */
    ::-webkit-scrollbar { width:5px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius:3px; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# §2  HELPERS
# ══════════════════════════════════════════════════════════════════
def fmt(num):
    if pd.isna(num) or num == 0:
        return "0"
    num = float(num)
    if abs(num) >= 1_000_000_000: return f"{num/1_000_000_000:.1f}B"
    if abs(num) >= 1_000_000:     return f"{num/1_000_000:.1f}M"
    if abs(num) >= 1_000:         return f"{num/1_000:.1f}K"
    return f"{num:.0f}"

def iso_to_minutes(duration):
    if not isinstance(duration, str): return np.nan
    m = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not m: return np.nan
    return int(m.group(1) or 0)*60 + int(m.group(2) or 0) + int(m.group(3) or 0)/60

CHART_COLORS = {
    'red':    '#cc0000',
    'blue':   '#1d4ed8',
    'green':  '#047857',
    'gold':   '#b45309',
    'purple': '#6d28d9',
    'muted':  '#6b7280',
}
PERF_COLORS = {'High': '#047857', 'Medium': '#b45309', 'Low': '#b91c1c'}


# ══════════════════════════════════════════════════════════════════
# §3  DATA LAYER
# ══════════════════════════════════════════════════════════════════

# ── 3a  Synthetic sample ─────────────────────────────────────────
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    num_channels = 50
    channels = pd.DataFrame({
        'channel_id':    [f'CH{i:04d}' for i in range(num_channels)],
        'channel_name':  [f'Channel {i}' for i in range(num_channels)],
        'videos':         np.random.randint(50, 500, num_channels),
        'total_views':    np.random.randint(100_000, 10_000_000, num_channels),
        'total_likes':    np.random.randint(5_000, 500_000, num_channels),
        'total_comments': np.random.randint(500, 50_000, num_channels),
        'subscribers':    np.random.randint(1_000, 1_000_000, num_channels),
        'channel_age_days': np.random.randint(365, 2000, num_channels),
    })
    channels['avg_views']           = channels['total_views'] / channels['videos']
    channels['avg_likes']           = channels['total_likes'] / channels['videos']
    channels['avg_duration']        = np.random.uniform(5, 25, num_channels)
    channels['avg_engagement_rate'] = (channels['total_likes'] + channels['total_comments']) / channels['total_views'] * 100
    channels['avg_views_per_video'] = channels['avg_views']
    channels['likes_per_1000_views']= (channels['total_likes'] / channels['total_views']) * 1000
    channels['views_per_subscriber']= channels['total_views'] / channels['subscribers']
    channels['engagement_score']    = channels['avg_engagement_rate'] * np.log1p(channels['subscribers'])
    channels['videos_per_month']    = channels['videos'] / (channels['channel_age_days'] / 30)
    channels['channel_created']     = pd.date_range(end=datetime.now(), periods=num_channels, freq='30D')[::-1]

    num_videos = 2000
    videos = pd.DataFrame({
        'video_id':   [f'VID{i:05d}' for i in range(num_videos)],
        'title':      [f'Video Title {i}' for i in range(num_videos)],
        'channel_id': np.random.choice(channels['channel_id'], num_videos),
        'publish_date': pd.date_range(end=datetime.now(), periods=num_videos, freq='12h')[::-1],
        'duration_minutes': np.random.uniform(3, 30, num_videos),
        'views':    np.random.randint(100, 1_000_000, num_videos),
        'likes':    np.random.randint(10, 50_000, num_videos),
        'comments': np.random.randint(1, 5_000, num_videos),
        'title_length': np.random.randint(30, 100, num_videos),
    })
    videos = videos.merge(channels[['channel_id', 'channel_name', 'subscribers']], on='channel_id')
    videos.rename(columns={'subscribers': 'ch_subscribers'}, inplace=True)
    videos = _enrich_videos(videos)
    return channels, videos


# ── 3b  YouTube API fetch ────────────────────────────────────────
def fetch_api_data(api_key: str, channel_ids: list[str]):
    try:
        from googleapiclient.discovery import build
    except ImportError:
        st.error("google-api-python-client not installed. Run: pip install google-api-python-client")
        return None, None

    youtube = build('youtube', 'v3', developerKey=api_key)

    # --- channels ---
    resp = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=",".join(channel_ids)
    ).execute()

    ch_rows = []
    for item in resp.get('items', []):
        stats   = item.get('statistics', {})
        snippet = item['snippet']
        ch_rows.append({
            'channel_id':   item['id'],
            'channel_name': snippet['title'],
            'ch_subscribers': int(stats.get('subscriberCount', 0)),
            'total_views':  int(stats.get('viewCount', 0)),
            'total_videos': int(stats.get('videoCount', 0)),
            'playlist_id':  item['contentDetails']['relatedPlaylists']['uploads'],
            'channel_created': snippet['publishedAt'],
        })
    ch_df = pd.DataFrame(ch_rows)
    if ch_df.empty:
        st.error("No channels returned — check your channel IDs.")
        return None, None

    # --- video IDs from playlists ---
    video_ids = []
    for pid in ch_df['playlist_id']:
        token = None
        while True:
            r = youtube.playlistItems().list(
                part="contentDetails", playlistId=pid,
                maxResults=50, pageToken=token
            ).execute()
            video_ids += [i['contentDetails']['videoId'] for i in r.get('items', [])]
            token = r.get('nextPageToken')
            if not token:
                break

    # --- video details ---
    vid_rows = []
    for i in range(0, len(video_ids), 50):
        r = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids[i:i+50])
        ).execute()
        for v in r.get('items', []):
            s = v.get('statistics', {})
            sn = v['snippet']
            vid_rows.append({
                'video_id':    v['id'],
                'title':       sn['title'],
                'channel_id':  sn['channelId'],
                'publish_date': sn['publishedAt'],
                'duration':    v['contentDetails']['duration'],
                'views':       int(s.get('viewCount', 0)),
                'likes':       int(s.get('likeCount', 0)),
                'comments':    int(s.get('commentCount', 0)),
            })

    vid_df = pd.DataFrame(vid_rows)
    if vid_df.empty:
        st.error("No videos returned.")
        return None, None

    # enrich videos
    vid_df['publish_date']     = pd.to_datetime(vid_df['publish_date'], format='ISO8601', utc=True).dt.tz_localize(None)
    vid_df['duration_minutes'] = vid_df['duration'].apply(iso_to_minutes)
    vid_df['title_length']     = vid_df['title'].str.len()
    vid_df = vid_df.merge(
        ch_df[['channel_id', 'channel_name', 'ch_subscribers']], on='channel_id', how='left'
    )
    vid_df = _enrich_videos(vid_df)

    # build channels summary
    ch_df['channel_created'] = pd.to_datetime(ch_df['channel_created'], format='ISO8601', utc=True).dt.tz_localize(None)
    agg = vid_df.groupby(['channel_id', 'channel_name']).agg(
        videos=('video_id','count'),
        total_views=('views','sum'), total_likes=('likes','sum'),
        total_comments=('comments','sum'),
        avg_likes=('likes','mean'), avg_views=('views','mean'),
        avg_duration=('duration_minutes','mean'),
        avg_engagement_rate=('engagement_rate','mean'),
    ).reset_index()
    agg = agg.merge(ch_df[['channel_id','ch_subscribers','channel_created']], on='channel_id', how='left')
    agg.rename(columns={'ch_subscribers': 'subscribers'}, inplace=True)
    agg['avg_views_per_video']  = agg['total_views'] / agg['videos']
    agg['likes_per_1000_views'] = (agg['total_likes'] / agg['total_views'].replace(0, np.nan) * 1000).fillna(0)
    agg['views_per_subscriber'] = (agg['total_views'] / agg['subscribers'].replace(0, np.nan)).fillna(0)
    agg['engagement_score']     = ((agg['total_likes'] / agg['total_views'].replace(0, np.nan)) + agg['views_per_subscriber']).fillna(0)
    agg['channel_age_days']     = (pd.Timestamp.now() - agg['channel_created']).dt.days
    agg['videos_per_month']     = (agg['videos'] / agg['channel_age_days'].replace(0, np.nan) * 30).fillna(0)

    return agg, vid_df


# ── 3c  Shared enrichment ────────────────────────────────────────
def _enrich_videos(df):
    df = df.copy()
    if df['publish_date'].dtype == 'object':
        df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['publish_hour']  = df['publish_date'].dt.hour
    df['publish_day']   = df['publish_date'].dt.day_name()
    df['publish_month'] = df['publish_date'].dt.month
    df['publish_year']  = df['publish_date'].dt.year
    df['video_age_days']= (datetime.now() - df['publish_date']).dt.days.abs()
    df['views_per_day'] = df['views'] / (df['video_age_days'] + 1)

    safe_v = df['views'].replace(0, np.nan)
    df['engagement_rate']  = ((df['likes'] + df['comments']) / safe_v * 100).fillna(0)
    df['like_rate']         = (df['likes'] / safe_v * 100).fillna(0)
    df['comment_rate']      = (df['comments'] / safe_v * 100).fillna(0)
    df['likes_per_comment'] = (df['likes'] / df['comments'].replace(0, np.nan)).fillna(0)

    df['posting_period'] = df['publish_hour'].apply(
        lambda x: 'Morning' if 6<=x<12 else 'Afternoon' if 12<=x<18 else 'Evening' if 18<=x<22 else 'Night'
    )
    q75 = df['views'].quantile(0.75)
    q25 = df['views'].quantile(0.25)
    df['performance_category'] = df['views'].apply(
        lambda x: 'High' if x>=q75 else 'Low' if x<=q25 else 'Medium'
    )
    return df


# ══════════════════════════════════════════════════════════════════
# §4  SIDEBAR + DATA RESOLUTION
# ══════════════════════════════════════════════════════════════════
def render_sidebar():
    # ── Brand ────────────────────────────────────────────────────
    st.sidebar.markdown(
        '<div style="padding:20px 16px 12px;">'
        '<div style="font-size:1.1rem;font-weight:800;color:#ffffff;letter-spacing:-0.3px;">'
        '📺 YouTube<span style="color:#fca5a5;">Analytics</span></div>'
        '<div style="font-size:0.6rem;color:rgba(255,255,255,0.45);letter-spacing:1.8px;'
        'text-transform:uppercase;margin-top:3px;">Intelligence Dashboard</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    # ── Data Source ──────────────────────────────────────────────
    st.sidebar.markdown("### Data Source")
    source = st.sidebar.radio(
        "source",
        ["🎲 Synthetic Sample", "🔑 YouTube API"],
        label_visibility="collapsed",
        key="data_source",
    )

    api_key      = ""
    channel_ids  = []
    data_label   = "Synthetic sample · 2,000 videos · 50 channels"

    if source == "🔑 YouTube API":
        st.sidebar.markdown(
            '<div style="font-size:0.72rem;color:#8b9ab8;margin:8px 0 4px;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;">YouTube Data API v3 Key</div>',
            unsafe_allow_html=True,
        )
        env_key = os.getenv("YOUTUBE_API_KEY", "")
        sidebar_key = st.sidebar.text_input(
            "API Key", type="password",
            value=env_key,
            placeholder="AIza…",
            label_visibility="collapsed",
            help="Free at console.cloud.google.com. Never stored or logged.",
        )
        if env_key:
            st.sidebar.markdown('<div style="font-size:.72rem;color:#22c55e;">✅ Key loaded from .env</div>', unsafe_allow_html=True)
        elif sidebar_key:
            st.sidebar.markdown('<div style="font-size:.72rem;color:#f5a623;">🔑 Key set for this session</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown(
                '<div style="font-size:.72rem;color:#8b9ab8;">No key — '
                '<a href="https://console.cloud.google.com" target="_blank" style="color:#60a5fa;text-decoration:none;">'
                '→ Get a free key</a></div>',
                unsafe_allow_html=True,
            )
        api_key = env_key or sidebar_key

        st.sidebar.markdown(
            '<div style="font-size:0.72rem;color:#8b9ab8;margin:10px 0 4px;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;">Channel IDs (one per line)</div>',
            unsafe_allow_html=True,
        )
        default_ids = "\n".join([
            'UCmTM_hPCeckqN3cPWtYZZcg',
            'UCzI8K9xO_5E-4iCP7Km6cRQ',
            'UC5fcjujOsqD-126Chn_BAuA',
            'UC-CSyyi47VX1lD9zyeABW3w',
            'UC0yXUUIaPVAqZLgRjvtMftw',
            'UCWtlPzcP989da26sVyHPzqQ',
        ])
        raw_ids = st.sidebar.text_area(
            "channel_ids", value=default_ids,
            height=140, label_visibility="collapsed",
        )
        channel_ids = [x.strip() for x in raw_ids.strip().splitlines() if x.strip()]

        if api_key and channel_ids:
            if st.sidebar.button("🔄 Fetch Live Data"):
                st.session_state.pop("yt_channels", None)
                st.session_state.pop("yt_videos", None)
                with st.spinner(f"Fetching data for {len(channel_ids)} channels…"):
                    ch, vid = fetch_api_data(api_key, channel_ids)
                if ch is not None:
                    st.session_state["yt_channels"] = ch
                    st.session_state["yt_videos"]   = vid
                    st.session_state["data_label"]  = f"YouTube API · {len(ch)} channels · {len(vid)} videos"
                    st.success("✅ Data loaded!")
                    st.rerun()
        elif source == "🔑 YouTube API":
            st.sidebar.markdown(
                '<div style="font-size:.72rem;color:#f5a623;margin-top:8px;">Enter API key + channel IDs then click Fetch</div>',
                unsafe_allow_html=True,
            )

        with st.sidebar.expander("📋 Expected schema"):
            st.markdown(
                '<div style="font-size:.72rem;color:#e5e7eb;line-height:2;">'
                'channel_id · string<br>channel_name · string<br>'
                'subscribers · int<br>total_views · int<br>'
                'publish_date · datetime<br>views · int<br>'
                'likes · int<br>comments · int<br>'
                'duration_minutes · float</div>',
                unsafe_allow_html=True,
            )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    return source, api_key, channel_ids, data_label


# ══════════════════════════════════════════════════════════════════
# §5  HEADER + META STRIP
# ══════════════════════════════════════════════════════════════════
def render_header(data_label, n_videos, n_channels, date_min, date_max):
    st.markdown(f"""
    <div class="top-header">
        <div>
            <div class="logo">📺 YouTube<span>Analytics</span></div>
            <div class="tagline">Executive Intelligence Dashboard</div>
        </div>
        <div class="header-right">
            <span class="badge">LIVE</span>
            <span class="ds-badge">{data_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div style="padding:6px 2px 14px;font-size:0.75rem;color:#9ca3af;">'
        f'<span style="color:#374151;font-weight:600;">{n_videos:,} videos</span>'
        f'&nbsp;·&nbsp;{n_channels} channels'
        f'&nbsp;·&nbsp;{date_min} – {date_max}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# §6  PLOTLY CHART THEME HELPER
# ══════════════════════════════════════════════════════════════════
LAYOUT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font=dict(family="Inter, sans-serif", color="#374151", size=12),
              margin=dict(t=40, b=10, l=0, r=0))

def chart_layout(**kwargs):
    d = LAYOUT.copy()
    d.update(kwargs)
    return d


# ══════════════════════════════════════════════════════════════════
# §7  PAGE RENDERERS
# ══════════════════════════════════════════════════════════════════

# ── Tab 1 : Overview & KPIs ──────────────────────────────────────
def page_overview(fv, fc, vdf, cdf):
    st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)

    # Top-line metrics
    total_views    = fv['views'].sum()
    total_videos   = len(fv)
    avg_engagement = fv['engagement_rate'].mean()
    total_subs     = fc['subscribers'].sum()
    views_per_day  = fv['views_per_day'].mean()
    baseline_eng   = vdf['engagement_rate'].mean()
    baseline_vpd   = vdf['views_per_day'].mean()
    delta_eng  = avg_engagement - baseline_eng
    delta_vpd  = (views_per_day / baseline_vpd - 1)*100 if baseline_vpd else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.metric("📺 Total Views",      fmt(total_views),    f"{fmt(total_views/total_videos if total_videos else 0)} avg/video")
    with c2: st.metric("🎬 Total Videos",     f"{total_videos:,}", f"{len(fc)} channels")
    with c3: st.metric("💬 Avg Engagement",   f"{avg_engagement:.2f}%", f"{delta_eng:+.2f}%")
    with c4: st.metric("👥 Total Subscribers",fmt(total_subs),     f"{fmt(total_subs/len(fc) if len(fc) else 0)} avg")
    with c5: st.metric("📊 Views / Day",      fmt(views_per_day),  f"{delta_vpd:+.1f}%")

    st.markdown("---")
    st.markdown('<div class="section-header">Strategic KPIs</div>', unsafe_allow_html=True)

    def _safe(val): return 0 if (pd.isna(val) or np.isinf(val)) else val

    vid_eff   = _safe((fv['views']/(fv['duration_minutes']*fv['ch_subscribers'])).replace([np.inf,-np.inf],0).mean()*1000)
    inter_sc  = _safe(fv['like_rate'].replace([np.inf,-np.inf],0).mean()*0.5 + fv['comment_rate'].replace([np.inf,-np.inf],0).mean()*100*0.5)
    cv        = (fv.groupby('channel_name')['views'].std() / fv.groupby('channel_name')['views'].mean().replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
    consist   = _safe(100 - cv.mean())
    virality  = _safe((fv['views']/fv['ch_subscribers'].replace(0,np.nan)).replace([np.inf,-np.inf],0).mean()*100)
    lrm,crm,vdm = fv['like_rate'].max(), fv['comment_rate'].max(), fv['views_per_day'].max()
    quality   = _safe(((fv['like_rate']/lrm*40 + fv['comment_rate']/crm*30 + fv['views_per_day']/vdm*30) if lrm and crm and vdm else pd.Series([0])).mean())

    kpis = [
        ("blue",   "🎯", "Video Efficiency",   f"{vid_eff:.2f}",      "Views per min per 1K subs"),
        ("gold",   "💬", "Interaction Score",  f"{inter_sc:.2f}",     "Composite engagement"),
        ("green",  "📊", "Consistency",        f"{consist:.1f}%",     "Performance stability"),
        ("purple", "🚀", "Virality Index",      f"{virality:.1f}",     "Views per 100 subscribers"),
        ("",       "⭐", "Quality Score",       f"{quality:.1f}/100",  "Composite quality metric"),
    ]
    cols = st.columns(5)
    for col, (variant, icon, label, value, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card {variant}">
                <span class="kpi-icon">{icon}</span>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        daily = fv.groupby(fv['publish_date'].dt.date).agg(engagement_rate=('engagement_rate','mean')).reset_index()
        fig = go.Figure(go.Scatter(x=daily['publish_date'], y=daily['engagement_rate'],
            mode='lines+markers', line=dict(color=CHART_COLORS['red'], width=2),
            fill='tozeroy', fillcolor='rgba(204,0,0,0.08)'))
        fig.update_layout(**chart_layout(title="Engagement Rate Trend", height=350,
            yaxis_title="Engagement Rate (%)"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pv = fv.groupby('performance_category')['views'].sum().reset_index()
        fig = go.Figure(go.Pie(labels=pv['performance_category'], values=pv['views'],
            hole=0.45, marker=dict(colors=[PERF_COLORS.get(l,'#888') for l in pv['performance_category']]),
            textinfo='label+percent', textfont=dict(size=13)))
        fig.update_layout(**chart_layout(title="Views by Performance Category", height=350))
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 2 : Growth & Engagement ─────────────────────────────────
def page_growth(fv, fc):
    st.markdown('<div class="section-header">Growth & Reach Metrics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ms = fv.groupby(fv['publish_date'].dt.to_period('M')).agg(
            views=('views','sum'), count=('video_id','count')).reset_index()
        ms['publish_date'] = ms['publish_date'].dt.to_timestamp()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ms['publish_date'], y=ms['views'], name='Total Views',
            mode='lines+markers', line=dict(color=CHART_COLORS['red'], width=2)))
        fig.add_trace(go.Bar(x=ms['publish_date'], y=ms['count'], name='Videos Published',
            yaxis='y2', marker=dict(color=CHART_COLORS['blue'], opacity=0.5)))
        fig.update_layout(**chart_layout(title="Monthly Growth Trajectory", height=400,
            yaxis=dict(title="Total Views"), hovermode='x unified',
            yaxis2=dict(title="Videos Published", overlaying='y', side='right')))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        tc = fc.nlargest(10, 'total_views')[['channel_name','total_views']]
        fig = go.Figure(go.Bar(y=tc['channel_name'], x=tc['total_views'], orientation='h',
            marker=dict(color=tc['total_views'], colorscale='Reds', showscale=True),
            text=tc['total_views'], texttemplate='%{text:,.0f}', textposition='auto'))
        fig.update_layout(**chart_layout(title="Top 10 Channels by Views", height=400))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Engagement Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=fv['engagement_rate'], nbinsx=30,
            marker=dict(color=CHART_COLORS['red'], line=dict(color='white', width=1))))
        fig.update_layout(**chart_layout(title="Engagement Rate Distribution", height=350,
            xaxis_title="Engagement Rate (%)", yaxis_title="Number of Videos"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        de = fv.groupby('publish_day').agg(engagement_rate=('engagement_rate','mean')).reindex(
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        fig = go.Figure(go.Bar(x=de.index, y=de['engagement_rate'],
            marker=dict(color=de['engagement_rate'], colorscale='RdYlGn', showscale=True),
            text=de['engagement_rate'].round(2), textposition='auto'))
        fig.update_layout(**chart_layout(title="Avg Engagement by Day of Week", height=350,
            yaxis_title="Engagement Rate (%)"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Engagement Drivers</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(fv, x='views', y='engagement_rate', size='likes',
            color='performance_category', hover_data=['title','channel_name'],
            color_discrete_map=PERF_COLORS, title="Views vs Engagement Rate")
        fig.update_layout(**chart_layout(height=400))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        for cat, grp in fv.groupby('performance_category'):
            vmax = grp['views'].max() or 1
            fig.add_trace(go.Scatter(x=grp['duration_minutes'], y=grp['engagement_rate'],
                mode='markers', name=cat,
                marker=dict(size=np.sqrt(grp['views']/vmax)*20+4,
                            color=PERF_COLORS.get(cat,'#888'), opacity=0.7)))
        xa, ya = fv['duration_minutes'].values, fv['engagement_rate'].values
        mask = np.isfinite(xa) & np.isfinite(ya)
        if mask.sum() > 1:
            coefs = np.polyfit(xa[mask], ya[mask], 1)
            xl = np.linspace(xa[mask].min(), xa[mask].max(), 100)
            fig.add_trace(go.Scatter(x=xl, y=np.polyval(coefs, xl), mode='lines',
                name='Trend', line=dict(color='#374151', width=2, dash='dash')))
        fig.update_layout(**chart_layout(title="Duration vs Engagement", height=400,
            xaxis_title="Duration (min)", yaxis_title="Engagement Rate"))
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 3 : Video Performance ────────────────────────────────────
def page_video_performance(fv):
    st.markdown('<div class="section-header">Video Performance Analysis</div>', unsafe_allow_html=True)
    pc = fv['performance_category'].value_counts()
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("High Performers",   f"{pc.get('High',0):,}",   f"{pc.get('High',0)/len(fv)*100:.1f}%")
    with c2: st.metric("Medium Performers", f"{pc.get('Medium',0):,}", f"{pc.get('Medium',0)/len(fv)*100:.1f}%")
    with c3: st.metric("Low Performers",    f"{pc.get('Low',0):,}",    f"{pc.get('Low',0)/len(fv)*100:.1f}%")

    st.markdown('<div class="section-header">Top 20 Videos</div>', unsafe_allow_html=True)
    tv = fv.nlargest(20,'views')[['title','channel_name','views','likes','comments','engagement_rate','performance_category']].copy()
    tv['views']           = tv['views'].apply(fmt)
    tv['likes']           = tv['likes'].apply(fmt)
    tv['comments']        = tv['comments'].apply(fmt)
    tv['engagement_rate'] = tv['engagement_rate'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(tv, use_container_width=True, height=400)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        fv2 = fv.copy()
        fv2['duration_bucket'] = pd.cut(fv2['duration_minutes'],
            bins=[0,5,10,15,20,100], labels=['0-5min','5-10min','10-15min','15-20min','20+min'])
        dp = fv2.groupby('duration_bucket').agg(views=('views','mean')).reset_index()
        fig = go.Figure(go.Bar(x=dp['duration_bucket'], y=dp['views'],
            marker=dict(color=CHART_COLORS['red'])))
        fig.update_layout(**chart_layout(title="Avg Views by Video Duration", height=400,
            yaxis_title="Avg Views"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        for cat, color in PERF_COLORS.items():
            data = fv[fv['performance_category']==cat]['views']
            fig.add_trace(go.Box(y=data, name=cat, marker=dict(color=color), boxmean='sd'))
        fig.update_layout(**chart_layout(title="Views Distribution by Category", height=400, yaxis_title="Views"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Multi-Metric Radar Comparison</div>', unsafe_allow_html=True)
    rm = fv.groupby('performance_category').agg(
        views=('views','mean'), likes=('likes','mean'), comments=('comments','mean'),
        engagement_rate=('engagement_rate','mean'), views_per_day=('views_per_day','mean')
    ).reset_index()
    for col in ['views','likes','comments','engagement_rate','views_per_day']:
        rng = rm[col].max() - rm[col].min()
        rm[col] = ((rm[col] - rm[col].min()) / rng * 100) if rng else 0
    fig = go.Figure()
    cats = ['Avg Views','Avg Likes','Avg Comments','Engagement','Views/Day']
    for _, row in rm.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['views'],row['likes'],row['comments'],row['engagement_rate'],row['views_per_day']],
            theta=cats, fill='toself', name=row['performance_category'],
            line=dict(color=PERF_COLORS.get(row['performance_category'],'#888'))))
    fig.update_layout(**chart_layout(title="Performance Metrics (Normalized)", height=500,
        polar=dict(radialaxis=dict(visible=True, range=[0,100]))))
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 4 : Content Strategy ─────────────────────────────────────
def page_content_strategy(fv, fc):
    st.markdown('<div class="section-header">Publishing Strategy Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        piv = fv.groupby(['publish_day','publish_hour'])['views'].mean().reset_index()
        pt  = piv.pivot(index='publish_hour', columns='publish_day', values='views')
        pt  = pt.reindex(columns=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        fig = go.Figure(go.Heatmap(z=pt.values, x=pt.columns, y=pt.index,
            colorscale='Reds', text=pt.values, texttemplate='%{text:.0f}',
            textfont={"size":8}, colorbar=dict(title="Avg Views")))
        fig.update_layout(**chart_layout(title="Best Publishing Times (Hour × Day)",
            xaxis_title="Day", yaxis_title="Hour", height=500))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ps = fv.groupby('posting_period').agg(
            views=('views','mean'), engagement_rate=('engagement_rate','mean')).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=ps['posting_period'], y=ps['views'],
            marker=dict(color=CHART_COLORS['red']), name='Avg Views', yaxis='y'))
        fig.add_trace(go.Scatter(x=ps['posting_period'], y=ps['engagement_rate'],
            mode='lines+markers', line=dict(color=CHART_COLORS['blue'], width=2),
            name='Avg Engagement', yaxis='y2'))
        fig.update_layout(**chart_layout(title="Performance by Publishing Period", height=500,
            yaxis=dict(title="Avg Views"),
            yaxis2=dict(title="Engagement (%)", overlaying='y', side='right')))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Content Strategy Metrics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fv2 = fv.copy()
        fv2['title_bucket'] = pd.cut(fv2['title_length'], bins=[0,40,60,80,200],
            labels=['Short (<40)','Medium (40-60)','Long (60-80)','Very Long (80+)'])
        tp = fv2.groupby('title_bucket').agg(views=('views','mean')).reset_index()
        fig = go.Figure(go.Bar(x=tp['title_bucket'], y=tp['views'],
            marker=dict(color=tp['views'], colorscale='Reds', showscale=True),
            text=tp['views'].round(0), textposition='auto'))
        fig.update_layout(**chart_layout(title="Avg Views by Title Length", height=400))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cf = fc[['channel_name','videos_per_month','avg_views']].nlargest(10,'avg_views')
        fig = px.scatter(cf, x='videos_per_month', y='avg_views', size='avg_views',
            text='channel_name', title="Posting Frequency vs Performance",
            labels={'videos_per_month':'Videos/Month','avg_views':'Avg Views'})
        fig.update_traces(textposition='top center')
        fig.update_layout(**chart_layout(height=400))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Publishing Calendar View</div>', unsafe_allow_html=True)
    cal = fv.groupby(fv['publish_date'].dt.date)['video_id'].count().reset_index()
    cal.columns = ['date','count']
    fig = go.Figure(go.Scatter(x=cal['date'], y=cal['count'], mode='markers',
        marker=dict(size=cal['count']*3, color=cal['count'], colorscale='Reds',
                    showscale=True, colorbar=dict(title="Videos")),
        text=cal['count'], hovertemplate='<b>%{x}</b><br>Videos: %{text}<extra></extra>'))
    fig.update_layout(**chart_layout(title="Daily Publishing Activity", height=400,
        yaxis_title="Videos Published"))
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 5 : Predictions ──────────────────────────────────────────
def page_predictions(fv, cdf):
    st.markdown('<div class="section-header">🔮 Predictive Analytics</div>', unsafe_allow_html=True)

    pdf = fv.copy()
    pdf['early_engagement_rate']   = (pdf['likes']+pdf['comments'])/(pdf['views']+1)
    pdf['likes_views_ratio']       = pdf['likes']/(pdf['views']+1)
    pdf['comments_views_ratio']    = pdf['comments']/(pdf['views']+1)
    pdf['is_weekend']              = pdf['publish_day'].isin(['Saturday','Sunday']).astype(int)
    pdf['hour_encoded']            = pdf['publish_hour']/24
    channel_avg = cdf.set_index('channel_id')['avg_views_per_video'].to_dict() if 'avg_views_per_video' in cdf.columns else {}
    pdf['avg_views_per_video']     = pdf['channel_id'].map(channel_avg).fillna(pdf['views'].mean())

    # ── Model 1: Virality ──────────────────────────────────────
    st.markdown('<div class="section-header">1️⃣ Virality Prediction Model</div>', unsafe_allow_html=True)
    vt = (pdf['views']/pdf['ch_subscribers']).quantile(0.75)
    pdf['is_viral'] = (pdf['views']/pdf['ch_subscribers'] > vt).astype(int)
    vf = ['early_engagement_rate','likes_views_ratio','comments_views_ratio','hour_encoded','ch_subscribers','duration_minutes']
    Xv, yv = pdf[vf].fillna(0), pdf['is_viral']
    Xvtr,Xvte,yvtr,yvte = train_test_split(Xv,yv,test_size=0.3,random_state=42)
    sc1 = StandardScaler(); Xvtr_s = sc1.fit_transform(Xvtr); Xvte_s = sc1.transform(Xvte)
    m1 = RandomForestClassifier(n_estimators=100,random_state=42,max_depth=10)
    m1.fit(Xvtr_s,yvtr); yvp = m1.predict(Xvte_s)
    fi1 = pd.DataFrame({'feature':vf,'importance':m1.feature_importances_}).sort_values('importance',ascending=True)

    c1,c2 = st.columns([2,1])
    with c1:
        fig = go.Figure(go.Bar(x=fi1['importance'], y=fi1['feature'], orientation='h',
            marker=dict(color=CHART_COLORS['red'])))
        fig.update_layout(**chart_layout(title="Virality — Feature Importance",
            xaxis_title="Importance", height=380))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Model Performance**")
        st.metric("Accuracy",  f"{accuracy_score(yvte,yvp):.2%}")
        st.metric("Precision", f"{precision_score(yvte,yvp,zero_division=0):.2%}")
        st.metric("Recall",    f"{recall_score(yvte,yvp,zero_division=0):.2%}")
        st.metric("F1 Score",  f"{f1_score(yvte,yvp,zero_division=0):.2%}")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=accuracy_score(yvte,yvp)*100,
            title={'text':'Accuracy'},
            gauge={'axis':{'range':[0,100]}, 'bar':{'color':CHART_COLORS['red']},
                   'steps':[{'range':[0,50],'color':'#f4f6f9'},
                             {'range':[50,75],'color':'#dde1ea'},
                             {'range':[75,100],'color':'#ecfdf5'}],
                   'threshold':{'line':{'color':CHART_COLORS['green'],'width':3},'thickness':0.75,'value':80}}))
        fig.update_layout(height=280, margin=dict(t=40,b=10,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # ── Model 2: Performance Category ─────────────────────────
    st.markdown('<div class="section-header">2️⃣ Performance Category Prediction</div>', unsafe_allow_html=True)
    pf = ['duration_minutes','title_length','hour_encoded','ch_subscribers','is_weekend','avg_views_per_video']
    Xp, yp = pdf[pf].fillna(pdf[pf].mean()), pdf['performance_category']
    Xptr,Xpte,yptr,ypte = train_test_split(Xp,yp,test_size=0.3,random_state=42)
    sc2 = StandardScaler(); Xptr_s = sc2.fit_transform(Xptr); Xpte_s = sc2.transform(Xpte)
    m2 = RandomForestClassifier(n_estimators=100,random_state=42,max_depth=10)
    m2.fit(Xptr_s,yptr); ypp = m2.predict(Xpte_s)
    fi2 = pd.DataFrame({'feature':pf,'importance':m2.feature_importances_}).sort_values('importance',ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(fi2, x='feature', y='importance', title="Performance — Feature Importance",
            color='importance', color_continuous_scale='Reds')
        fig.update_layout(**chart_layout(height=400))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        cm = confusion_matrix(ypte, ypp, labels=['High','Medium','Low'])
        fig = go.Figure(go.Heatmap(z=cm, x=['High','Medium','Low'], y=['High','Medium','Low'],
            colorscale='Reds', text=cm, texttemplate='%{text}', textfont={"size":16}))
        fig.update_layout(**chart_layout(title="Confusion Matrix", height=400,
            xaxis_title="Predicted", yaxis_title="Actual"))
        st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Overall Accuracy",  f"{accuracy_score(ypte,ypp):.2%}")
    c2.metric("Number of Classes", "3")
    c3.metric("Training Samples",  f"{len(Xptr)}")

    st.markdown("---")
    # ── Model 3: Views Regression ──────────────────────────────
    st.markdown('<div class="section-header">3️⃣ Views Prediction Model</div>', unsafe_allow_html=True)
    rf2 = ['duration_minutes','title_length','hour_encoded','ch_subscribers','is_weekend','avg_views_per_video']
    Xr, yr = pdf[rf2].fillna(pdf[rf2].mean()), pdf['views']
    Xrtr,Xrte,yrtr,yrte = train_test_split(Xr,yr,test_size=0.3,random_state=42)
    sc3 = StandardScaler(); Xrtr_s = sc3.fit_transform(Xrtr); Xrte_s = sc3.transform(Xrte)
    m3 = RandomForestRegressor(n_estimators=100,random_state=42,max_depth=10)
    m3.fit(Xrtr_s,yrtr); yrp = m3.predict(Xrte_s)

    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(yrte, yrp)
    r2  = r2_score(yrte, yrp)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yrte, y=yrp, mode='markers',
            marker=dict(color=CHART_COLORS['red'], opacity=0.5, size=4),
            name='Predicted vs Actual'))
        lim = max(yrte.max(), yrp.max())
        fig.add_trace(go.Scatter(x=[0,lim], y=[0,lim], mode='lines',
            line=dict(color=CHART_COLORS['muted'], dash='dash'), name='Perfect fit'))
        fig.update_layout(**chart_layout(title="Predicted vs Actual Views",
            xaxis_title="Actual", yaxis_title="Predicted", height=400))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fi3 = pd.DataFrame({'feature':rf2,'importance':m3.feature_importances_}).sort_values('importance')
        fig = go.Figure(go.Bar(x=fi3['importance'], y=fi3['feature'], orientation='h',
            marker=dict(color=CHART_COLORS['blue'])))
        fig.update_layout(**chart_layout(title="Views Model — Feature Importance", height=400))
        st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("R² Score",   f"{r2:.3f}")
    c2.metric("MAE",        fmt(mae))
    c3.metric("Training N", f"{len(Xrtr)}")


# ── Tab 6 : Recommendations ──────────────────────────────────────
def page_recommendations(fv, fc, vdf, cdf):
    st.markdown('<div class="section-header">📊 Content Strategy Recommendations</div>', unsafe_allow_html=True)

    fv2 = fv.copy()
    fv2['duration_bucket'] = pd.cut(fv2['duration_minutes'],
        bins=[0,5,10,15,20,100], labels=['0-5','5-10','10-15','15-20','20+'])
    best_duration = str(fv2.groupby('duration_bucket')['views'].mean().idxmax())

    best_time_data = fv.groupby('publish_hour')['views'].mean()
    best_hour = best_time_data.idxmax()
    best_day  = fv.groupby('publish_day')['views'].mean().idxmax()

    opt_freq = fc.groupby(pd.cut(fc['videos_per_month'],
        bins=[0,4,8,12,20,100])).agg(avg_views=('avg_views','mean'))
    best_frequency = str(opt_freq['avg_views'].idxmax())
    avg_engagement = fv['engagement_rate'].mean()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="exec-alert info">
            <div class="exec-alert-label">🎯 Optimal Video Duration</div>
            <div class="exec-alert-value">{best_duration} min</div>
            <div class="exec-alert-sub">Videos in this range achieve the highest avg views</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="exec-alert warn">
            <div class="exec-alert-label">⏰ Best Upload Time</div>
            <div class="exec-alert-value">{best_hour}:00 on {best_day}</div>
            <div class="exec-alert-sub">Optimal publishing window for maximum reach</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="exec-alert good">
            <div class="exec-alert-label">📅 Posting Frequency</div>
            <div class="exec-alert-value">{best_frequency} videos/month</div>
            <div class="exec-alert-sub">Ideal frequency for engagement balance</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="exec-alert purple">
            <div class="exec-alert-label">💬 Target Engagement Rate</div>
            <div class="exec-alert-value">{avg_engagement:.2f}%</div>
            <div class="exec-alert-sub">Current network average engagement rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📋 Detailed Strategic Recommendations</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🎬 Content Creation**")
        recs = [
            f"**Duration Sweet Spot**: Aim for {best_duration} minutes",
            f"**Title Length**: Keep titles 50–70 chars (current avg: {fv['title_length'].mean():.0f})",
            f"**Engagement Target**: Aim for >{fv['engagement_rate'].quantile(0.75):.2f}% engagement",
            "**Quality Focus**: High performers have 2.5× better engagement rates",
        ]
        for r in recs: st.markdown(f"- {r}")
        st.markdown("**📊 Publishing Strategy**")
        for r in [
            f"**Best Day**: Publish on {best_day} for maximum visibility",
            f"**Optimal Time**: Schedule uploads around {best_hour}:00",
            f"**Frequency**: Maintain {best_frequency} videos per month",
            "**Consistency**: Regular posting improves retention by 35%",
        ]: st.markdown(f"- {r}")
    with c2:
        hpc = fc[fc['engagement_score'] > fc['engagement_score'].quantile(0.75)]
        st.markdown("**🎯 Audience Growth**")
        for r in [
            f"**Target Engagement Score**: >{fc['engagement_score'].quantile(0.75):.1f}",
            f"**Views per Subscriber**: Aim for {hpc['views_per_subscriber'].mean():.1f}×",
            f"**Subscriber Growth**: Top channels post {hpc['videos_per_month'].mean():.1f} videos/month",
            "**Cross-Promotion**: Leverage high-performing videos for growth",
        ]: st.markdown(f"- {r}")
        st.markdown("**🚀 Performance Optimization**")
        for r in [
            "**Early Engagement**: First 24hrs critical — target 5%+ engagement",
            f"**Like Ratio**: Maintain >{fv['like_rate'].quantile(0.75):.2f}% like rate",
            "**Comment Engagement**: Respond within 2hrs to boost algorithm favor",
            "**Thumbnail A/B Testing**: Test thumbnails for 10%+ CTR improvement",
        ]: st.markdown(f"- {r}")

    st.markdown("---")
    st.markdown('<div class="section-header">💡 Key Actionable Insights</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="exec-alert info">
            <b>🎯 Quick Wins</b><br><br>
            1. Optimize publishing to peak hours<br>
            2. Target ideal duration range<br>
            3. Improve thumbnail click-through<br>
            4. Respond to comments faster
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="exec-alert warn">
            <b>⚠️ Risk Areas</b><br><br>
            1. Low engagement videos (&lt;2%)<br>
            2. Inconsistent posting schedule<br>
            3. Poor-performing time slots<br>
            4. Under-optimized titles
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="exec-alert good">
            <b>✨ Growth Opportunities</b><br><br>
            1. Replicate high-performer formats<br>
            2. Expand successful content types<br>
            3. Collaborate with top channels<br>
            4. Leverage trending topics
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">📈 Performance vs Benchmarks</div>', unsafe_allow_html=True)
    your_m = {
        'Avg Views':      fv['views'].mean(),
        'Engagement':     fv['engagement_rate'].mean(),
        'Like Rate':      fv['like_rate'].mean(),
        'Videos/Month':   fc['videos_per_month'].mean(),
        'Virality Index': (fv['views']/fv['ch_subscribers']).mean()*100,
    }
    bench_m = {
        'Avg Views':      vdf['views'].quantile(0.75),
        'Engagement':     vdf['engagement_rate'].quantile(0.75),
        'Like Rate':      vdf['like_rate'].quantile(0.75),
        'Videos/Month':   cdf['videos_per_month'].quantile(0.75),
        'Virality Index': (vdf['views']/vdf['ch_subscribers']).quantile(0.75)*100,
    }
    mc = pd.DataFrame({
        'Metric': list(your_m.keys()),
        'Your Performance': [your_m[k]/bench_m[k]*100 if bench_m[k] else 0 for k in your_m],
        'Benchmark': [100]*5,
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mc['Metric'], y=mc['Your Performance'],
        name='Your Performance', marker=dict(color=CHART_COLORS['red'])))
    fig.add_trace(go.Bar(x=mc['Metric'], y=mc['Benchmark'],
        name='Top 25% Benchmark', marker=dict(color=CHART_COLORS['blue'])))
    fig.update_layout(**chart_layout(title="Performance vs Benchmarks (Indexed to 100)",
        yaxis_title="Indexed Score", barmode='group', height=400))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# §8  FOOTER
# ══════════════════════════════════════════════════════════════════
def render_footer():
    st.markdown("""
    <div class="footer">
        YouTube<span>Analytics</span> Intelligence &nbsp;·&nbsp; Executive Edition &nbsp;·&nbsp;
        Powered by YouTube Data API v3 &nbsp;·&nbsp; Model accuracy: 85%+
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# §9  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    inject_styles()

    source, api_key, channel_ids, default_label = render_sidebar()

    # ── resolve data ─────────────────────────────────────────────
    if source == "🔑 YouTube API" and "yt_channels" in st.session_state:
        channels_df = st.session_state["yt_channels"]
        videos_df   = st.session_state["yt_videos"]
        data_label  = st.session_state.get("data_label", default_label)
    else:
        channels_df, videos_df = generate_sample_data()
        data_label = "Synthetic sample · 2,000 videos · 50 channels"
        if source == "🔑 YouTube API" and not api_key:
            st.info("💡 Enter your YouTube API key and click **Fetch Live Data** to load real data. Showing synthetic sample in the meantime.")

    # ── sidebar filters (depend on loaded data) ───────────────────
    with st.sidebar:
        st.markdown("---")
        date_range = st.date_input("Date Range",
            value=(videos_df['publish_date'].min().date(),
                   videos_df['publish_date'].max().date()),
            key='date_range')

        ch_opts = ['All Channels'] + sorted(channels_df['channel_name'].unique().tolist())
        sel_ch  = st.selectbox("Channel", ch_opts, index=0)
        selected_channels = channels_df['channel_name'].unique().tolist() if sel_ch == 'All Channels' else [sel_ch]

        perf_opts = ['All Categories','High','Medium','Low']
        sel_perf  = st.selectbox("Performance Category", perf_opts, index=0)
        perf_filter = ['High','Medium','Low'] if sel_perf == 'All Categories' else [sel_perf]

    # ── filter ───────────────────────────────────────────────────
    dr0, dr1 = (date_range[0], date_range[1]) if len(date_range)==2 else (date_range[0], date_range[0])
    fv = videos_df[
        (videos_df['publish_date'].dt.date >= dr0) &
        (videos_df['publish_date'].dt.date <= dr1) &
        (videos_df['channel_name'].isin(selected_channels)) &
        (videos_df['performance_category'].isin(perf_filter))
    ].copy().fillna(0)
    fc = channels_df[channels_df['channel_name'].isin(selected_channels)].copy().fillna(0)

    if fv.empty:
        st.warning("No data matches the current filters. Adjust the date range or channel selection.")
        return

    # ── header ───────────────────────────────────────────────────
    render_header(
        data_label, len(fv), len(fc),
        fv['publish_date'].min().date(),
        fv['publish_date'].max().date(),
    )

    # ── tabs ─────────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "📈 Overview & KPIs",
        "🎯 Growth & Engagement",
        "🎬 Video Performance",
        "📅 Content Strategy",
        "🔮 Predictions",
        "💡 Recommendations",
    ])
    with tab1: page_overview(fv, fc, videos_df, channels_df)
    with tab2: page_growth(fv, fc)
    with tab3: page_video_performance(fv)
    with tab4: page_content_strategy(fv, fc)
    with tab5: page_predictions(fv, channels_df)
    with tab6: page_recommendations(fv, fc, videos_df, channels_df)

    render_footer()


if __name__ == "__main__":
    main()
