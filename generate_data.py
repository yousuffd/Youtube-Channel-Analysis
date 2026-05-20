# ============================================================================
# YOUTUBE DATA GENERATOR
# ============================================================================
# This script fetches YouTube data and saves it to CSV files for use in the
# static dashboard. Run this script once to generate the data snapshot.
#
# Usage: python generate_data.py
# Output: Creates channel_stats.csv and video_stats.csv
# ============================================================================

import pandas as pd
import numpy as np
import warnings
import re
from datetime import datetime
from googleapiclient.discovery import build

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Add your API key here or use environment variable
API_KEY = "API_KEY"  # Replace with your actual API key

CHANNEL_IDS = [
    'UCmTM_hPCeckqN3cPWtYZZcg',
    'UCzI8K9xO_5E-4iCP7Km6cRQ',
    'UC5fcjujOsqD-126Chn_BAuA',
    'UC-CSyyi47VX1lD9zyeABW3w',
    'UC0yXUUIaPVAqZLgRjvtMftw',
    'UCWtlPzcP989da26sVyHPzqQ'
]

# ============================================================================
# HELPER FUNCTIONS
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


def posting_period(date):
    """Categorize posting period in month"""
    d = date.day
    return "First" if d <= 10 else "Middle" if d <= 20 else "Last"


def categorize_video_performance(views, percentile_75, percentile_25):
    """Categorize videos as High, Medium, or Low performance"""
    if views >= percentile_75:
        return "High"
    elif views >= percentile_25:
        return "Medium"
    else:
        return "Low"


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def get_channel_stats(channel_ids, api_key):
    """Fetch channel-level statistics"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    all_channels = []
    
    print("Fetching channel statistics...")
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
            "channel_created": item["snippet"]["publishedAt"]
        })
    
    return pd.DataFrame(all_channels)


def get_videos_from_playlists(playlist_ids, api_key):
    """Fetch all video IDs from playlists"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_ids = []
    
    print("Fetching video IDs from playlists...")
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
    
    print(f"Found {len(video_ids)} videos")
    return video_ids


def get_videos_data(video_ids, api_key):
    """Fetch detailed video statistics"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_details = []
    
    print("Fetching video details...")
    total_batches = (len(video_ids) + 49) // 50
    
    for i in range(0, len(video_ids), 50):
        batch_num = (i // 50) + 1
        print(f"  Processing batch {batch_num}/{total_batches}...")
        
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
                "comments": stats.get("commentCount", 0),
            })
    
    return pd.DataFrame(video_details)


# ============================================================================
# MAIN DATA PROCESSING
# ============================================================================

def generate_youtube_data():
    """Main function to fetch and process YouTube data"""
    
    print("="*70)
    print("YouTube Analytics Data Generator")
    print("="*70)
    print()
    
    try:
        # 1. Fetch channel stats
        yt_channels = get_channel_stats(CHANNEL_IDS, API_KEY)
        print(f"✓ Fetched data for {len(yt_channels)} channels")
        print()
        
        # 2. Fetch video IDs
        video_ids = get_videos_from_playlists(
            yt_channels['playlist_id'].tolist(), 
            API_KEY
        )
        print()
        
        # 3. Fetch video details
        yt_video_data = get_videos_data(video_ids, API_KEY)
        print(f"✓ Fetched details for {len(yt_video_data)} videos")
        print()
        
        # 4. Merge channel and video data
        print("Processing data...")
        yt_data = pd.merge(
            yt_video_data,
            yt_channels[['channel_id', 'channel_name', 'ch_subscribers', 'channel_created']],
            on='channel_id',
            how='left'
        )
        
        # 5. Data preprocessing
        # Convert data types
        yt_data['views'] = pd.to_numeric(yt_data['views'], errors='coerce').fillna(0)
        yt_data['likes'] = pd.to_numeric(yt_data['likes'], errors='coerce').fillna(0)
        yt_data['comments'] = pd.to_numeric(yt_data['comments'], errors='coerce').fillna(0)
        yt_data['ch_subscribers'] = pd.to_numeric(yt_data['ch_subscribers'], errors='coerce')
        
        # Fix datetime parsing
        yt_data['publish_date'] = pd.to_datetime(yt_data['publish_date'], format='ISO8601')
        
        # Create channel_created mapping
        channel_created_map = yt_channels.set_index('channel_id')['channel_created'].to_dict()
        yt_data['channel_created'] = yt_data['channel_id'].map(channel_created_map)
        yt_data['channel_created'] = pd.to_datetime(yt_data['channel_created'], format='ISO8601')
        
        # Duration processing
        yt_data['duration_minutes'] = yt_data['duration'].astype(str).apply(iso_duration_to_minutes).round(2)
        
        # Time-based features
        yt_data['publish_hour'] = yt_data['publish_date'].dt.hour
        yt_data['publish_day'] = yt_data['publish_date'].dt.day_name()
        yt_data['publish_month'] = yt_data['publish_date'].dt.to_period('M').astype(str)
        yt_data['publish_year'] = yt_data['publish_date'].dt.year
        yt_data['posting_period'] = yt_data['publish_date'].apply(posting_period)
        
        # Video age in days
        yt_data['video_age_days'] = (pd.Timestamp.now(tz='UTC') - yt_data['publish_date']).dt.days
        
        # Engagement metrics (with safe division)
        safe_views = yt_data['views'].replace(0, np.nan)
        yt_data['engagement_rate'] = ((yt_data['likes'] + yt_data['comments']) / safe_views * 100).fillna(0)
        yt_data['like_rate'] = (yt_data['likes'] / safe_views * 100).fillna(0)
        yt_data['comment_rate'] = (yt_data['comments'] / safe_views * 100).fillna(0)
        yt_data['likes_per_comment'] = (yt_data['likes'] / yt_data['comments'].replace(0, np.nan)).fillna(0)
        
        # Views per day (velocity)
        yt_data['views_per_day'] = (yt_data['views'] / (yt_data['video_age_days'] + 1)).fillna(0)
        
        # Title length
        yt_data['title_length'] = yt_data['title'].str.len()
        
        # Performance categories
        percentile_75 = yt_data['views'].quantile(0.75)
        percentile_25 = yt_data['views'].quantile(0.25)
        yt_data['performance_category'] = yt_data['views'].apply(
            lambda x: categorize_video_performance(x, percentile_75, percentile_25)
        )
        
        # 6. Create aggregated channel stats
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
        
        # Calculate channel-level metrics
        df_channels['avg_views_per_video'] = df_channels['total_views'] / df_channels['videos']
        df_channels['likes_per_1000_views'] = ((df_channels['total_likes'] / df_channels['total_views'].replace(0, np.nan)) * 1000).fillna(0)
        df_channels['views_per_subscriber'] = (df_channels['total_views'] / df_channels['subscribers'].replace(0, np.nan)).fillna(0)
        df_channels['engagement_score'] = ((df_channels['total_likes'] / df_channels['total_views'].replace(0, np.nan)) + df_channels['views_per_subscriber']).fillna(0)
        df_channels['channel_age_days'] = (pd.Timestamp.now(tz='UTC') - df_channels['channel_created']).dt.days
        df_channels['videos_per_month'] = ((df_channels['videos'] / df_channels['channel_age_days'].replace(0, np.nan)) * 30).fillna(0)
        
        # 7. Save to CSV files
        print()
        print("Saving data to CSV files...")
        
        # Save video-level data
        yt_data.to_csv('video_stats.csv', index=False)
        print(f"✓ Saved video_stats.csv ({len(yt_data)} rows)")
        
        # Save channel-level data
        df_channels.to_csv('channel_stats.csv', index=False)
        print(f"✓ Saved channel_stats.csv ({len(df_channels)} rows)")
        
        # 8. Generate data summary
        print()
        print("="*70)
        print("DATA SUMMARY")
        print("="*70)
        print(f"Total Channels: {len(df_channels)}")
        print(f"Total Videos: {len(yt_data)}")
        print(f"Total Views: {yt_data['views'].sum():,.0f}")
        print(f"Total Likes: {yt_data['likes'].sum():,.0f}")
        print(f"Date Range: {yt_data['publish_date'].min().date()} to {yt_data['publish_date'].max().date()}")
        print(f"Data Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print()
        print("✓ Data generation complete!")
        print()
        print("Next steps:")
        print("1. Run: streamlit run yt_dashboard_static.py")
        print("2. Share both CSV files with your dashboard script")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check:")
        print("1. Your API key is correct")
        print("2. You have internet connection")
        print("3. YouTube Data API v3 is enabled in your Google Cloud project")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    generate_youtube_data()
