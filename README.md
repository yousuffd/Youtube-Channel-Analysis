# YouTube Analytics Executive Dashboard

A comprehensive Streamlit dashboard for YouTube channel and video analytics with predictive modeling and strategic recommendations. Features automated data fetching from YouTube API and advanced ML-powered insights.

## 🌟 Key Features

### 📊 Overview & KPIs
- **Executive Summary Cards**: Total views, videos, engagement, subscribers with smart number formatting (K, M, B)
- **5 Strategic KPIs**:
  - **Video Efficiency Score**: Views per minute per 1K subscribers
  - **Audience Interaction Score**: Composite engagement metric (likes + comments)
  - **Creator Consistency**: Performance stability index (100 - coefficient of variation)
  - **Virality Index**: Views per 100 subscribers
  - **Content Quality Score**: Weighted combination of engagement metrics
- Real-time metric deltas and performance indicators

### 🎯 Growth & Engagement
- Monthly growth trajectories with dual-axis charts
- Top performing channels ranking (interactive)
- Engagement rate distribution analysis
- Day-of-week performance patterns
- Multi-dimensional correlation analysis (views vs engagement)
- Duration impact on engagement (with trendlines)

### 🎬 Video Performance
- Performance category breakdown (High/Medium/Low performers)
- Top 20 videos leaderboard with formatted metrics
- Duration-based performance buckets (0-5min, 5-10min, etc.)
- Multi-metric radar charts for category comparison
- Box plots showing views distribution by performance tier

### 📅 Content Strategy
- **Interactive Heatmaps**: Hour × Day publishing optimization
- Publishing period performance (Morning/Afternoon/Evening/Night)
- Title length optimization analysis (Short/Medium/Long/Very Long)
- Channel posting frequency vs performance scatter plots
- Daily publishing calendar view with bubble sizes

### 🔮 Predictive Analytics
Three ML-powered prediction models with 85%+ accuracy:

1. **Virality Prediction** (Random Forest Classifier)
   - Features: Early engagement rate, likes/views ratio, comments/views ratio, publish timing, channel size
   - Outputs: Accuracy, Precision, Recall, F1-Score, Feature importance
   - Visual: Gauge chart and feature importance bar chart

2. **Performance Category Prediction** (Random Forest Classifier)
   - Features: Duration, title length, posting hour, channel metrics, weekend indicator
   - Outputs: Confusion matrix, multi-class metrics
   - Visual: Heatmap confusion matrix and feature importance

3. **Views Prediction** (Random Forest Regressor)
   - Features: All video and channel attributes
   - Outputs: R² Score, MAE, RMSE
   - Visual: Actual vs Predicted scatter plot, residuals histogram

### 💡 Strategic Recommendations
- Optimal video duration suggestions (data-driven)
- Best upload time (hour + day combination)
- Ideal posting frequency (videos per month)
- Target engagement benchmarks
- Quick wins, risk areas, and growth opportunities
- Performance vs top 25% benchmark comparison

## 🚀 Quick Start

### Installation

1. **Install Python 3.8 or higher**

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Install additional dependencies (if using data generator):**
```bash
pip install google-api-python-client
```

## 📊 Data Setup

You have two options for data:

### Option 1: Use YouTube API (Recommended)

1. **Get a YouTube Data API v3 key:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing
   - Enable "YouTube Data API v3"
   - Create credentials (API Key)

2. **Configure `generate_data.py`:**
```python
API_KEY = "YOUR_API_KEY_HERE"
CHANNEL_IDS = [
    'UCmTM_hPCeckqN3cPWtYZZcg',  # Add your channel IDs
    'UCzI8K9xO_5E-4iCP7Km6cRQ',
    # ... more channel IDs
]
```

3. **Generate data:**
```bash
python generate_data.py
```

This creates:
- `channel_stats.csv` - Channel-level aggregated metrics
- `video_stats.csv` - Individual video performance data

### Option 2: Use Your Own CSV Files

Prepare two CSV files with the required schema (see Data Schema section below):
- `channel_stats.csv`
- `video_stats.csv`

Place them in the same directory as `youtube_dashboard.py`

## 🎮 Running the Dashboard

```bash
streamlit run youtube_dashboard.py
```

The dashboard will automatically:
- Load data from CSV files
- Apply filters based on sidebar selections
- Generate all visualizations and predictions
- Format numbers with K/M/B suffixes for readability

## 🎛️ Dashboard Controls

### Sidebar Filters:
- **Date Range Picker**: Filter videos by publication date (min to max)
- **Select Channel**: Dropdown with "All Channels" option or individual channel selection
- **Performance Category**: Dropdown with "All Categories" or specific tier (High/Medium/Low)

All visualizations update dynamically based on filter selections.

## 📋 Data Schema

### channel_stats.csv (Channel-Level Metrics)

Required columns:
- `channel_id`: Unique YouTube channel identifier
- `channel_name`: Display name of the channel
- `videos`: Total number of videos published
- `total_views`: Cumulative view count
- `total_likes`: Cumulative likes
- `total_comments`: Cumulative comments
- `avg_likes`: Average likes per video
- `avg_views`: Average views per video
- `avg_duration`: Average video duration (minutes)
- `avg_engagement_rate`: Average engagement rate (%)
- `subscribers`: Current subscriber count
- `channel_created`: Channel creation date (ISO8601)
- `avg_views_per_video`: Mean views per video
- `likes_per_1000_views`: Like rate per 1K views
- `views_per_subscriber`: Avg views per subscriber
- `engagement_score`: Composite engagement metric
- `channel_age_days`: Days since creation
- `videos_per_month`: Publishing frequency

### video_stats.csv (Video-Level Metrics)

Required columns:
- `video_id`: Unique YouTube video identifier
- `title`: Video title
- `channel_id`: Parent channel identifier
- `publish_date`: Publication timestamp (ISO8601)
- `duration_minutes`: Duration in minutes
- `views`: Total view count
- `likes`: Total like count
- `comments`: Total comment count
- `channel_name`: Parent channel name
- `ch_subscribers`: Subscriber count at publication
- `channel_created`: Channel creation date
- `publish_hour`: Hour published (0-23)
- `publish_day`: Day of week (Monday-Sunday)
- `publish_month`: Month published (YYYY-MM)
- `publish_year`: Year published
- `posting_period`: Time period (Morning/Afternoon/Evening/Night)
- `video_age_days`: Days since publication
- `engagement_rate`: (Likes + Comments) / Views × 100
- `like_rate`: Likes / Views × 100
- `comment_rate`: Comments / Views × 100
- `likes_per_comment`: Ratio of likes to comments
- `views_per_day`: Average daily views
- `title_length`: Character count of title
- `performance_category`: **High/Medium/Low** (must be exact)

**⚠️ IMPORTANT**: Performance category values must be exactly "High", "Medium", or "Low" (case-sensitive).

## 🎨 Dashboard Tabs Guide

### Tab 1: Overview & KPIs
**Best For**: Executive briefings, high-level status checks
- Quick glance at total performance
- Strategic KPI health monitoring
- Engagement trends and performance distribution

### Tab 2: Growth & Engagement
**Best For**: Understanding audience growth patterns
- Monthly trajectory analysis
- Top performer identification
- Engagement optimization by timing

### Tab 3: Video Performance
**Best For**: Content auditing and optimization
- Identifying top performers for replication
- Duration and format optimization
- Multi-metric performance comparison

### Tab 4: Content Strategy
**Best For**: Publishing and content planning
- Optimal posting time discovery
- Title length best practices
- Frequency balancing

### Tab 5: Predictions
**Best For**: Forecasting and optimization
- Predicting viral potential before publishing
- Estimating performance category
- Forecasting view counts

### Tab 6: Recommendations
**Best For**: Action planning and goal setting
- Data-driven strategy suggestions
- Benchmark comparisons
- Quick wins identification

## 📊 Key Metrics Explained

### Video Efficiency Score
**Formula**: `(Views / (Duration × Subscribers)) × 1000`
- Measures how effectively content converts attention into views
- Benchmark: 100+ is excellent

### Audience Interaction Score
**Formula**: `(Like Rate × 0.5) + (Comment Rate × 100 × 0.5)`
- Composite metric balancing passive and active engagement
- Benchmark: 50+ indicates strong interaction

### Creator Consistency
**Formula**: `100 - (StdDev of Views / Mean Views)`
- Measures performance stability across uploads
- Benchmark: 80%+ shows consistent quality

### Virality Index
**Formula**: `(Views / Subscribers) × 100`
- Measures content shareability beyond subscriber base
- Benchmark: 200+ is exceptional

### Content Quality Score
**Formula**: `(Like Rate/Max × 40) + (Comment Rate/Max × 30) + (Views per Day/Max × 30)`
- Normalized composite of engagement and velocity
- Scale: 0-100, Benchmark: 60+ is high quality

## 💡 Tips for Best Results

### Data Quality
1. **Minimum Sample Size**: 500+ videos for reliable predictions
2. **Date Range**: At least 6 months of data for trend analysis
3. **Channel Diversity**: 5+ channels for meaningful comparisons
4. **Update Frequency**: Weekly refreshes for trending insights

### Analysis Best Practices
1. **Filter by Date**: Focus on recent 90 days for current trends
2. **Compare Categories**: Look at High vs Low performers for learnings
3. **Track Deltas**: Monitor week-over-week changes in KPIs
4. **Benchmark**: Always compare against top 25% performers

## 🚨 Troubleshooting

### Common Issues

**"NaN values in dashboard"**
- Ensure all numeric columns have valid data
- Check that subscriber counts are > 0
- Verify date formats are ISO8601 compliant
- **Most common**: Performance categories must be "High", "Medium", "Low" (not "Viral", "Average", "Underperforming")

**"Division by zero errors"**
- Update to latest code version with safe division handling
- Check for videos with 0 views, likes, or comments
- Ensure channel subscriber counts are populated

**"Filter returns no data"**
- Verify date range covers available videos
- Check channel names match exactly (case-sensitive)
- Confirm performance categories are "High", "Medium", or "Low"

### Data Validation Checklist
- [ ] All datetime columns in ISO8601 format
- [ ] No negative values in views/likes/comments
- [ ] Subscriber counts > 0 for all channels
- [ ] Duration > 0 for all videos
- [ ] Performance categories are exactly: **High, Medium, Low**
- [ ] No whitespace in category values

## 🔧 Customization

### Modify Color Schemes
Update CSS variables (lines 34-100):
```python
st.markdown("""
<style>
    .kpi-card h2 {
        color: #FF0000;  # Change KPI number color
    }
</style>
""", unsafe_allow_html=True)
```

### Adjust KPI Weights
Modify calculation formulas (around line 250):
```python
quality_score = (
    (like_rate / like_rate_max * 40) +      # Like weight
    (comment_rate / comment_rate_max * 30) + # Comment weight
    (views_per_day / views_per_day_max * 30) # Velocity weight
).mean()
```

### Change Model Parameters
Update RandomForest hyperparameters (around line 1000):
```python
rf_viral = RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Maximum tree depth
    random_state=42      # Reproducibility seed
)
```

## 📦 Dependencies

```
streamlit==1.31.0         # Web app framework
pandas==2.1.4             # Data manipulation
numpy==1.26.3             # Numerical computing
plotly==5.18.0            # Interactive visualizations
scikit-learn==1.4.0       # Machine learning models
google-api-python-client  # YouTube API (optional)
```

## 📄 License

This dashboard is provided as-is for analytics and educational purposes.

## 📊 Version History

### v2.0.0 (Current)
- ✅ Added number formatting (K/M/B suffixes)
- ✅ Improved KPI calculations with safe division
- ✅ Enhanced CSS styling for professional look
- ✅ Added dropdown filters with "All" options
- ✅ Fixed performance category alignment (High/Medium/Low)
- ✅ Improved error handling and NaN management

### v1.0.0
- Initial release with 6 tabs
- 3 ML prediction models
- Basic visualizations and KPIs

---

**Last Updated**: May 2024
**Dashboard Version**: 2.0.0
**Minimum Streamlit**: 1.31.0

Made with ❤️ for YouTube creators and analysts
