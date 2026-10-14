import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, date, timedelta

def main():
    # Load the data from data.json
    try:
        with open('data.json', 'r', encoding='utf-8') as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    # Filtering: removing rows with '@Bot' in title and empty/unavailable titles
    data_filtered = [d for d in data_list if '@Bot' not in d.get('title', '')]

    data_unique = dict()
    for d in data_filtered:
        title = d.get("title", "")
        if title in ['', 'This post is anywhere.', 'This post is unavailable.', 'No content']:
            continue
        
        date_val = d.get("date", "Unknown")
        if date_val == 'Unknown':
            d["date"] = "2026-10-14T00:00:00+03:00"
            
        # Using title and likes as a key to ensure unicity as requested
        key = (title, d.get("likes", "0"))
        data_unique[key] = d

    data = list(data_unique.values())
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    if df.empty:
        print("No data to process.")
        return

    # Clean the data
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    metrics = ['shares', 'likes', 'bookmarks', 'views']
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce').fillna(0)

    # Drop rows with invalid dates and ensure uniqueness by title and date
    df = df.dropna(subset=['date'])
    df = df.drop_duplicates(subset=['title', 'date'])

    # Sort by date and set as index for time-based rolling window
    df = df.sort_values('date').set_index('date')

    if df.empty:
        print("No valid data to plot after processing.")
        return

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    fig.suptitle('Social Media Metrics (3-Day Time-Window Smoothing)', fontsize=16)
    axes_flat = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        
        # 1. Plot the actual data as a scatter plot
        ax.scatter(df.index, df[metric], label=metric.capitalize(), alpha=0.5, color='blue', s=30)
        
        # 2. Calculate and plot 3-day smoothed curve (Time-based Window)
        # '3D' window looks back 3 days in time from each timestamp
        print(df[metric].rolling(window=timedelta(days=3)))
        smoothed_values = df[metric].rolling(window=timedelta(days=3), center=True).mean()
        ax.plot(df.index, smoothed_values, color='red', linestyle='-', linewidth=2, label='3-Day Smooth')

        ax.set_title(f'{metric.capitalize()} vs Date')
        ax.set_xlabel('Date')
        ax.set_ylabel(metric.capitalize())
        ax.tick_params(axis='x', rotation=45)
        
        # Use log scale because metrics vary significantly
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = 'metrics_plot.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    print("\nTop 5 posts by views:")
    # Reset index to access 'date' for printing
    print(df.sort_values('views', ascending=False)[['title', 'views']].head())

if __name__ == "__main__":
    main()
