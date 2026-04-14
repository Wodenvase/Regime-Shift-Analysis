#!/usr/bin/env python3
"""
Resilience & Liquidity ETL Pipeline (Python)
Processes all Citi Bike trip data and aggregates to station-hour level
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import pickle
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def discover_csv_files(base_dir="."):
    """Discover all CSV files in monthly directories"""
    csv_files = []
    base_path = Path(base_dir).parent  # Go up to main directory

    for month_dir in sorted(base_path.glob("202*-citibike-tripdata")):
        if month_dir.is_dir():
            for csv_file in month_dir.glob("*.csv"):
                csv_files.append(csv_file)

    return sorted(csv_files)

def load_and_combine_csvs(csv_files):
    """Load and concatenate all CSV files"""
    logger.info("🚴 Resilience & Liquidity ETL Pipeline v0.1.0")
    logger.info("Processing 2023 NYC Citi Bike Trip Data\n")
    logger.info(f"📂 Found {len(csv_files)} CSV files to process")
    logger.info("🔄 Loading and combining trip data...\n")

    dfs = []
    for i, csv_file in enumerate(csv_files, 1):
        try:
            logger.info(f"  [{i}/{len(csv_files)}] Reading: {csv_file.name}")
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"    ⚠️ Error reading {csv_file.name}: {e}")

    if not dfs:
        raise ValueError("No CSV files successfully loaded")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"\n✅ Loaded {len(combined):,} total trips\n")
    return combined

def aggregate_to_station_hour(df):
    """Aggregate trips to station-hour level"""
    logger.info("📊 Aggregating to station-hour granularity...")

    # Convert timestamps
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])

    # Extract features
    df['hour_bucket'] = df['started_at'].dt.floor('h')
    df['duration_minutes'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    df['is_member'] = (df['member_casual'] == 'member').astype(int)

    # Detect Black Swan events (June 7-9, Sept 29-30)
    df['is_black_swan'] = (
        ((df['started_at'].dt.month == 6) & (df['started_at'].dt.day.isin([7, 8, 9]))) |
        ((df['started_at'].dt.month == 9) & (df['started_at'].dt.day.isin([29, 30])))
    ).astype(int)

    # Aggregate
    aggregated = df.groupby(['start_station_id', 'hour_bucket']).agg(
        trip_count=('ride_id', 'count'),
        avg_duration=('duration_minutes', 'mean'),
        member_trips=('is_member', 'sum'),
        unique_users=('ride_id', 'nunique'),
        station_name=('start_station_name', 'first'),
        is_black_swan=('is_black_swan', 'max'),
    ).reset_index()

    aggregated['member_ratio'] = aggregated['member_trips'] / aggregated['trip_count']

    logger.info(f"✅ Aggregated to {len(aggregated):,} station-hour records\n")
    return aggregated

def save_output(aggregated):
    """Save aggregated data"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save as pickle for Python usage
    output_path = output_dir / "station_hour_matrix.pkl"
    aggregated.to_pickle(output_path)
    logger.info(f"💾 Saved to {output_path}")

    # Also save as CSV for inspection
    csv_path = output_dir / "station_hour_matrix.csv"
    aggregated.to_csv(csv_path, index=False)
    logger.info(f"💾 Saved to {csv_path}\n")

    return aggregated

def main():
    try:
        csv_files = discover_csv_files()
        if not csv_files:
            logger.error("❌ No CSV files found")
            sys.exit(1)

        df = load_and_combine_csvs(csv_files)
        aggregated = aggregate_to_station_hour(df)
        saved_data = save_output(aggregated)

        logger.info("📈 Summary Statistics:")
        logger.info(f"  Total station-hours: {len(aggregated):,}")
        logger.info(f"  Unique stations: {aggregated['start_station_id'].nunique():,}")
        logger.info(f"  Date range: {aggregated['hour_bucket'].min()} to {aggregated['hour_bucket'].max()}")
        logger.info(f"  Black Swan records: {aggregated['is_black_swan'].sum():,}")
        logger.info(f"\n✅ ETL Pipeline completed successfully\n")

        return saved_data

    except Exception as e:
        logger.error(f"❌ ETL Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
