use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Raw trip record from CSV
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripRecord {
    pub ride_id: String,
    pub rideable_type: String,
    pub started_at: String,
    pub ended_at: String,
    pub start_station_name: String,
    pub start_station_id: Option<String>,
    pub end_station_name: String,
    pub end_station_id: Option<String>,
    pub start_lat: Option<f64>,
    pub start_lng: Option<f64>,
    pub end_lat: Option<f64>,
    pub end_lng: Option<f64>,
    pub member_type: String,
}

/// Processed trip with temporal and spatial features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedTrip {
    pub trip_duration_minutes: f64,
    pub start_hour: u32,
    pub start_day_of_week: u32,
    pub start_month: u32,
    pub start_timestamp: i64,
    pub station_id: String,
    pub is_member: bool,
    pub distance_km: Option<f64>,
}

/// Aggregated station-hour metrics for BSTS input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationHourStats {
    pub timestamp: i64,
    pub station_id: String,
    pub station_name: String,
    pub trip_count: u32,
    pub avg_duration_minutes: f64,
    pub member_ratio: f64,
    pub unique_users: u32,
    pub hour_of_day: u32,
    pub day_of_week: u32,
    pub month: u32,
}
