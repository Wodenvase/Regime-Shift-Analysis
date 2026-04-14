use crate::data::ProcessedTrip;
use chrono::{DateTime, Utc, Timelike, Datelike};

/// Extract temporal features from ISO8601 timestamp string
pub fn extract_temporal_features(timestamp_str: &str) -> Option<(u32, u32, u32, i64)> {
    // Expected format: "2023-06-15 08:30:00"
    let dt = DateTime::parse_from_rfc3339(&format!("{}Z", timestamp_str.replace(" ", "T")))
        .or_else(|_| {
            // Fallback: manual parsing
            let parts: Vec<&str> = timestamp_str.split(&[' ', ':', '-'][..]).collect();
            if parts.len() >= 5 {
                Ok((parts[3].parse::<u32>().ok(),
                    parts[4].parse::<u32>().ok()))
            } else {
                Err(chrono::format::ParseError::OutOfRange)
            }
        })
        .ok()?;

    let dt = dt.with_timezone(&Utc);

    let hour = dt.hour();
    let day_of_week = dt.weekday().number_from_monday(); // Mon=1, Sun=7
    let month = dt.month();
    let timestamp = dt.timestamp();

    Some((hour, day_of_week, month, timestamp))
}

/// Calculate haversine distance between two coordinates (in km)
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6371.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_KM * c
}

/// Detect "Black Swan" event days based on heuristics
pub fn is_black_swan_day(month: u32, day: u32) -> bool {
    // June 7-9: Wildfire smoke
    // Sept 29-30: Flash flood
    (month == 6 && day >= 7 && day <= 9) || (month == 9 && day >= 29 && day <= 30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_distance() {
        // NYC (40.7128, -74.0060) to LA (34.0522, -118.2437) should be ~3,944 km
        let dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437);
        assert!((dist - 3944.0).abs() < 50.0);
    }
}
