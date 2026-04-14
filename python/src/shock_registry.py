"""
Black Swan Event Registry for 2023 NYC
Exogenous shock events that affect Citi Bike liquidity
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict


@dataclass
class BlackSwanEvent:
    """Definition of a systemic shock event"""
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    event_type: str  # "weather", "transit", "external"
    magnitude: float  # Normalized impact scale (0-1)
    affected_stations: str  # Geographic region or "all"
    external_variables: Dict[str, float]  # Exogenous regressors


# Registry of 2023 Black Swan Events

CANADIAN_WILDFIRE_SMOKE = BlackSwanEvent(
    name="Canadian Wildfire Smoke",
    description="Air quality crisis from Canadian wildfires. AQI exceeded 200+ (hazardous). "
    "Reduced visibility and respiratory concerns led to sharp decline in outdoor mobility.",
    start_date=datetime(2023, 6, 7, 0, 0, 0),
    end_date=datetime(2023, 6, 9, 23, 59, 59),
    event_type="external",
    magnitude=0.85,
    affected_stations="all",
    external_variables={
        "aqi": 250,
        "pm25": 150,
        "visibility_km": 2.5,
        "precipitation_mm": 0.0,
        "temperature_c": 24
    }
)

SEPTEMBER_FLASH_FLOOD = BlackSwanEvent(
    name="Flash Flood & Transit Collapse",
    description="Record-breaking precipitation (3.5 inches) causing widespread flooding. "
    "NYC subway system partially shut down. Created 'Liquidity Vacuum' where demand spiked "
    "but system capacity collapsed. Emergency biking spike as commuters sought alternatives.",
    start_date=datetime(2023, 9, 29, 0, 0, 0),
    end_date=datetime(2023, 9, 30, 23, 59, 59),
    event_type="weather",
    magnitude=0.95,
    affected_stations="all",
    external_variables={
        "precipitation_mm": 88.9,  # 3.5 inches
        "temperature_c": 18,
        "humidity_percent": 85,
        "transit_delay_hours": 8,
        "subway_closure_percent": 0.40
    }
)

JULY_HEAT_WAVE = BlackSwanEvent(
    name="Intense Heat Wave",
    description="Extreme heat event with temperatures >85°C. Reduced outdoor activity "
    "but increased daytime usage due to AC failure in apartments.",
    start_date=datetime(2023, 7, 18, 0, 0, 0),
    end_date=datetime(2023, 7, 22, 23, 59, 59),
    event_type="weather",
    magnitude=0.45,
    affected_stations="all",
    external_variables={
        "temperature_c": 32,
        "heat_index_c": 38,
        "humidity_percent": 70,
        "precipitation_mm": 0.0
    }
)

# Complete registry
BLACK_SWAN_REGISTRY: List[BlackSwanEvent] = [
    CANADIAN_WILDFIRE_SMOKE,
    SEPTEMBER_FLASH_FLOOD,
    JULY_HEAT_WAVE,
]


def get_events_by_month(month: int) -> List[BlackSwanEvent]:
    """Filter events by calendar month (1-12)"""
    return [e for e in BLACK_SWAN_REGISTRY if e.start_date.month == month]


def is_shock_date(date: datetime) -> bool:
    """Check if date falls within any Black Swan event"""
    return any(
        e.start_date <= date <= e.end_date
        for e in BLACK_SWAN_REGISTRY
    )


def get_event_magnitude(date: datetime) -> float:
    """Get normalized shock magnitude for a given date (0-1 scale)"""
    for event in BLACK_SWAN_REGISTRY:
        if event.start_date <= date <= event.end_date:
            return event.magnitude
    return 0.0


# Time series indicators for external regressors
EXTERNAL_REGRESSORS_2023 = {
    "precipitation_mm": {
        "source": "NOAA Weather Data",
        "frequency": "daily",
        "units": "millimeters",
        "notes": "Total daily precipitation at Central Park"
    },
    "aqi": {
        "source": "EPA AirNow",
        "frequency": "hourly",
        "units": "index (0-500+)",
        "notes": "Air Quality Index for Manhattan"
    },
    "transit_delay_hours": {
        "source": "MTA Estimated Impact",
        "frequency": "daily",
        "units": "hours",
        "notes": "Cumulative system delay"
    },
    "temperature_c": {
        "source": "NOAA",
        "frequency": "hourly",
        "units": "Celsius",
    }
}
