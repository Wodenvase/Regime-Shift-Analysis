use thiserror::Error;

#[derive(Error, Debug)]
pub enum ResilienceError {
    #[error("Data processing error: {0}")]
    DataError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Polars error: {0}")]
    PolarsError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}
