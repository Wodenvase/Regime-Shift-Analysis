mod etl;
mod data;
mod features;
mod errors;

use anyhow::Result;
use env_logger::Env;
use std::path::PathBuf;
use log::info;

#[derive(Debug, Clone)]
pub struct Config {
    pub data_dir: PathBuf,
    pub output_dir: PathBuf,
    pub months: Vec<String>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    info!("🚴 Resilience & Liquidity ETL Pipeline v0.1.0");
    info!("Processing 2023 NYC Citi Bike Trip Data");

    let config = Config {
        data_dir: PathBuf::from("./"),
        output_dir: PathBuf::from("./output"),
        months: (1..=12)
            .map(|m| format!("202{:02}", m))
            .collect(),
    };

    info!("Loading and aggregating monthly data...");
    etl::run_etl(&config)?;

    info!("✅ ETL Pipeline completed successfully");
    Ok(())
}
