"""Runtime configuration from environment (prefix ``QT_``) or a local ``.env`` file."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QT_", env_file=".env", extra="ignore")

    database_url: SecretStr = SecretStr("postgresql://quantterm@127.0.0.1:5432/quantterm")
    db_pool_min: int = Field(default=2, ge=1)
    db_pool_max: int = Field(default=20, ge=1)
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    api_token: SecretStr = Field(min_length=24)
    ingest_token: SecretStr = Field(min_length=24)

    cache_ttl_seconds: int = Field(default=300, ge=1)
    candle_interval_seconds: int = Field(default=60, ge=1)
    tick_flush_size: int = Field(default=500, ge=1)
    tick_flush_interval_seconds: float = Field(default=1.0, gt=0)
    reconciliation_tolerance: float = Field(default=0.005, gt=0)
    hygiene_window: int = Field(default=50, ge=5)
    hygiene_mad_multiplier: float = Field(default=5.0, gt=0)
    auto_migrate: bool = True

    # SEC fair-access policy requires a descriptive UA with contact e-mail.
    sec_user_agent: str = "QuantTerm research ops@example.invalid"
    fmp_api_key: SecretStr | None = None
    polygon_api_key: SecretStr | None = None
    polygon_base_url: str = "https://api.polygon.io"
    polygon_requests_per_minute: float = Field(default=5.0, gt=0)  # free tier; raise for paid plans
    kite_api_key: SecretStr | None = None
    kite_api_secret: SecretStr | None = None  # only needed by `quantterm.cli kite-login`
    kite_access_token: SecretStr | None = None
    # e.g. QT_KITE_SYMBOLS='["NSE:INFY", "NSE:RELIANCE"]'; empty disables the live Kite feed.
    kite_symbols: list[str] = Field(default_factory=list)
    # Upstox read-only Analytics Token (1-year validity) from account.upstox.com/developer/apps#analytics.
    upstox_analytics_token: SecretStr | None = None
    upstox_base_url: str = "https://api.upstox.com"
    # e.g. QT_UPSTOX_SYMBOLS='["NSE:INFY", "NSE:RELIANCE"]'; empty disables the live Upstox feed.
    upstox_symbols: list[str] = Field(default_factory=list)
    # The terminal runs the live feeds itself; set false when the API service already runs them.
    terminal_live_feed: bool = True

    # Listings (EXCHANGE:TICKER) whose raw_market_bars serve as the BETA benchmark per market.
    beta_benchmarks: dict[str, str] = Field(default_factory=lambda: {"US": "NYSE:SPX", "IN": "NSE:NIFTY"})
    # macro_observations series (annualised %, e.g. 3M T-bill) used as the OVME risk-free rate per market.
    risk_free_series: dict[str, str] = Field(default_factory=lambda: {"US": "FRED:DTB3", "IN": "RBI:TBILL91"})
    # Regulatory filing authority per exchange (primary) and the vendor it is reconciled against.
    fundamentals_primary_sources: dict[str, str] = Field(default_factory=lambda: {
        "NYSE": "SEC_EDGAR", "NASDAQ": "SEC_EDGAR", "NSE": "NSE_XBRL", "BSE": "BSE_XBRL"})
    fundamentals_secondary_source: str = "FMP"
    option_quote_max_age_days: int = Field(default=5, ge=1)
    watchlist: list[str] = Field(default_factory=list)  # EXCHANGE:TICKER

    agent_base_url: str = "https://api.openai.com/v1"  # any OpenAI-compatible chat-completions endpoint
    agent_model: str = "gpt-4o-mini"
    agent_api_key: SecretStr | None = None
    agent_max_steps: int = Field(default=8, ge=1, le=32)
    agent_timeout_seconds: float = Field(default=90.0, gt=0)
    agent_retries: int = Field(default=2, ge=0, le=5)
    # e.g. "low" for Gemini/OpenAI reasoning models (much faster); unset for models that reject the field.
    agent_reasoning_effort: str | None = None
    # Tried in order when the main model is out of quota / overloaded, e.g. '["gemini-3-flash-preview"]'.
    agent_fallback_models: list[str] = Field(default_factory=list)
