"""
Configuration Schema and Validation
Provides comprehensive validation for trading bot configuration using Pydantic models
"""

import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import os

logger = logging.getLogger(__name__)


class TradingMode(str, Enum):
    """Trading mode enumeration"""
    PAPER = "paper"
    LIVE = "live"


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CUSTOM = "CUSTOM"


class TradingConfig(BaseModel):
    """Complete trading configuration schema"""

    # Core settings
    mode: TradingMode = Field(default=TradingMode.PAPER, description="Trading mode")
    tickers: List[str] = Field(default_factory=list, description="List of stock tickers to trade")

    # Financial settings
    starting_balance: float = Field(gt=0, default=10000, description="Initial portfolio balance")
    current_portfolio_value: float = Field(default=10000, description="Current portfolio value")
    current_pnl: float = Field(default=0, description="Current profit/loss")

    # Risk management
    riskLevel: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Risk level preset")
    stop_loss_pct: float = Field(ge=0, le=1, default=0.05, description="Stop loss percentage")
    max_capital_per_trade: float = Field(ge=0, le=1, default=0.25, description="Max capital per trade")
    max_trade_limit: int = Field(ge=1, le=1000, default=150, description="Maximum number of trades")

    # Risk-reward settings
    use_risk_reward: bool = Field(default=True, description="Enable risk-reward ratio")
    risk_reward_ratio: float = Field(ge=1, le=10, default=2.0, description="Risk-reward ratio")
    target_profit_pct: float = Field(ge=0, le=1, default=0.10, description="Target profit percentage")

    # Technical settings
    period: str = Field(default="3y", description="Historical data period")
    prediction_days: int = Field(ge=1, le=365, default=30, description="Prediction horizon in days")
    sleep_interval: int = Field(ge=5, le=3600, default=30, description="Sleep interval between trades")

    # Benchmark and reference
    benchmark_tickers: List[str] = Field(default_factory=lambda: ["^NSEI"], description="Benchmark tickers")

    # API credentials (optional for paper trading)
    dhan_client_id: Optional[str] = Field(default=None, description="Dhan API client ID")
    dhan_access_token: Optional[str] = Field(default=None, description="Dhan API access token")

    # Advanced settings
    drawdown_limit_pct: float = Field(ge=0, le=0.5, default=0.20, description="Maximum drawdown limit")
    max_concurrent_trades: int = Field(ge=1, le=50, default=5, description="Maximum concurrent trades")

    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        use_enum_values = True
        # Preserve field names as they are defined (camelCase)
        alias_generator = None

    @validator('tickers')
    def validate_tickers(cls, v):
        """Validate ticker format"""
        if not v:
            return v

        # Basic validation for Indian stock tickers
        for ticker in v:
            if not isinstance(ticker, str) or len(ticker.strip()) == 0:
                raise ValueError(f"Invalid ticker: {ticker}")
            if len(ticker) > 20:
                raise ValueError(f"Ticker too long: {ticker}")

        return [ticker.upper().strip() for ticker in v]

    @validator('stop_loss_pct', 'max_capital_per_trade', 'target_profit_pct', 'drawdown_limit_pct')
    def validate_percentages(cls, v):
        """Validate percentage values are reasonable"""
        if v < 0 or v > 1:
            raise ValueError("Percentage values must be between 0 and 1")
        return v

    @root_validator(skip_on_failure=True)
    def validate_live_trading_requirements(cls, values):
        """Validate that live trading has required credentials"""
        mode = values.get('mode')
        client_id = values.get('dhan_client_id')
        access_token = values.get('dhan_access_token')

        if mode == TradingMode.LIVE:
            if not client_id or not access_token:
                raise ValueError(
                    "Live trading mode requires dhan_client_id and dhan_access_token to be set"
                )

        return values

    @root_validator(skip_on_failure=True)
    def validate_risk_parameters(cls, values):
        """Validate risk parameters are consistent"""
        stop_loss = values.get('stop_loss_pct', 0.05)
        target_profit = values.get('target_profit_pct', 0.10)
        risk_reward_ratio = values.get('risk_reward_ratio', 2.0)
        use_rr = values.get('use_risk_reward', True)

        if use_rr and target_profit < stop_loss * risk_reward_ratio * 0.8:
            logger.warning(
                f"Target profit ({target_profit:.1%}) is less than recommended "
                f"for risk-reward ratio ({risk_reward_ratio}:1) with stop loss ({stop_loss:.1%})"
            )

        return values


class ConfigValidator:
    """
    Enhanced configuration validator using Pydantic schema
    """

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration against schema

        Args:
            config: Raw configuration dictionary

        Returns:
            Validated and sanitized configuration dictionary

        Raises:
            ConfigurationError: If validation fails
        """
        try:
            # Create validated config using Pydantic model
            validated_config = TradingConfig(**config)

            # Convert back to dictionary for compatibility
            validated_dict = validated_config.dict(by_alias=False)

            logger.info("Configuration validation successful")
            return validated_dict

        except Exception as e:
            from utils.exceptions import ConfigurationError
            error_msg = f"Configuration validation failed: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return TradingConfig().dict(by_alias=False)

    @staticmethod
    def validate_and_merge_configs(base_config: Dict[str, Any],
                                 override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and merge two configurations

        Args:
            base_config: Base configuration
            override_config: Configuration to override with

        Returns:
            Merged and validated configuration
        """
        # Merge configs (override takes precedence)
        merged = {**base_config, **override_config}

        # Validate merged config
        return ConfigValidator.validate_config(merged)

    @staticmethod
    def get_config_schema() -> Dict[str, Any]:
        """Get JSON schema for configuration"""
        return TradingConfig.schema()

    @staticmethod
    def validate_environment_variables() -> List[str]:
        """
        Validate required environment variables for current configuration

        Returns:
            List of missing or invalid environment variables
        """
        issues = []

        # Check for required environment variables based on mode
        mode = os.getenv('MODE', 'paper')

        if mode.lower() == 'live':
            required_vars = ['DHAN_CLIENT_ID', 'DHAN_ACCESS_TOKEN']
            for var in required_vars:
                if not os.getenv(var):
                    issues.append(f"Missing required environment variable: {var}")

        # Check for optional but recommended variables
        recommended_vars = ['LOG_LEVEL', 'DATA_SERVICE_URL']
        for var in recommended_vars:
            if not os.getenv(var):
                logger.warning(f"Recommended environment variable not set: {var}")

        return issues


def load_and_validate_config(mode: str = "paper") -> Dict[str, Any]:
    """
    Load configuration from file and validate against schema

    Args:
        mode: Trading mode ("paper" or "live")

    Returns:
        Validated configuration dictionary
    """
    try:
        # Load raw config from file
        raw_config = load_config_from_file(mode)

        # Get default config
        default_config = ConfigValidator.get_default_config()

        # Merge and validate
        if raw_config:
            validated_config = ConfigValidator.validate_and_merge_configs(default_config, raw_config)
        else:
            validated_config = default_config

        # Set mode
        validated_config['mode'] = mode

        # Validate environment variables
        env_issues = ConfigValidator.validate_environment_variables()
        if env_issues:
            for issue in env_issues:
                logger.error(issue)

        logger.info(f"Configuration loaded and validated for mode: {mode}")
        return validated_config

    except Exception as e:
        logger.error(f"Failed to load and validate config: {e}")
        # Return default config as fallback
        return ConfigValidator.get_default_config()


# Import here to avoid circular imports
def load_config_from_file(mode: str) -> Dict[str, Any]:
    """Load configuration from the appropriate JSON file"""
    try:
        import os
        import json

        # Get the data directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_dir = os.path.join(project_root, "data")

        # Determine the config file path
        config_file = os.path.join(data_dir, f"{mode}_config.json")

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return config_data
        else:
            logger.info(f"Config file {config_file} not found, using defaults")
            return {}

    except Exception as e:
        logger.error(f"Error loading config from file: {e}")
        return {}
