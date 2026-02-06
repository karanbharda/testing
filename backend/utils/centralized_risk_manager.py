#!/usr/bin/env python3
"""
Centralized Risk Management System
==================================
Production-grade centralized risk management with:
- Portfolio-level risk controls
- Position-level risk monitoring
- Dynamic risk parameter adjustment
- Real-time risk limits enforcement
- Comprehensive risk reporting
- Integration with all trading components
"""
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import logging
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskLimits:
    """Centralized risk limits configuration"""
    # Portfolio-level limits
    max_portfolio_var: float = 0.05  # 5% daily VaR
    max_portfolio_drawdown: float = 0.20  # 20% maximum drawdown
    max_daily_loss: float = 0.03  # 3% maximum daily loss
    max_exposure: float = 1.0  # 100% maximum portfolio exposure
    
    # Position-level limits
    max_position_size: float = 0.25  # 25% maximum single position
    max_sector_concentration: float = 0.40  # 40% maximum sector exposure
    min_liquidity_score: float = 0.3  # Minimum liquidity threshold
    
    # Market and correlation limits
    max_correlation_risk: float = 0.8  # Maximum correlation between positions
    max_beta_exposure: float = 1.5  # Maximum market beta
    volatility_threshold: float = 0.04  # 4% volatility threshold for adjustments
    
    # Dynamic adjustment parameters
    volatility_sensitivity: float = 0.3
    market_regime_sensitivity: float = 0.5
    performance_sensitivity: float = 0.4

@dataclass
class PortfolioRiskMetrics:
    """Current portfolio risk metrics"""
    portfolio_value: float = 0.0
    total_exposure: float = 0.0
    current_var: float = 0.0
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    concentration_metrics: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PositionRiskMetrics:
    """Risk metrics for individual positions"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    exposure: float
    unrealized_pnl: float
    stop_loss_price: float
    take_profit_price: float
    risk_amount: float
    reward_potential: float
    risk_reward_ratio: float
    atr: float
    volatility: float
    beta: float
    liquidity_score: float
    sector: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

class CentralizedRiskManager:
    """Production-grade centralized risk management system"""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        
        # Configuration
        self.config_file = Path(config.get("config_file", "config/risk_limits.json"))
        self.risk_limits = self._load_risk_limits()
        self.monitoring_frequency = config.get("monitoring_frequency", 60)  # seconds
        self.alert_thresholds = config.get("alert_thresholds", {
            "warning": 0.8,
            "alert": 0.9,
            "critical": 0.95
        })
        
        # Risk monitoring components
        self.portfolio_metrics: Optional[PortfolioRiskMetrics] = None
        self.position_metrics: Dict[str, PositionRiskMetrics] = {}
        self.risk_history: List[Dict] = []
        self.alerts: List[Dict] = []
        
        # Dynamic risk adjustment
        self.current_risk_level = RiskLevel.MEDIUM
        self.risk_adjustment_factor = 1.0
        self.last_adjustment = datetime.now()
        
        # Integration points
        self.trading_system = None  # Will be set by main system
        self.portfolio_manager = None  # Will be set by main system
        self.data_service = None  # Will be set by main system
        
        logger.info("Centralized Risk Manager initialized")
    
    def _load_risk_limits(self) -> RiskLimits:
        """Load risk limits from configuration file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                return RiskLimits(**config_data)
            else:
                # Create default configuration file
                default_limits = RiskLimits()
                self._save_risk_limits(default_limits)
                return default_limits
        except Exception as e:
            logger.warning(f"Failed to load risk limits, using defaults: {e}")
            return RiskLimits()
    
    def _save_risk_limits(self, limits: RiskLimits):
        """Save risk limits to configuration file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(limits.__dict__, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save risk limits: {e}")
    
    def pre_trade_check(self, 
                       symbol: str,
                       quantity: float,
                       price: float,
                       order_type: str = "market") -> Tuple[bool, str]:
        """
        Comprehensive pre-trade risk check
        Returns: (approved, reason)
        """
        try:
            # Update current risk metrics
            self._update_risk_metrics()
            
            # Check portfolio-level limits
            portfolio_check = self._check_portfolio_limits(symbol, quantity, price)
            if not portfolio_check[0]:
                return portfolio_check
            
            # Check position-level limits
            position_check = self._check_position_limits(symbol, quantity, price)
            if not position_check[0]:
                return position_check
            
            # Check market conditions
            market_check = self._check_market_conditions(symbol)
            if not market_check[0]:
                return market_check
            
            # Check correlation and concentration
            concentration_check = self._check_concentration_risk(symbol, quantity)
            if not concentration_check[0]:
                return concentration_check
            
            # All checks passed
            return (True, "All risk checks passed")
            
        except Exception as e:
            logger.error(f"Pre-trade check failed: {e}")
            return (False, f"Risk check error: {str(e)}")
    
    def _check_portfolio_limits(self, symbol: str, quantity: float, price: float) -> Tuple[bool, str]:
        """Check portfolio-level risk limits"""
        if not self.portfolio_metrics:
            return (True, "Portfolio metrics not available")
        
        # Calculate proposed new exposure
        trade_value = quantity * price
        new_total_exposure = self.portfolio_metrics.total_exposure + trade_value
        
        # Check maximum exposure limit
        if new_total_exposure > self.risk_limits.max_exposure * self.portfolio_metrics.portfolio_value:
            return (False, f"Exposure limit exceeded: {new_total_exposure/self.portfolio_metrics.portfolio_value:.1%}")
        
        # Check maximum position size
        if trade_value > self.risk_limits.max_position_size * self.portfolio_metrics.portfolio_value:
            return (False, f"Position size limit exceeded: {trade_value/self.portfolio_metrics.portfolio_value:.1%}")
        
        # Check portfolio VaR
        proposed_var = self._calculate_proposed_var(symbol, quantity, price)
        if proposed_var > self.risk_limits.max_portfolio_var:
            return (False, f"Portfolio VaR limit exceeded: {proposed_var:.1%}")
        
        # Check drawdown limits
        if self.portfolio_metrics.current_drawdown < -self.risk_limits.max_portfolio_drawdown:
            return (False, f"Maximum drawdown limit exceeded: {self.portfolio_metrics.current_drawdown:.1%}")
        
        return (True, "Portfolio limits check passed")
    
    def _check_position_limits(self, symbol: str, quantity: float, price: float) -> Tuple[bool, str]:
        """Check position-level risk limits"""
        # Check existing position size
        existing_position = self.position_metrics.get(symbol)
        if existing_position:
            new_quantity = existing_position.quantity + quantity
            new_exposure = new_quantity * price
            
            if new_exposure > self.risk_limits.max_position_size * self.portfolio_metrics.portfolio_value:
                return (False, f"Position size limit would be exceeded: {new_exposure/self.portfolio_metrics.portfolio_value:.1%}")
        
        # Check stop-loss and take-profit
        position_risk = self._calculate_position_risk(symbol, quantity, price)
        if position_risk.risk_reward_ratio < 1.0:
            return (False, f"Poor risk-reward ratio: {position_risk.risk_reward_ratio:.2f}")
        
        # Check liquidity
        if position_risk.liquidity_score < self.risk_limits.min_liquidity_score:
            return (False, f"Insufficient liquidity: {position_risk.liquidity_score:.2f}")
        
        return (True, "Position limits check passed")
    
    def _check_market_conditions(self, symbol: str) -> Tuple[bool, str]:
        """Check current market conditions"""
        # Check market volatility
        current_volatility = self._get_market_volatility(symbol)
        if current_volatility > self.risk_limits.volatility_threshold:
            # Apply dynamic risk adjustment
            self._adjust_risk_for_volatility(current_volatility)
            if self.risk_adjustment_factor < 0.5:
                return (False, f"High market volatility: {current_volatility:.2%}")
        
        # Check market regime
        market_regime = self._get_market_regime()
        if market_regime == "crisis":
            return (False, "Market in crisis regime")
        elif market_regime == "high_volatility":
            # Reduce position sizes by 30%
            self.risk_adjustment_factor *= 0.7
        
        return (True, "Market conditions acceptable")
    
    def _check_concentration_risk(self, symbol: str, quantity: float) -> Tuple[bool, str]:
        """Check sector and correlation concentration risk"""
        # Get sector information
        sector = self._get_symbol_sector(symbol)
        
        # Calculate new sector exposure
        sector_exposure = self.portfolio_metrics.sector_exposures.get(sector, 0.0)
        trade_value = quantity * self._get_current_price(symbol)
        new_sector_exposure = sector_exposure + trade_value
        
        if new_sector_exposure > self.risk_limits.max_sector_concentration * self.portfolio_metrics.portfolio_value:
            return (False, f"Sector concentration limit exceeded: {sector} exposure {new_sector_exposure/self.portfolio_metrics.portfolio_value:.1%}")
        
        # Check correlation with existing positions
        correlation_risk = self._calculate_correlation_risk(symbol)
        if correlation_risk > self.risk_limits.max_correlation_risk:
            return (False, f"High correlation risk: {correlation_risk:.2f}")
        
        return (True, "Concentration risk check passed")
    
    def real_time_monitoring(self):
        """Continuous real-time risk monitoring"""
        try:
            while True:
                # Update risk metrics
                self._update_risk_metrics()
                
                # Check for risk violations
                violations = self._check_risk_violations()
                
                # Generate alerts for violations
                for violation in violations:
                    self._generate_alert(violation)
                
                # Adjust risk parameters dynamically
                self._dynamic_risk_adjustment()
                
                # Wait for next monitoring cycle
                import time
                time.sleep(self.monitoring_frequency)
                
        except KeyboardInterrupt:
            logger.info("Risk monitoring stopped by user")
        except Exception as e:
            logger.error(f"Risk monitoring error: {e}")
    
    def _update_risk_metrics(self):
        """Update all risk metrics from connected systems"""
        try:
            # Get portfolio data from portfolio manager
            if self.portfolio_manager:
                portfolio_data = self.portfolio_manager.get_current_portfolio()
                self.portfolio_metrics = self._calculate_portfolio_metrics(portfolio_data)
            
            # Get position data
            if self.trading_system:
                positions = self.trading_system.get_active_positions()
                self.position_metrics = self._calculate_position_metrics(positions)
            
            # Store in history
            self.risk_history.append({
                "timestamp": datetime.now(),
                "portfolio_metrics": self.portfolio_metrics.__dict__ if self.portfolio_metrics else {},
                "position_count": len(self.position_metrics),
                "risk_level": self.current_risk_level.value
            })
            
        except Exception as e:
            logger.error(f"Failed to update risk metrics: {e}")
    
    def _calculate_portfolio_metrics(self, portfolio_data) -> PortfolioRiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        # Calculate portfolio value and exposure
        portfolio_value = portfolio_data.cash + sum(
            pos.quantity * pos.current_price for pos in portfolio_data.positions.values()
        )
        
        total_exposure = sum(
            pos.quantity * pos.current_price for pos in portfolio_data.positions.values()
        )
        
        # Calculate VaR (simplified)
        returns = self._get_portfolio_returns_history()
        if len(returns) > 30:
            var_95 = np.percentile(returns, 5)  # 95% VaR
        else:
            var_95 = 0.02  # Default 2%
        
        # Calculate drawdown
        equity_curve = self._get_equity_history()
        if len(equity_curve) > 1:
            current_drawdown = self._calculate_drawdown(equity_curve)
        else:
            current_drawdown = 0.0
        
        # Calculate daily P&L
        daily_pnl = self._calculate_daily_pnl()
        
        # Calculate sector exposures
        sector_exposures = self._calculate_sector_exposures(portfolio_data.positions)
        
        # Calculate concentration metrics
        concentration_metrics = self._calculate_concentration_metrics(portfolio_data.positions)
        
        return PortfolioRiskMetrics(
            portfolio_value=portfolio_value,
            total_exposure=total_exposure,
            current_var=var_95,
            current_drawdown=current_drawdown,
            daily_pnl=daily_pnl,
            sector_exposures=sector_exposures,
            concentration_metrics=concentration_metrics
        )
    
    def _calculate_position_metrics(self, positions: Dict[str, Any]) -> Dict[str, PositionRiskMetrics]:
        """Calculate risk metrics for all positions"""
        metrics = {}
        
        for symbol, position in positions.items():
            metrics[symbol] = PositionRiskMetrics(
                symbol=symbol,
                quantity=position.quantity,
                avg_price=position.avg_price,
                current_price=position.current_price,
                exposure=position.quantity * position.current_price,
                unrealized_pnl=position.unrealized_pnl,
                stop_loss_price=position.stop_loss_price,
                take_profit_price=position.take_profit_price,
                risk_amount=abs(position.avg_price - position.stop_loss_price) * position.quantity,
                reward_potential=abs(position.take_profit_price - position.avg_price) * position.quantity,
                risk_reward_ratio=position.reward_potential / (position.risk_amount + 1e-10),
                atr=self._get_symbol_atr(symbol),
                volatility=self._get_symbol_volatility(symbol),
                beta=self._get_symbol_beta(symbol),
                liquidity_score=self._get_symbol_liquidity(symbol),
                sector=self._get_symbol_sector(symbol)
            )
        
        return metrics
    
    def _check_risk_violations(self) -> List[Dict[str, Any]]:
        """Check for current risk violations"""
        violations = []
        
        if not self.portfolio_metrics:
            return violations
        
        # Check portfolio VaR
        if self.portfolio_metrics.current_var > self.risk_limits.max_portfolio_var:
            violations.append({
                "type": "portfolio_var",
                "level": "CRITICAL" if self.portfolio_metrics.current_var > self.risk_limits.max_portfolio_var * 1.5 else "HIGH",
                "value": self.portfolio_metrics.current_var,
                "limit": self.risk_limits.max_portfolio_var,
                "description": f"Portfolio VaR {self.portfolio_metrics.current_var:.2%} exceeds limit {self.risk_limits.max_portfolio_var:.2%}"
            })
        
        # Check drawdown
        if self.portfolio_metrics.current_drawdown < -self.risk_limits.max_portfolio_drawdown:
            violations.append({
                "type": "drawdown",
                "level": "CRITICAL",
                "value": self.portfolio_metrics.current_drawdown,
                "limit": -self.risk_limits.max_portfolio_drawdown,
                "description": f"Portfolio drawdown {self.portfolio_metrics.current_drawdown:.2%} exceeds limit {-self.risk_limits.max_portfolio_drawdown:.2%}"
            })
        
        # Check daily loss
        if self.portfolio_metrics.daily_pnl < -self.risk_limits.max_daily_loss * self.portfolio_metrics.portfolio_value:
            violations.append({
                "type": "daily_loss",
                "level": "HIGH",
                "value": self.portfolio_metrics.daily_pnl / self.portfolio_metrics.portfolio_value,
                "limit": -self.risk_limits.max_daily_loss,
                "description": f"Daily loss {-self.portfolio_metrics.daily_pnl/self.portfolio_metrics.portfolio_value:.2%} exceeds limit {-self.risk_limits.max_daily_loss:.2%}"
            })
        
        # Check position-level violations
        for symbol, metrics in self.position_metrics.items():
            if metrics.exposure > self.risk_limits.max_position_size * self.portfolio_metrics.portfolio_value:
                violations.append({
                    "type": "position_size",
                    "level": "HIGH",
                    "symbol": symbol,
                    "value": metrics.exposure / self.portfolio_metrics.portfolio_value,
                    "limit": self.risk_limits.max_position_size,
                    "description": f"Position {symbol} size {metrics.exposure/self.portfolio_metrics.portfolio_value:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}"
                })
        
        return violations
    
    def _generate_alert(self, violation: Dict[str, Any]):
        """Generate risk alert"""
        alert = {
            "timestamp": datetime.now(),
            "type": violation["type"],
            "level": violation["level"],
            "description": violation["description"],
            "value": violation["value"],
            "limit": violation["limit"]
        }
        
        self.alerts.append(alert)
        logger.warning(f"RISK ALERT [{violation['level']}]: {violation['description']}")
        
        # Trigger appropriate actions based on alert level
        if violation["level"] == "CRITICAL":
            self._trigger_critical_actions()
        elif violation["level"] == "HIGH":
            self._trigger_high_risk_actions()
    
    def _dynamic_risk_adjustment(self):
        """Dynamically adjust risk parameters based on current conditions"""
        if not self.portfolio_metrics:
            return
        
        # Adjust based on portfolio performance
        recent_performance = self._get_recent_performance()
        if recent_performance < 0:
            self.risk_adjustment_factor *= 0.9  # Reduce risk
        elif recent_performance > 0.02:  # 2% positive
            self.risk_adjustment_factor = min(1.0, self.risk_adjustment_factor * 1.05)  # Increase risk cautiously
        
        # Adjust based on market volatility
        market_volatility = self._get_market_volatility("NIFTY")
        if market_volatility > self.risk_limits.volatility_threshold:
            self.risk_adjustment_factor *= (1.0 - self.risk_limits.volatility_sensitivity)
        else:
            self.risk_adjustment_factor = min(1.0, self.risk_adjustment_factor * 1.02)
        
        # Update risk level classification
        self.current_risk_level = self._classify_risk_level()
        
        # Apply adjustments to risk limits
        self._apply_risk_adjustments()
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "current_risk_level": self.current_risk_level.value,
            "risk_adjustment_factor": self.risk_adjustment_factor,
            "portfolio_metrics": self.portfolio_metrics.__dict__ if self.portfolio_metrics else {},
            "position_metrics": {symbol: metrics.__dict__ for symbol, metrics in self.position_metrics.items()},
            "recent_alerts": self.alerts[-10:],  # Last 10 alerts
            "risk_limits": self.risk_limits.__dict__,
            "risk_history_length": len(self.risk_history)
        }
    
    # Helper methods (these would connect to your actual data sources)
    def _get_portfolio_returns_history(self) -> List[float]:
        """Get historical portfolio returns"""
        # Implementation would connect to your portfolio manager
        return []
    
    def _get_equity_history(self) -> List[float]:
        """Get historical equity values"""
        # Implementation would connect to your portfolio manager
        return []
    
    def _calculate_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_curve:
            return 0.0
        cumulative_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cumulative_max) / cumulative_max
        return np.min(drawdown)
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        # Implementation would connect to your portfolio manager
        return 0.0
    
    def _calculate_sector_exposures(self, positions: Dict) -> Dict[str, float]:
        """Calculate exposure by sector"""
        exposures = {}
        # Implementation would categorize positions by sector
        return exposures
    
    def _calculate_concentration_metrics(self, positions: Dict) -> Dict[str, float]:
        """Calculate concentration risk metrics"""
        # Implementation for Herfindahl index, etc.
        return {}
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        # Implementation would connect to your market data service
        return 0.0
    
    def _get_symbol_atr(self, symbol: str) -> float:
        """Get Average True Range for symbol"""
        # Implementation would connect to your technical analysis service
        return 0.0
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility for symbol"""
        # Implementation would connect to your market data service
        return 0.0
    
    def _get_symbol_beta(self, symbol: str) -> float:
        """Get beta for symbol"""
        # Implementation would connect to your market data service
        return 1.0
    
    def _get_symbol_liquidity(self, symbol: str) -> float:
        """Get liquidity score for symbol"""
        # Implementation would connect to your market data service
        return 0.5
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        # Implementation would connect to your market data service
        return "UNKNOWN"
    
    def _get_market_volatility(self, symbol: str) -> float:
        """Get current market volatility"""
        # Implementation would connect to your market data service
        return 0.02
    
    def _get_market_regime(self) -> str:
        """Get current market regime"""
        # Implementation would connect to your market analysis service
        return "normal"
    
    def _calculate_proposed_var(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate proposed portfolio VaR with new position"""
        # Simplified implementation
        return self.portfolio_metrics.current_var if self.portfolio_metrics else 0.02
    
    def _calculate_position_risk(self, symbol: str, quantity: float, price: float) -> PositionRiskMetrics:
        """Calculate risk metrics for a position"""
        # Simplified implementation
        return PositionRiskMetrics(
            symbol=symbol,
            quantity=quantity,
            avg_price=price,
            current_price=price,
            exposure=quantity * price,
            unrealized_pnl=0.0,
            stop_loss_price=price * 0.95,
            take_profit_price=price * 1.15,
            risk_amount=quantity * price * 0.05,
            reward_potential=quantity * price * 0.15,
            risk_reward_ratio=3.0,
            atr=price * 0.02,
            volatility=0.02,
            beta=1.0,
            liquidity_score=0.5
        )
    
    def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with existing positions"""
        # Simplified implementation
        return 0.5
    
    def _get_recent_performance(self) -> float:
        """Get recent portfolio performance"""
        # Implementation would connect to your portfolio manager
        return 0.0
    
    def _classify_risk_level(self) -> RiskLevel:
        """Classify current risk level"""
        if self.risk_adjustment_factor < 0.3:
            return RiskLevel.CRITICAL
        elif self.risk_adjustment_factor < 0.6:
            return RiskLevel.HIGH
        elif self.risk_adjustment_factor < 0.8:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _apply_risk_adjustments(self):
        """Apply risk adjustments to limits"""
        # Adjust position sizes
        self.risk_limits.max_position_size *= self.risk_adjustment_factor
        
        # Adjust portfolio limits
        self.risk_limits.max_portfolio_var *= self.risk_adjustment_factor
        
        # Ensure minimum limits
        self.risk_limits.max_position_size = max(0.05, self.risk_limits.max_position_size)
        self.risk_limits.max_portfolio_var = max(0.01, self.risk_limits.max_portfolio_var)
    
    def _trigger_critical_actions(self):
        """Trigger critical risk actions"""
        logger.critical("CRITICAL RISK LEVEL - Triggering emergency actions")
        # Implementation for emergency risk actions
        
    def _trigger_high_risk_actions(self):
        """Trigger high risk actions"""
        logger.warning("HIGH RISK LEVEL - Reducing position sizes and tightening stops")
        # Implementation for high risk actions

# Integration example
if __name__ == "__main__":
    # Example usage
    config = {
        "config_file": "config/risk_limits.json",
        "monitoring_frequency": 30
    }
    
    risk_manager = CentralizedRiskManager(config)
    
    # Example pre-trade check
    approved, reason = risk_manager.pre_trade_check("RELIANCE", 100, 2500.0)
    print(f"Trade approved: {approved}, Reason: {reason}")
    
    # Get risk report
    report = risk_manager.get_risk_report()
    print(f"Current risk level: {report['current_risk_level']}")