import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { formatCurrency, apiService } from '../services/apiService';

const PortfolioContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
  overflow: hidden;
`;

const Section = styled.div`
  h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 5px;
  }
`;

const HoldingsTable = styled.div`
  background: #f8f9fa;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #e9ecef;
  max-height: 200px;
  overflow-y: auto;

  /* Custom scrollbar styling */
  &::-webkit-scrollbar {
    width: 8px;
  }

  &::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
  }

  &::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 4px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
`;

const HoldingsHeader = styled.div`
  display: grid;
  grid-template-columns: 1.5fr 0.8fr 1fr 1fr 1fr 1fr 1fr 0.8fr;
  gap: 10px;
  padding: 15px;
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  font-weight: bold;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;

  @media (max-width: 768px) {
    grid-template-columns: 1fr 0.8fr 1fr 1fr;
    font-size: 0.7rem;
    gap: 5px;
  }
`;

const HoldingsRow = styled.div`
  display: grid;
  grid-template-columns: 1.5fr 0.8fr 1fr 1fr 1fr 1fr 1fr 0.8fr;
  gap: 10px;
  padding: 15px;
  align-items: center;
  border-bottom: 1px solid #e9ecef;
  background: white;
  transition: all 0.2s ease;

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    background: #f8f9fa;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }

  @media (max-width: 768px) {
    grid-template-columns: 1fr 0.8fr 1fr 1fr;
    font-size: 0.8rem;
    gap: 5px;
  }
`;

const TickerName = styled.div`
  font-weight: bold;
`;

const ProfitLoss = styled.div`
  font-weight: bold;
  color: ${props => {
    if (props.value > 0) return '#27ae60'; // Green for profit
    if (props.value < 0) return '#e74c3c'; // Red for loss
    return '#7f8c8d'; // Gray for neutral
  }};
`;

const TradeBadge = styled.span`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  background: ${props => props.type === 'BUY' ? '#27ae60' : '#e74c3c'};
  color: white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
`;

const StatusBadge = styled.span`
  display: inline-block;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  background: ${props => {
    switch (props.status) {
      case 'ACTIVE': return '#27ae60'; // Green for active holding
      case 'PROFIT': return '#2ecc71'; // Lighter green for profit
      case 'LOSS': return '#e74c3c'; // Red for loss
      default: return '#3498db'; // Blue default
    }
  }};
`;

const PortfolioSummary = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 20px;
  padding: 20px;
  background: linear-gradient(135deg, #f8f9fa, #e9ecef);
  border-radius: 10px;
  border: 1px solid #dee2e6;
`;

const SummaryCard = styled.div`
  text-align: center;
  padding: 15px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);

  h4 {
    margin: 0 0 10px 0;
    color: #2c3e50;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .value {
    font-size: 1.4rem;
    font-weight: bold;
    color: ${props => props.valueColor || '#2c3e50'};
  }

  .percentage {
    font-size: 0.9rem;
    margin-top: 5px;
    font-weight: 500;
  }
`;

const PerformanceSection = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
`;

const PerformanceCard = styled.div`
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  border-left: 4px solid ${props => props.borderColor || '#3498db'};

  .title {
    font-size: 0.9rem;
    color: #7f8c8d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
    font-weight: 600;
  }

  .main-value {
    font-size: 2rem;
    font-weight: bold;
    color: ${props => props.valueColor || '#2c3e50'};
    margin-bottom: 5px;
  }

  .sub-value {
    font-size: 1rem;
    color: ${props => props.subColor || '#7f8c8d'};
    font-weight: 500;
  }

  .change {
    font-size: 0.9rem;
    margin-top: 8px;
    padding: 4px 8px;
    border-radius: 4px;
    background: ${props => props.changeColor === 'positive' ? '#d4edda' : props.changeColor === 'negative' ? '#f8d7da' : '#e2e3e5'};
    color: ${props => props.changeColor === 'positive' ? '#155724' : props.changeColor === 'negative' ? '#721c24' : '#6c757d'};
    display: inline-block;
  }
`;

const LastUpdateTime = styled.div`
  font-size: 0.8rem;
  color: #7f8c8d;
  text-align: right;
  margin-bottom: 10px;
  font-style: italic;
`;

const RealTimeIndicator = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 0.8rem;
  color: #27ae60;
  margin-bottom: 10px;

  &::before {
    content: '';
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #27ae60;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

const WatchlistControls = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 20px;

  @media (max-width: 768px) {
    flex-direction: column;
  }
`;

const TickerInput = styled.input`
  flex: 1;
  padding: 10px;
  border: 2px solid #e9ecef;
  border-radius: 6px;
  font-size: 1rem;

  &:focus {
    outline: none;
    border-color: #3498db;
  }
`;

const AddButton = styled.button`
  background: #27ae60;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;

  &:hover {
    background: #229954;
  }

  &:disabled {
    background: #95a5a6;
    cursor: not-allowed;
  }
`;

const WatchlistGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px;

  @media (max-width: 768px) {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  }
`;

const WatchlistItem = styled.div`
  background: #3498db;
  color: white;
  padding: 10px;
  border-radius: 6px;
  text-align: center;
  font-weight: 500;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const RemoveButton = styled.button`
  position: absolute;
  top: -5px;
  right: -5px;
  background: #e74c3c;
  color: white;
  border: none;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  cursor: pointer;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;

  &:hover {
    background: #c0392b;
    transform: scale(1.1);
  }
`;

const NoHoldings = styled.div`
  text-align: center;
  color: #7f8c8d;
  font-style: italic;
  padding: 20px;
`;

const CurrentWatchlist = styled.div`
  h4 {
    color: #2c3e50;
    margin-bottom: 10px;
  }
`;

const Portfolio = ({ botData, onAddTicker, onRemoveTicker }) => {
  const [newTicker, setNewTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Fetch trade history on component mount
  useEffect(() => {
    const fetchTradeHistory = async () => {
      try {
        const trades = await apiService.getTrades(50);
        setTradeHistory(trades);
      } catch (error) {
        console.error('Error fetching trade history:', error);
      }
    };

    fetchTradeHistory();
  }, []);

  // Real-time updates every 10 seconds
  useEffect(() => {
    const interval = setInterval(async () => {
      setLastUpdate(new Date());
      try {
        // Refetch trade history for real-time updates
        const trades = await apiService.getTrades(50);
        setTradeHistory(trades);
      } catch (error) {
        console.error('Error updating real-time data:', error);
      }
    }, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, []);

  // Function to get the most recent trade type for a ticker
  const getLastTradeType = (ticker) => {
    const tickerTrades = tradeHistory.filter(trade => trade.asset === ticker);
    if (tickerTrades.length === 0) return 'BUY'; // Default to BUY if no trades found

    // Sort by timestamp and get the most recent
    const sortedTrades = tickerTrades.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    return sortedTrades[0].action.toUpperCase();
  };

  const handleAddTicker = async () => {
    if (!newTicker.trim()) {
      alert('Please enter a ticker symbol');
      return;
    }

    if (botData.config.tickers.includes(newTicker.toUpperCase())) {
      alert('Ticker already in watchlist');
      return;
    }

    setLoading(true);
    try {
      await onAddTicker(newTicker.toUpperCase());
      setNewTicker('');
    } catch (error) {
      console.error('Error adding ticker:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveTicker = async (ticker) => {
    setLoading(true);
    try {
      await onRemoveTicker(ticker);
    } catch (error) {
      console.error('Error removing ticker:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleAddTicker();
    }
  };

  const calculatePortfolioPercentage = (currentValue, totalValue) => {
    if (!totalValue || totalValue <= 0 || !currentValue || currentValue <= 0) {
      return '0.0';
    }
    const percentage = ((currentValue / totalValue) * 100);
    return isNaN(percentage) ? '0.0' : percentage.toFixed(1);
  };

  const holdings = botData.portfolio.holdings || {};
  const totalValue = botData.portfolio.totalValue || 0;
  const cash = botData.portfolio.cash || 0;
  const unrealizedPnL = botData.portfolio.unrealizedPnL || 0;
  const totalReturn = botData.portfolio.totalReturn || 0;
  const totalReturnPercentage = botData.portfolio.totalReturnPercentage || 0;
  const realizedPnL = botData.portfolio.realizedPnL || 0;
  const totalInvested = botData.portfolio.totalInvested || 10000; // Starting balance
  // const portfolioPerformance = botData.portfolio.performance || [];

  return (
    <PortfolioContainer>
      {/* Real-time Portfolio Performance */}
      <Section>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h3>Portfolio Performance</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
            <RealTimeIndicator>
              Live Performance
            </RealTimeIndicator>
            <LastUpdateTime>
              Last updated: {lastUpdate.toLocaleTimeString()}
            </LastUpdateTime>
          </div>
        </div>
        <PerformanceSection>
          <PerformanceCard
            borderColor="#3498db"
            valueColor="#2c3e50"
            changeColor={totalReturn >= 0 ? 'positive' : 'negative'}
          >
            <div className="title">Total Portfolio Value</div>
            <div className="main-value">{formatCurrency(totalValue)}</div>
            <div className="sub-value">Cash: {formatCurrency(cash)}</div>
            <div className="change">
              {totalReturn >= 0 ? '+' : ''}{formatCurrency(totalReturn)}
              ({totalReturnPercentage >= 0 ? '+' : ''}{totalReturnPercentage.toFixed(2)}%)
            </div>
          </PerformanceCard>

          <PerformanceCard
            borderColor={unrealizedPnL >= 0 ? "#27ae60" : "#e74c3c"}
            valueColor={unrealizedPnL >= 0 ? "#27ae60" : "#e74c3c"}
            changeColor={unrealizedPnL >= 0 ? 'positive' : 'negative'}
          >
            <div className="title">Unrealized P&L</div>
            <div className="main-value">
              {unrealizedPnL >= 0 ? '+' : ''}{formatCurrency(unrealizedPnL)}
            </div>
            <div className="sub-value">Open Positions</div>
            <div className="change">
              {((unrealizedPnL / totalInvested) * 100).toFixed(2)}% of invested
            </div>
          </PerformanceCard>

          <PerformanceCard
            borderColor={realizedPnL >= 0 ? "#27ae60" : "#e74c3c"}
            valueColor={realizedPnL >= 0 ? "#27ae60" : "#e74c3c"}
            changeColor={realizedPnL >= 0 ? 'positive' : 'negative'}
          >
            <div className="title">Realized P&L</div>
            <div className="main-value">
              {realizedPnL >= 0 ? '+' : ''}{formatCurrency(realizedPnL)}
            </div>
            <div className="sub-value">Closed Positions</div>
            <div className="change">
              {((realizedPnL / totalInvested) * 100).toFixed(2)}% of invested
            </div>
          </PerformanceCard>

          <PerformanceCard
            borderColor="#f39c12"
            valueColor="#2c3e50"
            changeColor="neutral"
          >
            <div className="title">Total Return</div>
            <div className="main-value">
              {totalReturn >= 0 ? '+' : ''}{formatCurrency(totalReturn)}
            </div>
            <div className="sub-value">Overall Performance</div>
            <div className="change">
              {totalReturnPercentage >= 0 ? '+' : ''}{totalReturnPercentage.toFixed(2)}% return
            </div>
          </PerformanceCard>
        </PerformanceSection>
      </Section>

      {/* Current Holdings */}
      <Section>
        <h3>Current Holdings</h3>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <RealTimeIndicator>
            Live Portfolio Data
          </RealTimeIndicator>
          <LastUpdateTime>
            Last updated: {lastUpdate.toLocaleTimeString()}
          </LastUpdateTime>
        </div>

        <HoldingsTable>
          {Object.keys(holdings).length > 0 ? (
            <>
              <HoldingsHeader>
                <div>Stock Symbol</div>
                <div>Status</div>
                <div>Last Trade</div>
                <div>Quantity</div>
                <div>Avg Price</div>
                <div>Current Price</div>
                <div>Profit/Loss</div>
                <div>% Portfolio</div>
              </HoldingsHeader>
              {Object.entries(holdings).length === 0 ? (
                <HoldingsRow>
                  <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '40px', color: '#7f8c8d' }}>
                    <i className="fas fa-chart-line" style={{ fontSize: '2rem', marginBottom: '10px' }}></i>
                    <div>No stocks in portfolio yet</div>
                    <div style={{ fontSize: '0.9rem', marginTop: '5px' }}>
                      Start the trading bot to begin building your portfolio
                    </div>
                  </div>
                </HoldingsRow>
              ) : (
                Object.entries(holdings).map(([ticker, data]) => {
                  // Use currentPrice if available, otherwise fall back to avg_price
                  const currentPrice = data.currentPrice || data.avg_price || 0;
                  const avgPrice = data.avg_price || 0;
                  const qty = data.qty || 0;
                  const currentValue = qty * currentPrice;
                  const costBasis = qty * avgPrice;
                  const profitLoss = currentValue - costBasis;
                  const portfolioPercentage = calculatePortfolioPercentage(currentValue, totalValue);

                  // Get the actual last trade type from trade history
                  const lastTradeType = getLastTradeType(ticker);

                  // Determine status based on profit/loss and quantity
                  const status = qty > 0 ? (profitLoss >= 0 ? 'PROFIT' : 'LOSS') : 'ACTIVE';

                  return (
                    <HoldingsRow key={ticker}>
                      <TickerName>{ticker}</TickerName>
                      <div>
                        <StatusBadge status={status}>{status}</StatusBadge>
                      </div>
                      <div>
                        <TradeBadge type={lastTradeType}>{lastTradeType}</TradeBadge>
                      </div>
                      <div>{qty.toFixed(2)}</div>
                      <div>{formatCurrency(avgPrice)}</div>
                      <div>{formatCurrency(currentPrice)}</div>
                      <ProfitLoss value={profitLoss}>
                        {profitLoss >= 0 ? '+' : ''}{formatCurrency(profitLoss)}
                      </ProfitLoss>
                      <div>{portfolioPercentage}%</div>
                    </HoldingsRow>
                  );
                })
              )}
            </>
          ) : (
            <NoHoldings>No current holdings</NoHoldings>
          )}
        </HoldingsTable>
      </Section>

      {/* Watchlist Management */}
      <Section>
        <h3>Watchlist Management</h3>

        <WatchlistControls>
          <TickerInput
            type="text"
            placeholder="Add Ticker (e.g., INFY.NS)"
            value={newTicker}
            onChange={(e) => setNewTicker(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
          />
          <AddButton
            onClick={handleAddTicker}
            disabled={loading || !newTicker.trim()}
          >
            {loading ? 'Adding...' : 'Add Ticker'}
          </AddButton>
        </WatchlistControls>

        <CurrentWatchlist>
          <h4>Current Watchlist:</h4>
          <WatchlistGrid>
            {botData.config.tickers.map(ticker => (
              <WatchlistItem key={ticker}>
                {ticker}
                <RemoveButton
                  onClick={() => handleRemoveTicker(ticker)}
                  disabled={loading}
                  title={`Remove ${ticker}`}
                >
                  ×
                </RemoveButton>
              </WatchlistItem>
            ))}
          </WatchlistGrid>
        </CurrentWatchlist>
      </Section>
    </PortfolioContainer>
  );
};

export default Portfolio;
