import React from 'react';
import styled from 'styled-components';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';
import { Line, Doughnut } from 'react-chartjs-2';
import { formatCurrency } from '../services/apiService';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, ArcElement);

const DashboardContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  min-height: 100%;
  overflow: visible;
  padding-bottom: 20px;
`;

const MetricsRow = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
`;

const MetricCard = styled.div`
  background: white;
  color: #333;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  padding: 20px;
  border-radius: 12px;
  text-align: center;
`;

const MetricLabel = styled.div`
  font-size: 0.9rem;
  color: #7f8c8d;
  margin-bottom: 8px;
`;

const MetricValue = styled.div`
  font-size: 1.8rem;
  font-weight: bold;
  color: #2c3e50;
`;

const ChartsSection = styled.div`
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 20px;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ChartContainer = styled.div`
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  border: 1px solid #e9ecef;

  h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    text-align: center;
  }
`;

const ActivitySection = styled.div`
  background: #f8f9fa;
  padding: 20px;
  border-radius: 10px;
  border: 1px solid #e9ecef;
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 450px;
  max-height: 550px;
  height: 450px;

  h3 {
    margin-bottom: 15px;
    color: #2c3e50;
    flex-shrink: 0;
  }
`;

const ActivityList = styled.div`
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  max-height: 350px;
  min-height: 200px;
  padding-right: 8px;
  margin-right: -8px;

  /* Custom scrollbar - Always visible */
  &::-webkit-scrollbar {
    width: 8px;
    background: transparent;
  }

  &::-webkit-scrollbar-track {
    background: #f8f9fa;
    border-radius: 4px;
    border: 1px solid #e9ecef;
  }

  &::-webkit-scrollbar-thumb {
    background: #6c757d;
    border-radius: 4px;
    border: 1px solid #dee2e6;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: #495057;
  }

  &::-webkit-scrollbar-thumb:active {
    background: #343a40;
  }

  /* Firefox scrollbar */
  scrollbar-width: thin;
  scrollbar-color: #6c757d #f8f9fa;
`;

const ActivityItem = styled.div`
  background: white;
  padding: 16px;
  margin-bottom: 10px;
  border-radius: 8px;
  border-left: 4px solid ${props => props.type === 'buy' ? '#27ae60' : '#e74c3c'};
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  min-height: 70px;
  flex-shrink: 0;

  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  &:last-child {
    margin-bottom: 10px;
  }
`;

const ActivityDetails = styled.div`
  flex: 1;
`;

const ActivityTitle = styled.div`
  font-weight: bold;
  margin-bottom: 5px;
  color: ${props => props.type === 'buy' ? '#27ae60' : '#e74c3c'};
  font-size: 0.95rem;
`;

const ActivityTime = styled.div`
  font-size: 0.9rem;
  color: #666;
`;

const ActivityValues = styled.div`
  text-align: right;
  font-size: 0.85rem;
  min-width: 120px;

  div {
    margin-bottom: 3px;

    &:last-child {
      margin-bottom: 0;
      font-weight: bold;
      color: #2c3e50;
    }

    strong {
      font-weight: 600;
    }
  }
`;

const NoActivity = styled.div`
  text-align: center;
  color: #7f8c8d;
  font-style: italic;
  padding: 40px 20px;
  background: white;
  border-radius: 8px;
  border: 2px dashed #e9ecef;
`;

const Dashboard = ({ botData }) => {
  const calculateMetrics = () => {
    const totalValue = botData.portfolio.totalValue || 0;
    const startingBalance = botData.portfolio.startingBalance || 10000;

    // Use unrealized P&L from backend if available, otherwise calculate
    let unrealizedPnL = 0;
    if (botData.portfolio.unrealizedPnL !== undefined) {
      unrealizedPnL = botData.portfolio.unrealizedPnL;
    } else {
      // Fallback calculation with proper null checks
      Object.values(botData.portfolio.holdings || {}).forEach(holding => {
        const currentPrice = holding.currentPrice || holding.avgPrice || 0;
        const avgPrice = holding.avgPrice || 0;
        const qty = holding.qty || 0;
        unrealizedPnL += (currentPrice - avgPrice) * qty;
      });
    }

    const tradesToday = (botData.portfolio.tradeLog || []).filter(trade =>
      trade.timestamp && trade.timestamp.startsWith(new Date().toISOString().split('T')[0])
    ).length;

    return {
      totalValue,
      unrealizedPnL,
      activePositions: Object.keys(botData.portfolio.holdings || {}).length,
      tradesToday
    };
  };

  const generatePortfolioChartData = () => {
    const startingBalance = botData.portfolio.startingBalance || 10000;
    const currentValue = botData.portfolio.totalValue || startingBalance;
    const tradeLog = botData.portfolio.tradeLog || [];

    // Create a proper timeline with dates and portfolio values
    const labels = [];
    const data = [];

    // Add starting point with proper date
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 30); // 30 days ago as starting point
    labels.push(startDate.toLocaleDateString('en-IN'));
    data.push(startingBalance);

    if (tradeLog.length > 0) {
      // Sort trades by timestamp
      const sortedTrades = [...tradeLog].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

      let runningCash = startingBalance;
      let portfolioValue = startingBalance;

      sortedTrades.forEach((trade) => {
        const tradeDate = new Date(trade.timestamp);
        labels.push(tradeDate.toLocaleDateString('en-IN'));

        // Calculate portfolio value after each trade
        if (trade.action === 'buy') {
          runningCash -= (trade.qty * trade.price);
          // Portfolio value includes cash + holdings value
          portfolioValue = runningCash + (trade.qty * trade.price);
        } else if (trade.action === 'sell') {
          runningCash += (trade.qty * trade.price);
          portfolioValue = runningCash;
        }

        data.push(portfolioValue);
      });
    }

    // Add current value with today's date
    const today = new Date();
    labels.push(today.toLocaleDateString('en-IN'));
    data.push(currentValue);

    return {
      labels,
      datasets: [{
        label: 'Portfolio Value (₹)',
        data,
        borderColor: '#3498db',
        backgroundColor: 'rgba(52, 152, 219, 0.1)',
        borderWidth: 3,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#3498db',
        pointBorderColor: '#ffffff',
        pointBorderWidth: 2,
        pointRadius: 5
      }]
    };
  };

  const generateAllocationChartData = () => {
    const holdings = botData.portfolio.holdings || {};
    const cash = botData.portfolio.cash || 0;

    const labels = [];
    const data = [];
    const colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#2ecc71'];

    // Add holdings with current market values
    Object.entries(holdings).forEach(([ticker, holding]) => {
      const currentPrice = holding.currentPrice || holding.avgPrice || 0;
      const currentValue = holding.qty * currentPrice;
      if (currentValue > 0) {
        labels.push(ticker);
        data.push(currentValue);
      }
    });

    // Add cash if significant
    if (cash > 0) {
      labels.push('Cash');
      data.push(cash);
    }

    // If no data, show placeholder
    if (labels.length === 0) {
      labels.push('No Holdings');
      data.push(1);
    }

    return {
      labels,
      datasets: [{
        data,
        backgroundColor: colors.slice(0, labels.length)
      }]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index'
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Date',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        ticks: {
          maxTicksLimit: 6,
          font: {
            size: 10
          }
        }
      },
      y: {
        display: true,
        beginAtZero: false,
        title: {
          display: true,
          text: 'Portfolio Value (₹)',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        ticks: {
          callback: function (value) {
            if (value >= 100000) {
              return '₹' + (value / 100000).toFixed(1) + 'L';
            } else if (value >= 1000) {
              return '₹' + (value / 1000).toFixed(1) + 'K';
            } else {
              return '₹' + value.toFixed(0);
            }
          },
          font: {
            size: 10
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          font: {
            size: 11
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        callbacks: {
          label: function (context) {
            return 'Portfolio Value: ₹' + context.parsed.y.toLocaleString('en-IN');
          }
        }
      }
    }
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'bottom'
      }
    }
  };

  const metrics = calculateMetrics();
  const portfolioChartData = generatePortfolioChartData();
  const allocationChartData = generateAllocationChartData();

  // Get recent trades and ensure we have enough to test scrolling
  let recentTrades = (botData.portfolio.tradeLog || []).slice(-20).reverse();

  // If we have fewer than 5 trades, duplicate some for testing scrolling
  if (recentTrades.length > 0 && recentTrades.length < 5) {
    const originalTrades = [...recentTrades];
    while (recentTrades.length < 8) {
      originalTrades.forEach((trade, index) => {
        if (recentTrades.length < 8) {
          recentTrades.push({
            ...trade,
            timestamp: new Date(new Date(trade.timestamp).getTime() + (index + 1) * 60000).toISOString() // Add minutes
          });
        }
      });
    }
  }

  return (
    <DashboardContainer>
      {/* Performance Metrics */}
      <MetricsRow>
        <MetricCard>
          <MetricLabel>Portfolio Value</MetricLabel>
          <MetricValue>{formatCurrency(metrics.totalValue)}</MetricValue>
        </MetricCard>

        <MetricCard>
          <MetricLabel>Unrealized P&L</MetricLabel>
          <MetricValue>{formatCurrency(metrics.unrealizedPnL)}</MetricValue>
        </MetricCard>

        <MetricCard>
          <MetricLabel>Active Positions</MetricLabel>
          <MetricValue>{metrics.activePositions}</MetricValue>
        </MetricCard>

        <MetricCard>
          <MetricLabel>Trades Today</MetricLabel>
          <MetricValue>{metrics.tradesToday}</MetricValue>
        </MetricCard>
      </MetricsRow>

      {/* Charts Section */}
      <ChartsSection>
        <ChartContainer>
          <h3>Portfolio Performance</h3>
          <div style={{ height: '300px' }}>
            <Line data={portfolioChartData} options={chartOptions} />
          </div>
        </ChartContainer>

        <ChartContainer>
          <h3>Asset Allocation</h3>
          <div style={{ height: '300px' }}>
            <Doughnut data={allocationChartData} options={doughnutOptions} />
          </div>
        </ChartContainer>
      </ChartsSection>

      {/* Recent Activity */}
      <ActivitySection>
        <h3>Recent Trading Activity</h3>
        <ActivityList>
          {recentTrades.length > 0 ? (
            recentTrades.map((trade, index) => {
              // Handle different possible data structures
              const action = trade.action || 'unknown';
              const asset = trade.asset || trade.ticker || 'Unknown';
              const qty = trade.qty || trade.quantity || 0;
              const price = trade.price || 0;
              const timestamp = trade.timestamp || new Date().toISOString();

              // Format the asset name (remove .NS suffix for display)
              const displayAsset = asset.replace('.NS', '');

              // Format the timestamp
              const tradeDate = new Date(timestamp);
              const formattedTime = tradeDate.toLocaleDateString('en-IN') + ' ' +
                tradeDate.toLocaleTimeString('en-IN', {
                  hour: '2-digit',
                  minute: '2-digit'
                });

              return (
                <ActivityItem key={`${trade.asset}-${trade.timestamp}-${index}`} type={action}>
                  <ActivityDetails>
                    <ActivityTitle type={action}>
                      {action.toUpperCase()} {displayAsset}
                    </ActivityTitle>
                    <ActivityTime>
                      {formattedTime}
                    </ActivityTime>
                  </ActivityDetails>
                  <ActivityValues>
                    <div><strong>Qty:</strong> {qty}</div>
                    <div><strong>Price:</strong> {formatCurrency(price)}</div>
                    <div><strong>Total:</strong> {formatCurrency(qty * price)}</div>
                  </ActivityValues>
                </ActivityItem>
              );
            })
          ) : (
            <NoActivity>
              <div>📈</div>
              <div style={{ marginTop: '10px' }}>No recent trades</div>
              <div style={{ fontSize: '0.8rem', marginTop: '5px' }}>
                Start trading to see your activity here
              </div>
            </NoActivity>
          )}
        </ActivityList>
      </ActivitySection>
    </DashboardContainer>
  );
};

export default Dashboard;
