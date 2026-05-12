import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import PropTypes from 'prop-types';

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

const ModalContent = styled.div`
  background: white;
  border-radius: 12px;
  width: 90%;
  max-width: 500px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  max-height: 90vh;
  overflow-y: auto;
`;

const ModalHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #e9ecef;

  h3 {
    color: #2c3e50;
    margin: 0;
  }
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: #7f8c8d;
  padding: 5px;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    color: #e74c3c;
    background: #f8f9fa;
  }
`;

const ModalBody = styled.div`
  padding: 20px;
`;

const SettingGroup = styled.div`
  margin-bottom: 20px;

  label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
    color: #2c3e50;
  }

  select, input {
    width: 100%;
    padding: 10px;
    border: 2px solid #e9ecef;
    border-radius: 6px;
    font-size: 1rem;
    box-sizing: border-box;

    &:focus {
      outline: none;
      border-color: #3498db;
    }
  }

  input[type="number"] {
    -moz-appearance: textfield;
    
    &::-webkit-outer-spin-button,
    &::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }
  }
`;

const ModalFooter = styled.div`
  padding: 20px;
  border-top: 1px solid #e9ecef;
  text-align: right;
  display: flex;
  gap: 10px;
  justify-content: flex-end;
`;

const Button = styled.button`
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
  border: none;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const SaveButton = styled(Button)`
  background: #27ae60;
  color: white;

  &:hover:not(:disabled) {
    background: #229954;
  }
`;

const CancelButton = styled(Button)`
  background: #95a5a6;
  color: white;

  &:hover:not(:disabled) {
    background: #7f8c8d;
  }
`;

const SettingsModal = ({ settings, onSave, onClose }) => {
  const [formData, setFormData] = useState({
    mode: 'paper',
    riskLevel: 'MEDIUM',
    productType: 'CNC',  // CNC (Delivery) or INTRADAY
    maxAllocation: 25,
    stopLossPct: 5,
    targetProfitLevel: 'MEDIUM',
    targetProfitPct: 8
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (settings) {
      setFormData({
        mode: settings.mode || 'paper',
        riskLevel: settings.riskLevel || 'MEDIUM',
        productType: settings.productType || 'CNC',  // Load product type from settings
        // Convert decimal to percentage - backend sends max_capital_per_trade and stop_loss_pct
        maxAllocation: settings.max_capital_per_trade
          ? (settings.max_capital_per_trade * 100)
          : 25,
        stopLossPct: settings.stop_loss_pct
          ? (settings.stop_loss_pct * 100)
          : 5,
        targetProfitLevel: settings.target_profit_level || 'MEDIUM',
        targetProfitPct: settings.target_profit_pct
          ? (settings.target_profit_pct * 100)
          : 8
      });
    }
  }, [settings]);

  const handleInputChange = (field, value) => {
    setFormData(prev => {
      const newData = { ...prev };

      // Handle risk level changes
      if (field === 'riskLevel') {
        newData.riskLevel = value;

        // Only auto-update if NOT switching TO CUSTOM
        if (value !== 'CUSTOM') {
          // For predefined risk levels, set the values
          const riskSettings = {
            'LOW': { stopLoss: 3, allocation: 15 },
            'MEDIUM': { stopLoss: 5, allocation: 25 },
            'HIGH': { stopLoss: 8, allocation: 35 }
          };

          if (riskSettings[value]) {
            newData.stopLossPct = riskSettings[value].stopLoss;
            newData.maxAllocation = riskSettings[value].allocation;
          }
        }
        // When switching to CUSTOM, keep current values (don't clear them)
      }
      // Handle target profit level changes
      else if (field === 'targetProfitLevel' || field === 'targetPriceLevel') {
        newData.targetPriceLevel = value;

        // Only auto-update if NOT switching TO CUSTOM
        if (value !== 'CUSTOM') {
          // For predefined target profit levels, set the values
          const targetSettings = {
            'LOW': 8,
            'MEDIUM': 8,
            'HIGH': 12
          };

          if (targetSettings[value]) {
            newData.targetPricePct = targetSettings[value];
          }
        }
        // When switching to CUSTOM, keep current value (don't clear it)
      }
      // Handle other field changes
      else {
        newData[field] = value;
      }

      console.log('Risk Level:', newData.riskLevel, 'Is Custom:', newData.riskLevel === 'CUSTOM');
      return newData;
    });
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      // Convert string values to numbers
      const maxAllocationNum = parseFloat(formData.maxAllocation) || 0;
      const stopLossPctNum = parseFloat(formData.stopLossPct) || 0;
      const targetPricePctNum = parseFloat(formData.targetPricePct) || 0;

      // Validate that we have values for custom mode
      if (formData.riskLevel === 'CUSTOM') {
        if (!formData.maxAllocation || !formData.stopLossPct || maxAllocationNum <= 0 || stopLossPctNum <= 0) {
          alert('Please enter valid values for both Max Allocation (greater than 0) and Stop Loss Percentage (greater than 0) when using Custom risk level.');
          setLoading(false);
          return;
        }

        // Validate ranges - allow any positive value
        if (maxAllocationNum <= 0) {
          alert('Max Allocation must be greater than 0.');
          setLoading(false);
          return;
        }

        if (maxAllocationNum > 100) {
          alert('Max Allocation should not exceed 100%.');
          setLoading(false);
          return;
        }

        if (stopLossPctNum <= 0) {
          alert('Stop Loss Percentage must be greater than 0 (e.g., 0.3, 0.5, 1.0).');
          setLoading(false);
          return;
        }

        if (stopLossPctNum > 100) {
          alert('Stop Loss Percentage should not exceed 100%.');
          setLoading(false);
          return;
        }
      }

      // Validate target profit for custom mode
      if (formData.targetPriceLevel === 'CUSTOM') {
        if (!formData.targetPricePct && formData.targetPricePct !== 0) {
          alert('Please enter a valid Target Profit Percentage (e.g., 0.3, 0.5, 1.0, 5).');
          setLoading(false);
          return;
        }

        if (targetPricePctNum <= 0) {
          alert('Target Profit Percentage must be greater than 0.');
          setLoading(false);
          return;
        }

        // Allow any positive value up to 100%
        if (targetPricePctNum > 100) {
          alert('Target Profit Percentage should not exceed 100%.');
          setLoading(false);
          return;
        }
      }

      // Use the converted numbers or defaults
      const maxAllocation = maxAllocationNum || 25;
      const stopLossPct = stopLossPctNum || 5;
      const targetProfitPct = targetPricePctNum || 8;

      const settingsToSave = {
        mode: formData.mode,
        riskLevel: formData.riskLevel,
        productType: formData.productType,  // Include product type in saved settings
        stop_loss_pct: stopLossPct / 100, // Convert percentage to decimal
        max_capital_per_trade: maxAllocation / 100, // Convert percentage to decimal
        target_profit_level: formData.targetProfitLevel,
        target_profit_pct: targetProfitPct / 100, // Convert percentage to decimal
        max_trade_limit: 150 // Default value
      };

      console.log('Saving settings:', settingsToSave);
      await onSave(settingsToSave);
    } catch (error) {
      console.error('Error saving settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <ModalOverlay onClick={handleOverlayClick}>
      <ModalContent>
        <ModalHeader>
          <h3>Settings</h3>
          <CloseButton onClick={onClose}>
            <i className="fas fa-times"></i>
          </CloseButton>
        </ModalHeader>

        <ModalBody>
          <SettingGroup>
            <label>Trading Mode:</label>
            <select
              value={formData.mode}
              onChange={(e) => handleInputChange('mode', e.target.value)}
              disabled={loading}
            >
              <option value="paper">Paper Trading</option>
              <option value="live">Live Trading</option>
            </select>
          </SettingGroup>

          <SettingGroup>
            <label>Product Type:</label>
            <select
              value={formData.productType}
              onChange={(e) => handleInputChange('productType', e.target.value)}
              disabled={loading}
            >
              <option value="CNC">CNC (Delivery) - Carry positions to next day</option>
              <option value="INTRADAY">Intraday (MIS) - Square off same day</option>
            </select>
            <small style={{ color: '#6c757d', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
              CNC: Positions carried forward overnight | Intraday: Auto square-off at 3:15 PM
            </small>
          </SettingGroup>

          <SettingGroup>
            <label>Risk Level:</label>
            <select
              value={formData.riskLevel}
              onChange={(e) => handleInputChange('riskLevel', e.target.value)}
              disabled={loading}
            >
              <option value="LOW">Low (3% stop-loss, 15% allocation)</option>
              <option value="MEDIUM">Medium (5% stop-loss, 25% allocation)</option>
              <option value="HIGH">High (8% stop-loss, 35% allocation)</option>
              <option value="CUSTOM">Custom (Set your own values)</option>
            </select>
          </SettingGroup>

          <SettingGroup>
            <label>Max Allocation per Trade (%):</label>
            <input
              type="number"
              min="1"
              max="100"
              value={formData.maxAllocation}
              placeholder={formData.riskLevel === 'CUSTOM' ? 'Enter percentage (1-100)' : ''}
              onChange={(e) => {
                const value = e.target.value;
                console.log('Max Allocation input changed:', value, 'Risk Level:', formData.riskLevel);
                // Always allow the change, let the input handle validation
                handleInputChange('maxAllocation', value);
              }}
              disabled={loading || formData.riskLevel !== 'CUSTOM'}
              style={{
                backgroundColor: formData.riskLevel !== 'CUSTOM' ? '#f8f9fa' : 'white',
                cursor: formData.riskLevel !== 'CUSTOM' ? 'not-allowed' : 'text',
                border: formData.riskLevel === 'CUSTOM' ? '2px solid #3498db' : '2px solid #e9ecef'
              }}
            />
            {formData.riskLevel !== 'CUSTOM' && (
              <small style={{ color: '#6c757d', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                Select "Custom" risk level to modify this value
              </small>
            )}
            {formData.riskLevel === 'CUSTOM' && (
              <small style={{ color: '#27ae60', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                ✓ Custom mode: You can edit this value
              </small>
            )}
          </SettingGroup>

          <SettingGroup>
            <label>Stop Loss Percentage (%):</label>
            <input
              type="number"
              min="0.01"
              max="100"
              step="0.01"
              value={formData.stopLossPct}
              placeholder={formData.riskLevel === 'CUSTOM' ? 'Enter percentage (e.g., 0.5, 1.5, 5)' : ''}
              onChange={(e) => {
                const value = e.target.value;
                console.log('Stop Loss input changed:', value, 'Risk Level:', formData.riskLevel);
                handleInputChange('stopLossPct', value);
              }}
              disabled={loading || formData.riskLevel !== 'CUSTOM'}
              style={{
                backgroundColor: formData.riskLevel !== 'CUSTOM' ? '#f8f9fa' : 'white',
                cursor: formData.riskLevel !== 'CUSTOM' ? 'not-allowed' : 'text',
                border: formData.riskLevel === 'CUSTOM' ? '2px solid #e74c3c' : '2px solid #e9ecef'
              }}
            />
            {formData.riskLevel !== 'CUSTOM' && (
              <small style={{ color: '#6c757d', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                Select "Custom" risk level to modify this value
              </small>
            )}
            {formData.riskLevel === 'CUSTOM' && (
              <small style={{ color: '#e74c3c', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                ✓ Custom mode: You can edit this value (e.g., 0.5, 0.3, 1.5)
              </small>
            )}
          </SettingGroup>

          <SettingGroup>
            <label>Target Price Level:</label>
            <select
              value={formData.targetPriceLevel}
              onChange={(e) => handleInputChange('targetPriceLevel', e.target.value)}
              disabled={loading}
            >
              <option value="LOW">Low (8% target price)</option>
              <option value="MEDIUM">Medium (4% target price)</option>
              <option value="HIGH">High (12% target price)</option>
              <option value="CUSTOM">Custom (Set your own percentage)</option>
            </select>
          </SettingGroup>

          <SettingGroup>
            <label>Target Price Percentage (%):</label>
            <input
              type="number"
              min="0.01"
              max="100"
              step="0.01"
              value={formData.targetPricePct || ''}
              placeholder={formData.targetPriceLevel === 'CUSTOM' ? 'Enter percentage (e.g., 0.3, 1.5, 5)' : ''}
              onChange={(e) => {
                const value = e.target.value;
                console.log('Target Price input changed:', value, 'Level:', formData.targetPriceLevel);
                handleInputChange('targetPricePct', value);
              }}
              disabled={loading || formData.targetPriceLevel !== 'CUSTOM'}
              style={{
                backgroundColor: formData.targetPriceLevel !== 'CUSTOM' ? '#f8f9fa' : 'white',
                cursor: formData.targetPriceLevel !== 'CUSTOM' ? 'not-allowed' : 'text',
                border: formData.targetPriceLevel === 'CUSTOM' ? '2px solid #e74c3c' : '2px solid #e9ecef'
              }}
            />
            {formData.targetPriceLevel !== 'CUSTOM' && (
              <small style={{ color: '#6c757d', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                Select "Custom" target price level to modify this value
              </small>
            )}
            {formData.targetPriceLevel === 'CUSTOM' && (
              <small style={{ color: '#e74c3c', fontSize: '0.85rem', marginTop: '5px', display: 'block' }}>
                ✓ Custom mode: You can edit this value (e.g., 0.3, 0.4, 1.5)
              </small>
            )}
          </SettingGroup>
        </ModalBody>

        <ModalFooter>
          <CancelButton onClick={onClose} disabled={loading}>
            Cancel
          </CancelButton>
          <SaveButton onClick={handleSave} disabled={loading}>
            {loading ? 'Saving...' : 'Save Settings'}
          </SaveButton>
        </ModalFooter>
      </ModalContent>
    </ModalOverlay>
  );
};

SettingsModal.propTypes = {
  settings: PropTypes.object,
  onSave: PropTypes.func.isRequired,
  onClose: PropTypes.func.isRequired
};

export default SettingsModal;
