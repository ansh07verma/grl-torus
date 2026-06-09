import React from 'react';

interface MetricsCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
  trend?: string;
  trendUp?: boolean;
}

const MetricsCard: React.FC<MetricsCardProps> = ({ title, value, icon, trend, trendUp }) => {
  return (
    <div className="glass-panel" style={{ padding: '20px', display: 'flex', alignItems: 'center', gap: '16px' }}>
      <div style={{ 
        width: '48px', 
        height: '48px', 
        borderRadius: '12px', 
        background: 'rgba(255,255,255,0.05)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px solid var(--border-color)'
      }}>
        {icon}
      </div>
      <div style={{ flex: 1 }}>
        <h4 style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '4px' }}>{title}</h4>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
          <span style={{ fontSize: '1.5rem', fontWeight: 600, color: 'var(--text-primary)' }}>{value}</span>
          {trend && (
            <span style={{ 
              fontSize: '0.85rem', 
              fontWeight: 500,
              color: trendUp ? 'var(--accent-success)' : (trend.startsWith('-') ? 'var(--accent-success)' : 'var(--text-muted)') 
            }}>
              {trend}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default MetricsCard;
