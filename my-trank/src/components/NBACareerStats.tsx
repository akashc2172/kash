import React, { useState } from 'react';
import { ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import './NBACareerStats.css';

interface NBACareerStatsProps {
    brStats: {
        'drb%'?: number;
        'trb%'?: number;
        'ast%'?: number;
        'obpm'?: number;
        'WS'?: number;
        'WS/48'?: number;
        'DBPM'?: number;
        'OWS'?: number;
        'DWS'?: number;
        'GS'?: number;
        'MP'?: number;
        'blk%'?: number;
        profile_url?: string;
    } | null;
}

export const NBACareerStats: React.FC<NBACareerStatsProps> = ({ brStats }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    if (!brStats) return null;

    const hasData = Object.values(brStats).some(v => v !== null && v !== undefined && v !== '');

    if (!hasData) return null;

    const formatStat = (value: any, decimals: number = 1): string => {
        if (value === null || value === undefined || value === '') return '-';
        const num = typeof value === 'string' ? parseFloat(value) : value;
        if (isNaN(num)) return '-';
        return num.toFixed(decimals);
    };

    return (
        <div className="nba-career-stats-container">
            <div
                className="nba-career-stats-header"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <div className="header-left">
                    <span className="nba-badge">🏀 NBA</span>
                    <span className="header-title">Career Totals</span>
                </div>
                <div className="header-right">
                    {brStats.profile_url && (
                        <a
                            href={brStats.profile_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="br-link"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <ExternalLink size={14} />
                            Basketball Reference
                        </a>
                    )}
                    {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                </div>
            </div>

            {isExpanded && (
                <div className="nba-career-stats-content">
                    <div className="stats-grid">
                        <div className="stat-category">
                            <h4>Impact Metrics</h4>
                            <div className="stat-row">
                                <span className="stat-label">Win Shares</span>
                                <span className="stat-value">{formatStat(brStats.WS)}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Offensive WS</span>
                                <span className="stat-value">{formatStat(brStats.OWS)}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Defensive WS</span>
                                <span className="stat-value">{formatStat(brStats.DWS)}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">WS/48</span>
                                <span className="stat-value">{formatStat(brStats['WS/48'], 3)}</span>
                            </div>
                        </div>

                        <div className="stat-category">
                            <h4>Plus-Minus</h4>
                            <div className="stat-row">
                                <span className="stat-label">Offensive BPM</span>
                                <span className="stat-value">{formatStat(brStats.obpm)}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Defensive BPM</span>
                                <span className="stat-value">{formatStat(brStats.DBPM)}</span>
                            </div>
                        </div>

                        <div className="stat-category">
                            <h4>Contribution Rates</h4>
                            <div className="stat-row">
                                <span className="stat-label">AST%</span>
                                <span className="stat-value">{formatStat(brStats['ast%'])}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">TRB%</span>
                                <span className="stat-value">{formatStat(brStats['trb%'])}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">DRB%</span>
                                <span className="stat-value">{formatStat(brStats['drb%'])}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">BLK%</span>
                                <span className="stat-value">{formatStat(brStats['blk%'])}</span>
                            </div>
                        </div>

                        <div className="stat-category">
                            <h4>Career Volume</h4>
                            <div className="stat-row">
                                <span className="stat-label">Games Started</span>
                                <span className="stat-value">{formatStat(brStats.GS, 0)}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Total Minutes</span>
                                <span className="stat-value">{formatStat(brStats.MP, 0)}</span>
                            </div>
                        </div>
                    </div>

                    <div className="nba-stats-note">
                        <small>NBA career totals from Basketball Reference</small>
                    </div>
                </div>
            )}
        </div>
    );
};
