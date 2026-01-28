import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import './ConsensusDraft.css';

interface Player {
    key: string;
    team: string;
    pos?: string;
    torvik_year: number;
    bpm?: number;
    ppg?: number;
    pick?: string | number;
    consensusRank?: number;
    voteCount?: number;
    trend?: 'up' | 'down' | 'same';
}

interface Vote {
    playerId: string;
    choice: 'start' | 'bench' | 'cut';
    timestamp: number;
}

interface ConsensusDraftProps {
    players: Player[];
    onClose: () => void;
}

export const ConsensusDraft: React.FC<ConsensusDraftProps> = ({ players, onClose }) => {
    const [currentMatchup, setCurrentMatchup] = useState<[Player, Player] | null>(null);
    const [votes, setVotes] = useState<Vote[]>([]);
    const [showComingSoon, setShowComingSoon] = useState(true);
    // hasSeenModal state reserved for future features

    useEffect(() => {
        // Check if user has seen the modal before
        const seen = localStorage.getItem('ctsSeenConsensusDraft');
        if (seen) {
            setShowComingSoon(false);
            // hasSeenModal used for future features
        }
    }, []);

    const handleDismissComingSoon = () => {
        localStorage.setItem('ctsSeenConsensusDraft', 'true');
        setShowComingSoon(false);
        // hasSeenModal used for future features
    };

    const getRandomMatchup = () => {
        // Filter drafted players only
        const draftedPlayers = players.filter(p => {
            const pick = p.pick;
            return pick && pick !== 'NA' && pick !== '';
        });

        if (draftedPlayers.length < 2) return null;

        // Get two random players with similar stats (BPM +/- 2)
        const player1 = draftedPlayers[Math.floor(Math.random() * draftedPlayers.length)];
        const similarPlayers = draftedPlayers.filter(p => {
            if (p.key === player1.key) return false;
            const bpm1 = player1.bpm || 0;
            const bpm2 = p.bpm || 0;
            return Math.abs(bpm1 - bpm2) < 3;
        });

        if (similarPlayers.length === 0) return null;

        const player2 = similarPlayers[Math.floor(Math.random() * similarPlayers.length)];
        return [player1, player2] as [Player, Player];
    };

    const handleVote = (playerKey: string, choice: 'start' | 'bench' | 'cut') => {
        if (!currentMatchup) return;

        const vote: Vote = {
            playerId: playerKey,
            choice,
            timestamp: Date.now()
        };

        const updatedVotes = [...votes, vote];
        setVotes(updatedVotes);

        // Save to localStorage
        localStorage.setItem('ctsConsensusDraftVotes', JSON.stringify(updatedVotes));

        // Get next matchup
        setTimeout(() => {
            setCurrentMatchup(getRandomMatchup());
        }, 500);
    };

    useEffect(() => {
        // Load votes from localStorage
        const savedVotes = localStorage.getItem('ctsConsensusDraftVotes');
        if (savedVotes) {
            try {
                setVotes(JSON.parse(savedVotes));
            } catch (e) {
                console.error('Failed to load votes:', e);
            }
        }

        // Set initial matchup
        if (!showComingSoon) {
            setCurrentMatchup(getRandomMatchup());
        }
    }, [showComingSoon]);

    if (showComingSoon) {
        return (
            <div className="consensus-modal-overlay" onClick={onClose}>
                <div className="consensus-coming-soon-modal" onClick={e => e.stopPropagation()}>
                    <button className="close-btn" onClick={onClose}>
                        <X size={20} />
                    </button>

                    <div className="coming-soon-content">
                        <div className="coming-soon-icon">🏀</div>
                        <h2>Consensus Draft Rankings</h2>
                        <p className="coming-soon-subtitle">Coming Soon!</p>

                        <div className="coming-soon-description">
                            <p>We're building a crowd-sourced consensus draft system, similar to KeepTradeCut.</p>

                            <div className="feature-list">
                                <div className="feature-item">
                                    <span className="feature-icon">📊</span>
                                    <div>
                                        <strong>Start / Bench / Cut</strong>
                                        <p>Vote on player matchups to build consensus rankings</p>
                                    </div>
                                </div>

                                <div className="feature-item">
                                    <span className="feature-icon">📈</span>
                                    <div>
                                        <strong>Live Rankings</strong>
                                        <p>See real-time consensus draft positions based on community votes</p>
                                    </div>
                                </div>

                                <div className="feature-item">
                                    <span className="feature-icon">🔥</span>
                                    <div>
                                        <strong>Trending Players</strong>
                                        <p>Track which players are rising or falling in the consensus</p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <button className="cta-button" onClick={handleDismissComingSoon}>
                            Got it! Take me back
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="consensus-modal-overlay" onClick={onClose}>
            <div className="consensus-draft-modal" onClick={e => e.stopPropagation()}>
                <div className="consensus-header">
                    <div>
                        <h3>Consensus Draft Voting</h3>
                        <p className="vote-count">{votes.length} votes cast</p>
                    </div>
                    <button className="close-btn" onClick={onClose}>
                        <X size={20} />
                    </button>
                </div>

                {currentMatchup && (
                    <div className="matchup-container">
                        <p className="matchup-prompt">In a redraft, which would you rather have?</p>

                        <div className="players-matchup">
                            {currentMatchup.map((player, _idx) => (
                                <div key={player.key} className="player-card">
                                    <div className="player-info">
                                        <h4>{player.key}</h4>
                                        <div className="player-meta">
                                            <span>{player.team}</span>
                                            <span>•</span>
                                            <span>{player.pos || 'N/A'}</span>
                                            <span>•</span>
                                            <span>'{String(player.torvik_year).slice(-2)}</span>
                                        </div>
                                        <div className="player-stats">
                                            <div className="stat">
                                                <span className="stat-label">PPG</span>
                                                <span className="stat-value">{player.ppg?.toFixed(1) || '-'}</span>
                                            </div>
                                            <div className="stat">
                                                <span className="stat-label">BPM</span>
                                                <span className="stat-value">{player.bpm?.toFixed(1) || '-'}</span>
                                            </div>
                                            {player.pick && player.pick !== 'NA' && (
                                                <div className="stat">
                                                    <span className="stat-label">Pick</span>
                                                    <span className="stat-value">#{player.pick}</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="vote-buttons">
                                        <button
                                            className="vote-btn vote-start"
                                            onClick={() => handleVote(player.key, 'start')}
                                        >
                                            Start
                                        </button>
                                        <button
                                            className="vote-btn vote-bench"
                                            onClick={() => handleVote(player.key, 'bench')}
                                        >
                                            Bench
                                        </button>
                                        <button
                                            className="vote-btn vote-cut"
                                            onClick={() => handleVote(player.key, 'cut')}
                                        >
                                            Cut
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <button className="skip-btn" onClick={() => setCurrentMatchup(getRandomMatchup())}>
                            Skip this matchup
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};
