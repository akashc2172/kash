// @ts-nocheck

import nerdBall from './assets/nerd_ball.png';
import React, { useEffect, useState, useMemo, useRef } from 'react';
import Papa from 'papaparse';
import _ from 'lodash';
import { TableVirtuoso } from 'react-virtuoso';
import { CustomStatBuilder } from './components/CustomStatBuilder';
import type { CustomStat } from './components/CustomStatBuilder';
import { ConsensusDraft } from './components/ConsensusDraft';
import { NBACareerStats } from './components/NBACareerStats';
import {
  LineChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer,
  BarChart, Bar, Legend, CartesianGrid, ReferenceLine,
  ScatterChart, Scatter, Cell, Label
} from 'recharts';
import {
  ChevronDown,
  ChevronUp,
  ChevronRight,
  ChevronLeft,
  X,
  Search,
  Eye,
  EyeOff,
  Twitter,
  Plus,
  Settings,
  Minus,
  Calculator
} from 'lucide-react';
import './App.css';

// --- Configuration ---
const FILES = {
  season: '/data/season.csv',
  career: '/data/career.csv',
  weights: '/data/weights.csv',
  archive: '/data/archive.csv',
  international: '/data/international_stat_history/internationalplayerarchive.csv',
  intl_2026: '/data/international_stat_history/2026_records.csv',
  br_advanced: '/data/br_advanced_stats.csv',
};

// --- CONSTANTS ---
const PERMANENT_COLS = [
  { key: 'exp', label: 'Exp' },
  { key: 'torvik_year', label: 'Year' },
  { key: 'key', label: 'Player' },
  { key: 'team', label: 'Team' },
  { key: 'pos', label: 'Pos' },
  { key: 'g', label: 'G', tooltip: 'Games Played' },
  { key: 'mpg', label: 'MPG', tooltip: 'Minutes Per Game' },
  { key: 'height', label: 'Hgt', tooltip: 'Height (Inches)' },
  { key: 'weight', label: 'Wt', tooltip: 'Weight (lbs)' },
  { key: 'bmi', label: 'BMI', tooltip: 'Body Mass Index' },
];

const CAREER_PERMANENT_COLS = [
  { key: 'key', label: 'Player' },
  {
    key: 'length',
    label: 'Length',
    tooltip:
      'Collegiate career length.'
  },
  { key: 'team', label: 'Team' },
  { key: 'pos', label: 'Pos' },
  { key: 'g', label: 'G', tooltip: 'Games Played' },
  { key: 'mpg', label: 'MPG', tooltip: 'Minutes Per Game' },
  { key: 'height', label: 'Hgt', tooltip: 'Height (Inches)' },
  { key: 'weight', label: 'Wt', tooltip: 'Weight (lbs)' },
  { key: 'bmi', label: 'BMI', tooltip: 'Body Mass Index' },
];

const COMPARE_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#9333ea', '#0891b2'];

// Columns that are TOTALS (need /G * 40/MPG conversion)
const PER40_FROM_TOTALS = new Set([
  'two_a', 'two_m', 'three_a', 'three_m', 'fta', 'ftm',
  'rim attempts', 'rim_makes', 'rim_makes_calc',
  'dunk_m', 'dunk_a',
  'middy attempts', 'middy_makes', 'middy_makes_calc',
  'fgm', 'fga',
]);

// Columns that are PER GAME (need * 40/MPG conversion)
const PER40_FROM_PER_GAME = new Set([
  'ppg', 'rpg', 'apg', 'spg', 'bpg', 'oreb', 'dreb', 'tov'
]);

type TableCol = {
  key: string;
  label: string;
  tooltip?: string;
  isGroupEnd?: boolean;
};

type TableGroup = {
  group: string;
  cols: TableCol[];
};

type SortConfig = {
  key: string;
  direction: 'asc' | 'desc';
} | null;

type StatsDistribution = Record<string, { mean: number; std: number }>;

const METRIC_DEFINITIONS: Record<string, string> = {
  "ts": "True Shooting %",
  "efg": "Effective FG%",
  "fg_pct": "Field Goal %",
  "bpm": "Box Plus-Minus",
  "porpag": "Points Over Replacement Per Adjusted Game",
  "dporpag": "Defensive PORPAG",
  "usg": "Usage Rate",
  "rim%": "Shooting % at the rim",
  "rim_makes": "Shots made at rim",
  "middy fg%": "Shooting % on midrange",
  "3p%": "Shooting % on 3s",
  "ast": "Assist Rate",
  "to": "Turnover Rate",
  "stops": "Estimate of defensive stops made (HIDDEN in UI)",
  "total RAPM": "Regularized Adjusted Plus-Minus",
  "adj_oe": "Adjusted Offensive Efficiency",
  "adj_de": "Adjusted Defensive Efficiency",
  "recruit rank": "Recruiting Rank",
  "pick": "NBA Draft Pick",
  "rim_makes_calc": "Estimated rim makes",
  "middy_makes_calc": "Midrange makes",
  "middy_makes": "Midrange Makes (Calculated)",

  // per-100
  "stls/100": "Steals per 100 defensive possessions (100 * steals / def poss).",
  "blks/100": "Blocks per 100 defensive possessions (100 * blocks / def poss).",
  "stops/100": "Stops per 100 defensive possessions.",
  "3pa/100": "3PA per 100 offensive possessions (attempts).",
  "2pa/100": "2PA per 100 offensive possessions (attempts).",
  "midfga/100": "Midrange FGA per 100 offensive possessions (attempts).",
  "rimfga/100": "Rim/finish FGA per 100 offensive possessions (attempts).",
  "dunkfga/100": "Dunk attempts per 100 offensive possessions (attempts).",
  "gbpm": "Game Box Plus-Minus (Torvik)",
  "per": "Player Efficiency Rating (PER)",
};

// --- STAT CONFIG ---
const STAT_CONFIG = [
  {
    group: "Overall Value",
    stats: [
      { key: 'total RAPM', source: 'H' },
      { key: 'offensive rapm', source: 'H' },
      { key: 'defensive rapm', source: 'H' },
      { key: 'porpag', source: 'B' },
      { key: 'dporpag', source: 'B' },
      { key: 'obpm', source: 'B' },
      { key: 'dbpm', source: 'B' },
      { key: 'gbpm', label: 'BPM', source: 'B' },
      { key: 'ortg', source: 'B' },
      { key: 'drtg', source: 'B' },
      { key: 'adj_oe', source: 'B' },
      { key: 'adj_de', source: 'B' },
      { key: 'per', label: 'PER', source: 'B' },
    ]
  },
  {
    group: "General Shooting",
    stats: [
      { key: 'ts', source: 'B', isPct: true },
      { key: 'efg', source: 'B', isPct: true },
      { key: 'fg_pct', label: 'FG%', source: 'B', isPct: true },
      { key: 'ft_pct', source: 'B', isPct: true },
      { key: 'ftr', label: 'FT Rate', source: 'B', isPct: true },
      { key: 'fta', label: 'FTA', source: 'B' },
      { key: 'ftm', label: 'FTM', source: 'B' },
    ]
  },
  {
    group: "2P: General",
    stats: [
      { key: 'two_a', label: '2PA', source: 'B' },
      { key: 'two_m', label: '2PM', source: 'B' },
      { key: '2p%', source: 'H', isPct: true },
      { key: '2pa/100', label: '2PA/100', source: 'H', tooltip: METRIC_DEFINITIONS['2pa/100'] },
      { key: 'transition 2p%', source: 'H', isPct: true },
    ]
  },
  {
    group: "2P: Rim/Finish",
    stats: [
      { key: 'rim_makes', label: 'Rim M', source: 'H' },
      { key: 'rim_m', label: 'Rim M', source: 'H' },
      { key: 'rim_makes_calc', label: 'Rim M', source: 'H', tooltip: METRIC_DEFINITIONS['rim_makes_calc'] },
      { key: 'rim attempts', source: 'H' },
      { key: 'rim%', source: 'H', isPct: true },
      { key: 'rimfga/100', label: 'Rim FGA/100', source: 'H', tooltip: METRIC_DEFINITIONS['rimfga/100'] },
      { key: 'rim attempt rate', source: 'H', isPct: true },
      { key: 'assisted rim fg%', source: 'H', isPct: true },
      { key: 'dunk_pct', label: 'dunk%', source: 'B', isPct: true, tooltip: 'Included in Rim Attempts' },
      { key: 'dunk_m', label: 'dunks', source: 'B', tooltip: 'Included in Rim Attempts' },
      { key: 'dunkfga/100', label: 'Dunk A/100', source: 'H', tooltip: METRIC_DEFINITIONS['dunkfga/100'] },
      { key: 'dunk_a', label: 'dunk A', source: 'B', tooltip: 'Included in Rim Attempts' },
      { key: 'opponent 2p rim fg%', label: 'opp rim%', source: 'H', isPct: true }
    ]
  },
  {
    group: "2P: Midrange",
    stats: [
      { key: 'middy_makes', label: 'Middy M', source: 'H' },
      { key: 'middy_m', label: 'Middy M', source: 'H' },
      { key: 'middy_makes_calc', label: 'Middy M', source: 'H', tooltip: METRIC_DEFINITIONS['middy_makes_calc'] },
      { key: 'middy attempts', label: 'Middy A', source: 'H' },
      { key: 'middy fg%', source: 'H', isPct: true },
      { key: 'midfga/100', label: 'Middy FGA/100', source: 'H', tooltip: METRIC_DEFINITIONS['midfga/100'] },
      { key: 'middy attempt rate', source: 'H', isPct: true },
      { key: 'assisted middy fg%', source: 'H', isPct: true },
      { key: 'transition midrange fg%', source: 'H', isPct: true },
    ]
  },
  {
    group: "3-Pointer",
    stats: [
      { key: 'three_m', label: '3PM', source: 'B' },
      { key: 'three_a', label: '3PA', source: 'B' },
      { key: '3p%', source: 'H', isPct: true },
      { key: '3pa/100', label: '3PA/100', source: 'H', tooltip: METRIC_DEFINITIONS['3pa/100'] },
      { key: '3 pt rate', source: 'H', isPct: true },
      { key: 'assisted 3p %', source: 'H', isPct: true },
      { key: 'transition 3p%', source: 'H', isPct: true },
    ]
  },
  {
    group: "Playmaking",
    stats: [
      { key: 'usg', label: 'usg%', source: 'B', isPct: true },
      { key: 'ast', label: 'ast%', source: 'B' },
      { key: 'to', label: 'to%', source: 'B' },
      { key: 'ast_to', label: 'A:TO', source: 'B' },

      { key: '% of assists end w/ rim', source: 'H', isPct: true },
      { key: '% of assists end w/ 3p', source: 'H', isPct: true },
      { key: '% of assists end w/ middy', source: 'H', isPct: true },

      { key: 'pure_point_rating', source: 'B' },
    ]
  },
  {
    group: "Defense & Reb",
    stats: [
      { key: 'stl', label: 'stl%', source: 'B' },
      { key: 'stls/100', label: 'STL/100', source: 'H', tooltip: METRIC_DEFINITIONS['stls/100'] },
      { key: 'oreb_rate', label: 'oreb%', source: 'B' },
      { key: 'dreb_rate', label: 'dreb%', source: 'B' },
      { key: 'blk', label: 'blk%', source: 'B' },
      { key: 'blks/100', label: 'BLK/100', source: 'H', tooltip: METRIC_DEFINITIONS['blks/100'] },
      { key: 'stops/100', label: 'STOPS/100', source: 'H', tooltip: METRIC_DEFINITIONS['stops/100'] },
      { key: 'pfr', label: 'Foul Rate', source: 'B' },
    ]
  },
  {
    group: "PRA / Box",
    stats: [
      { key: 'ppg', source: 'B' },
      { key: 'rpg', source: 'B' },
      { key: 'apg', source: 'B' },
      { key: 'spg', source: 'B' },
      { key: 'bpg', source: 'B' },
      { key: 'mpg', source: 'B' },
      { key: 'oreb', source: 'B' },
      { key: 'dreb', source: 'B' },
      { key: 'tov', label: 'tov', source: 'B' },
      { key: 'fgm', label: 'FGM', source: 'B' },
      { key: 'fga', label: 'FGA', source: 'B' },
    ]
  },
  {
    group: "Draft & Recruiting",
    stats: [
      { key: 'pick', source: 'B' },
      { key: 'recruit rank', source: 'B' }
    ]
  },
  {
    group: "NBA Career",
    stats: [
      { key: 'nba_WS', label: 'NBA WS', source: 'BR', tooltip: 'NBA Career Win Shares' },
      { key: 'nba_WS48', label: 'WS/48', source: 'BR', tooltip: 'NBA Career Win Shares per 48 minutes' },
      { key: 'nba_VORP', label: 'NBA VORP', source: 'BR', tooltip: 'NBA Career Value Over Replacement Player' },
      { key: 'nba_BPM', label: 'NBA BPM', source: 'BR', tooltip: 'NBA Career Box Plus-Minus' },
      { key: 'nba_OBPM', label: 'NBA OBPM', source: 'BR', tooltip: 'NBA Career Offensive Box Plus-Minus' },
      { key: 'nba_DBPM', label: 'NBA DBPM', source: 'BR', tooltip: 'NBA Career Defensive Box Plus-Minus' },
      { key: 'nba_PER', label: 'NBA PER', source: 'BR', tooltip: 'NBA Career Player Efficiency Rating' },
      { key: 'nba_TS', label: 'NBA TS%', source: 'BR', isPct: true, tooltip: 'NBA Career True Shooting %' },
      { key: 'nba_USG', label: 'NBA USG%', source: 'BR', isPct: true, tooltip: 'NBA Career Usage Rate' },
      { key: 'nba_AST_pct', label: 'NBA AST%', source: 'BR', isPct: true, tooltip: 'NBA Career Assist Percentage' },
      { key: 'nba_STL_pct', label: 'NBA STL%', source: 'BR', isPct: true, tooltip: 'NBA Career Steal Percentage' },
      { key: 'nba_BLK_pct', label: 'NBA BLK%', source: 'BR', isPct: true, tooltip: 'NBA Career Block Percentage' },
    ]
  }
];

// defaults
const DEFAULT_VISIBLE = new Set([
  'porpag', 'obpm', 'dbpm', 'gbpm', 'total RAPM', 'per',
  'fg_pct', 'ts', 'ft_pct',
  '2p%', '3p%', 'rim%',
  'dunk_m',
  'usg', 'ast_to', 'ast',
  'stl', 'blk', 'oreb_rate', 'dreb_rate',
  'midfga/100',
  'rimfga/100',
]);

const HIDDEN_COLS = new Set([
  'roster_ncaa_id', 'torvik_hgt_in', 'torvik_id',
  'min', 'total minutes',
  'rim_makes_calc', 'middy_makes_calc',
  'stops',
]);

const NO_COLOR_STATS = new Set([
  'g', 'height', 'weight', 'bmi', 'exp', 'year', 'num', 'pick',
  'recruit rank', 'pos', 'conf', 'torvik_year', 'length', 'mpg'
]);
const INVERT_COLOR_STATS = new Set(['defensive rapm', 'drtg', 'adj_de', 'to', 'tov']);
const TEXT_COLS = new Set(['key', 'team', 'pos', 'conf', 'exp', 'player', 'year', 'pick', 'recruit rank']);
const DEFAULT_FILTERS = { minGames: '10', year: '2026', team: 'All' };

// ... (Helper functions remain unchanged) ...
const getStatLabel = (key) => {
  for (const group of STAT_CONFIG) {
    const found = group.stats.find(s => s.key === key);
    if (found) return found.label || found.key;
  }
  return key;
};

const getColor = (val, stats, key) => {
  if (!stats || stats.std === 0 || val === null || val === undefined) return 'inherit';
  let z = (val - stats.mean) / stats.std;
  if (INVERT_COLOR_STATS.has(key)) z = -z;
  const clampedZ = Math.max(-2.5, Math.min(2.5, z));
  const normalized = (clampedZ + 2.5) / 5;
  const r = Math.round(255 * (1 - normalized));
  const g = Math.round(255 * normalized);
  return `rgba(${r}, ${g}, 0, 0.35)`;
};

const formatName = (name) => {
  if (!name || typeof name !== 'string') return name;
  if (name.includes(',')) {
    const parts = name.split(',').map(s => s.trim());
    if (parts.length === 2) return `${parts[1]} ${parts[0]}`;
  }
  return name;
};

const formatValue = (val, key) => {
  if (val === null || val === undefined) return val;
  if (typeof val !== 'number') return val;
  const conf = STAT_CONFIG.flatMap(g => g.stats).find(s => s.key === key);
  if (conf?.isPct) {
    let num = val;
    if (Math.abs(num) <= 1.0 && num !== 0) num = num * 100;
    return num.toFixed(2);
  }
  return Number.isInteger(val) ? val : val.toFixed(2);
};

const expBucket = (exp) => {
  const s = String(exp || '').toLowerCase();
  if (s.includes('fr')) return 'Fr';
  if (s.includes('so')) return 'So';
  if (s.includes('jr')) return 'Jr';
  if (s.includes('sr')) return 'Sr';
  return '';
};

const calcBMI = (weight, height) => {
  if (!weight || !height) return null;
  const w = parseFloat(weight);
  const h = parseFloat(height);
  if (isNaN(w) || isNaN(h) || h === 0) return null;
  return ((w * 703) / (h * h)).toFixed(1);
};

const cleanNumeric = (val) => {
  if (typeof val === 'number') return val;
  if (!val) return null;
  const cleaned = String(val).replace(/[^0-9.-]/g, '');
  const n = parseFloat(cleaned);
  return isNaN(n) ? null : n;
};

const dedupeInternationalRows = (rows) => {
  const keyFor = (r) => `${r.key}|${r.team}|${r.league}|${r.torvik_year}`;
  const scoreRow = (r) => {
    const preferred = ['ast', 'stl', 'ast_to', 'blk'];
    let score = 0;
    preferred.forEach(k => {
      const v = r[k];
      if (v !== null && v !== undefined && v !== '' && !(typeof v === 'number' && Number.isNaN(v))) {
        score += 10;
      }
    });
    // Tie-breaker: count total non-null numeric fields
    let nonNull = 0;
    Object.keys(r).forEach(k => {
      const v = r[k];
      if (v !== null && v !== undefined && v !== '' && !(typeof v === 'number' && Number.isNaN(v))) nonNull += 1;
    });
    return score * 1000 + nonNull;
  };

  const bestByKey = new Map();
  rows.forEach(r => {
    if (!r.isInternational || Number(r.torvik_year) !== 2026) return;
    const k = keyFor(r);
    const prev = bestByKey.get(k);
    if (!prev || scoreRow(r) > scoreRow(prev)) bestByKey.set(k, r);
  });

  return rows.filter(r => {
    if (!r.isInternational || Number(r.torvik_year) !== 2026) return true;
    const k = keyFor(r);
    return bestByKey.get(k) === r;
  });
};

const fetchCsv = (file) => new Promise((resolve) => {
  Papa.parse(file, {
    download: true,
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: (results) => resolve(results.data),
    error: () => resolve([])
  });
});

// --- UPDATED PROCESS ROWS: Includes LEAGUE Assignment + Intl Weight + (INTL) Team Prefix + PER-GAME to TOTAL FIX ---
// Added overrideYear parameter to force 2026 for specific files
const processRows = (rawData, weightData = [], isInternationalData = false, overrideYear = null) => {
  const weightMap = new Map();
  if (weightData && weightData.length > 0) {
    weightData.forEach(r => {
      if (r.torvik_id && r.weight) weightMap.set(String(r.torvik_id), r.weight);
    });
  }

  const temp = rawData.map((row, index) => {
    const nameRaw = row.key || row.player || row.player_name || row.Name;
    const fixedName = formatName(nameRaw);

    if (row.g === undefined) {
      row.g = row.games ?? row.G ?? row.gp ?? row.GP ?? 0;
    }

    const torvikId = row?.torvik_id ?? row?.roster_ncaa_id ?? row?.id ?? null;
    const baseId = torvikId !== null
      ? String(torvikId)
      : `${fixedName}-${row.team}-${row.torvik_year}-${index}`;

    return { row, index, fixedName, torvikId, baseId };
  });

  const lengthMap = new Map<string, number>();
  temp.forEach(({ baseId }) => {
    lengthMap.set(baseId, (lengthMap.get(baseId) || 0) + 1);
  });

  return temp.map(({ row, index, fixedName, torvikId, baseId }) => {
    const pickVal = cleanNumeric(row['pick']);
    const rankVal = cleanNumeric(row['recruit rank']);

    const toRate = (v) => {
      if (v === null || v === undefined) return null;
      const n = Number(v);
      if (Number.isNaN(n)) return null;
      return n > 1 ? n / 100 : n;
    };
    const toNum = (v) => {
      if (v === null || v === undefined) return null;
      const n = Number(v);
      return Number.isNaN(n) ? null : n;
    };

    const rimAtt = toNum(row['rim attempts'] ?? row['rim_a'] ?? row['rim_att']);
    const rimPct = toRate(row['rim%'] ?? row['rim_fg%'] ?? row['rim pct']);
    const middyAtt = toNum(row['middy attempts'] ?? row['middy_a'] ?? row['middy_att']);
    const middyPct = toRate(row['middy fg%'] ?? row['middy%'] ?? row['middy_fg_pct']);

    const rimMakesCalc = (rimAtt !== null && rimPct !== null) ? rimAtt * rimPct : null;
    let middyMakesCalc = (middyAtt !== null && middyPct !== null) ? middyAtt * middyPct : null;

    // New: Explicit middy_makes calculated field
    const middyMakes = middyMakesCalc;

    // --- WEIGHT LOGIC ---
    let weight = null;
    if (isInternationalData) {
      weight = row['weight'] ? cleanNumeric(row['weight']) : null;
    } else {
      weight = torvikId ? weightMap.get(String(torvikId)) : null;
    }

    const bmi = calcBMI(weight, row['hoop_hgt_in']);
    const length = lengthMap.get(baseId) || 1;

    // LEAGUE & TEAM NAME LOGIC
    let league = "NCAA D1";
    let teamName = row.team;

    // Handle Intl Data adjustments
    let adjustedStats = {};

    if (isInternationalData) {
      league = row['conf'] || 'INTL';
      // Prefix (INTL) to team name for visibility
      if (teamName) {
        teamName = `(INTL) ${teamName}`;
      }

      // CONVERT PER-GAME TO TOTALS for display consistency
      const games = row.g || 0;
      const statsToConvert = ['two_a', 'two_m', 'three_a', 'three_m', 'fta', 'ftm'];

      statsToConvert.forEach(k => {
        const val = row[k];
        if (typeof val === 'number') {
          adjustedStats[k] = val * games;
        }
      });
    }

    // OVERRIDE YEAR (for files that lack it or are implied 2026)
    let year = row['torvik_year'] || row['year'] || row['Season'];
    if (overrideYear) {
      year = overrideYear;
    }

    // Ensure middy_makes is present in the final object
    const processedRow = {
      ...row,
      ...adjustedStats,
      id: baseId,
      key: fixedName,
      _nameLower: (fixedName || '').toLowerCase(),
      height: row['hoop_hgt_in'],
      weight: weight,
      bmi: bmi ? parseFloat(bmi) : null,
      exp: row['exp'] || '-',
      pick: pickVal || row['pick'],
      'recruit rank': rankVal || row['recruit rank'],
      rim_makes_calc: rimMakesCalc,
      middy_makes_calc: middyMakesCalc,
      middy_makes: middyMakes, // Add new stat
      length,
      isInternational: isInternationalData,
      league: league,
      team: teamName,
      torvik_year: year
    };

    // FIX: Calculate MPG for career mode if missing but total minutes exists
    const totalMinutes = typeof row['total minutes'] === 'number' ? row['total minutes'] : parseFloat(row['total minutes']);
    const gamesPlayed = typeof row.g === 'number' ? row.g : parseFloat(row.g);

    if ((processedRow.mpg === null || processedRow.mpg === undefined || String(processedRow.mpg).toLowerCase() === 'na' || isNaN(processedRow.mpg)) &&
      !isNaN(totalMinutes) &&
      !isNaN(gamesPlayed) &&
      gamesPlayed > 0) {
      processedRow.mpg = Number((totalMinutes / gamesPlayed).toFixed(1));
    }

    return processedRow;
  });
};


interface ErrorBoundaryProps {
  onClose: () => void;
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(_: Error): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("UI crashed:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 20 }}>
          Something went wrong. <button onClick={this.props.onClose}>Close</button>
        </div>
      );
    }
    return this.props.children;
  }
}

// --- HELPER: GET DROPDOWN OPTIONS ---
const getDropdownOptions = () => {
  return STAT_CONFIG.flatMap(g => g.stats).map(s => ({
    key: s.key,
    label: s.label || s.key
  }));
};

// --- CUSTOM MULTI-SELECT COMPONENT FOR LEAGUES ---
const LeagueMultiSelect = ({ options, selected, onChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const toggleOption = (opt) => {
    const newSelected = new Set(selected);
    if (newSelected.has(opt)) newSelected.delete(opt);
    else newSelected.add(opt);
    onChange(newSelected);
  };

  const selectAll = () => {
    if (selected.size === options.length) onChange(new Set());
    else onChange(new Set(options));
  };

  return (
    <div className="league-select-container" ref={containerRef}>
      <button
        className="league-select-btn"
        onClick={() => setIsOpen(!isOpen)}
      >
        <span>
          {selected.size === 0 ? "Select Leagues" :
            selected.size === options.length ? "All Leagues" :
              `${selected.size} Selected`}
        </span>
        <ChevronDown size={14} />
      </button>

      {isOpen && (
        <div className="league-dropdown" style={{ background: 'white', border: '1px solid var(--border-subtle)', boxShadow: 'var(--glass-shadow)', borderRadius: '12px', padding: '12px', zindex: 1000 }}>
          <div className="league-dropdown-header" style={{ marginBottom: '10px', paddingBottom: '8px', borderBottom: '1px solid var(--border-subtle)' }}>
            <span onClick={selectAll} className="select-all-btn" style={{ fontSize: '0.75rem', color: 'var(--accent-primary)', fontWeight: 600, cursor: 'pointer' }}>
              {selected.size === options.length ? "Deselect All" : "Select All"}
            </span>
          </div>
          {options.map(opt => (
            <label key={opt} className="league-option" style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 0', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={selected.has(opt)}
                onChange={() => toggleOption(opt)}
              />
              <span style={{ color: opt === 'NCAA D1' ? 'var(--text-primary)' : 'var(--accent-secondary)', fontWeight: opt === 'NCAA D1' ? 600 : 500, fontSize: '0.875rem' }}>
                {opt}
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
};


// --- SINGLE PLAYER MODAL (Updated to respect visibleCols) ---
const PlayerDetailModal = ({ player, historyData, statConfig, visibleCols, brStats, onClose }) => {
  const [customStat, setCustomStat] = useState('gbpm');
  const dropdownOptions = getDropdownOptions();

  const effectiveVisibleCols = useMemo(() => {
    if (!player?.isInternational) {
      return new Set([...visibleCols].filter(k => k !== 'per'));
    }
    return visibleCols;
  }, [player, visibleCols]);

  // Filter the stat config to only show cols that are currently visible in the main table
  const filteredConfig = useMemo(() => (
    statConfig
      .map(g => ({ ...g, stats: g.stats.filter(s => effectiveVisibleCols.has(s.key)) }))
      .filter(g => g.stats.length > 0)
  ), [statConfig, effectiveVisibleCols]);

  if (!player) return null;

  const { playerHistory, radarData } = useMemo(() => {
    if (!historyData || historyData.length === 0) return { playerHistory: [], radarData: [] };

    const pid = player?.torvik_id ?? player?.roster_ncaa_id ?? player?.id;
    const history = historyData
      .filter(p => {
        const otherId = p?.torvik_id ?? p?.roster_ncaa_id ?? p?.id;
        if (pid && otherId) return String(otherId) === String(pid);
        return p._nameLower === player._nameLower;
      })
      .sort((a, b) => (a.torvik_year || 0) - (b.torvik_year || 0));

    if (history.length === 0) return { playerHistory: [], radarData: [] };

    const current = history[history.length - 1] || {};
    const safeVal = (val, mul = 1) => (val === undefined || val === null || Number.isNaN(val) ? 0 : val * mul);

    const rData = [
      { subject: 'Finishing', A: Math.min(100, safeVal(current['rim%'], 100)), fullMark: 100 },
      { subject: 'Shooting', A: Math.min(100, safeVal(current['3p%'], 100)), fullMark: 100 },
      { subject: 'Playmaking', A: safeVal(current['ast'], 2.5), fullMark: 100 },
      { subject: 'Rebounding', A: safeVal((current['oreb_rate'] || 0) + (current['dreb_rate'] || 0), 2), fullMark: 100 },
      { subject: 'Defense', A: safeVal((current['blk'] || 0) + (current['stl'] || 0), 12), fullMark: 100 },
    ];

    return { playerHistory: history, radarData: rData };
  }, [player, historyData]);

  //  Get BR stats for this player
  const playerBrStats = brStats?.get(player._nameLower) || null;


  if (playerHistory.length === 0) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}><X size={20} /></button>
        <div className="modal-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
          <div>
            <h2 className="player-title" style={{ fontSize: '2rem', fontWeight: 800, color: player.isInternational ? 'var(--accent-secondary)' : 'var(--text-primary)' }}>{player.key}</h2>
            <div className="player-meta" style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              {player.league} • {player.team} • {player.pos} • {player.height} • {player.weight ? `${player.weight}lbs` : ''} • BMI: {player.bmi || '-'}
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Latest RAPM</div>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: player['total RAPM'] > 0 ? 'var(--accent-tertiary)' : 'var(--accent-secondary)' }}>
              {typeof player['total RAPM'] === 'number' ? player['total RAPM']?.toFixed(1) : '-'}
            </div>
          </div>
        </div>

        {/* NBA Career Stats Panel */}
        <NBACareerStats brStats={playerBrStats} />

        <div className="dashboard-grid">
          <div className="chart-card">
            <div className="chart-header">
              <span className="chart-title">Custom Trajectory</span>
              <select className="stat-selector" value={customStat} onChange={(e) => setCustomStat(e.target.value)}>
                {dropdownOptions.map(opt => (<option key={opt.key} value={opt.key}>{opt.label}</option>))}
              </select>
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={playerHistory}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                <XAxis dataKey="torvik_year" tick={{ fontSize: 12 }} />
                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} />
                <RechartsTooltip contentStyle={{ background: '#333', border: 'none', borderRadius: '8px', color: '#fff' }} itemStyle={{ color: '#fff' }} />
                {customStat && (
                  <Line
                    type="monotone"
                    dataKey={customStat}
                    stroke="#2563eb"
                    strokeWidth={3}
                    dot={{ r: 4, fill: '#2563eb', strokeWidth: 2, stroke: '#fff' }}
                    activeDot={{ r: 6 }}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}><span className="chart-title" style={{ fontSize: '0.875rem', fontWeight: 700, color: 'var(--text-secondary)' }}>RAPM Impact (Off vs Def)</span></div>
            <ResponsiveContainer width="100%" height={250}>
              {playerHistory.some(r => r['offensive rapm'] !== undefined) ? (
                <BarChart data={playerHistory}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border-subtle)" />
                  <XAxis dataKey="torvik_year" tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
                  <YAxis domain={[-15, 15]} tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
                  <RechartsTooltip cursor={{ fill: 'transparent' }} contentStyle={{ borderRadius: '12px', border: '1px solid var(--border-subtle)', boxShadow: 'var(--glass-shadow)' }} />
                  <Legend iconType="circle" />
                  <ReferenceLine y={0} stroke="var(--text-muted)" />
                  <Bar dataKey="offensive rapm" name="Offense" fill="var(--accent-primary)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="defensive rapm" name="Defense" fill="var(--accent-secondary)" radius={[4, 4, 0, 0]} />
                </BarChart>
              ) : <div style={{ padding: 20, color: 'var(--text-muted)', fontSize: '0.875rem' }}>No RAPM Data</div>}
            </ResponsiveContainer>
          </div>

          <div className="chart-card">
            <div className="chart-header"><span className="chart-title">Playstyle Profile</span></div>
            <ResponsiveContainer width="100%" height={450}>
              <RadarChart cx="50%" cy="50%" outerRadius="120" data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: '#333', fontWeight: 600 }} />
                <Radar name={player.key} dataKey="A" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.5} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <h3 style={{ margin: '0 0 15px 0', fontSize: '1.2rem' }}>Complete Career Stats</h3>
        <div className="full-stats-container">
          <table className="full-stats-table">
            <thead>
              <tr>
                <th rowSpan={2} style={{ minWidth: '140px' }}>Year / Team</th>
                {filteredConfig.map((group, i) => (
                  <th key={i} colSpan={group.stats.length} className="table-section-header">{group.group}</th>
                ))}
              </tr>
              <tr>
                {filteredConfig.map(group => (
                  group.stats.map(stat => (
                    <th key={stat.key} title={stat.key}>{stat.label || stat.key.replace(/_/g, ' ')}</th>
                  ))
                ))}
              </tr>
            </thead>
            <tbody>
              {playerHistory.map((season, i) => (
                <tr key={i}>
                  <td>
                    <div style={{ fontWeight: 'bold' }}>{season.torvik_year}</div>
                    <div style={{ fontSize: '0.75rem', color: '#666', fontWeight: 'normal' }}>{season.team}</div>
                    <div style={{ fontSize: '0.65rem', color: '#999', fontStyle: 'italic' }}>{season.league}</div>
                  </td>
                  {filteredConfig.map(group => (
                    group.stats.map(groupStats => {
                      const val = season[groupStats.key];
                      return (
                        <td key={groupStats.key}>
                          {groupStats.isPct && typeof val === 'number'
                            ? ((val <= 1 && val !== 0 ? (val * 100) : val).toFixed(1) + '%')
                            : (typeof val === 'number' ? val.toFixed(1) : (val || '-'))}
                        </td>
                      );
                    })
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// --- MULTI-PLAYER COMPARISON SECTION ---
const ComparisonSection = ({ players, historyData, statsDistribution }) => {
  // ... (Same as before) ...
  const [customStat, setCustomStat] = useState('gbpm');
  const [radarConfig, setRadarConfig] = useState(['gbpm', 'usg', 'ts', 'ast', 'oreb_rate', 'dreb_rate']);
  const [showRadarConfig, setShowRadarConfig] = useState(false);

  const [scatterX, setScatterX] = useState('usg');
  const [scatterY, setScatterY] = useState('ts');

  const [visibleViz, setVisibleViz] = useState({
    trajectory: false,
    scatter: true,
    headToHead: true,
    radar: false
  });

  const dropdownOptions = getDropdownOptions();

  const { chartData, radarData, currentStats } = useMemo(() => {
    if (!players || players.length === 0 || !historyData) return { chartData: [], radarData: [], currentStats: [] };

    let mergedHistory = [];
    const currentStatsCalc = [];

    players.forEach((p, idx) => {
      const pid = p?.torvik_id ?? p?.roster_ncaa_id ?? p?.id;
      const pHistory = historyData.filter(h => {
        const otherId = h?.torvik_id ?? h?.roster_ncaa_id ?? h?.id;
        if (pid && otherId) return String(otherId) === String(pid);
        return h._nameLower === p._nameLower;
      }).sort((a, b) => a.torvik_year - b.torvik_year);

      pHistory.forEach((season, seasonIdx) => {
        const normalizedYear = `Y${seasonIdx + 1}`;
        let existing = mergedHistory.find(m => m.normYear === normalizedYear);
        if (!existing) {
          existing = { normYear: normalizedYear, sortIdx: seasonIdx };
          mergedHistory.push(existing);
        }
        Object.keys(season).forEach(key => {
          if (typeof season[key] === 'number') {
            existing[`${p.key}_${key}`] = season[key];
          }
        });
      });

      currentStatsCalc.push({ key: p.key, current: pHistory[pHistory.length - 1] || {} });
    });

    mergedHistory.sort((a, b) => a.sortIdx - b.sortIdx);

    const radarSeries = currentStatsCalc.map((p, idx) => {
      const dataPoints = radarConfig.map(metric => {
        const val = p.current[metric];
        let normalized = 50;
        if (statsDistribution && statsDistribution[metric] && typeof val === 'number') {
          const { mean, std } = statsDistribution[metric];
          if (std > 0) {
            let z = (val - mean) / std;
            if (['defensive rapm', 'drtg', 'adj_de', 'to', 'tov'].includes(metric)) z = -z;
            normalized = ((z + 2.5) / 5) * 100;
            normalized = Math.max(0, Math.min(100, normalized));
          }
        }
        return { subject: getStatLabel(metric), A: normalized };
      });
      return { name: p.key, color: COMPARE_COLORS[idx % COMPARE_COLORS.length], data: dataPoints };
    });

    return { chartData: mergedHistory, radarData: radarSeries, currentStats: currentStatsCalc };
  }, [players, historyData, radarConfig, statsDistribution]);

  const combinedRadar = useMemo(() => {
    if (radarData.length === 0) return [];
    const subjects = radarData[0].data.map(d => d.subject);
    return subjects.map((subj, i) => {
      const point = { subject: subj };
      radarData.forEach(p => { point[p.name] = p.data[i].A; });
      return point;
    });
  }, [radarData]);

  const scatterData = useMemo(() => {
    return currentStats.map((p, i) => ({
      name: p.key,
      x: p.current[scatterX] || 0,
      y: p.current[scatterY] || 0,
      color: COMPARE_COLORS[i % COMPARE_COLORS.length]
    }));
  }, [currentStats, scatterX, scatterY]);

  const xMean = statsDistribution?.[scatterX]?.mean || 0;
  const yMean = statsDistribution?.[scatterY]?.mean || 0;
  const xLabel = getStatLabel(scatterX);
  const yLabel = getStatLabel(scatterY);


  const EDGE_METRICS = ['bpm', 'usg', 'ts', 'ast', 'to', 'oreb_rate', 'dreb_rate', 'stl', 'blk'];
  const p1 = currentStats[0];
  const p2 = currentStats[1];

  const edgeData = useMemo(() => {
    if (!p1 || !p2) return [];
    return EDGE_METRICS.map(m => {
      const v1 = p1.current[m] || 0;
      const v2 = p2.current[m] || 0;

      let zScoreDiff = 0;
      if (statsDistribution && statsDistribution[m]) {
        const { std } = statsDistribution[m];
        if (std > 0) {
          let diff = v2 - v1;
          if (['defensive rapm', 'drtg', 'adj_de', 'to', 'tov'].includes(m)) diff = v1 - v2;
          zScoreDiff = (diff / std);
        }
      }
      return { metric: getStatLabel(m), diff: zScoreDiff };
    });
  }, [p1, p2, statsDistribution]);

  const addRadarMetric = () => {
    if (radarConfig.length >= 6) return;
    setRadarConfig([...radarConfig, dropdownOptions[0]?.key || 'bpm']);
  };
  const removeRadarMetric = (index) => {
    if (radarConfig.length <= 3) return;
    const newConfig = [...radarConfig];
    newConfig.splice(index, 1);
    setRadarConfig(newConfig);
  };

  const toggleViz = (key) => {
    setVisibleViz(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const activeCount = Object.values(visibleViz).filter(Boolean).length;
  const gridStyle = {
    display: 'grid',
    gridTemplateColumns: activeCount === 1 ? '1fr' : 'repeat(auto-fit, minmax(400px, 1fr))',
    gap: '20px',
    marginBottom: '30px'
  };

  return (
    <div style={{ margin: '2rem 0', padding: '2rem', border: '1px solid var(--border-subtle)', borderRadius: '24px', background: 'white', boxShadow: 'var(--glass-shadow)' }}>
      <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem' }}>
        <div>
          <h2 className="player-title" style={{ fontSize: '2rem', fontWeight: 800 }}>Comparison Visualization</h2>
          <div className="player-meta" style={{ marginTop: '0.5rem' }}>
            {players.map((p, i) => (
              <span key={p.id} style={{ color: COMPARE_COLORS[i % COMPARE_COLORS.length], marginRight: '1.25rem', fontWeight: 700, fontSize: '1rem' }}>
                {p.key}
              </span>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px', background: 'var(--bg-secondary)', padding: '6px 12px', borderRadius: '12px', fontSize: '0.8rem', fontWeight: 600, border: '1px solid var(--border-subtle)' }}>
          <span style={{ color: 'var(--text-muted)', marginRight: '4px' }}>Visualizations:</span>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
            <input type="checkbox" checked={visibleViz.headToHead} onChange={() => toggleViz('headToHead')} /> Head-to-Head
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
            <input type="checkbox" checked={visibleViz.scatter} onChange={() => toggleViz('scatter')} /> Scatter
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
            <input type="checkbox" checked={visibleViz.trajectory} onChange={() => toggleViz('trajectory')} /> Trajectory
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
            <input type="checkbox" checked={visibleViz.radar} onChange={() => toggleViz('radar')} /> Radar
          </label>
        </div>
      </div>

      <div style={gridStyle}>
        {visibleViz.headToHead && (
          <div className="chart-card">
            <div className="chart-header">
              <span className="chart-title">{players.length === 2 ? "Head-to-Head Advantage (SD)" : "Stat Face-Off"}</span>
            </div>
            <ResponsiveContainer width="100%" height={250}>
              {players.length === 2 ? (
                <BarChart layout="vertical" data={edgeData} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                  <XAxis type="number" domain={['auto', 'auto']} hide />
                  <YAxis dataKey="metric" type="category" width={100} tick={{ fontSize: 11, fill: '#333' }} />
                  <ReferenceLine x={0} stroke="#000" />
                  <RechartsTooltip cursor={{ fill: 'transparent' }} />
                  <Bar dataKey="diff" barSize={20}>
                    {edgeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.diff > 0 ? COMPARE_COLORS[1] : COMPARE_COLORS[0]} />
                    ))}
                  </Bar>
                </BarChart>
              ) : (
                <div style={{ padding: 20, color: '#999', fontStyle: 'italic', display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                  Head-to-Head requires exactly 2 players.
                </div>
              )}
            </ResponsiveContainer>
          </div>
        )}

        {visibleViz.scatter && (
          <div className="chart-card">
            <div className="chart-header" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '8px' }}>
              <span className="chart-title">Scatter Comparison</span>
              <div style={{ display: 'flex', gap: '10px', width: '100%' }}>
                <select className="stat-selector" value={scatterX} onChange={e => setScatterX(e.target.value)} style={{ flex: 1 }}>
                  {dropdownOptions.map(m => <option key={m.key} value={m.key}>X: {m.label}</option>)}
                </select>
                <select className="stat-selector" value={scatterY} onChange={e => setScatterY(e.target.value)} style={{ flex: 1 }}>
                  {dropdownOptions.map(m => <option key={m.key} value={m.key}>Y: {m.label}</option>)}
                </select>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <ScatterChart margin={{ top: 10, right: 10, bottom: 30, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" name={xLabel} tick={{ fontSize: 11 }} domain={['auto', 'auto']}>
                  <Label value={xLabel} offset={-10} position="insideBottom" style={{ fontSize: '0.8rem', fill: '#666' }} />
                </XAxis>
                <YAxis type="number" dataKey="y" name={yLabel} tick={{ fontSize: 11 }} domain={['auto', 'auto']}>
                  <Label value={yLabel} angle={-90} position="insideLeft" style={{ fontSize: '0.8rem', fill: '#666', textAnchor: 'middle' }} />
                </YAxis>
                <ReferenceLine x={xMean} stroke="#9ca3af" strokeDasharray="3 3" />
                <ReferenceLine y={yMean} stroke="#9ca3af" strokeDasharray="3 3" />
                <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '8px' }} />
                <Scatter name="Players" data={scatterData}>
                  {scatterData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '10px', marginTop: '-10px', paddingBottom: '10px' }}>
              {scatterData.map((entry, index) => (
                <div key={index} style={{ display: 'flex', alignItems: 'center', fontSize: '0.8rem' }}>
                  <span style={{ width: '10px', height: '10px', background: entry.color, borderRadius: '50%', marginRight: '5px' }}></span>
                  {entry.name}
                </div>
              ))}
            </div>
          </div>
        )}

        {visibleViz.trajectory && (
          <div className="chart-card">
            <div className="chart-header">
              <span className="chart-title">Trajectory (best w/ 3+ seasons)</span>
              <select className="stat-selector" value={customStat} onChange={(e) => setCustomStat(e.target.value)}>
                {dropdownOptions.map(m => (<option key={m.key} value={m.key}>{m.label}</option>))}
              </select>
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                <XAxis dataKey="normYear" tick={{ fontSize: 12 }} />
                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} />
                <RechartsTooltip contentStyle={{ background: '#333', border: 'none', borderRadius: '8px', color: '#fff' }} itemStyle={{ color: '#fff' }} />
                {players.map((p, i) => (
                  <Line
                    key={p.id}
                    type="monotone"
                    dataKey={`${p.key}_${customStat}`}
                    name={p.key}
                    stroke={COMPARE_COLORS[i % COMPARE_COLORS.length]}
                    strokeWidth={3}
                    dot={{ r: 4 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {visibleViz.radar && (
          <div className="chart-card">
            <div className="chart-header">
              <span className="chart-title">Shape (Normalized)</span>
              <div style={{ display: 'flex', gap: '5px' }}>
                <button
                  onClick={addRadarMetric}
                  title="Add Metric"
                  style={{ border: 'none', background: '#eee', borderRadius: '4px', cursor: 'pointer' }}
                >
                  <Plus size={16} />
                </button>
                <button
                  onClick={() => setShowRadarConfig(!showRadarConfig)}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#666' }}
                  title="Configure"
                >
                  <Settings size={16} />
                </button>
              </div>
            </div>

            {showRadarConfig && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px', marginBottom: '10px', padding: '10px', background: '#f9fafb', borderRadius: '8px' }}>
                {radarConfig.map((metric, i) => (
                  <div key={i} style={{ display: 'flex', gap: '4px' }}>
                    <select
                      value={metric}
                      onChange={(e) => {
                        const newConfig = [...radarConfig];
                        newConfig[i] = e.target.value;
                        setRadarConfig(newConfig);
                      }}
                      style={{ fontSize: '0.75rem', padding: '4px', width: '100%' }}
                    >
                      {dropdownOptions.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
                    </select>
                    <button
                      onClick={() => removeRadarMetric(i)}
                      style={{ color: 'red', border: 'none', background: 'none', cursor: 'pointer' }}
                    >
                      <Minus size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <ResponsiveContainer width="100%" height={500}>
              <RadarChart cx="50%" cy="50%" outerRadius="150" data={combinedRadar}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fontWeight: 600 }} />
                <Legend />
                {players.map((p, i) => (
                  <Radar
                    key={p.id}
                    name={p.key}
                    dataKey={p.key}
                    stroke={COMPARE_COLORS[i % COMPARE_COLORS.length]}
                    fill={COMPARE_COLORS[i % COMPARE_COLORS.length]}
                    fillOpacity={0.4}
                  />
                ))}
              </RadarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

// --- MAIN COMPONENT ---
function App() {
  // --- URL ROUTING INITIALIZATION ---
  const getInitialMode = () => {
    // UPDATED: Robust handling of path segments
    // e.g. "/career" -> ["", "career"] -> "career"
    // e.g. "/career/" -> ["", "career", ""] -> "career"
    const segments = window.location.pathname.split('/');
    const path = segments[1] ? segments[1].toLowerCase() : 'season';

    return ['season', 'career', 'compare', 'about'].includes(path) ? path : 'season';
  };

  const [datasetMode, setDatasetMode] = useState(getInitialMode());
  const [compareType, setCompareType] = useState('season');
  const [data, setData] = useState([]); // This holds the RAW merged data (domestic + intl)
  const [seasonData, setSeasonData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // LEAGUES STATE
  const [selectedLeagues, setSelectedLeagues] = useState(new Set(['NCAA D1']));
  const [nbaFilter, setNbaFilter] = useState('All');
  const [nbaLookup, setNbaLookup] = useState(new Map());

  // PER 40 STATE
  const [isPer40, setIsPer40] = useState(false);


  const [selectedYear, setSelectedYear] = useState(DEFAULT_FILTERS.year);
  const [sinceYear, setSinceYear] = useState('');
  const [playerSearch, setPlayerSearch] = useState('');
  const [selectedTeam, setSelectedTeam] = useState(DEFAULT_FILTERS.team);
  const [minGames, setMinGames] = useState(DEFAULT_FILTERS.minGames);
  const [minHeight, setMinHeight] = useState('');
  const [maxHeight, setMaxHeight] = useState('');
  const [expFilter, setExpFilter] = useState('All');
  const [minWeight, setMinWeight] = useState('');
  const [maxWeight, setMaxWeight] = useState('');
  const [minBmi, setMinBmi] = useState('');
  const [maxBmi, setMaxBmi] = useState('');

  // Only used in career mode
  const [minLength, setMinLength] = useState('');
  const [maxLength, setMaxLength] = useState('');

  const [filters, setFilters] = useState({});
  const [visibleCols, setVisibleCols] = useState(DEFAULT_VISIBLE);
  const [availableStats, setAvailableStats] = useState([]);
  const [sortConfig, setSortConfig] = useState<SortConfig>(null);
  const [compareList, setCompareList] = useState([]);
  const [compareSearchQuery, setCompareSearchQuery] = useState('');
  const [compareResults, setCompareResults] = useState([]);
  const [compareInitialized, setCompareInitialized] = useState(false);
  const [customStats, setCustomStats] = useState<CustomStat[]>([]);
  const [showCustomStatBuilder, setShowCustomStatBuilder] = useState(false);
  const [showConsensusDraft, setShowConsensusDraft] = useState(false);
  const [brAdvancedData, setBrAdvancedData] = useState<Map<string, any>>(new Map());

  // Ensure BPM/OBPM/DBPM defaults are always present on initial load
  useEffect(() => {
    setVisibleCols(prev => {
      const next = new Set(prev);
      ['gbpm', 'obpm', 'dbpm'].forEach(k => next.add(k));
      return next;
    });
  }, []);

  // ===== DEVTOOLS LOG =====
  useEffect(() => {
    console.log('%cCheck the Sheets %cAnalytics Mode', 'color: #6366f1; font-size: 20px; font-weight: bold;', 'color: #94a3b8; font-size: 16px;');
  }, []);

  // ===== LOAD CUSTOM STATS FROM LOCALSTORAGE =====
  useEffect(() => {
    const saved = localStorage.getItem('checkthesheets_custom_stats');
    if (saved) {
      try {
        setCustomStats(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load custom stats:', e);
      }
    }
  }, []);

  // ===== SAVE CUSTOM STATS TO LOCALSTORAGE =====
  useEffect(() => {
    if (customStats.length > 0) {
      localStorage.setItem('checkthesheets_custom_stats', JSON.stringify(customStats));
    }
  }, [customStats]);

  // ===== LOAD BR ADVANCED STATS =====
  useEffect(() => {
    fetch(FILES.br_advanced)
      .then(res => res.text())
      .then(csvText => {
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          complete: (results) => {
            const brMap = new Map();
            results.data.forEach((row: any) => {
              if (row.player_name) {
                // Normalize player name for lookup
                const nameNorm = row.player_name.toLowerCase().trim();
                brMap.set(nameNorm, row);
              }
            });
            setBrAdvancedData(brMap);
          }
        });
      })
      .catch(err => console.error('Failed to load BR advanced stats:', err));
  }, []);

  // --- SYNC URL WITH STATE ---
  useEffect(() => {
    window.history.pushState(null, '', `/${datasetMode}`);
  }, [datasetMode]);

  // Handle browser back button
  useEffect(() => {
    const onPopState = () => setDatasetMode(getInitialMode());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  // Load season+archive once for modal history
  useEffect(() => {
    Promise.all([
      fetchCsv(FILES.season),
      fetchCsv(FILES.weights),
      fetchCsv(FILES.archive),
      fetchCsv(FILES.international), // index 3
      fetchCsv(FILES.intl_2026),     // index 4
      fetchCsv('/data/nba_lookup.csv') // index 5
    ]).then(([stats, weights, archive, international, intl26, lookup]) => {
      const processedStats = processRows(stats || [], weights || [], false);
      const processedArchive = processRows(archive || [], weights || [], false);
      const processedInternational = processRows(international || [], weights || [], true);
      const processedIntl26 = processRows(intl26 || [], weights || [], true, 2026);

      // Map for quick NBA lookup
      const lookupMap = new Map();
      if (lookup) {
        lookup.forEach(l => lookupMap.set(String(l.name_lower), l));
      }
      setNbaLookup(lookupMap);

      const allStats = dedupeInternationalRows([
        ...processedStats,
        ...processedArchive,
        ...processedInternational,
        ...processedIntl26
      ]);
      setSeasonData(allStats);
    });
  }, []);

  // Load dataset for main table / compare
  useEffect(() => {
    if (datasetMode === 'about') return; // Don't fetch data for About page

    setLoading(true);
    let filePromises = [];

    // When fetching seasons, we now always fetch international too so we can merge them
    if (datasetMode === 'compare') {
      if (compareType === 'career') {
        filePromises = [fetchCsv(FILES.career)];
      } else {
        // Season Compare: Season + Archive + Intl + Intl2026
        filePromises = [
          fetchCsv(FILES.season),
          fetchCsv(FILES.archive),
          fetchCsv(FILES.international),
          fetchCsv(FILES.intl_2026)
        ];
      }
    } else if (datasetMode === 'career') {
      filePromises = [fetchCsv(FILES.career)];
    } else {
      // Season mode: Season + Intl + Intl2026
      filePromises = [
        fetchCsv(FILES.season),
        fetchCsv(FILES.international),
        fetchCsv(FILES.intl_2026)
      ];
    }

    Promise.all([...filePromises, fetchCsv(FILES.weights)]).then((results) => {
      const weights = results.pop();
      // Combine all loaded stats files (everything except weights)
      const rawCombinedStats = results.flatMap((res, index) => {
        // Identify if the chunk is international.
        let isIntl = false;
        let overrideYear = null;

        if (datasetMode === 'season') {
          // 0=season, 1=intl, 2=intl_26
          if (index === 1 || index === 2) isIntl = true;
          if (index === 2) overrideYear = 2026;
        }
        else if (datasetMode === 'compare' && compareType === 'season') {
          // 0=season, 1=archive, 2=intl, 3=intl_26
          if (index === 2 || index === 3) isIntl = true;
          if (index === 3) overrideYear = 2026;
        }

        return processRows(res, weights || [], isIntl, overrideYear);
      });

      if (rawCombinedStats.length > 0) {
        const keys = Object.keys(rawCombinedStats[0]);
        const allowed = keys.filter(k =>
          (typeof rawCombinedStats[0][k] === 'number' || k === 'pick' || k === 'recruit rank') &&
          !HIDDEN_COLS.has(k) &&
          !['key', 'id', '_nameLower', 'isInternational', 'league', 'team'].includes(k)
        );

        // Ensure stats in STAT_CONFIG are available even if the first row lacks them
        const statKeys = STAT_CONFIG.flatMap(g => g.stats.map(s => s.key));
        const hasValue = (row, key) => {
          const v = row[key];
          return v !== null && v !== undefined && v !== '' && !(typeof v === 'number' && Number.isNaN(v));
        };
        statKeys.forEach(k => {
          if (!allowed.includes(k)) {
            const existsInAny = rawCombinedStats.some(r => hasValue(r, k));
            if (existsInAny) allowed.push(k);
          }
        });

        setAvailableStats(allowed);
      }
      setData(dedupeInternationalRows(rawCombinedStats));
      setLoading(false);
    });
  }, [datasetMode, compareType]);

  // Default Career Sort
  useEffect(() => {
    if (datasetMode === 'career') {
      setSortConfig({ key: 'total RAPM', direction: 'desc' });
    }
  }, [datasetMode]);

  useEffect(() => {
    if (datasetMode === 'compare' && data.length > 0 && !compareInitialized) {
      const p1 = data.find(p => p._nameLower.includes("darius acuff") && String(p.torvik_year) === "2026");
      const p2 = data.find(p => p._nameLower.includes("tyler tanner") && String(p.torvik_year) === "2026");

      const newDefaults = [];
      if (p1) newDefaults.push(p1);
      if (p2) newDefaults.push(p2);

      if (newDefaults.length > 0) {
        setCompareList(newDefaults);
      }
      setCompareInitialized(true);
    }
  }, [datasetMode, data, compareInitialized]);

  useEffect(() => {
    if (datasetMode !== 'compare') {
      setCompareInitialized(false);
    }
  }, [datasetMode]);

  // Compare search
  useEffect(() => {
    if (datasetMode !== 'compare' || !compareSearchQuery || compareSearchQuery.length < 2) {
      setCompareResults([]);
      return;
    }
    const q = compareSearchQuery.toLowerCase().trim();
    // Split by space for tokenized matching
    const tokens = q.split(/\s+/).filter(t => t.length > 0);

    // Search across ALL data regardless of current league filters
    const searchUniverse = data;

    const results = searchUniverse
      .filter(p => {
        if (!p._nameLower) return false;
        return tokens.every(token => p._nameLower.includes(token));
      })
      .sort((a, b) => {
        const nameA = a._nameLower;
        const nameB = b._nameLower;

        // Scoring system for better relevance
        const getScore = (p, name, query) => {
          let score = 0;
          if (name === query) score += 1000;
          if (name.startsWith(query)) score += 500;
          if (new RegExp(`(^|\\s)${query}`).test(name)) score += 300;

          // Bonus for players with NBA status or specific draft picks
          if (p.pick && p.pick !== 'NA') score += 100;
          if (nbaLookup.has(name)) score += 80;

          // Bonus for more recent years
          if (p.torvik_year) score += (Number(p.torvik_year) - 2000);

          return score;
        };

        const scoreA = getScore(a, nameA, q);
        const scoreB = getScore(b, nameB, q);

        if (scoreA !== scoreB) return scoreB - scoreA;
        return nameA.localeCompare(nameB);
      })
      .slice(0, 50);

    setCompareResults(results);
  }, [compareSearchQuery, data, datasetMode, selectedLeagues]);

  const addToCompare = (player) => {
    if (!compareList.find(p => p.id === player.id)) setCompareList([...compareList, player]);
    setCompareSearchQuery('');
  };
  const removeFromCompare = (playerId) => setCompareList(compareList.filter(p => p.id !== playerId));
  const clearCompareList = () => setCompareList([]);

  const availableYears = useMemo(() => {
    if (datasetMode === 'career' || datasetMode === 'compare') return [];
    return Array.from(new Set(data.map(p => p.torvik_year).filter(Boolean))).sort().reverse();
  }, [data, datasetMode]);

  const availableTeams = useMemo(() => {
    if (datasetMode === 'compare') return [];
    // Only show teams from selected leagues
    const universe = data.filter(p => selectedLeagues.has(p.league));
    return Array.from(new Set(universe.map(p => p.team).filter(Boolean))).sort();
  }, [data, datasetMode, selectedLeagues]);

  const availableLeagues = useMemo(() => {
    // Extract unique leagues from the currently loaded data
    const leagues = new Set(data.map(d => d.league).filter(Boolean));
    // Ensure NCAA D1 is always first if present
    const arr = Array.from(leagues).sort();
    if (arr.includes('NCAA D1')) {
      return ['NCAA D1', ...arr.filter(l => l !== 'NCAA D1')];
    }
    return arr;
  }, [data]);

  const groupedStats = useMemo(() => {
    const exists = (key) => availableStats.includes(key);
    return STAT_CONFIG.map(g => ({
      ...g,
      stats: g.stats.filter(s => exists(s.key))
    })).filter(g => g.stats.length > 0);
  }, [availableStats]);

  // --- CUSTOM STAT CALCULATION ---
  const computedData = useMemo(() => {
    if (!data || data.length === 0 || customStats.length === 0) return data;

    return data.map(row => {
      const rowWithCustom = { ...row };
      customStats.forEach(cs => {
        let result = 0;
        cs.formula.forEach((f, idx) => {
          const val = Number(row[f.stat]) || 0;
          const weightedVal = val * f.coefficient;
          if (idx === 0) {
            result = weightedVal;
          } else {
            if (f.operation === '+') result += weightedVal;
            else if (f.operation === '-') result -= weightedVal;
            else if (f.operation === '*') result *= weightedVal;
            else if (f.operation === '/') result /= (weightedVal || 1);
          }
        });
        rowWithCustom[cs.id] = result;
      });
      return rowWithCustom;
    });
  }, [data, customStats]);

  // --- SYNC CUSTOM STATS TO AVAILABLE STATS ---
  const allAvailableStats = useMemo(() => {
    const customIds = customStats.map(s => s.id);
    return [...availableStats, ...customIds];
  }, [availableStats, customStats]);

  // --- SYNC CUSTOM STATS TO VISIBLE COLS ---
  useEffect(() => {
    if (customStats.length > 0) {
      const newVisible = new Set(visibleCols);
      customStats.forEach(s => newVisible.add(s.id));
      setVisibleCols(newVisible);
    }
  }, [customStats.length]);

  // --- DYNAMIC DISTRIBUTION LOGIC ---
  // The universe of players used for calculating mean/std dev depends on selected leagues
  const statsDistribution = useMemo<StatsDistribution>(() => {
    // Universe = players in currently selected leagues
    const validUniverse = computedData.filter(d => selectedLeagues.has(d.league));

    const dist: StatsDistribution = {};
    allAvailableStats.forEach(stat => {
      const values = validUniverse
        .map(d => Number(d[stat]))
        .filter(v => !isNaN(v) && v !== null);
      if (values.length > 1) {
        const mean = _.mean(values);
        const std = Math.sqrt(_.meanBy(values, (x) => Math.pow(x - mean, 2)));
        dist[stat] = { mean, std };
      }
    });
    return dist;
  }, [computedData, allAvailableStats, selectedLeagues]);

  const filteredData = useMemo(() => {
    if (datasetMode === 'compare') return compareList;
    if (datasetMode === 'about') return [];

    // Filter universe first based on league selection
    let filtered = computedData.filter(d => selectedLeagues.has(d.league));

    if (datasetMode === 'season') {
      if (selectedYear !== 'All') filtered = filtered.filter(p => String(p.torvik_year) === String(selectedYear));
      else if (sinceYear) filtered = filtered.filter(p => (p.torvik_year || 0) >= Number(sinceYear));
    }
    if (selectedTeam !== 'All') filtered = filtered.filter(p => p.team === selectedTeam);

    if (playerSearch) {
      const q = playerSearch.toLowerCase().trim();
      const tokens = q.split(/\s+/).filter(t => t.length > 0);
      filtered = filtered.filter(p => {
        if (!p._nameLower) return false;
        // Strict Search: All tokens must be present
        return tokens.every(token => p._nameLower.includes(token));
      });
    }

    if (minGames) filtered = filtered.filter(p => (p.g || 0) >= Number(minGames));
    if (minHeight) filtered = filtered.filter(p => Number(p.height) >= Number(minHeight));
    if (maxHeight) filtered = filtered.filter(p => Number(p.height) <= Number(maxHeight));
    if (minWeight) filtered = filtered.filter(p => Number(p.weight) >= Number(minWeight));
    if (maxWeight) filtered = filtered.filter(p => Number(p.weight) <= Number(maxWeight));
    if (minBmi) filtered = filtered.filter(p => Number(p.bmi) >= Number(minBmi));
    if (maxBmi) filtered = filtered.filter(p => Number(p.bmi) <= Number(maxBmi));
    if (datasetMode === 'career') {
      if (minLength) filtered = filtered.filter(p => (p.length || 0) >= Number(minLength));
      if (maxLength) filtered = filtered.filter(p => (p.length || 0) <= Number(maxLength));
    }

    if (expFilter !== 'All') {
      filtered = filtered.filter(p => {
        const bucket = expBucket(p.exp);
        if (expFilter === 'FrSo') return bucket === 'Fr' || bucket === 'So';
        return bucket === expFilter;
      });
    }

    if (nbaFilter !== 'All') {
      filtered = filtered.filter(p => {
        const hasPick = p.pick && p.pick !== 'NA' && p.pick !== '';
        const inLookup = nbaLookup.has(p._nameLower);
        const isNBA = hasPick || inLookup;

        if (nbaFilter === 'NBA Only') return isNBA;
        if (nbaFilter === 'Non-NBA') return !isNBA;
        if (nbaFilter === 'Drafted') return hasPick;
        if (nbaFilter === 'Undrafted') return inLookup && !hasPick;
        return true;
      });
    }

    Object.entries(filters).forEach(([stat, rule]) => {
      if (!rule.value) return;
      const limit = Number(rule.value);
      filtered = filtered.filter(p => {
        const val = Number(p[stat]);
        if (isNaN(val)) return false;
        switch (rule.operator) {
          case '>': return val > limit;
          case '>=': return val >= limit;
          case '<': return val < limit;
          case '<=': return val <= limit;
          case '=': return Math.abs(val - limit) < 0.001;
          default: return true;
        }
      });
    });

    if (sortConfig) {
      filtered = [...filtered].sort((a, b) => {
        let valA = a[sortConfig.key];
        let valB = b[sortConfig.key];

        // Handle NA, null, undefined - push to end
        const isAEmpty = valA === undefined || valA === null || valA === '' || valA === 'NA' || valA === 'na';
        const isBEmpty = valB === undefined || valB === null || valB === '' || valB === 'NA' || valB === 'na';
        if (isAEmpty && isBEmpty) return 0;
        if (isAEmpty) return 1;
        if (isBEmpty) return -1;

        // Try to parse as numbers for proper numeric sorting
        const numA = typeof valA === 'number' ? valA : parseFloat(String(valA));
        const numB = typeof valB === 'number' ? valB : parseFloat(String(valB));

        // If both are valid numbers, compare numerically
        if (!isNaN(numA) && !isNaN(numB)) {
          if (numA < numB) return sortConfig.direction === 'asc' ? -1 : 1;
          if (numA > numB) return sortConfig.direction === 'asc' ? 1 : -1;
          return 0;
        }

        // Fallback to string comparison
        if (valA < valB) return sortConfig.direction === 'asc' ? -1 : 1;
        if (valA > valB) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
      });
    }

    // Merge Basketball Reference advanced stats if available
    const mergedWithBR = filtered.map(p => {
      if (!brAdvancedData || brAdvancedData.size === 0) return p;
      const brStats = brAdvancedData.get(p._nameLower);
      if (!brStats) return p;
      return {
        ...p,
        nba_WS: brStats.career_ws,
        nba_WS48: brStats.career_ws48,
        nba_VORP: brStats.career_vorp,
        nba_BPM: brStats.career_bpm,
        nba_OBPM: brStats.career_obpm,
        nba_DBPM: brStats.career_dbpm,
        nba_PER: brStats.career_per,
        nba_TS: brStats.career_ts_pct,
        nba_USG: brStats.career_usg_pct,
        nba_AST_pct: brStats.career_ast_pct,
        nba_STL_pct: brStats.career_stl_pct,
        nba_BLK_pct: brStats.career_blk_pct,
      };
    });

    return mergedWithBR;
  }, [
    data, selectedYear, sinceYear, playerSearch, selectedTeam,
    minGames, minHeight, maxHeight, minWeight, maxWeight,
    minBmi, maxBmi, minLength, maxLength, nbaFilter, nbaLookup,
    expFilter, filters, sortConfig, datasetMode, compareList, selectedLeagues, brAdvancedData
  ]);

  const activeFilters = useMemo(() => {
    if (datasetMode === 'compare' || datasetMode === 'about') return [];
    const list = [];
    if (datasetMode === 'season') {
      if (selectedYear !== 'All') list.push({ label: 'Year', val: selectedYear });
      if (sinceYear) list.push({ label: 'Since', val: sinceYear });
    }
    if (selectedTeam !== 'All') list.push({ label: 'Team', val: selectedTeam });
    if (playerSearch) list.push({ label: 'Search', val: playerSearch });
    if (minGames !== DEFAULT_FILTERS.minGames) list.push({ label: 'Min G', val: minGames });
    if (minHeight) list.push({ label: 'Min Hgt', val: minHeight });
    if (expFilter !== 'All') list.push({ label: 'Exp', val: expFilter === 'FrSo' ? 'Fr+So' : expFilter });
    if (minWeight) list.push({ label: 'Min Wt', val: minWeight });
    if (maxWeight) list.push({ label: 'Max Wt', val: maxWeight });
    if (minBmi) list.push({ label: 'Min BMI', val: minBmi });
    if (maxBmi) list.push({ label: 'Max BMI', val: maxBmi });
    if (nbaFilter !== 'All') list.push({ label: 'NBA', val: nbaFilter });

    if (datasetMode === 'career') {
      if (minLength) list.push({ label: 'Min Len', val: minLength });
      if (maxLength) list.push({ label: 'Max Len', val: maxLength });
    }

    // Show active leagues in filter bar if not just default NCAA
    if (selectedLeagues.size !== 1 || !selectedLeagues.has('NCAA D1')) {
      const arr = Array.from(selectedLeagues);
      const display = arr.length > 3 ? `${arr.length} Leagues` : arr.join(', ');
      list.push({ label: 'Leagues', val: display });
    }

    if (isPer40) {
      list.push({ label: 'Mode', val: 'Per 40 Min' });
    }

    Object.entries(filters).forEach(([stat, rule]) => {
      if (rule.value) list.push({ label: stat, val: `${rule.operator} ${rule.value}` });
    });
    return list;
  }, [datasetMode, selectedYear, sinceYear, selectedTeam, playerSearch, minGames, minHeight, expFilter, filters, selectedLeagues, isPer40]);

  const toggleGroup = (groupName) => {
    const group = groupedStats.find(g => g.group === groupName);
    if (!group) return;
    const groupKeys = group.stats.map(s => s.key);
    const allVisible = groupKeys.every(k => visibleCols.has(k));
    const newSet = new Set(visibleCols);
    if (allVisible) groupKeys.forEach(k => newSet.delete(k));
    else groupKeys.forEach(k => newSet.add(k));
    setVisibleCols(newSet);
  };

  const handleSort = (key) => {
    let direction = 'desc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'desc') direction = 'asc';
    setSortConfig({ key, direction });
  };

  const toggleColumn = (stat) => {
    const newSet = new Set(visibleCols);
    if (newSet.has(stat)) newSet.delete(stat);
    else newSet.add(stat);
    setVisibleCols(newSet);
  };

  const handleShowAll = () => {
    const allStats = new Set(groupedStats.flatMap(g => g.stats.map(s => s.key)));
    setVisibleCols(allStats);
  };

  const handleHideAll = () => {
    setVisibleCols(new Set());
  };

  const handleBartStats = () => {
    const bartStats = new Set<string>();
    groupedStats.forEach(g => {
      g.stats.forEach(s => {
        if (s.source === 'B') bartStats.add(s.key);
      });
    });
    setVisibleCols(bartStats);
  };

  const updateFilter = (stat, field, val) => {
    setFilters(prev => {
      const existing = prev[stat] || { operator: '>=', value: '' };
      const updated = { ...existing, [field]: val };
      if (updated.value === '' && field === 'value') {
        const next = { ...prev };
        delete next[stat];
        return next;
      }
      return { ...prev, [stat]: updated };
    });
  };

  const clearAllFilters = () => {
    setFilters({});
    setPlayerSearch('');
    setSelectedTeam('All');
    setSelectedYear('2026');
    setSinceYear('');
    setMinHeight('');
    setMaxHeight('');
    setMinWeight('');
    setMaxWeight('');
    setMinBmi('');
    setMaxBmi('');
    setMinLength('');
    setMaxLength('');
    setMinGames(DEFAULT_FILTERS.minGames);
    setExpFilter('All');
    setSelectedLeagues(new Set(['NCAA D1']));
    setIsPer40(false);
  };

  const handleSaveCustomStat = (customStat: CustomStat) => {
    setCustomStats([...customStats, customStat]);
  };

  const handleDeleteCustomStat = (id: string) => {
    setCustomStats(customStats.filter(s => s.id !== id));
  };


  const getColLabel = (key, customLabel) => {
    let label = customLabel || key;
    if (key === 'total RAPM' && (datasetMode === 'career' || (datasetMode === 'compare' && compareType === 'career'))) {
      label = 'average RAPM';
    }

    // Append /40 if applicable
    if (isPer40) {
      if (PER40_FROM_TOTALS.has(key) || PER40_FROM_PER_GAME.has(key)) {
        label += '/40';
      }
    }
    return label;
  };

  const tableHeaders = useMemo<TableGroup[]>(() => {
    const headers: TableGroup[] = [];

    const permanentCols = datasetMode === 'career'
      ? CAREER_PERMANENT_COLS
      : PERMANENT_COLS;

    headers.push({
      group: '',
      cols: permanentCols.map(c => ({
        key: c.key,
        label: c.label,
        tooltip: c.tooltip
      }))
    });

    groupedStats.forEach(g => {
      const visibleStats = g.stats.filter(s => visibleCols.has(s.key));
      if (visibleStats.length > 0) {
        const cols: TableCol[] = visibleStats.map((s, idx) => ({
          key: s.key,
          label: getColLabel(s.key, s.label),
          tooltip: s.tooltip,
          isGroupEnd: idx === visibleStats.length - 1
        }));
        headers.push({ group: g.group, cols });
      }
    });

    // Add Custom Stats Group
    if (customStats.length > 0) {
      const customCols = customStats.map((s, idx) => ({
        key: s.id,
        label: s.name,
        tooltip: `Custom formula: ${s.formula.map(f => `${f.operation}${f.coefficient}*${f.stat}`).join(' ')}`,
        isGroupEnd: idx === customStats.length - 1
      }));
      headers.push({ group: 'Custom Stats', cols: customCols });
    }

    return headers;
  }, [groupedStats, visibleCols, datasetMode, compareType, isPer40, customStats]);


  const flattenedCols: TableCol[] = tableHeaders.flatMap(g => g.cols);
  const modalHistoryData = seasonData.length > 0 ? seasonData : data;

  return (
    <div className={`app-container ${sidebarCollapsed ? 'sidebar-collapsed' : ''} ${datasetMode === 'career' ? 'career-mode' : ''}`}>

      {/* KPRverse-style vignette glow overlay */}
      <div className="vignette-glow" />

      {/* --- FIXED TWITTER LINK --- */}
      <a
        href="https://x.com/checkthesheets"
        target="_blank"
        rel="noopener noreferrer"
        className="twitter-float-btn"
      >
        <Twitter size={14} fill="#fff" />
      </a>

      <div className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebar-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <img
              src={nerdBall}
              className="nerd-logo"
              alt="Nerd Ball Logo"
            />
            <h2>Check the Sheets</h2>
          </div>
          <div className="dataset-toggle">
            <button className={datasetMode === 'season' ? 'active' : ''} onClick={() => setDatasetMode('season')}>Season</button>
            <button className={datasetMode === 'career' ? 'active' : ''} onClick={() => setDatasetMode('career')}>Career</button>
            <button className={datasetMode === 'compare' ? 'active' : ''} onClick={() => setDatasetMode('compare')}>Compare</button>
            <button className={datasetMode === 'about' ? 'active' : ''} onClick={() => setDatasetMode('about')}>About CTS</button>
          </div>
          <div className="dataset-toggle" style={{ marginTop: '8px', gap: '6px' }}>
            <button onClick={() => setShowCustomStatBuilder(true)} style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem' }}>
              <Calculator size={14} /> Custom Stats
            </button>
            <button onClick={() => setShowConsensusDraft(true)} style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem' }}>
              🏀 Rankings
            </button>
          </div>
        </div>
        <div className="scroll-sidebar">
          {datasetMode === 'compare' ? (
            <>
              <div className="filter-group">
                <label>Compare Type</label>
                <div className="dataset-toggle">
                  <button
                    className={compareType === 'season' ? 'active' : ''}
                    onClick={() => { setCompareType('season'); setCompareList([]); setCompareInitialized(false); }}
                  >
                    Season
                  </button>
                  <button
                    className={compareType === 'career' ? 'active' : ''}
                    onClick={() => { setCompareType('career'); setCompareList([]); setCompareInitialized(false); }}
                  >
                    Career
                  </button>
                </div>
              </div>

              {/* SPECIAL COMPARISON NOTICES */}
              {compareType === 'career' && (
                <div style={{ padding: '0.75rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '8px', marginBottom: '1rem', border: '1px solid var(--border-accent)', fontSize: '0.75rem', color: 'var(--accent-primary)' }}>
                  Career data coverage starts from 2019.
                </div>
              )}
              {compareType === 'season' && (
                <div style={{ padding: '0.75rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', marginBottom: '1rem', border: '1px solid rgba(16, 185, 129, 0.3)', fontSize: '0.75rem', color: 'var(--accent-tertiary)' }}>
                  Search any season from the data pool.
                </div>
              )}

              {/* LEAGUE SELECTOR FOR COMPARE MODE */}
              {compareType === 'season' && availableLeagues.length > 0 && (
                <div className="filter-group">
                  <label>Leagues</label>
                  <LeagueMultiSelect
                    options={availableLeagues}
                    selected={selectedLeagues}
                    onChange={setSelectedLeagues}
                  />
                </div>
              )}

              <div className="filter-group">
                <label>Add Player</label>
                <div className="compare-search-container">
                  <div className="input-icon-wrap">
                    <Search size={14} />
                    <input
                      type="text"
                      placeholder={compareType === 'season' ? "Search Name..." : "Search Career..."}
                      value={compareSearchQuery}
                      onChange={e => setCompareSearchQuery(e.target.value)}
                    />
                  </div>
                  {compareResults.length > 0 && (
                    <div className="search-results-dropdown">
                      {compareResults.map(p => (
                        <div key={p.id} className="search-result-item" onClick={() => addToCompare(p)}>
                          <div>
                            <div style={{ color: p.isInternational ? 'red' : 'inherit' }}>{p.key}</div>
                            <div className="search-result-meta">
                              {compareType === 'season' ? `${p.torvik_year} - ${p.team}` : `${p.team}`}
                            </div>
                          </div>
                          <span style={{ fontSize: '1.2em', color: '#2563eb' }}>+</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="filter-group">
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <label>Selected ({compareList.length})</label>
                  {compareList.length > 0 && (
                    <span
                      style={{ fontSize: '0.75rem', color: '#ef4444', cursor: 'pointer' }}
                      onClick={clearCompareList}
                    >
                      Clear
                    </span>
                  )}
                </div>
                <div className="compare-list">
                  {compareList.map(p => (
                    <div key={p.id} className="compare-item">
                      <div className="compare-item-info">
                        <span style={{ fontWeight: 600, color: p.isInternational ? 'red' : 'inherit' }}>{p.key}</span>
                        <span className="compare-item-sub">
                          {compareType === 'season' ? `${p.torvik_year} • ${p.team}` : p.team}
                        </span>
                      </div>
                      <button className="btn-remove" onClick={() => removeFromCompare(p.id)}><X size={14} /></button>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : datasetMode === 'about' ? (
            <div style={{ padding: '20px', color: '#666', fontSize: '0.9rem' }}>
              Select other tabs to view stats.
            </div>
          ) : (
            <>
              {datasetMode === 'season' && (
                <div className="filter-group row-group">
                  <div>
                    <label>Year</label>
                    <select value={selectedYear} onChange={e => setSelectedYear(e.target.value)} className="year-select">
                      {availableYears.map(y => <option key={y} value={y}>{y}</option>)}
                      <option value="All">All</option>
                    </select>
                  </div>
                  <div>
                    <label>Since</label>
                    <select
                      value={sinceYear}
                      onChange={e => setSinceYear(e.target.value)}
                      disabled={selectedYear !== 'All'}
                      style={{ opacity: selectedYear !== 'All' ? 0.5 : 1 }}
                    >
                      <option value="">-</option>
                      {availableYears.map(y => <option key={y} value={y}>{y}</option>)}
                    </select>
                  </div>
                </div>
              )}

              <div className="filter-group">
                <label>Player Search</label>
                <div className="input-icon-wrap">
                  <Search size={14} />
                  <input
                    type="text"
                    placeholder="Type name..."
                    value={playerSearch}
                    onChange={e => setPlayerSearch(e.target.value)}
                  />
                </div>
              </div>

              <div className="filter-group">
                <label>Team</label>
                <select value={selectedTeam} onChange={e => setSelectedTeam(e.target.value)}>
                  <option value="All">All Teams</option>
                  {availableTeams.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>

              {/* LEAGUES DROPDOWN - Replaces "Include non-NCAA" */}
              {datasetMode === 'season' && availableLeagues.length > 0 && (
                <div className="filter-group">
                  <label>Leagues</label>
                  <LeagueMultiSelect
                    options={availableLeagues}
                    selected={selectedLeagues}
                    onChange={setSelectedLeagues}
                  />
                </div>
              )}

              {/* PER 40 TOGGLE */}
              {datasetMode !== 'compare' && (
                <div className="filter-group" style={{ marginTop: '10px' }}>
                  <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', fontSize: '0.85rem' }}>
                    <input
                      type="checkbox"
                      checked={isPer40}
                      onChange={e => setIsPer40(e.target.checked)}
                      style={{ width: 'auto', marginRight: '8px' }}
                    />
                    Per 40 Minute Stats
                  </label>
                </div>
              )}

              <div className="filter-group row-group">
                <div>
                  <label>Min G</label>
                  <input type="number" value={minGames} onChange={e => setMinGames(e.target.value)} />
                </div>
                <div>
                  <label>Hgt (in)</label>
                  <div className="mini-row">
                    <input
                      placeholder="Min"
                      type="number"
                      value={minHeight}
                      onChange={e => setMinHeight(e.target.value)}
                    />
                    <input
                      placeholder="Max"
                      type="number"
                      value={maxHeight}
                      onChange={e => setMaxHeight(e.target.value)}
                    />
                  </div>
                </div>
              </div>
              <div className="filter-group row-group">
                <div>
                  <label>Wt (lbs)</label>
                  <div className="mini-row">
                    <input
                      placeholder="Min"
                      type="number"
                      value={minWeight}
                      onChange={e => setMinWeight(e.target.value)}
                    />
                    <input
                      placeholder="Max"
                      type="number"
                      value={maxWeight}
                      onChange={e => setMaxWeight(e.target.value)}
                    />
                  </div>
                </div>
                <div>
                  <label>BMI</label>
                  <div className="mini-row">
                    <input
                      placeholder="Min"
                      type="number"
                      value={minBmi}
                      onChange={e => setMinBmi(e.target.value)}
                    />
                    <input
                      placeholder="Max"
                      type="number"
                      value={maxBmi}
                      onChange={e => setMaxBmi(e.target.value)}
                    />
                  </div>
                </div>
              </div>

              <div className="filter-group">
                <label>Experience</label>
                <select value={expFilter} onChange={e => setExpFilter(e.target.value)}>
                  <option value="All">All</option>
                  <option value="Fr">Freshman</option>
                  <option value="So">Sophomore</option>
                  <option value="Jr">Junior</option>
                  <option value="Sr">Senior</option>
                  <option value="FrSo">Freshman + Sophomore</option>
                </select>
              </div>

              <div className="filter-group">
                <label>NBA Status</label>
                <select value={nbaFilter} onChange={e => setNbaFilter(e.target.value)}>
                  <option value="All">All Players</option>
                  <option value="NBA Only">NBA Players (All)</option>
                  <option value="Drafted">Drafted Only</option>
                  <option value="Undrafted">Undrafted NBA Only</option>
                  <option value="Non-NBA">Non-NBA Only</option>
                </select>
              </div>
              {datasetMode === 'career' && (
                <div className="filter-group row-group">
                  <div>
                    <label>Career Length (yrs)</label>
                    <div className="mini-row">
                      <input
                        placeholder="Min"
                        type="number"
                        value={minLength}
                        onChange={e => setMinLength(e.target.value)}
                      />
                      <input
                        placeholder="Max"
                        type="number"
                        value={maxLength}
                        onChange={e => setMaxLength(e.target.value)}
                      />
                    </div>
                  </div>
                </div>
              )}


              <button className="btn-clear" onClick={clearAllFilters}>Reset All</button>
            </>
          )}

          {datasetMode !== 'about' && (
            <>
              <hr className="divider" />

              <div className="filter-group">
                <label className="section-title">Stat Options</label>

                {/* PRESETS ROW */}
                <div style={{ display: 'flex', gap: '8px', marginBottom: '10px' }}>
                  <button className="btn-secondary" onClick={handleShowAll} style={{ flex: 1 }}>Show All</button>
                  <button className="btn-secondary" onClick={handleHideAll} style={{ flex: 1 }}>Hide All</button>
                </div>
                <button className="btn-secondary" onClick={handleBartStats} style={{ width: '100%', marginBottom: '15px' }}>
                  Show Bart Stats Only
                </button>

                <div className="stat-legend">
                  <div className="legend-item"><span className="source-badge source-hoop">H</span> = Hoop Explorer</div>
                  <div className="legend-item"><span className="source-badge source-bart">B</span> = Bart Torvik</div>
                </div>

                {groupedStats.map(group => (
                  <div key={group.group}>
                    <div className="group-header-row">
                      <span className="group-header-title">{group.group}</span>
                      <button className="group-toggle-btn" onClick={() => toggleGroup(group.group)}>
                        {group.stats.every(s => visibleCols.has(s.key)) ? 'Hide All' : 'Show All'}
                      </button>
                    </div>
                    <div className="bart-list">
                      {group.stats.map(statObj => {
                        const stat = statObj.key;
                        const displayName = statObj.label || stat;
                        const activeFilter = datasetMode !== 'compare' ? filters[stat] : null;
                        const isVisible = visibleCols.has(stat);

                        return (
                          <div
                            key={stat}
                            className={`bart-row ${activeFilter ? 'active' : ''}`}
                            data-tooltip={METRIC_DEFINITIONS[stat]}
                          >
                            <div className="row-left">
                              <div
                                className="col-toggle"
                                onClick={() => toggleColumn(stat)}
                                style={{ color: isVisible ? '#2563eb' : '#9ca3af' }}
                              >
                                {isVisible ? <Eye size={16} /> : <EyeOff size={16} />}
                              </div>
                              <span className="stat-name">
                                {statObj.key === 'total RAPM' &&
                                  (datasetMode === 'career' || (datasetMode === 'compare' && compareType === 'career'))
                                  ? 'average RAPM'
                                  : displayName}
                                {statObj.source === 'B' && <span className="source-badge source-bart">B</span>}
                                {statObj.source === 'H' && <span className="source-badge source-hoop">H</span>}
                              </span>
                            </div>
                            {datasetMode !== 'compare' && (
                              <div className="bart-controls">
                                <select
                                  value={activeFilter?.operator || '>='}
                                  onChange={e => updateFilter(stat, 'operator', e.target.value)}
                                >
                                  <option value=">=">{'>='}</option>
                                  <option value=">">{'>'}</option>
                                  <option value="<=">{'<='}</option>
                                  <option value="<">{'<'}</option>
                                  <option value="=">{'='}</option>
                                </select>
                                <input
                                  type="number"
                                  placeholder="-"
                                  value={activeFilter?.value || ''}
                                  onChange={e => updateFilter(stat, 'value', e.target.value)}
                                />
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="main-content">
        <div className="top-bar" style={{ paddingBottom: '10px' }}>
          <div className="top-bar-left">
            <button
              className="sidebar-toggle"
              onClick={() => setSidebarCollapsed(prev => !prev)}
            >
              {sidebarCollapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
              <span>{sidebarCollapsed ? 'Show Filters' : 'Hide Filters'}</span>
            </button>

            <div>
              <h1 style={{ marginBottom: '4px' }}>
                {datasetMode === 'season' && `Season Data ${selectedYear !== 'All' ? `(${selectedYear})` : '(All Years)'}`}
                {datasetMode === 'career' && '2019-Present Career Data'}
                {datasetMode === 'compare' && `Comparison (${compareType === 'season' ? 'Season' : 'Career'})`}
                {datasetMode === 'about' && 'About Check The Sheets'}
              </h1>

              {datasetMode !== 'about' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: '#666', marginTop: '-2px' }}>
                  <img src={nerdBall} style={{ width: '20px', height: '20px', borderRadius: '50%' }} alt="" />
                  Generated by <strong style={{ color: '#2563eb' }}>checkthesheets.com</strong>
                </div>
              )}
            </div>
          </div>

          {datasetMode !== 'about' && (
            <span className="count-badge">{filteredData.length} Results</span>
          )}
        </div>


        {datasetMode !== 'about' && activeFilters.length > 0 && (
          <div className="active-filters-bar">
            <span style={{ fontWeight: 700, marginRight: '8px' }}>Active Filters:</span>
            {activeFilters.map((f, i) => (
              <div key={i} className="active-filter-tag">
                <span className="filter-label">{f.label}:</span>
                <span>{f.val}</span>
              </div>
            ))}
          </div>
        )}

        {/* --- ABOUT PAGE CONTENT --- */}
        {datasetMode === 'about' ? (
          <div style={{ padding: '30px', maxWidth: '800px', margin: '0 auto', lineHeight: '1.6', color: '#333' }}>
            <div style={{ background: 'white', padding: '30px', borderRadius: '12px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' }}>
              <h2 style={{ fontSize: '1.8rem', marginTop: 0, color: '#111' }}>Welcome to Check The Sheets</h2>

              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '20px', paddingBottom: '20px', borderBottom: '1px solid #eee' }}>
                <img src={nerdBall} alt="Nerd Ball" style={{ width: '60px', height: '60px', borderRadius: '50%' }} />
                <div>
                  <div style={{ fontWeight: 'bold', fontSize: '1.1rem' }}>Akash</div>
                  <a href="https://x.com/checkthesheets" style={{ color: '#2563eb', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Twitter size={14} /> @checkthesheets
                  </a>
                </div>
              </div>

              <h3>Data Sources</h3>
              <p>
                The data provided on this platform is aggregated from multiple trusted basketball analytics sources:
              </p>
              <ul style={{ paddingLeft: '20px' }}>
                <li><strong>Hoop-Explorer:</strong> Primary source for RAPM (Regularized Adjusted Plus-Minus) and other advanced impact metrics.</li>
                <li><strong>BartTorvik:</strong> Comprehensive box score stats, efficiency metrics, and player comparisons.</li>
                <li><strong>RealGM:</strong> International prospect statistics.</li>
                <li><strong>ESPN:</strong> BMI/Weight.</li>
              </ul>

              <div style={{ background: '#eff6ff', padding: '15px', borderRadius: '8px', borderLeft: '4px solid #2563eb', margin: '20px 0' }}>
                <strong>Note on Data Coverage:</strong> Season and Career data on this platform generally covers the <strong>2019 season onwards</strong>. While historical data exists on source sites, this dashboard focuses on the modern analytics era.
              </div>

              <h3>How to Use</h3>
              <p>
                <strong>Season Mode:</strong> Filter individual player seasons by year, team, or physical attributes. Use the "Stat Options" in the sidebar to toggle advanced metrics like RAPM, dunk rates, and rim finishing percentages.
              </p>
              <p>
                <strong>Career Mode:</strong> Aggregated stats for players' careers (post-2019). Useful for identifying long-term production and consistency.
              </p>
              <p>
                <strong>Comparison:</strong> Select multiple players to view head-to-head charts, scatter plots, and trajectory graphs. This tool is perfect for scouting and debating player value.
              </p>

              <p style={{ marginTop: '30px', borderTop: '1px solid #eee', paddingTop: '20px', fontSize: '0.9rem', color: '#666' }}>
                Built by <strong>checkthesheets.com</strong>. <br />
                Follow on Twitter for updates.
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="mega-intro">
              <h2>Season Data (2026)</h2>
              <div>
                This table lists all player seasons in the current dataset. Click column headers to sort, use the
                filters on the left to refine, and search by player name to jump to specific rows.
              </div>
              <div className="updated">Last updated: 3rd of February 2026</div>
            </div>

            {/* COMPARISON VIZ AT THE TOP */}
            {datasetMode === 'compare' && compareList.length > 0 && (
              <ErrorBoundary onClose={() => { }}>
                <ComparisonSection
                  players={compareList}
                  historyData={modalHistoryData}
                  statsDistribution={statsDistribution}
                />
              </ErrorBoundary>
            )}

            <div className="table-container">
              {loading ? (
                <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                  Loading Player Data...
                </div>
              ) : (
                <TableVirtuoso
                  style={{ height: '100%' }}
                  data={filteredData}
                  components={{
                    Table: (props) => <table {...props} style={{ borderCollapse: 'separate', tableLayout: 'fixed', width: 'max-content', minWidth: '100%' }} />,
                  }}
                  fixedHeaderContent={() => (
                    <>
                      <tr className="table-section-header">
                        {tableHeaders.map((group, i) => (
                          <th key={i} colSpan={group.cols.length}>
                            {group.group || <span dangerouslySetInnerHTML={{ __html: '&nbsp;' }} />}
                          </th>
                        ))}
                      </tr>
                      <tr>
                        {flattenedCols.map(col => (
                          <th
                            key={col.key}
                            onClick={() => handleSort(col.key)}
                            data-tooltip={col.tooltip || METRIC_DEFINITIONS[col.key] || METRIC_DEFINITIONS[col.label] || "Click to sort"}
                            title={col.label}
                            className={col.key === 'key' ? 'col-name' : col.key === 'team' ? 'col-team' : 'col-num'}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                              {col.label}
                              {sortConfig?.key === col.key && (
                                sortConfig.direction === 'asc'
                                  ? <ChevronUp size={12} />
                                  : <ChevronDown size={12} />
                              )}
                            </div>
                          </th>
                        ))}
                      </tr>
                    </>
                  )}
                  itemContent={(index, row) => (
                    <>
                      {flattenedCols.map(col => {
                        const val = row[col.key];
                        const isNum = typeof val === 'number';
                        const isText = TEXT_COLS.has(col.key);
                        let style = {};
                        if (isNum && !NO_COLOR_STATS.has(col.key)) {
                          style = { backgroundColor: getColor(val, statsDistribution[col.key], col.key) };
                        }

                        if (col.key === 'key' && row.isInternational) {
                          style = { ...style, color: 'var(--accent-secondary)', fontWeight: 'bold' };
                        }

                        let displayVal = val;
                        if (col.key === 'per' && !row.isInternational) {
                          displayVal = null;
                        }
                        if (isPer40 && isNum && val !== null) {
                          const mpg = row['mpg'] || 0;
                          const games = row['g'] || 0;
                          if (mpg > 0 && games > 0) {
                            if (PER40_FROM_TOTALS.has(col.key)) {
                              displayVal = (val * 40) / (games * mpg);
                            } else if (PER40_FROM_PER_GAME.has(col.key)) {
                              displayVal = (val * 40) / mpg;
                            }
                          }
                        }

                        const formattedVal = isNum ? formatValue(displayVal, col.key) : displayVal;

                        return (
                          <td
                            key={col.key}
                            style={style}
                            onClick={() => setSelectedPlayer(row)}
                            className={`${col.key === 'key' ? 'col-name' : col.key === 'team' ? 'col-team' : 'col-num'} ${!isText ? 'num' : ''}`}
                          >
                            {formattedVal}
                          </td>
                        );
                      })}
                    </>
                  )}
                />
              )}
            </div>
          </>
        )}
      </div>

      {
        selectedPlayer && (
          <ErrorBoundary onClose={() => setSelectedPlayer(null)}>
            <PlayerDetailModal
              player={selectedPlayer}
              historyData={modalHistoryData}
              statConfig={groupedStats}
              visibleCols={visibleCols}
              brStats={brAdvancedData}
              onClose={() => setSelectedPlayer(null)}
            />
          </ErrorBoundary>
        )
      }

      {/* Custom Stat Builder Modal */}
      {
        showCustomStatBuilder && (
          <CustomStatBuilder
            availableStats={availableStats}
            onSave={handleSaveCustomStat}
            onClose={() => setShowCustomStatBuilder(false)}
            existingStats={customStats}
          />
        )
      }

      {/* Consensus Draft Modal */}
      {
        showConsensusDraft && (
          <ConsensusDraft
            players={filteredData}
            onClose={() => setShowConsensusDraft(false)}
          />
        )
      }
    </div >
  );
}

export default App;
