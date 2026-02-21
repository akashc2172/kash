# Latent Space Model Analysis Report

**Generated**: 2026-02-18 22:47
**Model**: latent_model_20260218_224659

## Training Results

- **Best Epoch**: 13
- **Best Val Loss**: 7.0877
- **Test RMSE**: 1.59367036819458
- **Test Correlation**: -0.06656950858971084

## Discovered Archetypes

### High Ft Att Type (Archetype 0)

- **Players**: 71
- **Avg RAPM**: 0.31
- **Survival Rate**: 70.4%
- **Description**: Characterized by high Ft Att
- **Examples**: Erik McCree, Mangok Mathiang, Sindarius Thornwell

### Low Shots Total Type (Archetype 1)

- **Players**: 18
- **Avg RAPM**: 0.37
- **Survival Rate**: 61.1%
- **Description**: Characterized by low Shots Total
- **Examples**: Maurice Harkless, Andre Drummond, Vander Blue

### Low Rim Fg Pct Type (Archetype 2)

- **Players**: 12
- **Avg RAPM**: 1.05
- **Survival Rate**: 75.0%
- **Description**: Characterized by low Rim Fg Pct
- **Examples**: Rondae Hollis-Jefferson, Dakari Johnson, Noah Vonleh

### High Three Share Type (Archetype 3)

- **Players**: 85
- **Avg RAPM**: 0.24
- **Survival Rate**: 65.9%
- **Description**: Characterized by high Three Share
- **Examples**: Ike Anigbogu, Rodney Purvis, Justin Patton

### Low Team Srs Type (Archetype 4)

- **Players**: 12
- **Avg RAPM**: -0.94
- **Survival Rate**: 58.3%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Festus Ezeli, Dion Waiters

### High Ft Att Type (Archetype 5)

- **Players**: 42
- **Avg RAPM**: 1.15
- **Survival Rate**: 73.8%
- **Description**: Characterized by high Ft Att
- **Examples**: Zach Collins, Thomas Bryant, Jonathan Isaac

### High Blk Total Per100Poss Type (Archetype 6)

- **Players**: 6
- **Avg RAPM**: 1.56
- **Survival Rate**: 66.7%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Michael Kidd-Gilchrist, Jeff Withey

### High Poss Proxy Type (Archetype 7)

- **Players**: 89
- **Avg RAPM**: -0.06
- **Survival Rate**: 74.2%
- **Description**: Characterized by high Poss Proxy
- **Examples**: PJ Dozier, Nigel Williams-Goss, Caleb Swanigan

### High Breakout Rank Volume Type (Archetype 8)

- **Players**: 90
- **Avg RAPM**: 0.14
- **Survival Rate**: 75.6%
- **Description**: Characterized by high Breakout Rank Volume
- **Examples**: Josh Hart, Edmond Sumner, Donovan Mitchell

### Low Ft Att Type (Archetype 9)

- **Players**: 25
- **Avg RAPM**: -0.26
- **Survival Rate**: 68.0%
- **Description**: Characterized by low Ft Att
- **Examples**: London Perrantes, Harry Giles III, Davon Reed

### Low Games Played Type (Archetype 10)

- **Players**: 11
- **Avg RAPM**: -0.84
- **Survival Rate**: 90.9%
- **Description**: Characterized by low Games Played
- **Examples**: Solomon Hill, Peyton Siva, Darius Johnson-Odom

### High Trueshootingpct Team Resid Type (Archetype 11)

- **Players**: 22
- **Avg RAPM**: 1.14
- **Survival Rate**: 63.6%
- **Description**: Characterized by high Trueshootingpct Team Resid
- **Examples**: Amile Jefferson, Bam Adebayo, Stanley Johnson

## Usage

```python
from models import ProspectModel, ArchetypeAnalyzer

# Load model
model = ProspectModel(latent_dim=32, n_archetypes=8)
model.load_state_dict(torch.load('model.pt'))

# Analyze a prospect
analysis = analyzer.analyze_player(tier1, tier2, career, tier2_mask)
print(analysis.narrative)
```
