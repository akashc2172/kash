# Latent Space Model Analysis Report

**Generated**: 2026-02-18 23:00
**Model**: latent_model_20260218_230008

## Training Results

- **Best Epoch**: 32
- **Best Val Loss**: 8.5406
- **Test RMSE**: 1.7646725177764893
- **Test Correlation**: -0.060968150265631994

## Discovered Archetypes

### High Poss Total Type (Archetype 0)

- **Players**: 52
- **Avg RAPM**: 0.16
- **Survival Rate**: 65.4%
- **Description**: Characterized by high Poss Total
- **Examples**: Erik McCree, Tyler Cavanaugh, Dennis Smith Jr.

### High Recruiting Rating Type (Archetype 1)

- **Players**: 106
- **Avg RAPM**: 0.10
- **Survival Rate**: 70.8%
- **Description**: Characterized by high Recruiting Rating
- **Examples**: Josh Hart, Edmond Sumner, London Perrantes

### Low Recruiting Rating Type (Archetype 2)

- **Players**: 42
- **Avg RAPM**: 0.96
- **Survival Rate**: 73.8%
- **Description**: Characterized by low Recruiting Rating
- **Examples**: Mangok Mathiang, Chris Boucher, Jonathan Isaac

### Low Team Srs Type (Archetype 3)

- **Players**: 13
- **Avg RAPM**: -1.06
- **Survival Rate**: 61.5%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Festus Ezeli, DeQuan Jones

### High Blk Total Per100Poss Type (Archetype 4)

- **Players**: 13
- **Avg RAPM**: 1.67
- **Survival Rate**: 61.5%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Michael Kidd-Gilchrist, Andre Drummond

### Low Team Pace Type (Archetype 5)

- **Players**: 14
- **Avg RAPM**: -0.71
- **Survival Rate**: 78.6%
- **Description**: Characterized by low Team Pace
- **Examples**: Ike Anigbogu, Harry Giles III, Cheick Diallo

### Low Fga Total Type (Archetype 6)

- **Players**: 25
- **Avg RAPM**: 1.72
- **Survival Rate**: 72.0%
- **Description**: Characterized by low Fga Total
- **Examples**: Amile Jefferson, Bam Adebayo, Skal Labissiere

### High Three Share Type (Archetype 7)

- **Players**: 69
- **Avg RAPM**: -0.01
- **Survival Rate**: 75.4%
- **Description**: Characterized by high Three Share
- **Examples**: Donovan Mitchell, Nigel Williams-Goss, Lonzo Ball

### Low Poss Proxy Type (Archetype 8)

- **Players**: 14
- **Avg RAPM**: -0.59
- **Survival Rate**: 71.4%
- **Description**: Characterized by low Poss Proxy
- **Examples**: Deyonta Davis, Travis Wear, Pierre Jackson

### High Blk Total Per100Poss Type (Archetype 9)

- **Players**: 1
- **Avg RAPM**: -0.45
- **Survival Rate**: 0.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Nerlens Noel

### High Ft Att Type (Archetype 10)

- **Players**: 53
- **Avg RAPM**: 0.32
- **Survival Rate**: 67.9%
- **Description**: Characterized by high Ft Att
- **Examples**: Charles Cooke, Jayson Tatum, Jaylen Brown

### High Fga Total Type (Archetype 11)

- **Players**: 81
- **Avg RAPM**: 0.20
- **Survival Rate**: 74.1%
- **Description**: Characterized by high Fga Total
- **Examples**: PJ Dozier, Zach Collins, Caleb Swanigan

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
