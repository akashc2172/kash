# Latent Space Model Analysis Report

**Generated**: 2026-02-18 21:34
**Model**: latent_model_20260218_213404

## Training Results

- **Best Epoch**: 63
- **Best Val Loss**: 1.5660
- **Test RMSE**: 1.6113237142562866
- **Test Correlation**: -0.06596812615948919

## Discovered Archetypes

### High Fga Total Type (Archetype 0)

- **Players**: 210
- **Avg RAPM**: 0.05
- **Survival Rate**: 72.4%
- **Description**: Characterized by high Fga Total
- **Examples**: PJ Dozier, Josh Hart, Edmond Sumner

### High Blk Total Per100Poss Type (Archetype 1)

- **Players**: 24
- **Avg RAPM**: 1.49
- **Survival Rate**: 70.8%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Skal Labissiere, Rondae Hollis-Jefferson, Cliff Alexander

### High Recruiting Rank Type (Archetype 2)

- **Players**: 41
- **Avg RAPM**: 1.02
- **Survival Rate**: 75.6%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Jonathan Isaac, Jake Layman, Marcus Georges-Hunt

### High Shots Total Type (Archetype 3)

- **Players**: 89
- **Avg RAPM**: 0.22
- **Survival Rate**: 68.5%
- **Description**: Characterized by high Shots Total
- **Examples**: Erik McCree, Zach Collins, Sindarius Thornwell

### High Blk Total Per100Poss Type (Archetype 4)

- **Players**: 3
- **Avg RAPM**: 1.23
- **Survival Rate**: 100.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Jeff Withey, Mason Plumlee, Ryan Kelly

### Low Career Wt Three Fg Pct Type (Archetype 5)

- **Players**: 33
- **Avg RAPM**: 0.84
- **Survival Rate**: 66.7%
- **Description**: Characterized by low Career Wt Three Fg Pct
- **Examples**: Mangok Mathiang, Amile Jefferson, Bam Adebayo

### Low Recruiting Stars Type (Archetype 6)

- **Players**: 43
- **Avg RAPM**: 0.25
- **Survival Rate**: 69.8%
- **Description**: Characterized by low Recruiting Stars
- **Examples**: Luke Kornet, Charles Cooke, Jayson Tatum

### Low Poss Proxy Type (Archetype 7)

- **Players**: 40
- **Avg RAPM**: -0.63
- **Survival Rate**: 67.5%
- **Description**: Characterized by low Poss Proxy
- **Examples**: Jaylen Morris, Ike Anigbogu, Harry Giles III

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
