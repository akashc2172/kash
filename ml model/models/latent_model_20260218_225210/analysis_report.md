# Latent Space Model Analysis Report

**Generated**: 2026-02-18 22:52
**Model**: latent_model_20260218_225210

## Training Results

- **Best Epoch**: 14
- **Best Val Loss**: 6.9806
- **Test RMSE**: 1.619459867477417
- **Test Correlation**: 0.004790699429068355

## Discovered Archetypes

### High Shots Total Type (Archetype 0)

- **Players**: 86
- **Avg RAPM**: 0.10
- **Survival Rate**: 67.4%
- **Description**: Characterized by high Shots Total
- **Examples**: Erik McCree, Mangok Mathiang, Zach Collins

### High Recruiting Rank Type (Archetype 1)

- **Players**: 94
- **Avg RAPM**: 0.00
- **Survival Rate**: 71.3%
- **Description**: Characterized by high Recruiting Rank
- **Examples**: Edmond Sumner, London Perrantes, Luke Kornet

### Low Games Played Type (Archetype 2)

- **Players**: 24
- **Avg RAPM**: 0.02
- **Survival Rate**: 70.8%
- **Description**: Characterized by low Games Played
- **Examples**: Harry Giles III, Pierre Jackson, Maurice Harkless

### High Blk Total Per100Poss Type (Archetype 3)

- **Players**: 5
- **Avg RAPM**: 1.64
- **Survival Rate**: 60.0%
- **Description**: Characterized by high Blk Total Per100Poss
- **Examples**: Anthony Davis, Michael Kidd-Gilchrist, Jeff Withey

### High Fga Total Type (Archetype 4)

- **Players**: 81
- **Avg RAPM**: 0.22
- **Survival Rate**: 70.4%
- **Description**: Characterized by high Fga Total
- **Examples**: PJ Dozier, Donovan Mitchell, Nigel Williams-Goss

### Low Team Srs Type (Archetype 5)

- **Players**: 12
- **Avg RAPM**: -0.94
- **Survival Rate**: 58.3%
- **Description**: Characterized by low Team Srs
- **Examples**: Jaylen Morris, Festus Ezeli, Dion Waiters

### Low Three Share Type (Archetype 6)

- **Players**: 9
- **Avg RAPM**: 1.36
- **Survival Rate**: 88.9%
- **Description**: Characterized by low Three Share
- **Examples**: Bam Adebayo, Rondae Hollis-Jefferson, Karl-Anthony Towns

### High Breakout Rank Usage Type (Archetype 7)

- **Players**: 83
- **Avg RAPM**: 0.20
- **Survival Rate**: 75.9%
- **Description**: Characterized by high Breakout Rank Usage
- **Examples**: Josh Hart, Tyler Lydon, Justin Patton

### Low Team Pace Type (Archetype 8)

- **Players**: 5
- **Avg RAPM**: 1.09
- **Survival Rate**: 60.0%
- **Description**: Characterized by low Team Pace
- **Examples**: Terrence Jones, Cody Zeller, James Southerland

### Low Dev Quality Weight Type (Archetype 9)

- **Players**: 40
- **Avg RAPM**: 1.40
- **Survival Rate**: 80.0%
- **Description**: Characterized by low Dev Quality Weight
- **Examples**: Amile Jefferson, Jayson Tatum, De'Aaron Fox

### Low Shots Total Type (Archetype 10)

- **Players**: 34
- **Avg RAPM**: -0.07
- **Survival Rate**: 58.8%
- **Description**: Characterized by low Shots Total
- **Examples**: Ike Anigbogu, Rodney Purvis, Jarrett Allen

### Low Games Played Type (Archetype 11)

- **Players**: 10
- **Avg RAPM**: 1.20
- **Survival Rate**: 80.0%
- **Description**: Characterized by low Games Played
- **Examples**: Marquis Teague, Jae Crowder, Elias Harris

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
