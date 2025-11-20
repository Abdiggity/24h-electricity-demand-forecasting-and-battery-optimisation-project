# Forecasting Tomorrow's Energy, Optimising Today's Costs

This project tackles a real-world question at the intersection of machine learning and energy management. Can we forecast tomorrow's electricity demand accurately enough to run a battery and save money?

The answer is yes. For this commercial site, about £88 per day.

## What This Project Is

This is my submission for Allye's Data Science technical assessment. The brief was straightforward: use historical half-hourly electricity demand data to forecast the next 24 hours, then use that forecast to optimally charge and discharge a battery to minimise costs.

I approached this like I would a real engineering problem. Make it work end-to-end, evaluate honestly, and be clear about what I'd do next with more time.

## The Problem

Commercial electricity tariffs fluctuate throughout the day. A battery gives you flexibility. You can charge when electricity is cheap and discharge when it's expensive. But this only works if two things are true.

First, your demand forecast needs to be accurate enough to plan battery behaviour. Second, your battery schedule needs to be good enough to exploit the tariff differences.

Both pieces matter. A great forecast with a bad optimiser is wasted. A great optimiser with a bad forecast is dangerous. This project connects the two.

## What I Built

### The Forecasting Model

I trained a LightGBM regression model using features engineered specifically for 30-minute interval electricity demand.

The feature set includes lag features capturing recent history (1 period ago, 48 periods for yesterday, 96 periods for two days ago). I added rolling statistics to smooth out noise, like 3-hour and 24-hour moving averages and standard deviations. Time-based features use cyclical sine and cosine encoding so the model understands that 23:30 is close to 00:00. I also included the residual difference from yesterday to capture short-term deviations.

This lets the model learn the daily shape, weekly patterns, and recent changes all at once.

**Performance results.** The model achieved a mean absolute error of 1.14 kW and RMSE of 1.64 kW. Compare this to a naive baseline that just repeats yesterday's values, which scored 13.23 kW MAE and 18.69 kW RMSE. The model captures morning ramps, midday plateaus, and evening drops far more effectively than simply repeating yesterday.

### The Battery Optimisation

I implemented a simple but effective greedy heuristic. Sort all 48 half-hour periods by electricity price. Charge at maximum power during the cheapest third of periods. Discharge at maximum power during the most expensive third. Stay idle in the middle third. Throughout, enforce the battery constraints: maximum 220 kW charge or discharge rate, 440 kWh total capacity, and starting the day half full at 220 kWh.

**Cost results.** Without the battery, the daily cost would be £1,024.45. With the optimised battery schedule, the cost drops to £936.01. That's a saving of £88.44 per day, or about 8.6% reduction.

The battery charges overnight when prices are lowest and discharges during the late afternoon and evening price spikes. Exactly what you'd expect from sensible arbitrage.

## What This Demonstrates

**Technical fundamentals.** Strong feature engineering often beats overly complex models for time series problems. The lag and rolling features capture the key patterns, and LightGBM handles the non-linearities without needing much tuning.

**Sound problem decomposition.** I treated forecasting and optimisation as separable components. Each was tested independently but connected through a clean interface. This makes debugging easier and the logic clearer.

**Practical engineering thinking.** The greedy optimisation algorithm is not mathematically perfect. But it's transparent, it respects battery constraints, and it captures most of the available benefit. In a real system, you'd start here before adding complexity.

**Honest evaluation.** I compared against a naive baseline because it's easy to build a complicated model that still loses to "just repeat yesterday". The baseline keeps you honest and contextualises the improvement.

## Where This Could Go Next

I've thought carefully about how to extend this work. Here's what I'd prioritise if I had more time or was building this for production.

### Improving the Forecast

**Better validation strategy.** The single-day test set gives a point estimate but no sense of variability. I'd implement time-series cross-validation with multiple rolling windows to understand how performance varies across different days, weekdays versus weekends, and different demand conditions.

**Quantify uncertainty.** Real operations need more than point forecasts. Using quantile regression in LightGBM or residual bootstrapping would generate probabilistic forecasts at the 10th, 50th, and 90th percentiles. This enables robust optimisation that accounts for forecast risk.

**Feature importance analysis.** I'd use SHAP values to understand which features actually drive predictions. This could reveal surprises, like whether Friday afternoons have unique patterns or whether certain lag features are redundant. It would help remove dead weight and potentially suggest new features worth testing.

**Additional baselines.** Beyond the day-ago baseline, I'd compare against a weekly seasonal naive (same time 7 days ago) and a simple mean-by-hour model. This would better contextualise where the LightGBM model is adding value.

### Strengthening the Optimisation

**Proper mathematical optimisation.** Replace the greedy heuristic with a linear program that guarantees the optimal schedule. Formulate it as minimising total grid import cost subject to power limits, state-of-energy bounds, and energy balance constraints. This matters more when you add real-world complications.

**Account for battery wear.** Batteries degrade with cycling. I'd add a degradation cost term so the optimiser only charges and discharges when the price arbitrage genuinely exceeds the wear-and-tear cost. This prevents chasing tiny price differences that aren't economically justified.

**Model predictive control.** In reality, you don't know tomorrow's demand perfectly. I'd implement a rolling horizon approach where you re-forecast and re-optimise every few hours using the latest observations. This allows the system to adapt when reality diverges from the forecast.

**Different efficiency assumptions.** Real batteries aren't 100% efficient. Testing with 90-95% round-trip efficiency would show how much the arbitrage margin shrinks in practice.

### Integration Questions

**How much does forecast accuracy matter?** I'd quantify the value of perfect information by comparing battery savings using the forecast versus using actual future demand. This reveals how much to invest in better forecasting versus better optimisation.

**Sensitivity analysis.** How do savings scale with battery size? What if we had 500 kWh instead of 440? What about different power ratings or efficiency levels? These questions matter for business cases and procurement decisions.

**Performance by segment.** Break down forecast accuracy by hour of day, weekday versus weekend, and low versus high demand periods. This reveals specific time windows where the model struggles and guides targeted improvements.

### Engineering and Communication

**Modularise the code.** Break the notebook into reusable functions like build_features(), train_model(), forecast_24h(), and simulate_battery(). This makes the code easier to test, version, and reuse.

**Configuration management.** Pull all the magic numbers out into a config file. Lag windows, battery specifications, file paths, model hyperparameters. This makes experiments reproducible and comparable.

**Better visualisation.** Add plots showing charge and discharge power over time, not just state of energy. Show demand versus net grid import on the same axes during peak periods to make the battery's impact more visceral.

**Executive summary.** Add a brief summary at the end of the notebook. Two or three paragraphs covering forecast performance, battery savings, key assumptions, and the most important caveats.

## Reflections

This was a satisfying problem because the pieces genuinely fit together. The forecast informs the optimiser, and the optimiser's behaviour validates the forecast.

The time constraint forced prioritisation. You can't tune everything, you have to decide what matters most. That's much closer to production reality than endlessly optimising in isolation.

If I were building this for deployment, I'd focus first on the model predictive control loop and uncertainty quantification. Those are where real-world operations differ most from this idealised version. The greedy optimiser would probably stay until we had evidence it was leaving significant money on the table.

The core point stands. The forecast is good enough to be useful, and the optimisation is simple enough to trust. Everything beyond that is refinement.

---

## How to Run This Project

Clone the repository, install dependencies, and launch Jupyter.

```bash
pip install -r requirements.txt
jupyter lab
```

Open and run the notebook at `Notebooks/24h_demand_forecasting_and_battery_cost_optimisation.ipynb`.

## Project Structure

```
├── Data/
│   ├── assessment_demand_data.csv
│   └── assessment_tariff_data.csv
├── Notebooks/
│   └── 24h_demand_forecasting_and_battery_cost_optimisation.ipynb
└── README.md
```

## Tech Stack

Python, LightGBM, pandas, NumPy, Matplotlib, Jupyter

**Time spent:** Approximately 3 hours as suggested, plus thinking time on improvements.
