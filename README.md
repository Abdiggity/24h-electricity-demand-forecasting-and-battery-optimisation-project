# Forecasting Tomorrow's Energy and Optimising Today's Costs

Most forecasting projects stop at prediction accuracy. This one goes further: the forecast has to be good enough to make money.

The challenge is predicting tomorrow's electricity demand well enough that a battery can arbitrage the price differences between cheap overnight electricity and expensive peak hours. Get the forecast wrong and your battery schedule falls apart. Get it right and you save about £88 per day for this commercial site.

This work was completed as part of Allye's Data Science technical assessment. I developed the full solution in Google Colab. The brief was to forecast the next 24 hours of electricity demand using historical half-hourly data, then create a battery charging and discharging schedule that minimises cost under the given tariff structure.

## Summary of the Approach

I treated the problem as two connected but separable tasks: forecasting and optimisation. The forecasting model provides a 24-hour demand prediction, and the optimisation logic uses that prediction to plan battery behaviour subject to physical constraints.

### Forecasting Model

I trained a LightGBM regression model on historical half-hourly demand data. The feature set included lag values such as one step ago and one day ago, rolling means and standard deviations over short and long windows, time-based features such as hour of day and day of week with cyclical encoding, and a residual difference from the previous day to capture short-term changes.

The model achieved a mean absolute error of 1.14 kW and an RMSE of 1.64 kW on the final 24 hours of the dataset. A simple baseline that repeated yesterday's profile scored 13.23 kW MAE and 18.69 kW RMSE. The improvement is substantial and consistent across the forecast horizon, capturing both the daily structure and short-term variations that the naive baseline misses entirely.

### Battery Optimisation

Using the forecast and the tariff data, I implemented a simple greedy scheduling strategy. Each of the 48 half-hour periods was ranked by price. The battery charged at its maximum rate during the lowest-priced third of periods, discharged during the highest-priced third, and remained idle in the middle third. All actions respected the battery limits: a maximum power of 220 kW, a total capacity of 440 kWh, and an initial state of energy of 220 kWh. Grid export was not allowed, so net import was clipped at zero.

This strategy reduced the total energy cost for the day from £1,024.45 to £936.01, a saving of £88.44 or roughly 8.6 per cent. The battery naturally charged during the low-price night periods and discharged during the expensive late afternoon and evening windows. The simplicity of the approach is deliberate. It's transparent, respects all constraints, and captures most of the available value without unnecessary complexity.

## What This Demonstrates

The project shows that effective feature engineering can produce strong forecasts for half-hourly demand, and that even a simple optimisation approach can deliver meaningful cost savings when the forecast is reliable. Separating forecasting from optimisation makes the solution easier to test, debug, and extend. Comparing the model to a naive baseline provides an honest measure of improvement and ensures the added complexity is justified.

More broadly, it demonstrates the value of treating machine learning as part of a system. The forecast accuracy matters, but only insofar as it enables better decisions. The real measure of success is whether the battery schedule works and whether it saves money. It does both.

## Future Work

Several improvements would be worthwhile with more time or for a production setting.

**Validation and uncertainty.** Use rolling window cross-validation to understand performance across different days and conditions rather than relying on a single test period. Generate probabilistic forecasts using quantile regression or residual bootstrapping to quantify uncertainty and enable robust optimisation under forecast risk.

**Better optimisation.** Replace the greedy heuristic with a linear program that guarantees the optimal schedule given the constraints. Include battery efficiency losses and degradation costs to make the model more realistic and prevent cycling on marginal price differences that don't cover wear and tear.

**Adaptive control.** Implement model predictive control where the system re-forecasts and re-optimises every few hours using the latest observations. This allows it to adapt when reality diverges from the forecast, which it always does.

**Practical analysis.** Conduct sensitivity analysis across battery sizes, power ratings, and tariff structures to support operational and investment decisions. Quantify the value of forecast accuracy by comparing savings using the forecast versus perfect future knowledge.

**Engineering quality.** Modularise the notebook into reusable functions, extract configuration into a separate file, and improve visualisations to show charge and discharge power alongside state of energy.

The priority would be adaptive control and uncertainty quantification, as those are where this idealised version differs most from real-world operations. The greedy optimiser would likely remain until there was clear evidence it was leaving significant money on the table.

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

Python, LightGBM, pandas, NumPy, Matplotlib, Jupyter.

Time spent: approximately three hours as suggested, plus additional time for reflection on possible improvements.
