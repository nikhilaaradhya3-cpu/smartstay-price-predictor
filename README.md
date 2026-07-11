# SmartStay — Explainable Rental Price Predictor

**NVIDIA AI & GPU Computing Summer Internship Program (Summer 2026)**
**Author:** Nikhil D P · Roll No: 20241ISE0014

An end-to-end regression pipeline that predicts a fair nightly rental price from
listing details (location, room type, reviews, availability, host activity) and
explains **why**, instead of returning a black-box number.

## Results

| Metric (held-out test set) | Before tuning | After tuning |
|---|---|---|
| MAE | $28.38 | **$28.36** |
| RMSE | $38.66 | **$38.54** |
| R² | 0.630 | **0.632** |

Best model: **Gradient Boosting Regressor** (`n_estimators=250, max_depth=2, learning_rate=0.05`), selected after a fair 5-fold cross-validated comparison against Linear Regression and Random Forest.

## Project structure

```
.
├── generate_data.py                          # builds the NYC-Airbnb-style dataset
├── smartstay_pipeline.py                     # full pipeline: clean → engineer → model → tune → explain
├── app.py                                     # Streamlit deployment app
├── SmartStay_Nikhil_DP_20241ISE0014.ipynb     # notebook version of the pipeline (submission copy)
├── SmartStay_Nikhil_DP_20241ISE0014.pptx      # judge presentation deck
├── Presenter_Script_and_QA.md                 # talking points + anticipated Q&A
├── data/AB_NYC_2019.csv                       # dataset (NYC Airbnb-style, 8,500 listings)
├── charts/                                    # all generated evaluation/EDA charts
└── requirements.txt
```

## Pipeline

1. **Ingest & Clean** — fill missing reviews, cap price outliers, log-transform the skewed target.
2. **Feature Engineering** — distance-to-city-center, review activity score, host scale, availability ratio.
3. **Model & Compare** — Linear Regression, Random Forest, Gradient Boosting under identical 5-fold CV.
4. **Tune** — `GridSearchCV` over the winning model's hyperparameters.
5. **Explain** — global feature importance + top-3 reasons surfaced per prediction.
6. **Deploy** — a Streamlit form serving the tuned, serialized (`joblib`) pipeline.

## Running it locally

```bash
pip install -r requirements.txt

# 1. Regenerate the dataset (or use the one already in data/)
python generate_data.py

# 2. Run the full pipeline — trains, tunes, evaluates, saves charts + model
python smartstay_pipeline.py

# 3. Launch the deployment app
streamlit run app.py
```

## Dataset note

Built to mirror the public **AB_NYC_2019** dataset's structure and statistical
relationships (borough price gradient, room-type price gradient, review/activity
patterns, realistic missingness) since the original CSV wasn't reachable from
this build environment. The pipeline, model comparison, and evaluation
methodology apply identically to the real file — swap in the original
`AB_NYC_2019.csv` in `data/` and everything runs unchanged.

## License

Educational project — NVIDIA AI & GPU Computing Summer Internship Program, Summer 2026.
