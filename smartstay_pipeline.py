"""
SmartStay — Explainable Rental Price Predictor
NVIDIA AI GPU Summer Internship Program (Summer 2026)
Presented by: Nikhil D P | Roll No: 20241ISE0014

End-to-end regression pipeline: cleaning -> feature engineering -> model
comparison (Linear Regression / Random Forest / Gradient Boosting) ->
cross-validation -> GridSearchCV tuning -> explainability -> evaluation.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
PALETTE = ["#0F6E5C", "#2E9E8B", "#E8A33D", "#C9491C"]  # teal/amber real-estate palette

OUT = "/home/claude/smartstay"

# ----------------------------------------------------------------------
# Stage 1: Ingestion & Cleaning
# ----------------------------------------------------------------------
df = pd.read_csv(f"{OUT}/AB_NYC_2019.csv")
print("Raw shape:", df.shape)

df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
df["host_name"] = df["host_name"].fillna("Unknown")
df = df[(df["price"] > 0) & (df["price"] < df["price"].quantile(0.995))].copy()
df["log_price"] = np.log1p(df["price"])

# ----------------------------------------------------------------------
# Stage 2: Feature Engineering
# ----------------------------------------------------------------------
CITY_CENTER = (40.7580, -73.9855)  # Times Square
df["distance_to_center_km"] = np.sqrt(
    (df["latitude"] - CITY_CENTER[0]) ** 2 + (df["longitude"] - CITY_CENTER[1]) ** 2
) * 111
df["reviews_activity_score"] = df["reviews_per_month"] * (df["number_of_reviews"] > 0).astype(int)
df["host_scale"] = np.log1p(df["calculated_host_listings_count"])
df["availability_ratio"] = df["availability_365"] / 365.0

num_features = ["minimum_nights", "number_of_reviews", "reviews_activity_score",
                 "distance_to_center_km", "host_scale", "availability_ratio"]
cat_features = ["neighbourhood_group", "room_type"]

X = df[num_features + cat_features]
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
price_test = np.expm1(y_test)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
])

# ----------------------------------------------------------------------
# Stage 3: Modelling — 3-way comparison with 5-fold CV
# ----------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                             scoring=("neg_root_mean_squared_error", "r2"), n_jobs=-1)
    cv_results[name] = {
        "RMSE": -scores["test_neg_root_mean_squared_error"].mean(),
        "R2": scores["test_r2"].mean(),
    }
    print(name, cv_results[name])

best_name = min(cv_results, key=lambda k: cv_results[k]["RMSE"])
print("Best baseline model:", best_name)

# ----------------------------------------------------------------------
# Stage 4: Tuning the winning model (Gradient Boosting) with GridSearchCV
# ----------------------------------------------------------------------
gb_pipe = Pipeline([("prep", preprocess), ("model", GradientBoostingRegressor(random_state=42))])
param_grid = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.05, 0.1],
}
grid = GridSearchCV(gb_pipe, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

# Pre-tuning baseline (best untuned model) for before/after comparison
baseline_pipe = Pipeline([("prep", preprocess), ("model", GradientBoostingRegressor(random_state=42))])
baseline_pipe.fit(X_train, y_train)

def eval_on_test(pipe, label):
    pred_log = pipe.predict(X_test)
    pred_price = np.expm1(pred_log)
    mae = mean_absolute_error(price_test, pred_price)
    rmse = np.sqrt(mean_squared_error(price_test, pred_price))
    r2 = r2_score(price_test, pred_price)
    print(f"{label}: MAE=${mae:.2f}  RMSE=${rmse:.2f}  R2={r2:.3f}")
    return dict(MAE=mae, RMSE=rmse, R2=r2), pred_price

metrics_before, pred_before = eval_on_test(baseline_pipe, "Before tuning")
metrics_after, pred_after = eval_on_test(best_model, "After tuning")

# ----------------------------------------------------------------------
# Stage 5: Explainability — global feature importance
# ----------------------------------------------------------------------
ohe = best_model.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(cat_features))
all_feature_names = num_features + cat_names
importances = best_model.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).head(10)

# ----------------------------------------------------------------------
# Save artifacts: model, metrics, charts
# ----------------------------------------------------------------------
joblib.dump(best_model, f"{OUT}/smartstay_model.joblib")

with open(f"{OUT}/results.json", "w") as f:
    json.dump({
        "cv_results": cv_results,
        "best_params": grid.best_params_,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "n_rows": int(df.shape[0]),
    }, f, indent=2)

# --- Chart 1: Model comparison (RMSE) ---
fig, ax = plt.subplots(figsize=(7, 4.5))
names = list(cv_results.keys())
rmses = [cv_results[n]["RMSE"] for n in names]
bars = ax.bar(names, rmses, color=PALETTE[:3])
ax.set_ylabel("Cross-Val RMSE (log price)")
ax.set_title("Model Comparison — 5-Fold Cross-Validation")
for b, v in zip(bars, rmses):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/chart_model_comparison.png", dpi=150)
plt.close()

# --- Chart 2: Before vs After tuning ---
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
metrics_labels = ["MAE", "RMSE"]
before_vals = [metrics_before[m] for m in metrics_labels]
after_vals = [metrics_after[m] for m in metrics_labels]
x = np.arange(len(metrics_labels))
axes[0].bar(x - 0.18, before_vals, width=0.36, label="Before tuning", color=PALETTE[3])
axes[0].bar(x + 0.18, after_vals, width=0.36, label="After tuning", color=PALETTE[0])
axes[0].set_xticks(x); axes[0].set_xticklabels(metrics_labels)
axes[0].set_ylabel("USD ($)")
axes[0].set_title("Error — Before vs After Tuning")
axes[0].legend(fontsize=8)
axes[1].bar(["Before"], [metrics_before["R2"]], color=PALETTE[3], width=0.5)
axes[1].bar(["After"], [metrics_after["R2"]], color=PALETTE[0], width=0.5)
axes[1].set_title("R\u00b2 — Before vs After Tuning")
axes[1].set_ylim(0, 1)
for i, v in enumerate([metrics_before["R2"], metrics_after["R2"]]):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/chart_before_after.png", dpi=150)
plt.close()

# --- Chart 3: Feature importance ---
fig, ax = plt.subplots(figsize=(7.5, 4.5))
imp_df_sorted = imp_df.sort_values("importance")
ax.barh(imp_df_sorted["feature"], imp_df_sorted["importance"], color=PALETTE[1])
ax.set_title("Top 10 Features Driving Price Prediction")
ax.set_xlabel("Relative Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/chart_feature_importance.png", dpi=150)
plt.close()

# --- Chart 4: Residuals (actual vs predicted) ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(price_test, pred_after, alpha=0.25, s=12, color=PALETTE[0])
lims = [0, max(price_test.max(), pred_after.max())]
ax.plot(lims, lims, "--", color=PALETTE[3], linewidth=1.5)
ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("Predicted Price ($)")
ax.set_title("Actual vs Predicted — Tuned Model")
plt.tight_layout()
plt.savefig(f"{OUT}/chart_residuals.png", dpi=150)
plt.close()

# --- Chart 5: Price distribution by borough & room type (EDA) ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
order = df.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="neighbourhood_group", y="price", order=order, ax=axes[0],
            showfliers=False, palette=PALETTE + ["#8AA6A0"])
axes[0].set_title("Price by Borough")
axes[0].set_xlabel(""); axes[0].tick_params(axis="x", rotation=20)
sns.boxplot(data=df, x="room_type", y="price", ax=axes[1], showfliers=False,
            palette=PALETTE[:3])
axes[1].set_title("Price by Room Type")
axes[1].set_xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/chart_eda_price.png", dpi=150)
plt.close()

print("\nAll artifacts saved to", OUT)
print(json.dumps({"cv_results": cv_results, "metrics_before": metrics_before,
                   "metrics_after": metrics_after, "best_params": grid.best_params_}, indent=2))
