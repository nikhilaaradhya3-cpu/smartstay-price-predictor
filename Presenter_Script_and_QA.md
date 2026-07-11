# SmartStay — Presenter Script & Judge Q&A Prep
**Nikhil D P · 20241ISE0014 · NVIDIA AI GPU Summer Internship Program (Summer 2026)**

Total deck: 13 slides. Aim for ~6-7 minutes talking, leaving time for questions.

---

## Slide-by-slide talking points

**1. Title (15s)**
"Good [morning/afternoon], I'm Nikhil D P, roll number 20241ISE0014. My capstone for the NVIDIA AI & GPU Computing Summer Internship is SmartStay — an explainable rental price predictor."

**2. Problem Statement (30s)**
"Hosts don't know how to price a new listing, guests can't tell if a rate is fair, and most pricing tools give you a number with zero explanation. SmartStay solves both sides of that problem at once."

**3. Objectives (30s)**
"Five goals: clean a messy real-world dataset, fairly compare three regression models, tune the winner, explain every prediction instead of hiding behind a black box, and ship it as something anyone can actually use."

**4. Dataset (30s)**
"I worked with an NYC-style Airbnb listings dataset — 8,500 listings across 5 boroughs and 3 room types, structured exactly like the public AB_NYC_2019 dataset: skewed prices, missing review data, the real messiness you'd expect."

**5. Pipeline (45s)**
"Six stages, start to finish: ingest and clean, engineer features like distance-to-center and host activity, compare three models under cross-validation, tune the winner, explain its predictions, and deploy it as a live form."

**6. Tech Stack (20s)**
"Standard, production-relevant tools — pandas and numpy for data, scikit-learn for modelling and tuning, Streamlit for deployment, joblib to persist the final model."

**7. Model Comparison (45s)**
"I compared Linear Regression, Random Forest, and Gradient Boosting under identical 5-fold cross-validation — same preprocessing, same splits, so it's a fair fight. Linear Regression and Gradient Boosting land almost neck-and-neck; I picked Gradient Boosting to move forward because it captures non-linear interactions between location and room type, and gives me tree-based feature importances for the explainability piece."

**8. Tuning (30s)**
"GridSearchCV tuned the estimator count, tree depth, and learning rate. The final model lands at roughly $28 mean absolute error and an R² of 0.63 on a completely untouched test set — meaning it explains about 63% of the variance in nightly price from listing details alone."

**9. Explainability (30s)**
"This is the part that matters most to me: every prediction ships with its top contributing factors. Room type and borough dominate, which matches real-world intuition — an entire home in Manhattan costs more than a shared room in the Bronx — and distance-to-center adds a fine correction on top."

**10. Evaluation (30s)**
"Actual-vs-predicted on the held-out test set: most points sit close to the diagonal. The spread widens for high-end luxury listings, which is expected — those prices depend on things like interior quality that aren't in this dataset."

**11. Deployment (30s)**
"The tuned pipeline is wrapped in a Streamlit form. A host enters listing details, gets an instant price estimate, and sees exactly which factors drove that number — no black box."

**12. Team Split (15s)** *(skip or mention briefly if presenting solo)*
"For a full team build, work splits cleanly into four roles: data engineering, modelling, explainability/reporting, and deployment."

**13. Conclusion (20s)**
"SmartStay is low-risk, high-trust, and fully deployed end-to-end — a real dataset, a fair model comparison, measurable tuning gains, and predictions anyone can understand. Thank you — happy to take questions."

---

## Anticipated judge questions

**Q: Why didn't Gradient Boosting clearly beat Linear Regression?**
A: On this feature set, the price signal is largely linear-separable — location and room type dominate and don't need deep trees to capture. Gradient Boosting still wins narrowly after tuning and generalizes better to non-linear interactions (e.g., how distance-to-center compounds with borough), which matters more on a larger, richer dataset.

**Q: How do you avoid data leakage?**
A: All preprocessing (scaling, encoding) is inside a single `sklearn.Pipeline`, fit only on the training fold each cross-validation round. The test set is touched exactly once, at the very end, for final reporting.

**Q: Why log-transform the price target?**
A: Rental prices are heavily right-skewed — a few luxury listings stretch the range. Training on log(price) stabilizes variance and stops the model from being dominated by high-price outliers; predictions are converted back with `expm1` before reporting dollar metrics.

**Q: How would this scale to a real production system?**
A: Swap the Streamlit form for a lightweight API (FastAPI/Flask) behind the same joblib pipeline, add a retraining schedule as new listings/reviews come in, and log prediction confidence intervals for monitoring drift.

**Q: What's the biggest limitation?**
A: No text or image features — listing photos and descriptions likely carry price signal (e.g., "renovated," "rooftop view") this pipeline can't see yet. A stretch goal would be adding NLP on listing titles/descriptions.

**Q: Why R² of 0.63 and not higher?**
A: That's a realistic, honest number for price-from-metadata-only prediction — real Airbnb pricing studies land in a similar range. A higher number without richer features (photos, amenities, seasonality) would be a red flag for overfitting or leakage, not a strength.

---

## Day-of checklist
- [ ] Export deck to PDF as a backup (`SmartStay_Nikhil_DP_20241ISE0014.pdf` already generated)
- [ ] Load `SmartStay_Nikhil_DP_20241ISE0014.ipynb` and confirm it still runs top-to-bottom
- [ ] Have `app.py` open in a code editor in case judges ask to see the deployment code
- [ ] Print or screenshot the slide with model comparison numbers (Slide 7) for quick reference during Q&A
- [ ] Charge laptop / test HDMI or screen-share connection beforehand
