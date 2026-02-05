# Soccer-Outcome-Predictor


End-to-End Machine Learning Pipeline for Predicting Premier League Results

## Overview

This project builds and evaluates machine learning models to predict English Premier League match outcomes (home win, draw, away win) using historical match data from the football-data.org API.

Beyond prediction accuracy, the project focuses on understanding what actually drives match outcomes, specifically:

Home field advantage

Short-term team form

Contextual match factors

The goal is not just to predict results, but to evaluate which signals meaningfully improve out-of-sample performance and how those trade off with interpretability.

Problem Statement

Soccer outcomes are influenced by many overlapping factors: team strength, momentum, venue, and match context. This project asks:

Does home advantage remain statistically significant after controlling for team strength and form?

Does recent form materially improve predictive performance?

Do contextual features meaningfully improve generalization beyond team-level features?

## Data

Source: football-data.org REST API

League: English Premier League

Seasons: 2023 season through December 2025

Unit of analysis: One row per match

## Target Variable

match_outcome

1 → Home win

0 → Draw

-1 → Away win

## Key Features

Team identifiers (home / away)

Recent form metrics (rolling performance over prior matches)

Goal averages

Head-to-head win rates

Matchday

Contextual variables (venue, referee)

All features are engineered chronologically to prevent data leakage.

## Feature Engineering & Preprocessing

Converted raw JSON responses into structured tabular data

Validated and cleaned incomplete or inconsistent records

Engineered leakage-safe temporal features, including:

Recent form metrics

Rolling goal averages

Head-to-head statistics

Encoded categorical variables and scaled numerical features where required

Ensured one row per match with predictors available only before kickoff

## Modeling Approach

Three model families were trained and compared:

Logistic Regression

Baseline model

Used for interpretability (coefficients, odds ratios)

Random Forest

Captures nonlinear interactions

Evaluated via permutation importance

Gradient Boosting

Optimized for predictive performance

Used SHAP values for feature attribution

## Training Strategy

70 / 15 / 15 train / validation / test split

Cross-validation applied only on training data

Validation set used for model selection

Test set held out for final performance estimation

## Evaluation Metrics

Models were compared using:

Accuracy

Macro F1 (to account for class imbalance)

ROC-AUC

Calibration curves and ROC curves were also used to evaluate probabilistic reliability and sensitivity-specificity tradeoffs.

## Key Findings

Home advantage consistently shows a positive effect across models

Recent form is one of the strongest predictors of match outcome

Contextual features improve interpretability and marginally improve performance, but do not produce large gains in accuracy

Simpler models remain competitive, highlighting the importance of feature quality over model complexity

## Impact

This project demonstrates:

Strong problem framing beyond raw prediction

Careful handling of temporal data and leakage

Thoughtful model comparison and evaluation

Emphasis on interpretability, tradeoffs, and generalization

End-to-end ML workflow from data ingestion to insights

## Tech Stack

Python

pandas, NumPy

scikit-learn

SHAP

Matplotlib
