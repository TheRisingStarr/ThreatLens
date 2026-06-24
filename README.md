# ThreatLens

AI-powered cyber threat detection system that uses machine learning to classify incoming network connections as normal or potential attacks in real time.

## Overview

ThreatLens uses a machine learning model trained on historical network traffic data to identify patterns associated with cyber attacks. The system analyzes incoming connections and predicts whether the activity is legitimate or malicious.

The predictions are displayed through a live monitoring dashboard, providing real-time visibility into detected network activity.

## Features

* Machine learning-based attack detection
* Real-time network connection classification
* Live dashboard for monitoring predictions
* Logistic Regression model for threat classification
* Test script to simulate incoming connections
* Normal vs attack activity visualization

## How It Works

1. A dataset containing previous network activity is used to train the ML model.
2. The trained model learns patterns from normal and malicious connections.
3. Incoming connection data is processed and passed to the model.
4. The model predicts whether the connection is:

   * Normal
   * Cyber Attack
5. Results are displayed instantly on the dashboard.

## Tech Stack

* Python
* Machine Learning
* Logistic Regression
* Data Processing
* Web Dashboard
* Real-time Data Streaming

## Goal

To build a lightweight AI-based security monitoring system capable of detecting suspicious network activity in real time.
