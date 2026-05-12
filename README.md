**1️.Project Overview**
This project focuses on predicting high-frequency stock prices using a Transformer-based Deep Learning model. The system analyzes historical stock market data from S&P 500, DOW, and NASDAQ indices to forecast future stock price movements. The Transformer model learns temporal dependencies and sequential financial patterns to improve prediction accuracy and capture market fluctuations effectively.

**2️.About Secondary Data**
The project uses Secondary Data collected from the Stooq Financial Data Platform. Historical stock market data of:
S&P 500
Dow Jones Industrial Average (DOW)
NASDAQ Composite
were used for model training and evaluation. The dataset was preprocessed using normalization and detrending techniques before model training.

**3️.Results Obtained**
MetricValueMSE0.007834RMSE0.088512MAE0.063378R² Score0.994095
Observations
The predicted values closely matched actual stock prices.
The Transformer model successfully captured short-term and long-term market trends.
The model achieved high forecasting accuracy with low residual error.

**4️.Method Used**
The project uses a Transformer Neural Network Architecture for sequence-to-sequence stock price forecasting.
Techniques Used
Multi-Head Self Attention
Feed Forward Networks
Teacher Forcing Training
Sliding Window Sequence Generation
Z-Score Normalization
Linear Detrending
The model was trained using the Adam Optimizer and Mean Squared Error (MSE) loss function.

**5️.Tools and Technologies Used**
Tool / TechnologyPurposePythonProgramming LanguageTensorFlow / KerasDeep Learning FrameworkNumPyNumerical ComputationMatplotlibData VisualizationScikit-learnPerformance EvaluationStreamlitDashboard Development

**6️.Dashboard**
An interactive Streamlit dashboard was developed for:
Stock Trend Visualization
Actual vs Predicted Comparison
Future Stock Price Forecasting
Model Performance Metrics Display
The dashboard helps users visualize stock market behaviour and prediction performance in a simple and interactive manner.

**7️.Future Scope**
Real-time stock prediction using live financial APIs
Financial news sentiment analysis integration
Cryptocurrency forecasting support
Deployment on cloud platforms
Comparison with LSTM and GRU models
Advanced interactive visualization features

