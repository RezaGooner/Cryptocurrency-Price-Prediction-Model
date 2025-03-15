# Cryptocurrency Price Prediction Model

This project is a tool for predicting the price of **TON Coin** based on historical data. It uses the **CoinGecko API** to fetch data and employs a **Linear Regression** model to forecast future prices. The project can be easily extended to other cryptocurrencies by modifying the API.

## Features
- **Price Prediction**: Uses Linear Regression to predict future prices.
- **Simple and Attractive UI**: Eye-catching color schemes and interactive charts.
- **Threshold-Based Analysis**: By selecting a threshold value, the chart is drawn relative to it, and areas of profit and loss are color-coded.
- **Display of Predicted Values**: Predicted prices for future days are displayed.
- **Line Drawing Tool**: Allows drawing lines on the chart for further analysis and clearing them.
- **Interactive Tooltips**: Hover over the chart to display precise price information.

---

## How to Run

### Prerequisites
To run this project, you need to install the required libraries. These libraries are listed in the `requirements.txt` file.

### Installing Dependencies
1. Ensure **Python** and **pip** are installed on your system.
2. Run the following command in your terminal to install the required libraries:
   ```bash
       pip install -r requirements.txt

### Running the Project
After installing the dependencies, simply run the main script (main.py):

  ```bash
    python code.py
```

![image](https://github.com/user-attachments/assets/dfb9266a-a4ce-4171-9409-d6dae97357b8)

![image](https://github.com/user-attachments/assets/2695b576-7231-445d-88a0-4a0248848a83)

---

## How It Works

- Data Fetching: Historical price data for TON Coin is fetched from the CoinGecko API.
- Data Preprocessing: The data is prepared for use in the model.
- Modeling: Linear Regression is used to predict future prices.
- Displaying Results: The prediction results are displayed in an interactive chart.

---

## Chart Features

- Threshold-Based Coloring: By selecting a threshold value, the chart is drawn relative to it, and areas of profit and loss are color-coded.
- Interactive Tooltips: Hover over the chart to display precise price information.
- Line Drawing Tool: Draw lines on the chart for further analysis and clear them if needed.

---

## Extending to Other Cryptocurrencies
This project can be easily extended to other cryptocurrencies. Simply replace the CoinGecko API with the API of the desired cryptocurrency and fetch the data.

---

## Reporting Issues or Bugs
If you encounter any issues or bugs while running the project, please let me know through the [**Issues**](https://github.com/RezaGooner/Cryptocurrency-Price-Prediction-Model/issues) section on GitHub. I will try to resolve the problem as soon as possible.

---

## Libraries Used
- **`requests`**: For fetching data from the API.
- **`pandas`**: For data processing and management.
- **`matplotlib`**: For plotting charts.
- **`scikit-learn`**: For implementing the Linear Regression model.
- **`numpy`**: For numerical computations.
- **`mplcursors`**: For adding tooltips to the charts.

---

## License
This project is licensed under the **MIT License**. For more information, see the [LICENSE](LICENSE) file.

---

## Contributing
If you are interested in contributing to this project, please submit your changes via a **Pull Request**. I would be happy to review your ideas and improvements.

---
[RezaGooner](https://github.com/RezaGooner/)
> Best regard!
