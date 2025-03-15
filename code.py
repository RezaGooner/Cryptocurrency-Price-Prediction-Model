import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import mplcursors
from matplotlib import patheffects
from matplotlib.widgets import Button

#----------Variable----------
CoinGecko_API = "https://api.coingecko.com/api/v3/coins/the-open-network/market_chart"
vs_currency = 'usd'
days = '30'
interval = 'daily'
predict_default_day = 4
backGround_color = '#0c1421'
line_actual_color = '#ECFFDC'
line_actual_width = 1
line_actual_lable = 'Actual Price'
below_threshold_color = '#14ab74'
above_threshold_color = '#d93640'
past_predict_color = '#00FF7F'
past_predict_lable = 'Past Predictions (Train)'
past_predict_linestyle = '--'
future_predict_lable = 'Future Predictions'
future_predict_color = '#DA70D6'
future_predict_linestyle = '--'
future_predict_marker = 'o'
test_predict_color = '#FF5733'
test_predict_lable = 'Test Predictions'
test_predict_marker = 'o'
axis_text_color = 'white'
x_lable_color = 'white'
y_lable_color = 'white'
main_title_color = 'white'
grid_line_color = 'lightgray'
grid_line_linestyle = '--'
grid_line_width = 0.5
x_lable = 'Date'
y_lable = 'Price (USD)'
main_title = 'TON Coin Price Prediction'
draw_button_lable = 'Draw Line'
draw_button_color = '#0c1421'
draw_button_hovercolor = '#14ab74'
clear_button_lable = 'Clear Lines'
clear_button_color = '#0c1421'
clear_button_hovercolor = '#d93640'
drawing_line_color = 'red'
drawing_line_width = 2
drawing_line_linestyle = '--'
drawed_line_color = 'red'
drawed_line_width = 2


#----------------------------

# recive price data from CoinGecko API
def fetch_toncoin_data():
    url = CoinGecko_API
    params = {
        'vs_currency': vs_currency,
        'days': days,  
        'interval': interval
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop(columns=['timestamp'], inplace=True)
    return df

def preprocess_data(df):
    df['days'] = (df.index - df.index[0]).days
    return df

def predict_price(df, future_days=predict_default_day):
    X = df[['days']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # preiction for test data
    y_pred = model.predict(X_test)
    
    # prediction for trained data
    y_train_pred = model.predict(X_train)
    
    # prediction of future
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days)
    future_days_list = np.arange(df['days'].max() + 1, df['days'].max() + 1 + future_days).reshape(-1, 1)
    
    future_days_df = pd.DataFrame(future_days_list, columns=['days'])
    
    future_prices = model.predict(future_days_df)
    
    future_df = pd.DataFrame({'date': future_dates, 'price': future_prices})
    future_df.set_index('date', inplace=True)
    
    mse_test = mean_squared_error(y_test, y_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    print(f'Mean Squared Error (Test): {mse_test}')
    print(f'Mean Squared Error (Train): {mse_train}')
    
    return model, X_train, X_test, y_train, y_test, y_train_pred, y_pred, future_df

def plot_results(df, X_train, X_test, y_train, y_test, y_train_pred, y_pred, future_df, threshold):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.set_facecolor(backGround_color)
    fig.patch.set_facecolor(backGround_color)
    
    df = df.sort_index()
    
    line_actual, = plt.plot(df.index, df['price'], label=line_actual_lable, color=line_actual_color, linewidth=line_actual_width)
    
    plt.fill_between(df.index, df['price'], threshold, where=(df['price'] > threshold), color=below_threshold_color, alpha=0.3, interpolate=True)
    plt.fill_between(df.index, df['price'], threshold, where=(df['price'] <= threshold), color=above_threshold_color, alpha=0.3, interpolate=True)
    
    X_train_sorted = X_train.sort_index()
    y_train_pred_sorted = pd.Series(y_train_pred, index=X_train.index).sort_index()
    
    line_train, = plt.plot(X_train_sorted.index, y_train_pred_sorted, label=past_predict_lable, color=past_predict_color, linestyle=past_predict_linestyle)
    
    line_future, = plt.plot(future_df.index, future_df['price'], label=future_predict_lable, color=future_predict_color, linestyle=future_predict_linestyle, marker=future_predict_marker)

    scatter_test = plt.scatter(X_test.index, y_pred, color=test_predict_color, label=test_predict_lable, marker=test_predict_marker)
    
    ax.tick_params(colors=axis_text_color)
    ax.xaxis.label.set_color(x_lable_color)
    ax.yaxis.label.set_color(y_lable_color)
    ax.title.set_color(main_title_color)
    ax.grid(color=grid_line_color, linestyle=grid_line_linestyle, linewidth=grid_line_width)
    
    plt.xticks(rotation=70)
    
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.title(main_title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    cursor = mplcursors.cursor([line_actual, line_train, line_future, scatter_test], hover=True)

    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(f"Price: {sel.target[1]:.2f} USD")
        sel.annotation.get_bbox_patch().set(facecolor="black", edgecolor="white", alpha=0.8)
        sel.annotation.arrow_patch.set_color("white")
        sel.annotation.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])

    @cursor.connect("remove")
    def on_remove(sel):
        sel.annotation.set_visible(False)

    ax_button_draw = plt.axes([0.7, 0.02, 0.1, 0.05])
    ax_button_clear = plt.axes([0.81, 0.02, 0.1, 0.05])

    button_draw = Button(ax_button_draw, draw_button_lable, color=draw_button_hovercolor)
    button_clear = Button(ax_button_clear, clear_button_lable , color=clear_button_color, hovercolor=clear_button_hovercolor)

    for button in [button_draw, button_clear]:
        button.label.set_color('white')

    drawn_lines = []

    is_drawing = False
    line_start = None
    temp_line = None

    def toggle_draw_line(event):
        nonlocal is_drawing, line_start, temp_line
        is_drawing = not is_drawing
        
        if is_drawing:
            button_draw.color =draw_button_hovercolor
            button_draw.hovercolor = draw_button_hovercolor
            plt.setp(ax_button_draw, facecolor=draw_button_hovercolor)
        else:
            button_draw.color = draw_button_color
            button_draw.hovercolor = draw_button_hovercolor
            plt.setp(ax_button_draw, facecolor=draw_button_color)
            line_start = None
            if temp_line:
                temp_line.remove()
                temp_line = None
        plt.draw()

    def clear_lines(event):
        for line in drawn_lines:
            line.remove()
        drawn_lines.clear()
        plt.draw()

    def on_motion(event):
        nonlocal temp_line, line_start
        if is_drawing and line_start:
            if temp_line:
                temp_line.remove()
            x1, y1 = line_start
            x2, y2 = event.xdata, event.ydata
            temp_line = plt.Line2D([x1, x2], [y1, y2], color=drawing_line_color, linewidth=drawing_line_width, linestyle=drawing_line_linestyle)
            ax.add_line(temp_line)
            plt.draw()

    def on_click(event):
        nonlocal line_start, temp_line
        if is_drawing and event.inaxes == ax:
            if line_start is None:
                line_start = (event.xdata, event.ydata)
            else:
                x1, y1 = line_start
                x2, y2 = event.xdata, event.ydata
                line = plt.Line2D([x1, x2], [y1, y2], color=drawed_line_color, linewidth=drawed_line_width)
                ax.add_line(line)
                drawn_lines.append(line)
                line_start = None
                if temp_line:
                    temp_line.remove()
                    temp_line = None
                plt.draw()

    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event', on_click)

    button_draw.on_clicked(toggle_draw_line)
    button_clear.on_clicked(clear_lines)

    plt.show()

df = fetch_toncoin_data()
df = preprocess_data(df)

threshold = float(input("Enter Threshold >> "))

model, X_train, X_test, y_train, y_test, y_train_pred, y_pred, future_df = predict_price(df, future_days=4)

print("Prediction for next " + str(predict_default_day) + " days :" )
print(future_df)

plot_results(df, X_train, X_test, y_train, y_test, y_train_pred, y_pred, future_df, threshold)
