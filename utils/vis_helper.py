import matplotlib.pyplot as plt
import pandas as pd

def plot_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    column_names = list(df.columns)
    x_name = column_names[0]
    y_name = column_names[1]

    df.plot(x =x_name, y=y_name)
    plt.ylabel(y_name)
    plt.show()
