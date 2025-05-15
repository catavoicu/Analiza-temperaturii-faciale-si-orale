from flask import Flask, render_template, request, url_for
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from werkzeug.utils import secure_filename
from scipy.stats import linregress

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

df_cache = None
selected_column = None
ref_choice = None
comparison_url = None
regression_url = None
stats = None
regression_info = None

plt.style.use('dark_background')


def interpretare_corelatie(r):
    r = abs(r)
    if r < 0.2:
        return "corelație foarte slabă"
    elif r < 0.4:
        return "corelație slabă"
    elif r < 0.6:
        return "corelație moderată"
    elif r < 0.8:
        return "corelație puternică"
    else:
        return "corelație foarte puternică"


@app.route('/', methods=['GET', 'POST'])
def index():
    global df_cache, selected_column, ref_choice
    global comparison_url, regression_url, stats, regression_info

    columns = []

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                try:
                    df = pd.read_csv(filepath, header=2)
                    df.columns = df.columns.str.strip()
                    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
                    df_cache = df
                    selected_column = None
                    ref_choice = None
                    comparison_url = None
                    regression_url = None
                    stats = None
                    regression_info = None
                    columns = df.select_dtypes(include='number').columns.tolist()
                    return render_template('index.html', columns=columns)
                except Exception as e:
                    return render_template('index.html', error=f"Eroare la procesare: {str(e)}")

        elif 'column' in request.form and 'ref' in request.form:
            selected_column = request.form['column']
            ref_choice = request.form['ref']
            if df_cache is not None and selected_column in df_cache.columns and ref_choice in df_cache.columns:
                try:
                    mean_val = df_cache[selected_column].mean()
                    std_val = df_cache[selected_column].std()
                    var_val = df_cache[selected_column].var()
                    stats = f"Media: {mean_val:.2f} °C | Deviație standard: {std_val:.2f} °C | Variață: {var_val:.2f} °C²"

                    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#2E2E2E')
                    axes[0].set_facecolor('#2E2E2E')
                    axes[1].set_facecolor('#2E2E2E')
                    df_cache[selected_column].dropna().hist(bins=30, ax=axes[0], color='deepskyblue', edgecolor='black')
                    axes[0].set_title(f'{selected_column}')
                    axes[0].set_xlabel("Temperatură (°C)")
                    axes[0].set_ylabel("Frecvență")

                    df_cache[ref_choice].dropna().hist(bins=30, ax=axes[1], color='hotpink', edgecolor='black')
                    axes[1].set_title(ref_choice)
                    axes[1].set_xlabel("Temperatură (°C)")
                    axes[1].set_ylabel("Frecvență")

                    plt.tight_layout()
                    comp_path = os.path.join('static', 'comparison.png')
                    plt.savefig(comp_path, facecolor='#2E2E2E')
                    plt.close()
                    comparison_url = url_for('static', filename='comparison.png')

                    df_reg = df_cache[[selected_column, ref_choice]].dropna()
                    slope, intercept, r_value, _, _ = linregress(df_reg[selected_column], df_reg[ref_choice])
                    tip_corelatie = interpretare_corelatie(r_value)

                    regression_info = (
                        f"<strong>Ecuația regresiei liniare:</strong><br>"
                        f"y = {slope:.2f}·x + {intercept:.2f} <small>(unde x este temperatura facială, y este temperatura orală estimată)</small><br><br>"
                        f"<strong>Coeficient de corelație Pearson:</strong><br>"
                        f"r = {r_value:.2f} &nbsp;&nbsp; <small>({tip_corelatie})</small>"
                    )

                    fig = plt.figure(figsize=(6, 5), facecolor='#2E2E2E')
                    ax = plt.gca()
                    ax.set_facecolor('#2E2E2E')
                    plt.scatter(df_reg[selected_column], df_reg[ref_choice], alpha=0.5, color='cyan')
                    plt.plot(df_reg[selected_column], slope * df_reg[selected_column] + intercept, color='red')
                    plt.xlabel(selected_column + " (°C)")
                    plt.ylabel(ref_choice + " (°C)")
                    plt.title("Regresie liniară")
                    plt.tight_layout()
                    reg_path = os.path.join('static', 'regression.png')
                    plt.savefig(reg_path, facecolor='#2E2E2E')
                    plt.close()
                    regression_url = url_for('static', filename='regression.png')

                except Exception as e:
                    return render_template('index.html', error=f"Eroare la generarea graficelor: {str(e)}")

    columns = df_cache.select_dtypes(include='number').columns.tolist() if df_cache is not None else []
    return render_template(
        'index.html',
        columns=columns,
        comparison_url=comparison_url,
        regression_url=regression_url,
        regression_info=regression_info,
        stats=stats,
        selected_column=selected_column,
        ref_choice=ref_choice
    )

if __name__ == '__main__':
    app.run(debug=True)
    