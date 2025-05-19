import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, url_for
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from werkzeug.utils import secure_filename
from scipy.stats import linregress
from pandas.plotting import autocorrelation_plot

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
cumulativa_url = None
densitate_url = None
autocorelatie_url = None
clasificare_text = None

plt.style.use('dark_background')

GRAFICE_DISPONIBILE = {
    'comparativa': 'Histogramă comparativă',
    'regresie': 'Regresie liniară',
    'cumulativa': 'Histogramă cumulativă',
    'densitate': 'Densitate de probabilitate (KDE)',
    'clasificare': 'Clasificare temperaturi',
    'autocorelatie': 'Autocorelatie'
}

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
    global cumulativa_url, densitate_url, autocorelatie_url, clasificare_text

    columns = []
    selected_grafice = []

    if request.method == 'POST':
        comparison_url = None
        regression_url = None
        stats = None
        regression_info = None
        cumulativa_url = None
        densitate_url = None
        autocorelatie_url = None
        clasificare_text = None

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
                    columns = df.select_dtypes(include='number').columns.tolist()
                    return render_template('index.html', columns=columns, grafice=GRAFICE_DISPONIBILE)
                except Exception as e:
                    return render_template('index.html', error=f"Eroare la procesare: {str(e)}")

        elif 'column' in request.form and 'ref' in request.form:
            selected_column = request.form['column']
            ref_choice = request.form['ref']
            selected_grafice = request.form.getlist('grafice')

            if df_cache is not None:
                df_col = df_cache[selected_column].dropna()

                if 'comparativa' in selected_grafice:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#2E2E2E')
                    axes[0].set_facecolor('#2E2E2E')
                    axes[1].set_facecolor('#2E2E2E')
                    df_ref = df_cache[ref_choice].dropna()
                    df_col.hist(bins=30, ax=axes[0], color='#4C78A8', edgecolor='black')
                    df_ref.hist(bins=30, ax=axes[1], color='#F58518', edgecolor='black')
                    axes[0].set_title(selected_column)
                    axes[1].set_title(ref_choice)
                    media = df_col.mean()
                    dev_std = df_col.std()
                    varianta = df_col.var()
                    stats = f"Media: {media:.2f} °C | Deviație standard: {dev_std:.2f} °C | Varianță: {varianta:.2f} °C²"
                    comp_path = os.path.join('static', 'comparison.png')
                    plt.tight_layout()
                    plt.savefig(comp_path, facecolor='#2E2E2E')
                    plt.close()
                    comparison_url = url_for('static', filename='comparison.png')

                if 'regresie' in selected_grafice:
                    df_reg = df_cache[[selected_column, ref_choice]].dropna()
                    slope, intercept, r_value, _, _ = linregress(df_reg[selected_column], df_reg[ref_choice])
                    tip_corelatie = interpretare_corelatie(r_value)
                    regression_info = (
                        f"<strong>Ecuație:</strong> y = {slope:.2f}x + {intercept:.2f}<br>"
                        f"<strong>r:</strong> {r_value:.2f} ({tip_corelatie})"
                    )
                    fig = plt.figure(figsize=(6, 5), facecolor='#2E2E2E')
                    ax = fig.gca()
                    ax.set_facecolor('#2E2E2E')
                    plt.scatter(df_reg[selected_column], df_reg[ref_choice], alpha=0.5, color='#72B7B2')
                    plt.plot(df_reg[selected_column], slope * df_reg[selected_column] + intercept, color='#F58518')
                    plt.xlabel(selected_column)
                    plt.ylabel(ref_choice)
                    plt.title(selected_column)
                    plt.tight_layout()
                    path = os.path.join('static', 'regression.png')
                    plt.savefig(path, facecolor='#2E2E2E')
                    plt.close()
                    regression_url = url_for('static', filename='regression.png')

                if 'cumulativa' in selected_grafice:
                    fig = plt.figure(figsize=(6, 4), facecolor='#2E2E2E')
                    ax = fig.gca()
                    ax.set_facecolor('#2E2E2E')
                    df_col.hist(bins=30, cumulative=True, density=True, color='#4C78A8', edgecolor='black')
                    plt.title(selected_column)
                    plt.xlabel("Temperatură (°C)")
                    plt.ylabel("Frecvență cumulativă")
                    plt.tight_layout()
                    path = os.path.join('static', 'cumulativa.png')
                    plt.savefig(path, facecolor='#2E2E2E')
                    plt.close()
                    cumulativa_url = url_for('static', filename='cumulativa.png')

                if 'densitate' in selected_grafice:
                    fig = plt.figure(figsize=(6, 4), facecolor='#2E2E2E')
                    ax = fig.gca()
                    ax.set_facecolor('#2E2E2E')
                    df_col.plot(kind='kde', color='#72B7B2', linewidth=2, ax=ax)
                    plt.title(selected_column)
                    plt.xlabel("Temperatură (°C)")
                    plt.ylabel("Densitate")
                    plt.tight_layout()
                    path = os.path.join('static', 'densitate.png')
                    plt.savefig(path, facecolor='#2E2E2E')
                    plt.close()
                    densitate_url = url_for('static', filename='densitate.png')

                if 'autocorelatie' in selected_grafice:
                    fig = plt.figure(figsize=(6, 4), facecolor='#2E2E2E')
                    ax = fig.gca()
                    ax.set_facecolor('#2E2E2E')
                    autocorrelation_plot(df_col, ax=ax, color='#F58518')
                    ax.set_title(f"{selected_column}")
                    ax.grid(True, linestyle='--', alpha=0.5)
                    path = os.path.join('static', 'autocorelatie.png')
                    plt.tight_layout()
                    plt.savefig(path, facecolor='#2E2E2E')
                    plt.close()
                    autocorelatie_url = url_for('static', filename='autocorelatie.png')

    columns = df_cache.select_dtypes(include='number').columns.tolist() if df_cache is not None else []
    return render_template(
        'index.html',
        columns=columns,
        grafice=GRAFICE_DISPONIBILE,
        selected_column=selected_column,
        ref_choice=ref_choice,
        comparison_url=comparison_url,
        regression_url=regression_url,
        regression_info=regression_info,
        stats=stats,
        cumulativa_url=cumulativa_url,
        densitate_url=densitate_url,
        autocorelatie_url=autocorelatie_url,
        clasificare_text=clasificare_text
    )

if __name__ == '__main__':
    app.run(debug=True)
