import matplotlib
matplotlib.use('Agg')

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
cumulativa_url = None
densitate_url = None
cumulativa_info = None
densitate_info = None

plt.style.use('dark_background')

GRAFICE_DISPONIBILE = {
    'comparativa': 'Histogramă comparativă',
    'regresie': 'Regresie liniară',
    'cumulativa': 'Histogramă cumulativă',
    'densitate': 'Densitate de probabilitate (KDE)'
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
    global cumulativa_url, densitate_url, cumulativa_info, densitate_info

    columns = []
    selected_grafice = []

    if request.method == 'POST':
        # Resetăm toate valorile
        comparison_url = None
        regression_url = None
        stats = None
        regression_info = None
        cumulativa_url = None
        densitate_url = None
        cumulativa_info = None
        densitate_info = None

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

        elif 'column' in request.form and 'ref' in request.form and 'grafice' in request.form:
            selected_column = request.form['column']
            ref_choice = request.form['ref']
            selected_grafice = request.form.getlist('grafice')

            if df_cache is not None:
                df_col = df_cache[selected_column].dropna()
                mean_val = df_col.mean()
                std_val = df_col.std()
                var_val = df_col.var()
                stats = f"Media: {mean_val:.2f} °C | Deviație standard: {std_val:.2f} °C | Variație: {var_val:.2f} °C²"

                if 'comparativa' in selected_grafice:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#2E2E2E')
                    plt.subplots_adjust(left=0.08, right=0.92, wspace=0.5)
                    axes[0].set_facecolor('#2E2E2E')
                    axes[1].set_facecolor('#2E2E2E')

                    df_cache[selected_column].dropna().hist(
                        bins=30, ax=axes[0], color='deepskyblue', edgecolor='black'
                    )
                    axes[0].set_title(f'{selected_column}')
                    axes[0].set_xlabel("Temperatură (°C)")
                    axes[0].set_ylabel("Frecvență")

                    df_cache[ref_choice].dropna().hist(
                        bins=30, ax=axes[1], color='hotpink', edgecolor='black'
                    )
                    axes[1].set_title(ref_choice)
                    axes[1].set_xlabel("Temperatură (°C)")
                    axes[1].set_ylabel("Frecvență")

                    comp_path = os.path.join('static', 'comparison.png')
                    plt.savefig(comp_path, facecolor='#2E2E2E')
                    plt.close()
                    comparison_url = url_for('static', filename='comparison.png')

                if 'regresie' in selected_grafice:
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

                if 'cumulativa' in selected_grafice:
                    p25 = df_col.quantile(0.25)
                    median = df_col.median()
                    p75 = df_col.quantile(0.75)
                    min_val = df_col.min()
                    max_val = df_col.max()

                    cumulativa_info = (
                        f"<strong>Percentile:</strong><br>"
                        f"P25: {p25:.2f} °C<br>"
                        f"Mediana: {median:.2f} °C<br>"
                        f"P75: {p75:.2f} °C<br>"
                        f"Interval: [{min_val:.2f} °C – {max_val:.2f} °C]"
                    )

                    fig = plt.figure(figsize=(6, 4), facecolor='#2E2E2E')
                    ax = fig.gca()
                    ax.set_facecolor('#2E2E2E')
                    df_col.hist(bins=30, cumulative=True, density=True, color='orange', edgecolor='black')
                    plt.title(f"Histogramă cumulativă - {selected_column}")
                    plt.xlabel("Valori")
                    plt.ylabel("Frecvență cumulativă")
                    path = os.path.join('static', 'cumulativa.png')
                    plt.savefig(path, facecolor='#2E2E2E')
                    plt.close()
                    cumulativa_url = url_for('static', filename='cumulativa.png')

                if 'densitate' in selected_grafice:
                    kde = df_col.plot.kde()
                    x = kde.get_lines()[0].get_xdata()
                    y = kde.get_lines()[0].get_ydata()
                    peak_x = x[np.argmax(y)]

                    densitate_info = (
                        f"<strong>Media:</strong> {df_col.mean():.2f} °C<br>"
                        f"<strong>Densitate maximă estimată la:</strong> ~{peak_x:.2f} °C"
                    )

                    fig = plt.figure(figsize=(6, 4), facecolor='#2E2E2E')
                    ax = fig.gca()
                    ax.set_facecolor('#2E2E2E')
                    df_col.plot(kind='kde', color='lime', linewidth=2, ax=ax)
                    plt.title(f"Densitate de probabilitate (KDE) - {selected_column}")
                    plt.xlabel("Valori")
                    plt.ylabel("Densitate")
                    path = os.path.join('static', 'densitate.png')
                    plt.savefig(path, facecolor='#2E2E2E')
                    plt.close()
                    densitate_url = url_for('static', filename='densitate.png')

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
        cumulativa_info=cumulativa_info,
        densitate_url=densitate_url,
        densitate_info=densitate_info
    )

if __name__ == '__main__':
    app.run(debug=True)
