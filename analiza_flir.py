import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from matplotlib.font_manager import FontProperties

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

df_cache = None
csv_filename = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global df_cache, csv_filename

    selected_column = None
    selected_factor = None
    temperature_columns = []
    factor_columns = []
    plot_url = None
    regression_info = None
    group_stats = None
    distributie_url = None
    regression_data = None

    def clasifica(ser):
        bins = [0, 36.0, 37.5, float('inf')]
        etichete = ['Subnormală', 'Normală', 'Ridicată']
        categorii = pd.cut(ser, bins=bins, labels=etichete, right=False)
        return categorii.value_counts().reindex(etichete, fill_value=0).to_dict()

    def generate_pie_chart(valori, filename):
        fig, ax = plt.subplots(figsize=(16, 5), facecolor='#1E1E2E')  # fundal exterior
        ax.set_facecolor('#1E1E2E')  # fundal interior (același ca în al doilea grafic)

        etichete = ['Subnormală', 'Normală', 'Ridicată']
        culori = ['#4C78A8', '#56B870', '#F58518']
        total = sum(valori)
        procente = [f"{label}: {val} ({val / total:.1%})" for label, val in zip(etichete, valori)]

        wedges, _ = ax.pie(
            valori,
            colors=culori,
            startangle=140,
            wedgeprops={'edgecolor': 'black'},
            textprops={'color': 'white'}
        )

        ax.legend(
            wedges, procente, title="Categorii", loc="center left",
            bbox_to_anchor=(1, 0.5),
            facecolor='#1E1E2E',  # fundalul legendei
            labelcolor='white',
            title_fontsize='10',
            fontsize='9'
        )

        path = os.path.join('static', filename)
        plt.savefig(path, facecolor='#1E1E2E', bbox_inches='tight')  # salvează cu fundalul dorit
        plt.close()
        return url_for('static', filename=filename)

    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                csv_filename = filename
                try:
                    df = pd.read_csv(filepath, header=2)
                    df.columns = df.columns.str.strip()
                    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
                    df = df.dropna(how='all')
                    df_cache = df
                except Exception as e:
                    return render_template('index.html', error=f"Eroare: {e}")

        selected_column = request.form.get('column')
        selected_factor = request.form.get('factor')

        if df_cache is not None and selected_column and selected_factor:
            df = df_cache[[selected_column, selected_factor]].dropna()

            if df[selected_factor].dtype == 'object' or df[selected_factor].nunique() <= 10:
                if selected_factor == 'Age':
                    age_order = ['<18', '18-20', '21-25', '26-30', '31-40', '41-50', '51-60', '>60']
                    order = [x for x in age_order if x in df[selected_factor].unique()]
                    df[selected_factor] = pd.Categorical(df[selected_factor], categories=order, ordered=True)
                else:
                    order = sorted(df[selected_factor].unique())

                df[selected_column] = pd.to_numeric(df[selected_column], errors='coerce')
                df = df.dropna(subset=[selected_column, selected_factor])

                plt.style.use("dark_background")
                fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1E1E2E')
                ax.set_facecolor('#1E1E2E')

                palette_profesionala = ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#EECA3B', '#B279A2',
                                        '#FF9DA6']

                sns.violinplot(
                    data=df,
                    x=selected_factor,
                    y=selected_column,
                    palette=palette_profesionala,
                    ax=ax,
                    inner="point"
                )

                ax.set_title(f'Temperatura {selected_column} în funcție de {selected_factor}', color='white',
                             fontsize=14, pad=15)
                ax.set_xlabel(selected_factor, color='white', fontsize=12)
                ax.set_ylabel(f'{selected_column} (°C)', color='white', fontsize=12)
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color('#888')

                plt.tight_layout()
                path = os.path.join('static', 'plot.png')
                plt.savefig(path, dpi=150, facecolor='#1E1E2E', bbox_inches='tight')
                plt.close()
                plot_url = url_for('static', filename='plot.png')

                # statistici
                group_stats = df.groupby(selected_factor)[selected_column].agg(['mean', 'std', 'count']).round(
                    2).reset_index()

                # KDE numai dacă factorul are mai multe valori unice și este coloană validă
                if selected_factor in df.columns and df[selected_factor].nunique() > 1 and selected_column in df.columns and selected_factor != 'Age':
                    plt.style.use("dark_background")
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1E1E2E')
                    ax.set_facecolor('#1E1E2E')

                    culori_kde = ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#EECA3B', '#B279A2',
                                  '#FF9DA6']

                    sns.kdeplot(
                        data=df,
                        x=selected_column,
                        hue=selected_factor,
                        fill=True,
                        alpha=0.4,
                        palette=culori_kde,
                        ax=ax,
                        legend=True
                    )

                    legend = ax.get_legend()
                    if legend:
                        legend.get_frame().set_facecolor('#1E1E2E')  # ✅ fundal legendă
                        legend.get_frame().set_edgecolor('#444')  # ✅ contur
                        legend.get_frame().set_linewidth(1.0)
                        title_font = FontProperties(size=10, weight='bold')
                        legend.set_title(selected_factor, prop=title_font)
                        legend.get_title().set_color('white')
                        for text in legend.get_texts():
                            text.set_color("white")

                        plt.tight_layout()
                        distrib_path = os.path.join('static', 'distributie.png')
                        plt.savefig(distrib_path, facecolor='#1E1E2E', bbox_inches='tight')
                        plt.close()
                        distributie_url = url_for('static', filename='distributie.png')




            else:

                from scipy.stats import linregress
                slope, intercept, r_value, p_value, _ = linregress(df[selected_factor], df[selected_column])
                regression_info = {
                    "ecuatie": f'y = {slope:.2f}x + {intercept:.2f}',
                    "r": round(r_value, 2),
                    "p": round(p_value, 4)

                }

                # === GRAFIC DARK PERSONALIZAT ===
                plt.style.use("dark_background")
                fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1E1E2E')
                ax.set_facecolor('#1E1E2E')
                sns.regplot(
                    data=df,
                    x=selected_factor,
                    y=selected_column,
                    ax=ax,
                    scatter_kws={'color': '#62B6CB', 'alpha': 0.6},
                    line_kws={'color': '#F4A261', 'linewidth': 2.5}

                )

                ax.set_title(f'Temperatura {selected_column} în funcție de {selected_factor}', color='white',
                             fontsize=14, pad=15)

                ax.set_xlabel(selected_factor, color='white', fontsize=12)
                ax.set_ylabel(f'{selected_column} (°C)', color='white', fontsize=12)
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color('#888')
                path = os.path.join('static', 'plot.png')
                plt.tight_layout()
                plt.savefig(path, dpi=150)
                plt.close()
                plot_url = url_for('static', filename='plot.png')


        # === REGRESIE MULTIPLA ===
        if df_cache is not None and selected_column:
            factors = ["Humidity", "T_atm", "Age", "Distance", "Gender", "Ethnicity", "Cosmetics"]
            available = [col for col in factors if col in df_cache.columns and col != selected_column]
            df_model = df_cache[[selected_column] + available].dropna()

            if not df_model.empty:
                X = df_model[available]
                y = df_model[selected_column]
                cat_cols = X.select_dtypes(include='object').columns.tolist()
                num_cols = X.select_dtypes(include='number').columns.tolist()

                pipeline = Pipeline(steps=[
                    ('preprocessor', ColumnTransformer([
                        ('num', 'passthrough', num_cols),
                        ('cat', OneHotEncoder(drop='first'), cat_cols)
                    ])),
                    ('regressor', LinearRegression())
                ])
                pipeline.fit(X, y)
                y_pred = pipeline.predict(X)
                r2 = r2_score(y, y_pred)

                regressor = pipeline.named_steps['regressor']
                encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
                encoded_features = list(encoder.get_feature_names_out(cat_cols))
                all_features = num_cols + encoded_features
                coef_list = list(zip(all_features, regressor.coef_))
                coef_list.sort(key=lambda x: abs(x[1]), reverse=True)
                regression_table = [{"factor": f, "coef": round(c, 3)} for f, c in coef_list]

                # Interpretare automată
                if r2 < 0.2:
                    interpretare_text = ("🔴 Rezultatul indică un model slab. Variabilele analizate par să aibă o influență redusă asupra temperaturii."
                                         "Este posibil ca alți factori (fiziologici, emoționali sau tehnici) să joace un rol mai important.")
                elif r2 < 0.6:
                    interpretare_text = "🟡 Modelul are o putere explicativă moderată. Variabilele analizate explică parțial variația temperaturii, dar influențe externe pot fi prezente."
                else:
                    interpretare_text = "🟢 Modelul este puternic. Variabilele analizate explică o proporție semnificativă din variația temperaturii."

                regression_data = {
                    "r2": round(r2, 3),
                    "coeficients": regression_table,
                    "interpretare": interpretare_text
                }

    clasificare_text = None
    if df_cache is not None and selected_column and 'Gender' in df_cache.columns:
        df_filtered = df_cache[[selected_column, 'Gender']].dropna()
        if not df_filtered.empty:
            femei = df_filtered[df_filtered['Gender'].str.lower() == 'female'][selected_column].dropna()
            barbati = df_filtered[df_filtered['Gender'].str.lower() == 'male'][selected_column].dropna()
            if not femei.empty and not barbati.empty:
                cl_femei = clasifica(femei)
                cl_barbati = clasifica(barbati)
                femei_url = generate_pie_chart(list(cl_femei.values()), "clasificare_femei.png")
                barbati_url = generate_pie_chart(list(cl_barbati.values()), "clasificare_barbati.png")
                clasificare_text = {
                    'femei_url': femei_url,
                    'barbati_url': barbati_url,
                    'femei_stats': cl_femei,
                    'barbati_stats': cl_barbati,
                    'femei_count': len(femei),
                    'barbati_count': len(barbati)
                }

    if df_cache is not None:
        temperature_columns = df_cache.select_dtypes(include='number').columns.tolist()
        candidate_factors = ['Humidity', 'T_atm', 'Age', 'Gender', 'Ethnicity', 'Distance', 'Cosmetics']
        factor_columns = [col for col in candidate_factors if col in df_cache.columns]

    return render_template(
        'index.html',
        columns=temperature_columns,
        factor_columns=factor_columns,
        selected_column=selected_column,
        selected_factor=selected_factor,
        uploaded_filename=csv_filename,
        plot_url=plot_url,
        group_stats=group_stats,
        regression_info=regression_info,
        distributie_url=distributie_url,
        regression_data=regression_data,
        clasificare_text=clasificare_text
    )


if __name__ == '__main__':
    app.run(debug=True)