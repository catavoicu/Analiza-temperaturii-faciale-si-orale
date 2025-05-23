import os
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
                else:
                    order = sorted(df[selected_factor].unique())

                plt.figure(figsize=(10, 6))
                sns.set_theme(style="darkgrid")
                sns.violinplot(data=df, x=selected_factor, y=selected_column, palette='Set2', inner='box', order=order)
                sns.stripplot(data=df, x=selected_factor, y=selected_column, color='white', size=3, jitter=0.3, alpha=0.3, order=order)
                plt.title(f'Temperatura {selected_column} în funcție de {selected_factor}', fontsize=14)
                plt.xlabel(selected_factor)
                plt.ylabel(f'{selected_column} (°C)')
                plt.xticks(rotation=30)
                plt.tight_layout()
                path = os.path.join('static', 'plot.png')
                plt.savefig(path)
                plt.close()
                plot_url = url_for('static', filename='plot.png')

                group_stats = df.groupby(selected_factor)[selected_column].agg(['mean', 'std', 'count']).round(2).reset_index()

                if df[selected_factor].nunique() <= 5:
                    plt.figure(figsize=(10, 5))
                    sns.kdeplot(data=df, x=selected_column, hue=selected_factor, fill=True, alpha=0.4)
                    plt.title(f'Distribuții KDE pentru {selected_column} după {selected_factor}')
                    plt.xlabel(f'{selected_column} (°C)')
                    plt.ylabel('Densitate')
                    plt.tight_layout()
                    distrib_path = os.path.join('static', 'distributie.png')
                    plt.savefig(distrib_path)
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

                plt.figure(figsize=(9, 6))
                sns.set_theme(style="darkgrid")
                sns.regplot(data=df, x=selected_factor, y=selected_column, line_kws={'color': 'orange'}, scatter_kws={'alpha': 0.4})
                plt.title(f'Temperatura {selected_column} în funcție de {selected_factor}', fontsize=14)
                plt.xlabel(selected_factor)
                plt.ylabel(f'{selected_column} (°C)')
                plt.tight_layout()
                path = os.path.join('static', 'plot.png')
                plt.savefig(path)
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
                    interpretare_text = "🔴 Rezultatul indică un model slab. Variabilele analizate par să aibă o influență redusă asupra temperaturii. Este posibil ca alți factori (fiziologici, emoționali sau tehnici) să joace un rol mai important."
                elif r2 < 0.6:
                    interpretare_text = "🟡 Modelul are o putere explicativă moderată. Variabilele analizate explică parțial variația temperaturii, dar influențe externe pot fi prezente."
                else:
                    interpretare_text = "🟢 Modelul este puternic. Variabilele analizate explică o proporție semnificativă din variația temperaturii."

                regression_data = {
                    "r2": round(r2, 3),
                    "coeficients": regression_table,
                    "interpretare": interpretare_text
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
        regression_data=regression_data
    )


if __name__ == '__main__':
    app.run(debug=True)