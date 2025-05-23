# Facial and Oral Temperature Analysis

## Project Overview

This project is an interactive web application developed in **Python** using **Flask**, which analyzes facial and oral temperature data from a CSV file. The user can generate statistical charts, apply linear regression, analyze distributions and correlations, and automatically classify temperature values.

## Key Features

- User-friendly web interface 
- Upload and process CSV files containing experimental data  
- Interactive selection of chart types:
  - Histogram
  - KDE (Kernel Density Estimation)
  - Linear regression representation
  - Temperature classification: subnormal, normal, high
- Display of detailed row information and prediction of oral temperature
- Multi-selection support and dynamic option retrieval without page reload

## Project Structure

The project includes the following files and folders:

- `app.py` – Main Flask backend source code
- `templates/` – HTML (Jinja2) templates:
- `index.html` – Main interface
- `static/` – Static resources:
- `style.css` – Application styling
- `scripts.js` – JavaScript scripts for dynamic selections
- `uploads/` – Temporary folder for uploaded CSV files 

## Usage Instructions

### 1. Install Dependencies

Make sure **Python 3.x** is installed, then run:

```bash
pip install flask pandas matplotlib seaborn scikit-learn
```

### 2. **Run the Application**

```bash
python app.py
```
Open the application in your browser at: http://localhost:5000

### 3. **How to Play**

- Upload a CSV file containing temperature data.
- Select the desired chart types and click Generate.
- For classification, select a temperature column and click Classify.
- You can select a specific row to view oral temperature prediction.

## Screenshots


 ![Image](https://github.com/catavoicu/Analiza-temperaturii-faciale-si-orale/blob/d693f3fd90031d45d61008c0166499f680a61daa/main_page.png)
 

 ![Image](https://github.com/catavoicu/Analiza-temperaturii-faciale-si-orale/blob/master/rezultat_analiza.png)


 ![Image](https://github.com/catavoicu/Analiza-temperaturii-faciale-si-orale/blob/master/regresie_multipla.png)

 
![Image](https://github.com/catavoicu/Analiza-temperaturii-faciale-si-orale/blob/327a7555ac2f1d021582252ab62aa6fec854d75a/statistici_grup.png)


![Image](https://github.com/catavoicu/Analiza-temperaturii-faciale-si-orale/blob/327a7555ac2f1d021582252ab62aa6fec854d75a/clasificare_temperaturi.png)

 
## Technologies Used

**Backend:** Flask  
**Frontend:** HTML, CSS, JavaScript (Bootstrap)  

**Python Libraries:**
- `pandas` – data manipulation  
- `matplotlib`, `seaborn` – chart generation  
- `sklearn.linear_model` – linear regression


## Authors
- **Names:** Catalin Voicu & Victor Enache
- **Emails:** catavoicu01@gmail.com & enachevictor887@gmail.com
- Project developed as part of the course: Decision and Estimation in Information Processing.
- **University:** Faculty of Electronics, Telecommunications and Information Technology, Polytechnic University of Bucharest.

