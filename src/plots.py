import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.config import PLOT_CONFIG, OUTPUT_DIR
from src.utils  import save_fig

def plot_missing_matrix(df):
    plt.clf()
    msno.matrix(df,
        color=(0.4, 0.8, 0.6)
    )
    save_fig(plt.gcf(), OUTPUT_DIR + 'missing_matrix.png')

def plot_gender_churn(df):
    cfg = PLOT_CONFIG['pie']
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type':'domain'},{'type':'domain'}]])
    colors = cfg['color_sequence']

    for i, col in enumerate(['gender','Churn'], start=1):
        labels = df[col].value_counts().index.tolist()
        values = df[col].value_counts().tolist()
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),     # ← here!
                hole=cfg['hole'],
                textinfo=cfg['textinfo'],
                textfont=dict(size=cfg['font_size'])
            ),
            1, i
        )
    fig.update_traces(
        hole=cfg['hole'],
        textinfo=cfg['textinfo'],
        textfont_size=cfg['font_size']
    )
    w = cfg.get('width', 800)
    h = cfg.get('height', 400)
    fig.update_layout(title_text='Gender & Churn', width=w, height=h)
    save_fig(fig, OUTPUT_DIR + 'gender_churn.png')

def plot_contract_distribution(df):
    cfg = PLOT_CONFIG['hist']
    fig = px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="relative",
        barnorm="percent",
        text_auto=".1%",
        title="Churn % by Contract Type",
        color_discrete_sequence=cfg['color_sequence']
    )
    fig.update_layout(
        width=cfg['width'],
        height=cfg['height'],
        bargap=cfg['bargap'],
        xaxis_title="Contract Type",
        yaxis_title="Percentage",
        legend_title="Churn"
    )
    save_fig(fig, OUTPUT_DIR + "customer_contract_distribution.png")

def plot_payment_method_distribution(df):
    labels = df['PaymentMethod'].value_counts().index.tolist()
    values = df['PaymentMethod'].value_counts().tolist()
    cfg = PLOT_CONFIG['pie']
    colors = cfg['color_sequence']

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=cfg['hole'],
            marker=dict(colors=colors),
            textinfo=cfg['textinfo'],
            textfont=dict(size=cfg['font_size'])
        )
    ])
    fig.update_layout(title_text='Payment Method Distribution',
                      width=800, height=600)
    save_fig(fig, OUTPUT_DIR + 'payment_method_dist.png')

def plot_payment_method_churn(df):
    cfg = PLOT_CONFIG['hist']
    fig = px.histogram(
        df,
        x='Churn',
        color='PaymentMethod',
        barmode='group',
        title='Churn by Payment Method',
        color_discrete_sequence=cfg['color_sequence']
    )
    fig.update_layout(width=cfg['width'], height=cfg['height'], bargap=cfg['bargap'])
    save_fig(fig, OUTPUT_DIR + 'payment_method_churn.png')

def plot_internet_gender_churn(df):
    df_counts = df.groupby(['Churn','gender','InternetService']).size().reset_index(name='count')
    cfg = PLOT_CONFIG['hist']
    fig = px.bar(
        df_counts,
        x='Churn', y='count', color='InternetService', facet_col='gender',
        title='Churn by Internet Service & Gender',
        color_discrete_sequence=cfg['color_sequence']
    )
    fig.update_layout(width=cfg['width'], height=cfg['height'], bargap=cfg['bargap'])
    save_fig(fig, OUTPUT_DIR + 'internet_gender_churn.png')

def plot_binary_churn(df, features):
    cmap = PLOT_CONFIG['bar_colors']['binary']
    cfg = PLOT_CONFIG['hist']
    for feat in features:
        fig = px.histogram(
            df, x='Churn', color=feat, barmode='group',
            title=f'Churn by {feat}',
            color_discrete_sequence=cfg['color_sequence'],
            color_discrete_map=cmap
        )
        fig.update_layout(width=cfg['width'], height=cfg['height'], bargap=cfg['bargap'])
        save_fig(fig, OUTPUT_DIR + f'{feat.lower()}_churn.png')

def plot_monthly_total_charges(df):
    for col, colors in PLOT_CONFIG['kde_colors'].items():
        plt.clf()
        sns.set_context('paper', font_scale=1.1)
        sns.kdeplot(df[col][df.Churn=='No'], fill=True, color=colors[0])
        sns.kdeplot(df[col][df.Churn=='Yes'], fill=True, color=colors[1])
        plt.legend(['Retained','Churned'], loc='upper right')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.title(f'Distribution of {col} by Churn')
        plt.tight_layout()
        save_fig(plt.gcf(), OUTPUT_DIR + f'{col.lower()}_distribution.png')

def plot_correlation(df):
    plt.clf()
    corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    save_fig(plt.gcf(), OUTPUT_DIR + 'correlation_heatmap.png')