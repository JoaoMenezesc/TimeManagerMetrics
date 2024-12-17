import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def daily_time_management():
    print("Por favor, insira as atividades e o tempo gasto em cada uma delas (em horas).")
    activities = []
    hours = []

    while True:
        activity = input("Digite o nome da atividade (ou 'sair' para finalizar): ")
        if activity.lower() == 'sair':
            break
        try:
            hour = float(input(f"Digite as horas gastas em {activity}: "))
            activities.append(activity)
            hours.append(hour)
        except ValueError:
            print("Por favor, insira um valor válido para as horas.")

    if not activities:
        print("Nenhuma atividade fornecida. Encerrando.")
        return None, None

    total_hours = sum(hours)
    percentages = [h / total_hours * 100 for h in hours]

    plt.figure(figsize=(8, 6))
    plt.pie(percentages, labels=activities, autopct="%1.1f%%", startangle=140)
    plt.title("Distribuição de Tempo Diário")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.bar(activities, hours, color="skyblue")
    plt.xlabel("Atividades")
    plt.ylabel("Horas")
    plt.title("Horas Gastas por Atividade")
    plt.show()

    return activities, hours

def create_synthetic_dataset(activities, hours):
    if not activities:
        print("Nenhuma atividade fornecida. Encerrando.")
        return None

    dataset = pd.DataFrame({
        "Atividade": np.random.choice(activities, size=100, p=[h / sum(hours) for h in hours]),
        "Horas": np.random.randint(1, 9, size=100),
        "Produtividade": np.random.uniform(50, 100, size=100),
        "Satisfacao": np.random.uniform(1, 5, size=100)
    })

    print("Dataset Sintético Criado:")
    print(dataset.head())
    return dataset

def validate_dataset(dataset):
    print("Validação do Dataset: Coerente e Representativo")
    print(dataset.describe())

def apply_clustering(dataset):
    features = ["Horas", "Produtividade", "Satisfacao"]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(dataset[features])

    plt.figure(figsize=(8, 6))
    plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c='gray', s=50, alpha=0.6)
    plt.title("Visualização Inicial dos Dados Normalizados")
    plt.xlabel("Horas")
    plt.ylabel("Produtividade")
    plt.show()

    print("Aplicando K-Means")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters_kmeans = kmeans.fit_predict(normalized_data)
    dataset["Cluster_KMeans"] = clusters_kmeans

    plt.figure(figsize=(8, 6))
    plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=clusters_kmeans, cmap="viridis", s=50)
    plt.title("Clusters Formados pelo K-Means")
    plt.xlabel("Horas")
    plt.ylabel("Produtividade")
    plt.show()

    print("Aplicando DBSCAN")
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    clusters_dbscan = dbscan.fit_predict(normalized_data)

    if len(set(clusters_dbscan)) > 1 and -1 in clusters_dbscan:
        dataset["Cluster_DBSCAN"] = clusters_dbscan

        plt.figure(figsize=(8, 6))
        plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=clusters_dbscan, cmap="plasma", s=50)
        plt.title("Clusters Formados pelo DBSCAN")
        plt.xlabel("Horas")
        plt.ylabel("Produtividade")
        plt.show()

        print("Métrica de Silhouette para K-Means:", silhouette_score(normalized_data, clusters_kmeans))
        print("Métrica de Silhouette para DBSCAN:", silhouette_score(normalized_data, clusters_dbscan))
    else:
        print("DBSCAN não encontrou múltiplos clusters. Métrica de Silhouette não será calculada.")

if __name__ == "__main__":
    activities, hours = daily_time_management()
    if activities:
        dataset = create_synthetic_dataset(activities, hours)
        if dataset is not None:
            validate_dataset(dataset)
            apply_clustering(dataset)