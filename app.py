import os
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))  
    
    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index')) 
    
    # reading the CSV and performing clustering
    df = pd.read_csv(file)
    
    X = df[['MURDER', 'THEFT']]
    
    # Determine the number of clusters
    k = 3
    
    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=0)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Find centroids of clusters
    centroids = kmeans.cluster_centers_

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green']
    for i in range(k):
        plt.scatter(X[df['Cluster'] == i]['MURDER'], X[df['Cluster'] == i]['THEFT'], 
                    color=colors[i], label=f'Cluster {i}', marker='o')
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, c='yellow', label='Centroids', marker='o')
    
    # Title and labels
    plt.title('K-means Clustering of Crime Data')
    plt.xlabel('Number of Murders')
    plt.ylabel('Number of Thefts')
    plt.legend()
    plt.grid()

    static_dir = os.path.join(app.static_folder) 
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    plot_path = os.path.join(static_dir, 'plot.png')
    plt.savefig(plot_path)
    plt.close() 

    # Group cities by cluster
    city_data = df[['DISTRICT','YEAR', 'MURDER', 'THEFT', 'Cluster']]
    grouped_data = city_data.groupby('Cluster').apply(lambda x: x[['DISTRICT','YEAR', 'MURDER', 'THEFT']].to_dict('records')).to_dict()
    
    # Prepare centroids for display
    centroid_list = [{'MURDER': centroids[i][0], 'THEFT': centroids[i][1]} for i in range(k)]
   # Calculate average values for each cluster
    cluster_summary = df.groupby('Cluster').agg({
        'MURDER': 'mean',
        'THEFT': 'mean',
        'DISTRICT': 'count'
    }).rename(columns={'DISTRICT': 'City Count'}).reset_index()

    return render_template('index.html', plot_url='static/plot.png', grouped_data=grouped_data, 
                           centroids=centroid_list, cluster_summary=cluster_summary)

if __name__ == '__main__':
    app.run(debug=True)