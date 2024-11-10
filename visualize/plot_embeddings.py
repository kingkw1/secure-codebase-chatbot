import faiss
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# File paths
embedding_index_path = os.path.join(os.path.dirname(__file__), '..', 'sample_metadata', 'embedding_index.faiss')
metadata_path = os.path.join(os.path.dirname(__file__), '..', 'sample_metadata', 'test_metadata.json')

# Load embeddings from FAISS index
def load_faiss(file_path):
    index = faiss.read_index(file_path)
    embeddings = index.reconstruct_n(0, index.ntotal)  # Retrieve all embeddings in the index
    return embeddings, index

# Load metadata (optional, to label points if needed)
def load_metadata(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Sampling strategy for small and large repositories
def get_embedding_sample(index, sample_size=1000):
    num_embeddings = index.ntotal
    
    if num_embeddings <= sample_size:
        # For smaller datasets, use all embeddings
        _, distances = index.search(index.reconstruct_n(0, num_embeddings), num_embeddings)
    else:
        # For larger datasets, sample a random subset of embeddings
        random_indices = np.random.choice(num_embeddings, sample_size, replace=False)
        sample_embeddings = np.array([index.reconstruct(i) for i in random_indices])
        _, distances = index.search(sample_embeddings, 1)
    
    return distances.ravel()  # Flatten distances for easier analysis

# Plot embeddings in 2D
def plot_embeddings(embeddings, metadata=None):
    # Use TSNE to reduce dimensionality to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    x = reduced_embeddings[:, 0]
    y = reduced_embeddings[:, 1]
    plt.scatter(x, y, s=10, alpha=0.7)

    # Optional: Label each point using metadata (e.g., filenames or function names)
    if metadata:
        for i, (x_coord, y_coord) in enumerate(zip(x, y)):
            # Customize labels according to your metadata structure
            label = metadata['files'][i]['file_path'] if i < len(metadata['files']) else f"Point {i}"
            plt.text(x_coord, y_coord, label, fontsize=6, alpha=0.6)

    plt.title("2D Visualization of Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

    plt.figure(figsize=(10, 8))


# Function to plot the distribution of distances to a query embedding
def plot_distance_distribution(index, k=5):
    # Randomly select a query embedding from the FAISS index
    query_embedding_id = np.random.randint(0, index.ntotal)
    query_embedding = index.reconstruct(query_embedding_id)

    # Search for the k nearest neighbors
    distances, _ = index.search(query_embedding.reshape(1, -1), k)

    # Plot histogram of distances
    plt.hist(distances[0], bins=100, color='skyblue', edgecolor='black')
    plt.title("Distribution of Distances to Query Embedding")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()

    # Calculate mean and standard deviation
    mean_distance = distances[0].mean()
    std_distance = distances[0].std()
    print("Mean distance:", mean_distance)
    print("Standard deviation:", std_distance)
    return mean_distance, std_distance

# Plot histogram of distances
def plot_distance_histogram(distances):
    plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Distances in Embedding Space")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.show()

    # Calculate and print statistics
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    print("Mean distance:", mean_distance)
    print("Standard deviation:", std_distance)
    
    return mean_distance, std_distance


if __name__ == "__main__":
    # Load the embeddings and metadata
    embeddings, index = load_faiss(embedding_index_path)
    metadata = load_metadata(metadata_path)  # Optional, for labeling

    # Plot the embeddings
    # plot_embeddings(embeddings, metadata)

    distances = get_embedding_sample(index)
    mean_distance, std_distance = plot_distance_histogram(distances)

    # Plot the distribution of distances
    # mean_distance, std_distance = plot_distance_distribution(index)
    
    # Print suggested threshold range based on statistics
    print("Suggested distance threshold range: ", mean_distance - std_distance, "to", mean_distance + std_distance)