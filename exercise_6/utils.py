import numpy as np
import matplotlib.pyplot as plt
import torch


def make_spiral(n_samples=100):
    np.random.seed(42)

    # Class 0
    t = 0.75 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
    x1 = t * np.cos(t)
    x2 = t * np.sin(t)
    y = np.zeros_like(t)

    # Class 1
    t = 0.75 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
    x1 = np.hstack([-x1, t * np.cos(t)])
    x2 = np.hstack([-x2, t * np.sin(t)])
    y = np.hstack([y, np.ones_like(t)])

    # Concatenate data points for both classes
    X = np.concatenate((x1, x2))
    # Add some noise
    X += 0.50 * np.random.randn(2, 2 * n_samples)

    return X.T, y[0]

def generate_spiral_dataset(n_samples_per_class=300, dimensions=2, n_classes=3, noise=0.2, seed=None):
    """
    Generiert ein spiralförmiges Dataset mit n_classes, wobei jede Klasse n_samples_per_class Datenpunkte enthält.
    
    :param n_samples_per_class: Anzahl der Punkte pro Klasse
    :param dimensions: Dimensionalität der Daten (in diesem Fall fest auf 2 für 2D-Punkte)
    :param n_classes: Anzahl der Klassen
    :param noise: Rausch-Level für die Spiralstruktur
    :param seed: Zufallssamen für die Reproduzierbarkeit
    
    :return: Tuple bestehend aus Feature-Matrix X und Label-Vektor y
    """
    if seed is not None:
        np.random.seed(seed)
    
    X = np.zeros((n_samples_per_class * n_classes, dimensions))
    y = np.zeros(n_samples_per_class * n_classes, dtype='uint8')
    
    for j in range(n_classes):
        ix = range(n_samples_per_class * j, n_samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, n_samples_per_class)  # Radius
        t = np.linspace(j * 4, (j + 1) * 4, n_samples_per_class) + np.random.randn(n_samples_per_class) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    
    return X.T, y

# Plot dataset
def plot_data(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='purple', label='Class 0', marker='x')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='orange', label='Class 1', marker='x')
    plt.title("DataSet Plot")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        Z = model(grid_tensor)
        _, predicted = torch.max(Z, 1)
    Z = predicted.numpy().reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolor='k', marker='x')
    plt.title("Decision Boundary")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def plot_decision_boundary_roundwise(model, X, y, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        Z = model(grid_tensor)
        _, predicted_classes = torch.max(Z, 1)
    
    Z = predicted_classes.numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.5, cmap='plasma')  
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Spectral, s=40, marker='x')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

# Plot training process
def plot_training_progress(iterations, X, y, losses, accs, model):
    n_steps = len(losses)
    fig, axes = plt.subplots(3, n_steps//2, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        if i >= n_steps:
            break
        ax.plot(range(iterations), losses[i], label='Loss')
        ax.plot(range(iterations), accs[i], label='Accuracy')
        ax.set_title(f'Training Step {i*2+1}')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Metrics')
        ax.legend(loc='upper right')
        
        # Plot decision boundary
        plot_decision_boundary(model, X, y)
    plt.tight_layout()
    plt.show()