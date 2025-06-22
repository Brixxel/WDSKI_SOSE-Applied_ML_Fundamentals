import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Generate a spiral dataset with two classes
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

########################################################
########################################################
#####              Plotting functions              #####
########################################################
########################################################

# Plot just the dataset
def plot_data(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolor='k', marker='x')
    plt.title("DataSet Plot")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot Dataset with decision boundary
def plot_data_and_decision_boundary(model, X, y, ax=None, title="Decision Boundary"):
    # Create axis if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        Z = model(grid_tensor)
        _, predicted = torch.max(Z, 1)
    
    Z = predicted.numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.Spectral)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolor='g', marker='x')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Show plot only if we created a new figure
    if ax is None:
        plt.show()
    
    return ax

########################################################
#### Plotting the Network Architekture & Parameters ####
########################################################

def draw_mlp_network(model, architecture_dict=None, show_weights=True, figsize=(12, 6), ax=None):

    """
    Visualisiert ein MLP als schematische Darstellung – mit optionaler Gewichtsanzeige.

    Args:
        model: Ein PyTorch-Modell (nn.Sequential oder benutzerdefinierte Subklasse)
        architecture_dict: Optionales Dictionary mit input_dim, hidden_layers, output_dim
        show_weights: Bool – falls True, werden die Gewichte farbcodiert gezeichnet
        figsize: Tuple – Größe des Diagramms
    """

    # Architektur ableiten
    if architecture_dict is not None:
        layer_sizes = [architecture_dict['input_dim']] + architecture_dict['hidden_layers'] + [architecture_dict['output_dim']]
    else:
        # Versuche automatisch die Layer-Größen aus dem Modell abzuleiten
        layer_sizes = []
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                if not layer_sizes:
                    layer_sizes.append(layer.in_features)
                layer_sizes.append(layer.out_features)

    # Parameter extrahieren
    weight_tensors = []
    bias_tensors = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_tensors.append(param.data.clone().detach())
        elif 'bias' in name:
            bias_tensors.append(param.data.clone().detach())

    # Zeichnung vorbereiten
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    positions = []
    for i, n_neurons in enumerate(layer_sizes):
        x = i / (len(layer_sizes) - 1)
        y_positions = [(j + 1) / (n_neurons + 1) for j in range(n_neurons)]
        layer_pos = [(x, y) for y in y_positions]
        positions.append(layer_pos)

        for j, (px, py) in enumerate(layer_pos):
            bias_val = 0
            if i > 0 and bias_tensors and len(bias_tensors) >= i:
                bias_val = bias_tensors[i - 1][j].item()
            circle_color = 'black'
            if bias_val != 0:
                circle_color = plt.cm.coolwarm((bias_val + 1) / 2)
            circle = plt.Circle((px, py), 0.02, color=circle_color)
            ax.add_patch(circle)

    # Verbindungen einzeichnen
    for l in range(len(layer_sizes) - 1):
        weights = weight_tensors[l].cpu().numpy() if l < len(weight_tensors) else np.zeros((layer_sizes[l+1], layer_sizes[l]))
        vmin = np.min(np.abs(weights)) if show_weights else 0
        vmax = np.max(np.abs(weights)) if show_weights else 1

        for i, dst in enumerate(positions[l + 1]):
            for j, src in enumerate(positions[l]):
                w = weights[i, j] if show_weights else 0
                norm_w = np.clip((abs(w) - vmin) / (vmax - vmin + 1e-6), 0, 1) if show_weights else 0.5
                color = plt.cm.seismic((w + 1) / 2) if show_weights else 'gray'
                ax.plot(
                    [src[0], dst[0]], [src[1], dst[1]],
                    color=color,
                    alpha=0.9,
                    linewidth=0.5 + 2 * norm_w
                )

    title_suffix = "with Weights" if show_weights else "without Weights"
    plt.title(f"Neural Network Architecture ({title_suffix})", fontsize=14)
    plt.tight_layout()
    plt.show()


########################################################
#### Plotting Training Process ####
########################################################

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
    ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.Spectral)  
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Spectral, s=40, marker='x')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

def plot_training_progress(iterations, X, y, losses, accs, model_snapshots, architectures, plot_interval):
    for i, model in enumerate(model_snapshots):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        plot_data_and_decision_boundary(model, X, y, ax=ax1)
        ax1.set_title(f"Decision Boundary (Step {i*plot_interval})")
        
        draw_mlp_network(model, architecture_dict=architectures[0]['model_params'], show_weights=True)
        plt.show()


#######
#### Comparing Modells and their Training Process
#######

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Definiere eine anpassbare Modellklasse für den erweiterten Parametervergleich
class FlexibleNeuralNetwork(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=[10, 10], output_dim=2, activation=nn.ReLU()):

        super(FlexibleNeuralNetwork, self).__init__()
        
        # Speichere Parameter für Referenz
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation
        
        # Erstelle die Schichten basierend auf den Parametern
        layers = []
        
        # Füge erste Schicht hinzu (input -> erste versteckte Schicht)
        prev_dim = input_dim
        
        # Füge versteckte Schichten hinzu
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation)
            prev_dim = h_dim
        
        # Füge die Ausgabeschicht hinzu
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=1))
        
        # Erstelle das sequenzielle Modell
        self.model = nn.Sequential(*layers)
        
        # Initialisiere die Gewichte
        self.init_weights()
    
    def init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
    
        return self.model(x)
    
    def __str__(self):
    
        return f"FlexibleNN(input={self.input_dim}, hidden={self.hidden_layers}, output={self.output_dim})"

# Erweiterte Trainingsfunktion für den flexiblen Parametervergleich
def train_flexible_model(model_params, training_params, dataset):

    # Erstelle ein neues Modell mit den gegebenen Parametern
    model = FlexibleNeuralNetwork(
        input_dim=model_params.get('input_dim', 2),
        hidden_layers=model_params.get('hidden_layers', [10, 10]),
        output_dim=model_params.get('output_dim', 2),
        activation=model_params.get('activation', nn.ReLU())
    )
    
    # Bereite Trainingsdaten vor
    batch_size = training_params.get('batch_size', 32)
    local_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialisiere Verlustfunktion und Optimierer
    loss_fn = training_params.get('loss_fn', nn.CrossEntropyLoss)
    criterion = loss_fn()
    optimizer = training_params.get('optimizer_fn', optim.Adam)(
        model.parameters(), 
        lr=training_params.get('learning_rate', 0.01)
    )
    
    # Training-Historie
    history = {
        'train_loss': [],
        'train_acc': []
    }
    
    # Training durchführen
    num_epochs = training_params.get('num_epochs', 100)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        
        for batch_X, batch_y in local_dataloader:
            y_pred = model(batch_X)

            if isinstance(criterion, nn.MSELoss):
                # Für MSE: Konvertiere Klassenlabels zu One-Hot-Vektoren
                y_one_hot = torch.zeros(batch_y.size(0), model_params.get('output_dim', 2))
                y_one_hot.scatter_(1, batch_y.unsqueeze(1), 1.0)
                loss = criterion(y_pred, y_one_hot)
            else:
                # Für CrossEntropy und andere: Verwende die Labels direkt
                loss = criterion(y_pred, batch_y)
                            
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                predictions = torch.argmax(y_pred, dim=1)
                correct += (predictions == batch_y).sum().item()
        
        # Berechne durchschnittlichen Verlust und Genauigkeit
        train_loss = epoch_loss / len(local_dataloader)
        accuracy = 100 * correct / len(dataset)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(accuracy)
        
        # Fortschritt ausgeben
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return model, history

# Funktion zum Vergleichen einer bestimmten Parametergruppe
def compare_parameter_group(parameter_group, dataset,  default_model_params={}, default_training_params={}):
    
    results = []
    
    # Standardwerte festlegen
    std_model_params = {
        'input_dim': 2,
        'hidden_layers': [10, 10],
        'output_dim': 2,
        'activation': nn.ReLU()
    }
    std_model_params.update(default_model_params)
    
    std_training_params = {
        'optimizer_fn': optim.Adam,
        'learning_rate': 0.01,
        'batch_size': 32,
        'loss_fn': nn.CrossEntropyLoss,
        'num_epochs': 200
    }
    std_training_params.update(default_training_params)
    
    # Für jedes Parameter-Set in der Gruppe
    for params in parameter_group:
        print(f"\nTraining mit {params['name']}...")
        
        # Modell- und Trainingsparameter zusammenführen
        model_params = std_model_params.copy()
        training_params = std_training_params.copy()
        
        if 'model_params' in params:
            model_params.update(params['model_params'])
        
        if 'training_params' in params:
            training_params.update(params['training_params'])
        
        # Trainiere das Modell mit den aktuellen Parametern
        trained_model, history = train_flexible_model(model_params, training_params, dataset)
        
        # Speichere die Ergebnisse
        res = {
            'params': params,
            'model': trained_model,
            'history': history
        }
        results.append(res)
        
        print(f"Training für {params['name']} abgeschlossen! Finale Genauigkeit: {history['train_acc'][-1]:.2f}%")
    
    return results


# Funktion zum Visualisieren der Ergebnisse
def visualize_results(results, X, y, title="Parametervergleich"):

    n = len(results)
    if n == 0:
        print("Keine Ergebnisse zum Visualisieren.")
        return
    
    # Visualisiere die Trainingskurven
    plt.figure(figsize=(15, 10))
    
    # Plot Loss
    plt.subplot(2, 1, 1)
    for res in results:
        plt.plot(res['history']['train_loss'], label=res['params']['name'])
    plt.title(f'{title}: Vergleich des Trainingsverlusts')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(2, 1, 2)
    for res in results:
        plt.plot(res['history']['train_acc'], label=res['params']['name'])
    plt.title(f'{title}: Vergleich der Trainingsgenauigkeit')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Visualisiere nur die finalen Decision Boundaries in einem 2x2 Grid
    # Unabhängig von der Anzahl der Modelle verwenden wir ein 2x2 Layout für Konsistenz
    rows, cols = 2, 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten()  # Flatten für einfacheren Zugriff
    
    # Überschrift für die gesamte Grafik
    fig.suptitle(f'Decision Boundary: {title}', fontsize=16)
    
    # Für jedes Ergebnis eine Entscheidungsgrenze plotten
    for i, res in enumerate(results):
        if i < rows * cols:  # Nur so viele plotten, wie in das Grid passen
            ax = axes[i]
            plot_data_and_decision_boundary(res['model'], X, y, ax=ax, title=f"Decision Boundary: {res['params']['name']}")
    
    # Verstecke nicht benötigte Subplots
    for i in range(len(results), rows * cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Platz für den Haupttitel
    plt.show()

