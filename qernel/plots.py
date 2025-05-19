import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_qubit(states, labels=None, title="Bloch Sphere"):
    """
    Plot qubit states on the Bloch sphere.
    
    Parameters:
    -----------
    states : list of numpy.ndarray
        List of qubit states in vector representation [α, β]
        where |ψ⟩ = α|0⟩ + β|1⟩
    labels : list of str, optional
        Labels for each state point
    title : str, optional
        Title of the plot
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the sphere
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightgray', alpha=0.3, rstride=4, cstride=4)
    
    # Plot the axes
    ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.3)
    
    # Add axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Convert states to Bloch sphere coordinates
    for i, state in enumerate(states):
        # Normalize the state
        state_norm = np.linalg.norm(state)
        if state_norm == 0: # Avoid division by zero for zero vector
            print(f"Warning: State {i} is a zero vector and cannot be plotted.")
            continue
        state = state / state_norm
        
        # Calculate Bloch sphere coordinates
        # For state |ψ⟩ = α|0⟩ + β|1⟩
        # x_pt = 2Re(α*β_conj)
        # y_pt = 2Im(α*β_conj)
        # z_pt = |α|² - |β|²
        alpha, beta = state
        x_pt = 2 * np.real(alpha * np.conj(beta))
        y_pt = 2 * np.imag(alpha * np.conj(beta))
        z_pt = np.abs(alpha)**2 - np.abs(beta)**2
        
        # Plot the point
        ax.scatter([x_pt], [y_pt], [z_pt], s=100, depthshade=True)
        
        # Line to XY plane (projection at z=-1, assuming sphere bottom)
        ax.plot([x_pt, x_pt], [y_pt, y_pt], [z_pt, -1], color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        # Line to XZ plane (projection at y=-1, assuming sphere back)
        ax.plot([x_pt, x_pt], [y_pt, -1], [z_pt, z_pt], color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        # Line to YZ plane (projection at x=-1, assuming sphere left)
        ax.plot([x_pt, -1], [y_pt, y_pt], [z_pt, z_pt], color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        # Add label if provided
        if labels is not None and i < len(labels):
            # Offset label slightly for better visibility
            ax.text(x_pt * 1.1, y_pt * 1.1, z_pt * 1.1, labels[i], fontsize=12)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set view limits to ensure the sphere is nicely framed
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    
    # Set title
    plt.title(title)
    
    # Show the plot
    plt.show()