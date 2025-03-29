import pygame
import numpy as np
import imageio

# Grid settings
grid_size = 10
cell_size = 70
window_size = grid_size * cell_size

# Colors
colors = {
    'bg': (25, 25, 112),  # Dark blue background
    'grid': (70, 130, 180),  # Light blue grid
    'human_director': (0, 0, 255),  # Blue agent
    'person': (255, 255, 0),  # Yellow people
    'flood_zone': (220, 20, 60),  # Red danger area
    'safe_zone': (34, 139, 34),  # Green safe area
}

# Define positions for flood zone and safe zone
flood_zone = [(x, 2) for x in range(3)]  # Red zone on row 2
safe_zone = [(x, 8) for x in range(7, 10)]  # Green zone on row 8


def draw_grid(surface):
    """Draws the grid with flood and safe zones."""
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            
            # Color flood zone red
            if (x, y) in flood_zone:
                pygame.draw.rect(surface, colors['flood_zone'], rect)
            # Color safe zone green
            elif (x, y) in safe_zone:
                pygame.draw.rect(surface, colors['safe_zone'], rect)

            pygame.draw.rect(surface, colors['grid'], rect, 2, border_radius=8)


def draw_human_director(surface, director_pos):
    """Draws the Human Director (Agent)."""
    pos_pix = ((director_pos + 0.5) * cell_size).astype(int)
    pygame.draw.rect(surface, colors['human_director'], (*pos_pix - cell_size//4, cell_size//2, cell_size//2), border_radius=10)


def draw_people(surface, people_positions):
    """Draws people who need evacuation."""
    for pos in people_positions:
        pos_pix = ((pos + 0.5) * cell_size).astype(int)
        pygame.draw.circle(surface, colors['person'], pos_pix, cell_size//4)


def move_towards(agent_pos, target_pos):
    """Moves agent (human director) towards target (people in danger)."""
    direction = np.sign(target_pos - agent_pos)
    new_pos = agent_pos + direction
    return np.clip(new_pos, 0, grid_size - 1)


def visualize_environment():
    pygame.init()
    surface = pygame.Surface((window_size, window_size))

    frames = []

    # Initial positions
    human_director_pos = np.array([5, 5])
    people_positions = [np.array([3, 2]), np.array([2, 2]), np.array([1, 3])]

    for _ in range(50):  # Run for 50 frames
        surface.fill(colors['bg'])
        draw_grid(surface)
        draw_human_director(surface, human_director_pos)
        draw_people(surface, people_positions)

        # Move the human director towards people
        for i, person_pos in enumerate(people_positions):
            if tuple(person_pos) in flood_zone:
                human_director_pos = move_towards(human_director_pos, person_pos)
                people_positions[i] += np.random.choice([-1, 0, 1], size=2)
                people_positions[i] = np.clip(people_positions[i], 0, grid_size - 1)

        # Capture frame
        frame = pygame.surfarray.array3d(surface)
        frames.append(np.transpose(frame, (1, 0, 2)))

    imageio.mimsave('evacuation_simulation.gif', frames, duration=0.2)
    pygame.quit()


if __name__ == "__main__":
    visualize_environment()

