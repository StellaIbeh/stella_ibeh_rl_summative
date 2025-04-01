import sys
import numpy as np
import time
import imageio

# Import OpenGL modules
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# ----------------------------
# Custom Gym-like Environment
# ----------------------------
class HypertensionMonitoringEnv:
    """
    A simplified custom environment for AI-powered remote monitoring.
    The agent (or user) chooses an action that affects the patient’s vitals.
    """
    def __init__(self):
        # Define action space (Discrete: 0 to 6)
        # 0: No medication, 1: Low-dose medication, 2: Moderate-dose medication,
        # 3: High-dose medication, 4: Recommend rest, 5: Recommend exercise,
        # 6: Call emergency response.
        self.action_space = list(range(7))
        
        # Observation: [SBP, DBP, HR, Stress, Physical Activity, Last Medication, Time since dose, Sleep Quality]
        self.state = None
        self.time_step = 0
        self.reset()

    def reset(self):
        # Reset state to average values.
        self.state = np.array([120, 80, 70, 5, 0, 0, 0, 1], dtype=np.float32)
        self.time_step = 0
        return self.state

    def step(self, action):
        sbp, dbp, hr, stress, phys_act, last_med, time_since, sleep = self.state
        
        # Add random fluctuations
        sbp += np.random.uniform(-2, 2)
        dbp += np.random.uniform(-1, 1)
        hr += np.random.uniform(-1, 1)
        stress += np.random.uniform(-0.5, 0.5)
        stress = np.clip(stress, 0, 10)
        
        # Effects of actions
        if action == 0:  # No medication
            reward = 0
        elif action == 1:  # Low-dose medication
            sbp -= 3
            dbp -= 2
            reward = 2
        elif action == 2:  # Moderate-dose medication
            sbp -= 5
            dbp -= 3
            reward = 4
        elif action == 3:  # High-dose medication
            sbp -= 8
            dbp -= 5
            reward = -2  # risk of too much drop
        elif action == 4:  # Recommend rest
            stress -= 1
            reward = 2
        elif action == 5:  # Recommend exercise
            sbp += 2
            dbp += 1
            reward = 1
        elif action == 6:  # Call emergency response
            # Only beneficial if patient is in critical condition.
            if sbp > 160 or dbp > 100 or sbp < 80 or dbp < 50:
                reward = 10
            else:
                reward = -5
        
        # Update time and medication info
        time_since += 1
        if action in [1, 2, 3]:
            last_med = action  # record which medication was given
            time_since = 0

        # Update and clip state values
        self.state = np.array([
            np.clip(sbp, 50, 200),
            np.clip(dbp, 30, 150),
            np.clip(hr, 40, 180),
            stress,
            np.clip(phys_act, 0, 2),
            last_med,
            np.clip(time_since, 0, 120),
            sleep
        ], dtype=np.float32)

        # Additional reward shaping: bonus if BP is in the optimal range
        optimal_sbp = (90, 120)
        optimal_dbp = (60, 80)
        if optimal_sbp[0] <= self.state[0] <= optimal_sbp[1] and optimal_dbp[0] <= self.state[1] <= optimal_dbp[1]:
            reward += 10
        elif self.state[0] > 130 or self.state[1] > 90 or self.state[0] < 80 or self.state[1] < 50:
            reward -= 10

        self.time_step += 1
        done = self.time_step >= 100  # End episode after 100 steps
        info = {}
        return self.state, reward, done, info

# ----------------------------
# Global variables for rendering
# ----------------------------
env = HypertensionMonitoringEnv()
state = env.reset()
window_width = 800
window_height = 600

# Variables to track interactive action feedback
last_action = None
last_action_time = 0
action_flash_duration = 0.5  # seconds

# Map each action to a flash color
action_color_map = {
    0: (1.0, 1.0, 1.0),   # White for No Medication
    1: (0.5, 1.0, 0.5),   # Light green for Low-dose
    2: (0.5, 0.5, 1.0),   # Light blue for Moderate-dose
    3: (1.0, 0.5, 0.5),   # Light red for High-dose
    4: (1.0, 1.0, 0.5),   # Yellow for Recommend rest
    5: (1.0, 0.7, 0.3),   # Orange for Recommend exercise
    6: (0.8, 0.0, 0.8)    # Purple for Emergency response
}

# Global list to store frames for GIF creation
frames = []

# ----------------------------
# Utility function: Draw text on screen
# ----------------------------
def drawText(x, y, text_string):
    glRasterPos2f(x, y)
    for ch in text_string:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))

# ----------------------------
# OpenGL Display Callback
# ----------------------------
def display():
    global state, last_action, last_action_time
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Setup 2D orthographic view for text overlay.
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window_width, 0, window_height)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # ----------------------------
    # Render the 3D Patient Cube
    # ----------------------------
    # Switch to 3D view for drawing the cube.
    glViewport(0, 0, window_width, window_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (window_width/window_height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5)
    
    # Determine the cube color:
    # Default: color based on SBP (green if optimal, red if high, blue if low)
    sbp = state[0]
    default_color = (0.0, 1.0, 0.0) if 90 <= sbp <= 120 else (1.0, 0.0, 0.0) if sbp > 120 else (0.0, 0.0, 1.0)
    
    # If an action was taken recently, override with the corresponding flash color.
    current_time = time.time()
    if last_action is not None and (current_time - last_action_time) < action_flash_duration:
        cube_color = action_color_map.get(last_action, default_color)
    else:
        cube_color = default_color

    glColor3f(*cube_color)
    glutSolidCube(1.5)

    # ----------------------------
    # Render the Vital Stats Text (2D overlay)
    # ----------------------------
    # Switch back to 2D orthographic projection.
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window_width, 0, window_height)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glColor3f(1.0, 1.0, 1.0)
    
    stats = f"SBP: {state[0]:.1f}  DBP: {state[1]:.1f}  HR: {state[2]:.1f}  Stress: {state[3]:.1f}"
    drawText(10, window_height - 30, stats)
    
    instructions = "Keys: [0]NoMed [1]Low [2]Mod [3]High [4]Rest [5]Exercise [6]Emergency | [g] GIF Run"
    drawText(10, window_height - 60, instructions)
    
    if last_action is not None:
        action_names = {
            0: "No Medication",
            1: "Low-dose Medication",
            2: "Moderate-dose Medication",
            3: "High-dose Medication",
            4: "Recommend Rest",
            5: "Recommend Exercise",
            6: "Call Emergency Response"
        }
        action_text = f"Last Action: {action_names.get(last_action, 'N/A')}"
        drawText(10, window_height - 90, action_text)

    glutSwapBuffers()

# ----------------------------
# Keyboard Callback to control the environment
# ----------------------------
def keyboard(key, x, y):
    global state, env, last_action, last_action_time
    # Check for GIF simulation trigger ('g' or 'G')
    if key in (b'g', b'G'):
        print("Starting GIF simulation run...")
        simulation_for_gif(num_steps=50, delay=0.2)
        return

    try:
        # Map key presses (as bytes) to actions (0 to 6)
        action = int(key.decode("utf-8"))
    except:
        return

    if action in env.action_space:
        state, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward:.1f}, New State: {state}")
        last_action = action
        last_action_time = time.time()
        if done:
            print("Episode finished. Resetting environment.")
            state = env.reset()
    glutPostRedisplay()

# ----------------------------
# Idle Callback for periodic updates (optional)
# ----------------------------
def idle():
    glutPostRedisplay()

# ----------------------------
# Utility: Capture current OpenGL frame
# ----------------------------
def capture_frame():
    """Capture the current frame from the OpenGL buffer and return as an image array."""
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(window_height, window_width, 3)
    return np.flipud(image)

# ----------------------------
# GIF Simulation Function
# ----------------------------
def simulation_for_gif(num_steps=50, delay=0.2):
    """Run simulation with random actions, capture frames, and create a GIF."""
    global state, env, last_action, last_action_time, frames

    frames = []  # Clear previous frames
    state = env.reset()
    for i in range(num_steps):
        # Choose a random action
        action = np.random.choice(env.action_space)
        state, reward, done, _ = env.step(action)
        last_action = action
        last_action_time = time.time()
        glutPostRedisplay()
        time.sleep(delay)  # Delay to allow scene update
        frame = capture_frame()
        frames.append(frame)
        if done:
            state = env.reset()
    gif_filename = 'simulation.gif'
    imageio.mimsave(gif_filename, frames, duration=delay)
    print(f"GIF saved as {gif_filename}")

# ----------------------------
# Main: Set up the GLUT window and start the loop
# ----------------------------
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(window_width, window_height)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"Hypertension Monitoring Simulation")
    
    glEnable(GL_DEPTH_TEST)
    
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)
    
    glutMainLoop()

if __name__ == "__main__":
    main()
