import pygame
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import sys
import pickle
import pathlib

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Flappy Bird')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Bird settings
bird = pygame.Rect(100, 300, 30, 30)
bird_velocity = 0
gravity = 0.5

# Pipe settings
pipe_width = 50
pipe_gap = 150
pipe_velocity = -3

# Ground settings
ground_height = 0
ground_speed = 2
ground_offset = 0

# Define the path to the asset directory
asset_path = pathlib.Path().resolve() / 'Asset'
raw_path = pathlib.Path().resolve() / 'Raw'

# Load assets
try:
    bird_up_image = pygame.image.load(str(asset_path / 'bird0.png'))
    bird_down_image = pygame.image.load(str(asset_path / 'bird1.png'))
    bird_up_image = pygame.transform.scale(bird_up_image, (45, 40))
    bird_down_image = pygame.transform.scale(bird_down_image, (45, 40))
    
    pipe_image = pygame.image.load(str(asset_path / 'pipe1.png'))
    pipe_image_flip = pygame.image.load(str(asset_path / 'pipe0.png'))

    background_image = pygame.image.load(str(asset_path / 'flappy_bird_bg.jpg'))
    background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

    ground_image = pygame.image.load(str(asset_path / 'ground.jpg'))
    ground_image = pygame.transform.scale(ground_image, (WIDTH, 300))

    jump_sound = pygame.mixer.Sound(str(raw_path / 'jump.mp3'))
    collision_sound = pygame.mixer.Sound(str(raw_path / 'collision.mp3'))
    score_sound = pygame.mixer.Sound(str(raw_path / 'tunn.wav'))

except pygame.error as e:
    print(f"Error loading assets: {e}")
    pygame.quit()
    sys.exit()

# Create a neural network model
model = Sequential([
    Dense(24, activation='relu', input_shape=(8,)),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

def get_state(bird, pipes, bird_velocity):
    bird_y = bird.y / HEIGHT
    bird_velocity /= 10
    pipe_x = pipes[0][0].x / WIDTH
    top_pipe_bottom = pipes[0][1].top
    bottom_pipe_top = pipes[0][0].bottom
    pipe_gap_top = top_pipe_bottom / HEIGHT
    pipe_gap_bottom = bottom_pipe_top / HEIGHT
    pipe_gap_size = (top_pipe_bottom - bottom_pipe_top) / HEIGHT
    bird_height = bird.height / HEIGHT
    bird_width = bird.width / WIDTH
    if bird.y > bottom_pipe_top:
        distance_from_gap = bird.y - bottom_pipe_top
    else:
        distance_from_gap = top_pipe_bottom - bird.y
    gap_height = top_pipe_bottom - bottom_pipe_top
    normalized_distance = distance_from_gap / gap_height
    
    return np.array([
        bird_y, bird_velocity, pipe_x, pipe_gap_top, pipe_gap_bottom, pipe_gap_size,
        bird_height, normalized_distance
    ])

def get_reward(bird, pipes):
    top_pipe = pipes[0][0]
    bottom_pipe = pipes[0][1]
    if bird.colliderect(top_pipe) or bird.colliderect(bottom_pipe) or bird.y > HEIGHT or bird.y < 0:
        return -1  # Collision penalty
    gap_top = top_pipe.bottom
    gap_bottom = bottom_pipe.top
    bird_center = bird.y + bird.height / 2
    distance_from_bottom = bird_center - gap_bottom
    gap_size = gap_bottom - gap_top
    normalized_distance = abs(distance_from_bottom) / gap_size
    if bird.y > gap_top and (bird.y + bird.height) < gap_bottom:
        reward = 1 - normalized_distance
    else:
        reward = 1 - normalized_distance
    return reward

def draw_text(text, font, color, x, y):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

def show_start_screen():
    screen.blit(background_image, (0, 0))
    
    title_font = pygame.font.Font(None, 64)
    instruction_font = pygame.font.Font(None, 24)
    
    # Title
    title_text = "Flappy Bird AI"
    title_surface = title_font.render(title_text, True, (255, 165, 0))  # Orange color
    title_rect = title_surface.get_rect(center=(WIDTH // 2, HEIGHT // 4))
    screen.blit(title_surface, title_rect)
    
    instructions = [
        ("Press SPACE or click START to begin", (255, 0, 0)),
        ("Avoid the pipes and survive as long as possible", BLACK),
        ("The game will train an AI to play", BLACK),
        ("After training, watch the AI play automatically", BLACK)
    ]
    
    for i, (instruction, color) in enumerate(instructions):
        text_surface = instruction_font.render(instruction, True, color)
        text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + i * 40))
        screen.blit(text_surface, text_rect)
    
    # Start button
    start_button = pygame.Rect(WIDTH // 2 - 50, HEIGHT - 80, 100, 50)
    pygame.draw.rect(screen, (0, 215, 0), start_button)
    start_text = instruction_font.render("START", True, WHITE)
    start_text_rect = start_text.get_rect(center=start_button.center)
    screen.blit(start_text, start_text_rect)
    
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                waiting = False
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_pos = pygame.mouse.get_pos()
                if start_button.collidepoint(mouse_pos):
                    waiting = False

def create_pipe():
    gap_start = random.randint(100, HEIGHT - ground_height - 100 - pipe_gap)
    top_pipe = pygame.Rect(WIDTH, 0, pipe_width, gap_start)
    bottom_pipe = pygame.Rect(WIDTH, gap_start + pipe_gap, pipe_width, HEIGHT - gap_start - pipe_gap - ground_height)
    return [top_pipe, bottom_pipe, False]  # Using a list instead of a tuple

def draw_bird_and_pipes(bird, pipes, score, game_number, autoplay=False):
    screen.blit(background_image, (0, 0))

    # Draw ground
    first_ground_pos = ground_offset % WIDTH
    screen.blit(ground_image, (first_ground_pos - WIDTH, HEIGHT - 300))
    screen.blit(ground_image, (first_ground_pos, HEIGHT - 300))
    screen.blit(ground_image, (first_ground_pos + WIDTH, HEIGHT - 300))
    
    # Draw bird with continuous flapping animation
    flap_speed = 1  # Adjust this value to change flapping speed
    if pygame.time.get_ticks() % (1.5 * flap_speed) < flap_speed:
        screen.blit(bird_up_image, (bird.x, bird.y))
    else:
        screen.blit(bird_down_image, (bird.x, bird.y))
    
    # Draw pipes
    for pipe in pipes:
        top_pipe_image = pygame.transform.scale(pipe_image, (pipe_width, pipe[0].height))
        bottom_pipe_image = pygame.transform.scale(pipe_image_flip, (pipe_width, pipe[1].height))
        screen.blit(top_pipe_image, (pipe[0].x, pipe[0].y))
        screen.blit(bottom_pipe_image, (pipe[1].x, pipe[1].y))

    # Draw the score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f'Score: {score}', True, (0, 0, 215))
    screen.blit(score_text, (WIDTH - score_text.get_width() - 10, 10))

    # Draw game number or autoplay mode
    game_mode_text = "Autoplay mode" if autoplay else f"Game: {game_number}"
    mode_text = font.render(game_mode_text, True, BLACK)
    screen.blit(mode_text, (10, 10))
    
    pygame.display.flip()

def reset_game():
    global bird, bird_velocity, pipes, score, pipe_velocity, ground_offset
    bird = pygame.Rect(100, 100, 30, 30)
    bird_velocity = 0
    pipes = [create_pipe()]
    score = 0
    pipe_velocity = -3
    ground_offset = 0

def display_message(text, color=BLACK):
    # Display a message in the center of the screen
    font = pygame.font.Font(None, 50)
    message_text = font.render(text, True, color)
    screen.fill(WHITE)  # Clear screen
    screen.blit(message_text, (WIDTH // 2 - message_text.get_width() // 2, HEIGHT // 2))
    pygame.display.flip()
    pygame.time.wait(2000)  # Pause to show the message

def show_game_over():
    font = pygame.font.Font(None, 74)
    text = font.render("Game Over", True, (255, 0, 0))
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))
    pygame.display.flip()
    pygame.time.wait(2000)

def collect_training_data(game_number):
    global bird_velocity, score, ground_offset, pipe_velocity
    training_data = []
    reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird_velocity = -8
                jump_sound.play()

        # Bird movement
        bird_velocity += gravity
        bird.y += bird_velocity

        # Ground movement
        ground_offset = (ground_offset - ground_speed) % WIDTH

        # Pipe movement and score counting
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity
            
            # Check if the bird has passed the pipe
            if bird.x > pipe[0].x + pipe_width and not pipe[2]:
                score += 1
                pipe[2] = True  # Mark this pipe as passed
                score_sound.play()
                
                # Increase pipe speed with every score
                pipe_velocity -= 1  # Gradually increase speed

        # Remove pipes off the screen and add new ones
        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())

        # Collision detection and reward calculation
        reward = get_reward(bird, pipes)
        state = get_state(bird, pipes, bird_velocity)
        training_data.append((state, reward))

        if reward == -1 or bird.y > HEIGHT - ground_height or bird.y < 0:
            collision_sound.play()
            show_game_over()
            break

        # Update the display
        draw_bird_and_pipes(bird, pipes, score, game_number)

        # Frame rate control
        clock.tick(30)
    
    # Save training data for the current game
    with open(f'training_data_game_{game_number}.pkl', 'wb') as f:
        pickle.dump(training_data, f)

def autoplay_game(game_number):
    global bird_velocity, score, ground_offset, pipe_velocity
    reset_game()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()

        # Bird movement
        bird_velocity += gravity
        bird.y += bird_velocity

        # Ground movement
        ground_offset = (ground_offset - ground_speed) % WIDTH

        # Pipe movement and score counting
        for pipe in pipes:
            pipe[0].x += pipe_velocity
            pipe[1].x += pipe_velocity

            if bird.x > pipe[0].x + pipe_width and not pipe[2]:
                score += 1
                pipe[2] = True  # Mark this pipe as passed
                score_sound.play()
                
                # Increase pipe speed with every score
                pipe_velocity -= 1  # Gradually increase speed

        # Remove pipes off the screen and add new ones
        if pipes[0][0].x < -pipe_width:
            pipes.pop(0)
            pipes.append(create_pipe())

        # Neural network prediction
        state = get_state(bird, pipes, bird_velocity)
        state = np.reshape(state, [1, 8])
        action_prob = model.predict(state)[0][0]
        if action_prob > 0.5:
            bird_velocity = -8  # Flap action
            jump_sound.play()  # Add jump sound in autoplay mode

        # Collision detection
        if bird.colliderect(pipes[0][0]) or bird.colliderect(pipes[0][1]) or bird.y > HEIGHT - ground_height or bird.y < 0:
            collision_sound.play()
            show_game_over()
            break

        # Update the display
        draw_bird_and_pipes(bird, pipes, score, game_number, autoplay=True)

        # Frame rate control
        clock.tick(30)

def train_model():
    # Load and prepare training data
    all_data = []
    num_games = 5
    for game_number in range(1, num_games + 1):
        display_message(f"Game {game_number}")
        collect_training_data(game_number)
        with open(f'training_data_game_{game_number}.pkl', 'rb') as f:
            game_data = pickle.load(f)
            states, rewards = zip(*game_data)
            all_data.extend(zip(states, rewards))
    
    # Convert to NumPy arrays
    states, rewards = zip(*all_data)
    states = np.array(states)
    rewards = np.array(rewards)

    # Train the model
    display_message("Training Neural Network")
    model.fit(states, rewards, epochs=10, verbose=1)
    display_message("Training completed", color=(0, 215, 0))

def main():
    # Show start screen
    show_start_screen()
    
    # Main loop for the game
    while True:
        for game_number in range(1):
            display_message("Game started", color=(0, 215, 0))
            train_model()
            
            # Autoplay mode
            for autoplay_game_number in range(1):
                display_message("Autoplay mode of Training")
                autoplay_game(autoplay_game_number)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()