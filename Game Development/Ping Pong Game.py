#pip install pygame
import pygame
import sys


# Initialize pygame
pygame.init()


# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong Game")


# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


# Define the paddles and ball
paddle_width, paddle_height = 15, 100
ball_size = 20


# Left paddle
left_paddle = pygame.Rect(30, HEIGHT // 2 - paddle_height // 2, paddle_width, paddle_height)
# Right paddle
right_paddle = pygame.Rect(WIDTH - 30 - paddle_width, HEIGHT // 2 - paddle_height // 2, paddle_width, paddle_height)
# Ball
ball = pygame.Rect(WIDTH // 2 - ball_size // 2, HEIGHT // 2 - ball_size // 2, ball_size, ball_size)


# Set initial speeds
ball_speed_x, ball_speed_y = 5, 5
paddle_speed = 7


# Game loop
clock = pygame.time.Clock()


def draw():
   screen.fill(BLACK)
   pygame.draw.rect(screen, WHITE, left_paddle)
   pygame.draw.rect(screen, WHITE, right_paddle)
   pygame.draw.ellipse(screen, WHITE, ball)
   pygame.draw.rect(screen, WHITE, ball)
   pygame.display.flip()


def handle_input():
   keys = pygame.key.get_pressed()
   if keys[pygame.K_w] and left_paddle.top > 0:
       left_paddle.y -= paddle_speed
   if keys[pygame.K_s] and left_paddle.bottom < HEIGHT:
       left_paddle.y += paddle_speed
   if keys[pygame.K_UP] and right_paddle.top > 0:
       right_paddle.y -= paddle_speed
   if keys[pygame.K_DOWN] and right_paddle.bottom < HEIGHT:
       right_paddle.y += paddle_speed


def move_ball():
   global ball_speed_x, ball_speed_y


   # Move the ball
   ball.x += ball_speed_x
   ball.y += ball_speed_y


   # Ball collision with top and bottom
   if ball.top <= 0 or ball.bottom >= HEIGHT:
       ball_speed_y = -ball_speed_y


   # Ball collision with paddles
   if ball.colliderect(left_paddle) or ball.colliderect(right_paddle):
       ball_speed_x = -ball_speed_x


   # Ball out of bounds (score)
   if ball.left <= 0 or ball.right >= WIDTH:
       ball.x = WIDTH // 2 - ball_size // 2
       ball.y = HEIGHT // 2 - ball_size // 2
       return True  # Ball is out of bounds, reset the game


   return False


def main():
   running = True
   while running:
       # Event handling
       for event in pygame.event.get():
           if event.type == pygame.QUIT:
               running = False


       # Handle user input
       handle_input()


       # Move the ball and check for out of bounds
       if move_ball():
           # Optionally reset score here


           # Display score or reset
           print("Ball out of bounds! Resetting...")


       # Draw everything
       draw()


       # Frame rate
       clock.tick(60)


   pygame.quit()
   sys.exit()


if __name__ == "__main__":
   main()
