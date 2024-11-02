"""
Initial script that will initialize Bob's environment and startup Bob.
"""
from app.env.computer import Computer
from app.bob.bob import Bob



def main():
    computer = Computer()
    bob = Bob(computer)

    while True:
        bob.alive(computer)



if __name__ == "__main__":
    main()
