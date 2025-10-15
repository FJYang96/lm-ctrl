color_map = {
    "red": "\033[1m\033[38;5;196m",
    "orange": "\033[1m\033[38;5;208m",
    "green": "\033[1m\033[38;5;46m",
    "blue": "\033[1m\033[38;5;21m",
    "reset": "\033[0m",  # Resets to default terminal settings
}


def color_print(color: str, text: str) -> None:
    if color not in color_map:
        raise ValueError(f"Invalid color: {color}")
    print(color_map[color] + text + color_map["reset"])
