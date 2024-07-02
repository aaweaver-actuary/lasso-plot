# Launch the FastAPI app, expose a port, and run the app
FROM python:3.11-slim

WORKDIR /app

# Install zsh and oh-my-zsh
RUN apt update && apt upgrade -y
RUN apt install git curl bash sudo zsh python3-pip python3-setuptools python3-wheel -y
RUN git clone http://github.com/aaweaver-actuary/dotfiles && \
    cp dotfiles/install_dotfiles /usr/bin/install_dotfiles && \
    chmod +x /usr/bin/install_dotfiles && \
    install_dotfiles /app .ruff.toml install_oh_my_zsh
RUN chmod +x install_oh_my_zsh
RUN ./install_oh_my_zsh
SHELL ["/bin/zsh", "-c"]

# Use uv to manage the app, venv, etc
RUN python3 -m pip install uv && uv venv && uv pip install -r requirements.lock

# Expose the port
EXPOSE 8000

# Run the app
# CMD fastapi dev --reload
CMD ["/usr/bin/zsh"]