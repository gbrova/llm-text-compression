# Port Configuration Guide

✅ **Port configuration is now available!** The default port has been changed from 5000 to 8080 to avoid conflicts.

## Quick Start

```bash
# Default port 8080 (changed from 5000)
uv run python rank_visualizer.py

# Custom port 
uv run python rank_visualizer.py --port 3000

# With convenience script (auto-opens browser)
uv run python start_ui.py --port 3000
```

## Configuration Options

### Command Line Arguments

```bash
# Basic options
uv run python rank_visualizer.py --port 3000 --host 0.0.0.0

# Full options
uv run python rank_visualizer.py --port 3000 --host 127.0.0.1 --debug

# Convenience script options
uv run python start_ui.py --port 3000 --no-browser
```

### Environment Variables

```bash
# Set port via environment variable
FLASK_PORT=3000 uv run python rank_visualizer.py

# Set both host and port
FLASK_HOST=0.0.0.0 FLASK_PORT=3000 uv run python rank_visualizer.py
```

## Available Ports

The app will now default to port **8080** instead of 5000. You can use any available port:

- `3000` - Common alternative for web development
- `8000` - Another common development port  
- `8080` - Default port (HTTP alternative)
- `9000` - Higher numbered port to avoid conflicts

## Help

```bash
# Get help for main app
uv run python rank_visualizer.py --help

# Get help for convenience script
uv run python start_ui.py --help
```

## Testing

```bash
# Test with custom port
uv run python test_flask_app.py --port 3000
```