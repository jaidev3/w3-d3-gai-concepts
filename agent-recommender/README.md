# AI Coding Agent Recommender

## Project Overview

The AI Coding Agent Recommender is an intelligent system designed to help developers find the most suitable AI coding assistant for their specific development tasks. By analyzing task descriptions and matching them against a comprehensive database of AI coding agents, this tool provides personalized recommendations.

## Features

- Task analysis using advanced NLP techniques
- Intelligent agent recommendation based on task complexity, domain, and requirements
- RESTful API for agent recommendations
- Web interface for easy interaction
- Supports multiple programming languages and domains

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git (optional, for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-agent-recommender.git
cd agent-recommender
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Additional NLP Model

```bash
python -m spacy download en_core_web_sm
```

## Configuration

### Environment Variables

No specific environment variables are required for basic setup. For advanced configurations (like OpenAI API integration), you may need to modify `app.py`.

## Running the Application

### Development Server

```bash
# Using Uvicorn
uvicorn app:app --reload

# Alternatively
python app.py
```

The application will be available at `http://localhost:8000`

### API Endpoints

- `GET /`: Web interface
- `POST /api/recommend`: Recommend AI agents for a task
- `GET /api/agents`: List all available agents
- `GET /api/health`: Health check endpoint

## Usage Examples

### Web Interface

1. Open `http://localhost:8000` in your browser
2. Enter a task description
3. Receive AI agent recommendations

### API Request (curl)

```bash
curl -X POST http://localhost:8000/api/recommend \
     -H "Content-Type: application/json" \
     -d '{"task_description": "Build a machine learning model for image classification"}'
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - your.email@example.com

Project Link: [https://github.com/your-username/ai-agent-recommender](https://github.com/your-username/ai-agent-recommender)
