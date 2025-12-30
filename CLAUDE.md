# CLAUDE.md - AI Assistant Guide for LoopForge

## Project Overview

**LoopForge** is a Flask-based web application implementing Human-in-the-Loop machine learning. Users chat with GPT-4 while simultaneously training a local GPT-2 model using their conversation data. Each user interaction:
1. Gets a response from GPT-4 API
2. Stores the conversation in SQLite database
3. Trains a local GPT-2 model on the conversation
4. Returns training loss metrics

## Tech Stack

| Category | Technology |
|----------|-----------|
| Backend | Flask >=2.3.0 |
| Database | SQLAlchemy >=2.0.0 + SQLite |
| Authentication | Flask-Login >=0.6.0 |
| Forms | Flask-WTF >=1.2.0, WTForms >=3.1.0 |
| ML/NLP | PyTorch >=2.0.0, Transformers >=4.30.0 |
| API Client | OpenAI >=1.0.0 |
| Frontend | Bootstrap 4.3.1, jQuery 3.3.1 (CDN) |
| Production | Gunicorn >=21.0.0 |

## File Structure

```
loopforge/
├── app.py                 # Main Flask application (all routes, models, ML logic)
├── requirements.txt       # Python dependencies
├── README.md             # Project description
├── CLAUDE.md             # This file
├── Conditional Commercial License  # Licensing terms
├── 外網部屬.md           # Deployment guide (Chinese)
├── model/                # Model storage directory
│   └── model.txt        # Placeholder
└── templates/           # Jinja2 HTML templates
    ├── index.html       # Main chat interface
    ├── login.html       # Login form
    ├── register.html    # Registration form
    └── history.html     # Paginated chat history
```

## Key Components in app.py

### Database Models (lines 39-60)
- **User**: id, username, email, password_hash, chats relationship
  - `set_password(password)`: Hash and store password securely using PBKDF2-SHA256
  - `check_password(password)`: Verify password against stored hash
- **Chat**: id, input_text, response_text, user_id (FK)

### Forms (lines 68-91)
- **LoginForm**: username, password, remember_me
- **RegistrationForm**: username, email, password with uniqueness validation

### ML Components (lines 93-106)
- GPT-2 model configuration: 768 embeddings, 10 layers, 12 heads
- `torch.optim.AdamW` optimizer with lr=5e-5
- Session-based model state management via `get_session_model()`

### Core Function (lines 109-167)
`chat_and_train(input_text: str, cot_mode: bool = False) -> Tuple[str, float]`:
- Calls GPT-4 API with optional Chain-of-Thought mode
- Handles specific API exceptions (RateLimitError, APIConnectionError, APIError)
- Tokenizes conversation, performs forward/backward pass
- Updates model in session storage
- Returns tuple of (response_text, loss_value)

### Input Validation (lines 183-202)
`_validate_chat_input(data: Optional[dict]) -> Tuple[str, bool]`:
- Validates JSON request body
- Enforces 2000 character limit on input
- Type-checks cot_mode boolean

### Routes
| Route | Method | Auth | Purpose |
|-------|--------|------|---------|
| `/`, `/index` | GET | Required | Main chat interface |
| `/login` | GET, POST | No | User authentication |
| `/logout` | GET | No | Session termination |
| `/register` | GET, POST | No | User registration |
| `/chat` | POST | Required | Process chat, train model |
| `/history` | GET | Required | Paginated chat history |

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SECRET_KEY="your-secret-key"
export TOGETHER_API_KEY="your-api-key"
export FLASK_DEBUG="true"  # Only for development

# Run development server
python app.py

# Production deployment
gunicorn -w 4 -b 0.0.0.0:8000 --timeout 120 app:app
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | Yes | `your_secret_key_here` | Flask session encryption |
| `TOGETHER_API_KEY` | Yes | None | API key for LLM service |
| `OPENAI_API_BASE` | No | `http://127.0.0.1:5001/v1` | OpenAI-compatible API base URL |
| `OPENAI_MODEL` | No | `gpt-4-turbo` | Model name to use for chat |
| `DATABASE_URL` | No | `sqlite:///app.db` | Database connection string |
| `FLASK_DEBUG` | No | `False` | Enable debug mode (dev only) |
| `POSTS_PER_PAGE` | No | `10` | Pagination size for history |

## Code Conventions

### Naming
- **Functions/variables**: snake_case
- **Classes**: PascalCase
- **Routes**: lowercase with hyphens

### Type Hints
All functions include type hints for parameters and return values:
```python
def chat_and_train(input_text: str, cot_mode: bool = False) -> Tuple[str, float]:
def load_user(user_id: int) -> Optional[User]:
def get_session_model() -> GPT2LMHeadModel:
```

### Patterns Used
- MVC architecture (Models + Templates + Routes in app.py)
- Decorator-based auth: `@login_required`
- Session-based ML model state
- WTForms for validation with custom validators
- Secure password hashing with Werkzeug

### Error Handling Pattern
```python
# OpenAI v1.0+ error classes (imported directly from openai)
from openai import RateLimitError, APIConnectionError, APIError, APIStatusError

try:
    # Operation
except RateLimitError:
    logging.warning("API rate limit exceeded")
    raise RuntimeError("Service temporarily unavailable.")
except APIConnectionError as e:
    logging.error(f"API connection failed: {str(e)}")
    raise RuntimeError("Cannot connect to AI service.")
except APIStatusError as e:
    logging.error(f"API status error: {e.status_code}")
    raise RuntimeError("Error communicating with AI service")
except ValueError as e:
    return jsonify({'error': str(e)}), 400
except RuntimeError as e:
    return jsonify({'error': str(e)}), 500
```

### Database Queries
```python
# Use SQLAlchemy ORM
User.query.filter_by(username=name).first()
current_user.chats.order_by(Chat.id.desc()).paginate(page, per_page, False)
```

## Security Features

1. **Password Hashing**: Uses PBKDF2-SHA256 via `werkzeug.security`
2. **CSRF Protection**: Flask-WTF on all forms
3. **Input Validation**: Length limits and type checking on `/chat` endpoint
4. **Environment-based Debug Mode**: Debug disabled by default
5. **Session-based Authentication**: Flask-Login
6. **SQL Injection Protection**: SQLAlchemy ORM

## Remaining Technical Debt

1. **No Test Suite**: No automated tests exist
2. **No Database Migrations**: Uses `db.create_all()` - consider adding Alembic/Flask-Migrate
3. **Session Model Storage**: Large model states in session may cause performance issues
4. **No Rate Limiting**: Consider adding Flask-Limiter for API protection

## When Modifying Code

### Adding a New Route
1. Add route handler in `app.py` with appropriate decorators
2. Create template in `templates/` if needed
3. Use `@login_required` for authenticated endpoints
4. Add input validation for POST endpoints
5. Include type hints and docstrings

### Modifying ML Components
- Model config at lines 96-98
- Training logic in `chat_and_train()` (lines 109-167)
- Session model accessed via `get_session_model()`

### Adding Form Fields
1. Add field to appropriate FlaskForm class
2. Add custom validator if needed (pattern: `validate_<fieldname>`)
3. Update corresponding template

## API Integration

The app uses OpenAI v1.0+ client with configurable endpoint:
```python
from openai import OpenAI

openai_client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE", "http://127.0.0.1:5001/v1")
)

# API call example
response = openai_client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
    temperature=0.7, top_p=0.9, max_tokens=400,
    messages=[...]
)
```

## Frontend Notes

- Templates use Jinja2 syntax with `{{ }}` and `{% %}`
- AJAX calls via jQuery for `/chat` endpoint
- Bootstrap 4.3.1 for styling
- Chain-of-Thought mode toggle available in index.html

## License

Conditional Commercial License - Non-commercial use allowed. Commercial use requires permission from the author. OpenAI and affiliates are prohibited from using this software.
