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
| Backend | Flask 2.1.2 |
| Database | SQLAlchemy 1.4.27 + SQLite |
| Authentication | Flask-Login 0.5.0 |
| Forms | Flask-WTF 1.0.0, WTForms 3.0.0 |
| ML/NLP | PyTorch 1.10.0, Transformers 4.12.3 |
| API Client | OpenAI 0.10.2 |
| Frontend | Bootstrap 4.3.1, jQuery 3.3.1 (CDN) |
| Production | Gunicorn 20.1.0 |

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

### Database Models (lines 36-47)
- **User**: id, username, email, password_hash, chats relationship
- **Chat**: id, input_text, response_text, user_id (FK)

### Forms (lines 55-77)
- **LoginForm**: username, password, remember_me
- **RegistrationForm**: username, email, password with uniqueness validation

### ML Components (lines 80-91)
- GPT-2 model configuration: 768 embeddings, 10 layers, 12 heads
- AdamW optimizer with lr=5e-5
- Session-based model state management via `get_session_model()`

### Core Function (lines 93-129)
`chat_and_train(input_text, cot_mode=False)`:
- Calls GPT-4 API with optional Chain-of-Thought mode
- Tokenizes conversation, performs forward/backward pass
- Updates model in session storage

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

# Run development server
python app.py

# Production deployment
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | Yes | `your_secret_key_here` | Flask session encryption |
| `TOGETHER_API_KEY` | Yes | None | API key for LLM service |
| `DATABASE_URL` | No | `sqlite:///app.db` | Database connection string |

## Code Conventions

### Naming
- **Functions/variables**: snake_case
- **Classes**: PascalCase
- **Routes**: lowercase with hyphens

### Patterns Used
- MVC architecture (Models + Templates + Routes in app.py)
- Decorator-based auth: `@login_required`
- Session-based ML model state
- WTForms for validation with custom validators

### Error Handling Pattern
```python
try:
    # Operation
except Exception as e:
    logging.error(f"Context: {str(e)}")
    raise RuntimeError("User-friendly message")
```

### Database Queries
```python
# Use SQLAlchemy ORM
User.query.filter_by(username=name).first()
current_user.chats.order_by(Chat.id.desc()).paginate(page, per_page, False)
```

## Known Issues / Technical Debt

1. **Password Storage**: Passwords stored as plaintext in `password_hash` field (line 144). Should use bcrypt/werkzeug.security.
2. **Broad Exception Handling**: Uses generic `except Exception` - should catch specific exceptions.
3. **Missing Configuration**: `POSTS_PER_PAGE` referenced but not defined in config.
4. **No Test Suite**: No automated tests exist.
5. **No Database Migrations**: Uses `db.create_all()` - no Alembic/Flask-Migrate.

## When Modifying Code

### Adding a New Route
1. Add route handler in `app.py` with appropriate decorators
2. Create template in `templates/` if needed
3. Use `@login_required` for authenticated endpoints
4. Add database operations within request context

### Modifying ML Components
- Model config at lines 83-85
- Training logic in `chat_and_train()` (lines 93-129)
- Session model accessed via `get_session_model()`

### Adding Form Fields
1. Add field to appropriate FlaskForm class
2. Add custom validator if needed (pattern: `validate_<fieldname>`)
3. Update corresponding template

## API Integration

The app uses OpenAI-compatible API:
```python
openai.api_base = 'http://127.0.0.1:5001/v1'  # Default local endpoint
# Model: gpt-4.0-turbo
# Temperature: 0.7, top_p: 0.9, max_tokens: 400
```

## Frontend Notes

- Templates use Jinja2 syntax with `{{ }}` and `{% %}`
- AJAX calls via jQuery for `/chat` endpoint
- Bootstrap 4.3.1 for styling
- Chain-of-Thought mode toggle available in index.html

## Security Considerations

- CSRF protection via Flask-WTF on all forms
- Session-based authentication with Flask-Login
- Environment variables for secrets (use `.env` file)
- SQLAlchemy ORM provides SQL injection protection

## License

Conditional Commercial License - Non-commercial use allowed. Commercial use requires permission from the author. OpenAI and affiliates are prohibited from using this software.
