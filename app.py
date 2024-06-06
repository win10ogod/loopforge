from flask import Flask, jsonify, request, render_template, session, redirect, url_for, flash
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, ValidationError, Email, EqualTo
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
import openai
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
# 設置 OpenAI API Key 和本地 API 地址
openai.api_key = os.getenv("TOGETHER_API_KEY", "your_openai_api_key")
openai.api_base = 'http://127.0.0.1:5001/v1'

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key_here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'

db = SQLAlchemy(app)
Session(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Setup logging
logging.basicConfig(level=logging.INFO)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    chats = db.relationship('Chat', backref='author', lazy='dynamic')

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    input_text = db.Column(db.String(1024))
    response_text = db.Column(db.String(1024))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# User loader
@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')

# Initialize NLP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
configuration = GPT2Config(n_embd=768, n_layer=10, n_head=12)
model = GPT2LMHeadModel(configuration).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

def get_session_model():
    if 'model_state' not in session:
        session['model_state'] = model.state_dict()
    model.load_state_dict(session['model_state'])
    return model

def chat_and_train(input_text, cot_mode=False):
    model = get_session_model()
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}]
    if cot_mode:
        input_text = f"Let's think step by step to answer the user's question:\n\n{input_text}"
    messages.append({"role": "user", "content": input_text})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4.0-turbo",
            temperature=0.7,
            top_p=0.9,
            max_tokens=400,
            presence_penalty=0.3,
            frequency_penalty=0.3,
            messages=messages
        )
        model_response = response['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error communicating with OpenAI: {str(e)}")
        raise RuntimeError("Error communicating with OpenAI GPT-4 service")

    try:
        encodings = tokenizer(input_text + tokenizer.eos_token + model_response, return_tensors='pt')
        input_ids = encodings.input_ids.to(device)
        attn_mask = encodings.attention_mask.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, labels=input_ids, attention_mask=attn_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        session['model_state'] = model.state_dict()
        return model_response, loss.item()
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise RuntimeError("Error during model training")

@app.route('/')
@app.route('/index')
@login_required
def index():
    return render_template('index.html', title='Home')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.password_hash == form.password.data:
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data, password_hash=form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    user_input = data['input']
    cot_mode = data.get('cot_mode', False)
    try:
        response_text, loss = chat_and_train(user_input, cot_mode)
        chat = Chat(input_text=user_input, response_text=response_text, author=current_user)
        db.session.add(chat)
        db.session.commit()
        return jsonify({'response': response_text, 'loss': loss})
    except Exception as e:
        logging.error(f"Failed to process user input: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)  # 使用 request.args 而不是 args
    chats = current_user.chats.order_by(Chat.id.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('history', page=chats.next_num) \
        if chats.has_next else None
    prev_url = url_for('history', page=chats.prev_num) \
        if chats.has_prev else None
    return render_template('history.html', title='History', chats=chats.items,
                           next_url=next_url, prev_url=prev_url)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    app.run(debug=True)
