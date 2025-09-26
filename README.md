# Scriza AI Platform

Next-gen platform for creating, training, and deploying human-like AI agents.

## Project Structure

```
scriza_ai_platform/
│
├── apps/
│ ├── agents/
│ │ ├── agent_base.py # BaseAgent class (shared by all)
│ │ ├── sales/
│ │ │ ├── agent.py
│ │ │ ├── strategy_ingest.py
│ │ │ ├── call_handler.py
│ │ │ └── voice_handler.py
│ ├── users/
│ │ ├── models.py
│ │ ├── endpoints.py
│ │ ├── subscriptions.py
│ ├── admin_panel/
│ │ ├── user_management.py
│ │ ├── training_logs.py
│ │ ├── control_panel.py
│ ├── strategy_api/
│ │ ├── ingest_engine.py # Ingest strategy JSON/PDF
│ │ └── transformer.py # Turn into agent behavior
│ ├── integrations/
│ │ ├── telephony/
│ │ │ ├── twilio.py
│ │ ├── chat/
│ │ │ ├── telegram.py
│ │ └── crm/
│ │ ├── zoho.py
│ └── memory/
│ ├── memory_manager.py # Store & delete agent memory
│
├── core/
│ ├── config.py
│ ├── auth.py # JWT auth
│ ├── database.py # MongoDB setup
│ ├── scheduler.py # For future training/inference jobs
│
├── shared/
│ ├── utils.py
│ ├── constants.py
│ └── exceptions.py
│
├── tests/
├── scripts/
├── docker/
│ ├── Dockerfile
│ └── docker-compose.yml
└── main.py # FastAPI app entry
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start MongoDB:
```bash
docker-compose up -d mongodb
```

3. Initialize database:
```bash
python scripts/init_db.py
```

4. Start the application:
```bash
uvicorn main:app --reload
```

## API Endpoints

### Health Check
- `GET /health` - Health check endpoint

### User Management
- `POST /api/v1/user/register` - Register new user
- `POST /api/v1/user/login` - User login
- `GET /api/v1/user/profile` - Get user profile
- `PUT /api/v1/user/profile` - Update user profile

### Sales Agent
- `POST /api/v1/agent/sales/ingest_strategy` - Ingest strategy for sales agent
- `POST /api/v1/agent/sales/ingest_strategy_file` - Ingest strategy from file
- `POST /api/v1/agent/sales/place_call` - Place sales call
- `POST /api/v1/agent/sales/synthesize_voice` - Synthesize voice

### Admin Panel
- `GET /api/v1/admin/users` - List all users
- `GET /api/v1/admin/users/{user_id}` - Get specific user
- `PUT /api/v1/admin/users/{user_id}` - Update user
- `DELETE /api/v1/admin/users/{user_id}` - Delete user

## Environment Variables

Create a `.env` file with the following variables:

```env
SECRET_KEY=your-secret-key-here
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=scriza_db
ELEVENLABS_API_KEY=your-elevenlabs-api-key
TWILIO_ACCOUNT_SID=your-twilio-account-sid
TWILIO_AUTH_TOKEN=your-twilio-auth-token
TWILIO_PHONE_NUMBER=your-twilio-phone-number
ZOHO_CLIENT_ID=your-zoho-client-id
ZOHO_CLIENT_SECRET=your-zoho-client-secret
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
```

## Docker

To run the application with Docker:

```bash
docker-compose up
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

Follow PEP 8 guidelines. Use type hints for all function parameters and return values.

## License

This project is licensed under the MIT License.
